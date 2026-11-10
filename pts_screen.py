"""PTS step 1: estimate p(success) per HumanEval problem and pick the trajectories to search along.

    python pts_screen.py gen [<shard> <nshard>]   (GPU)  -> $PTS_WORK/screen_<shard>of<nshard>.parquet
    python pts_screen.py select                   (CPU)  -> $PTS_WORK/select.json

gen: N_SCREEN fresh draws of every problem with the PTS sampler, graded in the job.  Resumable: rows are
saved after every batch and a rerun only does the missing draws.
select: problems with SEL_LO <= p(success) <= SEL_HI (too easy / too hard problems have no pivots); for
each, the lowest-numbered passing and failing draw become the two trajectories (T_full of the paper).
A draw already ends at its first closed ``` block (see pts_lib), trailing stop tokens are dropped, and the
trajectory is regraded to make sure it keeps the draw's grade.
"""
import glob
import os
import sys
import time

import pandas as pd

import pts_lib as P


def gen(shard, nshard):
    t00 = time.time()
    m, tok = P.load_model()
    st, bt = P.stops(tok, m), P.backtick_ids(tok)
    its = [i for j, i in enumerate(P.items()) if j % nshard == shard]
    pids = {i: P.prompt_ids(tok, i) for i in its}
    out = f"{P.WORK}/screen_{shard}of{nshard}.parquet"
    rows = pd.read_parquet(out).to_dict("records") if os.path.exists(out) else []
    have = {(r["item"], r["draw"]) for r in rows}
    work = sorted([(i, d) for i in its for d in range(P.N_SCREEN) if (i, d) not in have],
                  key=lambda x: (len(pids[x[0]]), x))
    print(f"shard {shard}/{nshard}: {len(its)} problems, {len(work)} draws to do ({len(rows)} saved)", flush=True)
    G, slow = P.Grader(), 0.0

    def flush(wait):
        for (w, outs), passed, h in G.collect(wait):
            for (i, d), o, ok, hh in zip(w, outs, passed, h):
                rows.append(dict(item=i, draw=d, ids=o, passed=bool(ok), code=hh))
        pd.DataFrame(rows).to_parquet(out)

    for c in range(0, len(work), P.B):
        if time.time() - t00 + 1.1 * slow > P.TLIM:
            print("time limit: stopping (rerun to resume)", flush=True)
            break
        t0 = time.time()
        w = work[c:c + P.B]
        outs = P.rollout(m, tok, [pids[i] for i, _ in w], [[] for _ in w], [P.BUDGET] * len(w),
                         P.seed_of("screen", shard, w[0], w[-1]), st, bt)
        G.submit((w, outs), [i for i, _ in w], [tok.decode(o, skip_special_tokens=True) for o in outs])
        flush(False)
        slow = max(slow, time.time() - t0)
        print(f"  batch {c // P.B}: {len(w)} draws, {time.time() - t0:.0f}s (total {time.time() - t00:.0f}s)",
              flush=True)
    flush(True)
    print(f"total {time.time() - t00:.0f}s, {len(rows)} draws saved", flush=True)


def select():
    tok = P.tokenizer()
    st = set(P.stops(tok))
    files = sorted(glob.glob(f"{P.WORK}/screen_*of*.parquet"))
    if not files:
        sys.exit(f"no screen_*.parquet in {P.WORK}: run `python pts_screen.py gen` first")
    s = pd.concat([pd.read_parquet(f) for f in files])
    c = s.groupby("item").passed.agg(["sum", "count"])
    short = c[c["count"] < P.N_SCREEN]
    if len(short):
        print(f"warning: {len(short)} problems have fewer than {P.N_SCREEN} draws (unfinished screen?)")
    c["p"] = c["sum"] / c["count"]
    keep = c[(c.p >= P.SEL_LO) & (c.p <= P.SEL_HI) & (c["sum"] > 0) & (c["sum"] < c["count"])]
    sel = []
    order = {i: j for j, i in enumerate(P.he.problems())}
    for item in sorted(keep.index, key=order.get):
        d = s[s.item == item].sort_values("draw")
        for outcome in (True, False):
            r = d[d.passed == outcome].iloc[0]
            ids = [int(t) for t in r.ids]
            while ids and ids[-1] in st:
                ids = ids[:-1]
            sel.append(dict(tid=f"{item.split('/')[1]}{'P' if outcome else 'F'}", item=item, draw=int(r.draw),
                            outcome=bool(outcome), p0=float(keep.p[item]), k0=int(keep["sum"][item]),
                            n0=int(keep["count"][item]), prompt=P.prompt_ids(tok, item), traj=ids))
    g, _ = P.grade_many([x["item"] for x in sel], [tok.decode(x["traj"], skip_special_tokens=True) for x in sel])
    bad = [x["tid"] for x, ok in zip(sel, g) if ok != x["outcome"]]
    assert not bad, f"trajectories change grade after dropping stop tokens: {bad}"
    P.json.dump(sel, open(f"{P.WORK}/select.json", "w"))
    print(f"{len(c)} problems screened, pass@1 {c['sum'].sum() / c['count'].sum():.3f}; "
          f"{len(keep)} with {P.SEL_LO} <= p <= {P.SEL_HI} -> {len(sel)} trajectories, "
          f"{sum(len(x['traj']) for x in sel) / max(len(sel), 1):.0f} tokens on average")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        a = sys.argv[2:] or ["0", "1"]
        gen(int(a[0]), int(a[1]))
    else:
        select()
