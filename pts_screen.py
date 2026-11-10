"""PTS step 1: estimate p(success) per problem and pick the trajectories to search along.

    python pts_screen.py from-samples             (CPU)  -> $PTS_WORK/select.json   from an existing run ($PTS_DATA)
  or, for a model without recorded samples:
    python pts_screen.py gen [<shard> <nshard>]   (GPU)  -> $PTS_WORK/screen_<shard>of<nshard>.parquet
    python pts_screen.py select                   (CPU)  -> $PTS_WORK/select.json

from-samples: the run's own draws (20 per problem in data/), with the prompt token ids they were sampled from.

gen: N_SCREEN fresh draws of every problem with the PTS sampler, graded in the job.  Resumable: rows are
saved after every batch and a rerun only does the missing draws.
select: problems with SEL_LO <= p(success) <= SEL_HI (too easy / too hard problems have no pivots); for
each, the lowest-numbered passing and failing draw become the two trajectories (T_full of the paper).
Each trajectory is cut after the token that closes its first ``` block (the grader reads only that block),
trailing stop tokens are dropped, and it is regraded here; a draw whose grade differs here is skipped.
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


def cut(ids, tok, st, bt):
    """Drop trailing stop tokens; end after the token that closes the first ``` block (grader reads only it)."""
    ids = [int(t) for t in ids]
    while ids and ids[-1] in st:
        ids = ids[:-1]
    k = next((k for k in range(1, len(ids) + 1) if ids[k - 1] in bt and P.CLOSED.search(tok.decode(ids[:k]))), len(ids))
    return ids[:k], len(ids)


def choose(s, tok):
    """s: one row per draw with item, draw, passed, ids [, prompt].  -> select.json.

    Problems with SEL_LO <= p <= SEL_HI; per problem and outcome, the lowest-numbered draw whose cut text
    regrades to its recorded grade here (a different toolchain can flip a grade; such draws are skipped)."""
    st, bt = set(P.stops(tok)), P.backtick_ids(tok)
    c = s.groupby("item").passed.agg(["sum", "count"])
    c["p"] = c["sum"] / c["count"]
    keep = c[(c.p >= P.SEL_LO) & (c.p <= P.SEL_HI) & (c["sum"] > 0) & (c["sum"] < c["count"])]
    order = {i: j for j, i in enumerate(P.problems())}
    queue = {}
    for item in sorted(keep.index, key=order.get):
        d = s[s.item == item].sort_values("draw")
        for outcome in (True, False):
            queue[(item, outcome)] = list(d[d.passed == outcome].itertuples())
    chosen, flipped = {}, 0
    while queue:
        cand = {key: rows.pop(0) for key, rows in queue.items()}
        cuts = {key: cut(r.ids, tok, st, bt) for key, r in cand.items()}
        keys = list(cand)
        g, _ = P.grade_many([k[0] for k in keys], [tok.decode(cuts[k][0], skip_special_tokens=True) for k in keys],
                            P.GRADE_WORKERS * 2)
        for key, ok in zip(keys, g):
            if ok == key[1]:
                chosen[key] = (cand[key], cuts[key])
                del queue[key]
            else:
                flipped += 1
                if not queue[key]:
                    del queue[key]
    sel = []
    for item in sorted(keep.index, key=order.get):
        if not ((item, True) in chosen and (item, False) in chosen):
            print(f"  {item}: no draw keeps its grade for one outcome here; problem skipped")
            continue
        for outcome in (True, False):
            r, (traj, n) = chosen[(item, outcome)]
            prompt = [int(t) for t in r.prompt] if "prompt" in s.columns else P.prompt_ids(tok, item)
            sel.append(dict(tid=f"{item.split('/')[1]}{'P' if outcome else 'F'}", item=item, draw=int(r.draw),
                            outcome=bool(outcome), p0=float(keep.p[item]), k0=int(keep["sum"][item]),
                            n0=int(keep["count"][item]), prompt=prompt, traj=traj, cut_from=n))
    P.json.dump(sel, open(f"{P.WORK}/select.json", "w"))
    print(f"{P.TASK}, {P.MODEL}: {len(c)} problems, pass@1 {c['sum'].sum() / c['count'].sum():.3f}; "
          f"{len(keep)} with {P.SEL_LO} <= p <= {P.SEL_HI} -> {len(sel)} trajectories "
          f"({sum(len(x['traj']) for x in sel) / max(len(sel), 1):.0f} tokens on average); "
          f"{flipped} draws skipped because their regrade differs here -> {P.WORK}/select.json")


def select():
    files = sorted(glob.glob(f"{P.WORK}/screen_*of*.parquet"))
    if not files:
        sys.exit(f"no screen_*.parquet in {P.WORK}: run `python pts_screen.py gen` first")
    s = pd.concat([pd.read_parquet(f) for f in files])
    short = s.groupby("item").size() < P.N_SCREEN
    if short.any():
        print(f"warning: {int(short.sum())} problems have fewer than {P.N_SCREEN} draws (unfinished screen?)")
    choose(s, P.tokenizer())


def from_samples():
    if not P.DATA:
        sys.exit("set PTS_DATA to a run dir, e.g. PTS_DATA=data/humanevalx/cpp/olmoe")
    s = pd.read_parquet(f"{P.DATA}/samples.parquet", columns=["task_id", "draw", "passed", "prompt_ids", "generated_ids"])
    s = s.rename(columns={"task_id": "item", "generated_ids": "ids", "prompt_ids": "prompt"})
    if P.LIMIT:
        s = s[s.item.isin(P.items())]
    choose(s, P.tokenizer())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "gen":
        a = sys.argv[2:] or ["0", "1"]
        gen(int(a[0]), int(a[1]))
    elif cmd == "select":
        select()
    elif cmd == "from-samples":
        from_samples()
    else:
        sys.exit(__doc__)
