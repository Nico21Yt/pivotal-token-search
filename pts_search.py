"""PTS step 2 (GPU): Pivotal Token Search along every selected trajectory, then an optional fresh-sample check.

    python pts_search.py [<shard> <nshard>]     -> $PTS_WORK/roll_<shard>of<nshard>.parquet (+ done_<shard>of<nshard>)

Search (the paper's Fig. 4, memoised): Subdivide(prefix, T) stops on |T| <= 1 or |p(prefix) - p(prefix+T)|
< P_GAP, else splits T at the cumulative midpoint of the tokens' -log p (temperature 1) and recurses on both
halves.  A one-token segment with |delta p| >= P_GAP is a pivotal token.  p(prefix) = passes / N_SEARCH
continuations of prompt + prefix ('s' rows); p(full trajectory) is its own grade, exactly.  All trajectories
of the shard advance together: each round collects every p the recursions need next and samples them in
batches of PTS_B rows.
Check (PTS_CONFIRM=1): for each pivotal token at position k, N_CONFIRM fresh continuations from prefix[:k]
('cb', also gives the alternative next tokens) and from prefix[:k+1] ('ca'), so delta p is re-measured free
of the search's selection.  Resumable: rows are saved after every batch; a rerun tops up missing counts.
"""
import os
import sys
import time

import pandas as pd

import pts_lib as P

NEED = {"s": P.N_SEARCH, "cb": P.N_CONFIRM, "ca": P.N_CONFIRM}
CONFIRM = os.environ.get("PTS_CONFIRM") == "1"


def split(w, lo, hi):
    """Split point of [lo, hi): where the cumulative -log p first reaches half of the segment's total."""
    seg = [max(0.0, -x) for x in w[lo:hi]]
    tot = sum(seg)
    if tot <= 1e-9:
        return (lo + hi) // 2
    cum, mid = 0.0, hi - 1
    for j, x in enumerate(seg):
        cum += x
        if cum >= tot / 2:
            mid = lo + j + 1
            break
    return min(max(mid, lo + 1), hi - 1)


class Traj:
    def __init__(self, d, logp):
        self.__dict__.update(d)
        self.L = len(self.traj)
        self.w = logp

    def p(self, cnt, k, phase="s"):
        if k == self.L:
            return 1.0 if self.outcome else 0.0
        kp, n = cnt.get((self.tid, k, phase), (0, 0))
        return kp / n if n >= NEED[phase] else None

    def walk(self, cnt):
        """-> (pivots [(k, p_before, p_after)], missing prefix lengths).  Pivots only once nothing is missing."""
        segs, miss = [], set()

        def sub(lo, hi):
            if hi - lo <= 1:
                segs.append((lo, hi))
                return
            a, b = self.p(cnt, lo), self.p(cnt, hi)
            if a is None or b is None:
                miss.update(k for k, v in ((lo, a), (hi, b)) if v is None)
                return
            if abs(a - b) < P.P_GAP:
                segs.append((lo, hi))
                return
            mid = split(self.w, lo, hi)
            sub(lo, mid)
            sub(mid, hi)

        sub(0, self.L)
        for lo, hi in segs:
            if hi - lo == 1:
                miss.update(k for k in (lo, hi) if self.p(cnt, k) is None)
        if miss:
            return None, sorted(miss)
        piv = [(lo, self.p(cnt, lo), self.p(cnt, hi)) for lo, hi in segs
               if hi - lo == 1 and abs(self.p(cnt, lo) - self.p(cnt, hi)) >= P.P_GAP]
        return piv, []


def counts(rows):
    cnt = {}
    for r in rows:
        k = (r["tid"], r["k"], r["phase"])
        a, n = cnt.get(k, (0, 0))
        cnt[k] = (a + r["passed"], n + 1)
    return cnt


def requests(T, cnt):
    """[(tid, k, phase, n more)] for the next round; empty when the shard is finished."""
    req = []
    for t in T:
        piv, miss = t.walk(cnt)
        for k in miss:
            req.append((t.tid, k, "s", NEED["s"] - cnt.get((t.tid, k, "s"), (0, 0))[1]))
        for k, _, _ in (piv or []) if CONFIRM else []:
            for kk, ph in ((k, "cb"), (k + 1, "ca")):
                if kk < t.L:
                    n = NEED[ph] - cnt.get((t.tid, kk, ph), (0, 0))[1]
                    if n > 0:
                        req.append((t.tid, kk, ph, n))
    return req


def main(shard, nshard):
    t00 = time.time()
    if not os.path.exists(f"{P.WORK}/select.json"):
        sys.exit(f"no select.json in {P.WORK}: run `python pts_screen.py select` first")
    sel = [d for j, d in enumerate(P.load_json(f"{P.WORK}/select.json")) if j % nshard == shard]
    out = f"{P.WORK}/roll_{shard}of{nshard}.parquet"
    rows = pd.read_parquet(out).to_dict("records") if os.path.exists(out) else []
    lpf = f"{P.WORK}/logp_{shard}of{nshard}.json"
    logp = P.load_json(lpf, {})
    print(f"shard {shard}/{nshard}: {len(sel)} trajectories, {len(rows)} rollouts so far", flush=True)
    m, tok = P.load_model()
    for d in sel:
        if d["tid"] not in logp:
            logp[d["tid"]] = P.token_logprobs(m, d["prompt"], d["traj"])
    P.json.dump(logp, open(lpf, "w"))
    T = [Traj(d, logp[d["tid"]]) for d in sel]
    by = {t.tid: t for t in T}
    stops, bt = P.stops(tok, m), P.backtick_ids(tok)
    print(f"load {time.time() - t00:.0f}s", flush=True)
    G = P.Grader()
    slow, rnd = 0.0, 0

    def flush(wait):
        for (work, outs), passed, h in G.collect(wait):
            for (tid, k, ph), o, ok, hh in zip(work, outs, passed, h):
                rows.append(dict(tid=tid, k=k, phase=ph, first=o[0] if o else -1, n_new=len(o), passed=bool(ok), code=hh))
        pd.DataFrame(rows).to_parquet(out)

    stopped = False
    while not stopped:
        req = requests(T, counts(rows))
        if not req:
            open(f"{P.WORK}/done_{shard}of{nshard}", "w").write(time.strftime("%m-%d %H:%M\n"))
            print("shard finished", flush=True)
            break
        work = sorted([(tid, k, ph) for tid, k, ph, n in req for _ in range(n)],
                      key=lambda x: (len(by[x[0]].prompt) + x[1], x))
        print(f"round {rnd}: {len(req)} prefixes, {len(work)} rollouts "
              f"({sum(r[2] != 's' for r in req)} check prefixes)", flush=True)
        for c in range(0, len(work), P.B):
            if time.time() - t00 + 1.1 * slow > P.TLIM:
                print("time limit: stopping (rerun to resume)", flush=True)
                stopped = True
                break
            t0 = time.time()
            w = work[c:c + P.B]
            pre = [by[tid].traj[:k] for tid, k, _ in w]
            maxn = [P.BUDGET - k for _, k, _ in w]
            outs = P.rollout(m, tok, [by[tid].prompt for tid, _, _ in w], pre, maxn,
                             P.seed_of("pts", shard, len(rows), c, w[0], w[-1]), stops, bt)
            G.submit((w, outs), [by[tid].item for tid, _, _ in w],
                     [tok.decode(p + o, skip_special_tokens=True) for p, o in zip(pre, outs)])
            flush(False)
            slow = max(slow, time.time() - t0)
            print(f"  batch {c // P.B}: {len(w)} rows, max new {max(map(len, outs))}, {time.time() - t0:.0f}s "
                  f"(total {time.time() - t00:.0f}s)", flush=True)
        flush(True)
        rnd += 1
    print(f"total {time.time() - t00:.0f}s, {len(rows)} rollouts", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:] or ["0", "1"]
    main(int(a[0]), int(a[1]))
