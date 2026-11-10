"""PTS step 3 (CPU): one row per pivotal token found, plus a per-trajectory summary.

    python pts_table.py       -> $PTS_WORK/tables/{pts_tokens.csv, pts_tokens.md, pts_problems.csv}

Per pivotal token: problem, trajectory (P = passing draw, F = failing draw), position k in the response,
the token, the text just before it, p(before)/p(after) from the search (N_SEARCH each) and, if the search ran
with PTS_CONFIRM=1, re-measured from fresh samples (N_CONFIRM each; Fisher exact two-sided p) together with
the most frequent other next token in the fresh 'cb' rollouts and its pass rate there.
`confirmed` = fresh delta has the same sign and |delta| >= P_GAP.
"""
import glob
import os

import pandas as pd
from scipy.stats import fisher_exact

import pts_lib as P
from pts_search import NEED, Traj, counts

OUT = os.path.join(P.WORK, "tables")


def show(s, n=None):
    s = s if n is None else s[-n:]
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("|", "\\|").replace("`", "ˋ")


def main():
    tok = P.tokenizer()
    sel = P.load_json(f"{P.WORK}/select.json")
    files = sorted(glob.glob(f"{P.WORK}/roll_*of*.parquet"))
    if not files:
        raise SystemExit(f"no roll_*.parquet in {P.WORK}: run `python pts_search.py` first")
    R = pd.concat([pd.read_parquet(f) for f in files])
    cnt = counts(R.to_dict("records"))
    done = len(glob.glob(f"{P.WORK}/done_*"))
    logp = {}
    for f in glob.glob(f"{P.WORK}/logp_*of*.json"):
        logp.update(P.load_json(f))
    rows, probs = [], []
    for d in sel:
        if d["tid"] not in logp:
            continue
        t = Traj(d, logp[d["tid"]])   # same split points as the search
        piv, miss = t.walk(cnt)
        probs.append(dict(item=d["item"], tid=t.tid, outcome="pass" if t.outcome else "fail", p0_screen=round(d["p0"], 3),
                          p0_search=t.p(cnt, 0), length=t.L, n_pivots=None if piv is None else len(piv),
                          finished=piv is not None, rollouts=int((R.tid == t.tid).sum())))
        for k, a, b in piv or []:
            def fresh(kk, ph):
                if kk == t.L:
                    return (int(t.outcome) * NEED[ph], NEED[ph])
                return cnt.get((t.tid, kk, ph), (0, 0))
            (xa, na), (xb, nb) = fresh(k, "cb"), fresh(k + 1, "ca")
            fa, fb = (xa / na if na else None), (xb / nb if nb else None)
            fp = fisher_exact([[xa, na - xa], [xb, nb - xb]])[1] if na and nb else None
            cb = R[(R.tid == t.tid) & (R.k == k) & (R.phase == "cb")]
            alt = cb[cb["first"] != t.traj[k]].groupby("first").passed.agg(["size", "mean"]).sort_values("size", ascending=False)
            same = cb[cb["first"] == t.traj[k]]
            rows.append(dict(
                item=d["item"], tid=t.tid, k=k, token=tok.decode([t.traj[k]]),
                context=tok.decode(t.traj[max(0, k - 12):k]),
                p_before=a, p_after=b, delta=b - a,
                p_before_fresh=fa, p_after_fresh=fb, delta_fresh=None if fa is None else fb - fa, fisher_p=fp,
                confirmed=None if fa is None else (fb - fa) * (b - a) > 0 and abs(fb - fa) >= P.P_GAP,
                own_share_fresh=len(same) / len(cb) if len(cb) else None,
                alt_token=tok.decode([int(alt.index[0])]) if len(alt) else None,
                alt_share=alt["size"].iloc[0] / len(cb) if len(alt) else None,
                alt_pass=alt["mean"].iloc[0] if len(alt) else None))
    T, Pr = pd.DataFrame(rows), pd.DataFrame(probs)
    os.makedirs(OUT, exist_ok=True)
    T.to_csv(f"{OUT}/pts_tokens.csv", index=False)
    Pr.to_csv(f"{OUT}/pts_problems.csv", index=False)
    f2 = lambda x: "" if x is None or pd.isna(x) else f"{x:.2f}"
    lines = ["| problem | traj | k | pivotal token | text before | p before → after (search) | fresh | Fisher p | ok | alt token (share, pass) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        alt = "" if r.alt_token is None else f"`{show(r.alt_token)}` ({r.alt_share:.2f}, {r.alt_pass:.2f})"
        lines.append(f"| {r.item} | {r.tid[-1]} | {r.k} | `{show(r.token)}` | `{show(r.context, 40)}` | "
                     f"{r.p_before:.2f} → {r.p_after:.2f} | {f2(r.p_before_fresh)} → {f2(r.p_after_fresh)} | "
                     f"{'' if r.fisher_p is None or pd.isna(r.fisher_p) else f'{r.fisher_p:.1g}'} | "
                     f"{'' if r.confirmed is None or pd.isna(r.confirmed) else '✓' if r.confirmed else '✗'} | {alt} |")
    open(f"{OUT}/pts_tokens.md", "w").write("\n".join(lines) + "\n")
    print(f"{done} shards done; {int(Pr.finished.sum()) if len(Pr) else 0}/{len(Pr)} trajectories finished; "
          f"{len(T)} pivotal tokens, {int((T.confirmed == True).sum()) if len(T) else 0} confirmed; "  # noqa: E712
          f"{len(R)} rollouts -> {OUT}")


if __name__ == "__main__":
    main()
