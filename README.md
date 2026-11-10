# Pivotal Token Search on HumanEval

A small implementation of **Pivotal Token Search (PTS)** from the
[Phi-4 technical report](https://arxiv.org/abs/2412.08905) (sec. 4.3), applied to HumanEval
with the official unit tests. It works with any Hugging Face causal LM that has a chat template.
The default is `allenai/OLMoE-1B-7B-0125-Instruct`.

A *pivotal token* is a single token in a model's own response where the probability that the response ends
up correct, p(success | prefix), jumps by at least `p_gap`. p(success | prefix) is estimated by sampling
continuations of `prompt + prefix` and running the tests on each one.

## Pipeline

| step | script | device | output (in `$PTS_WORK`) |
|---|---|---|---|
| 1. screen | `pts_screen.py gen [shard nshard]` | GPU | `screen_*.parquet`: `N_SCREEN` draws per problem, graded |
| 1b. select | `pts_screen.py select` | CPU | `select.json`: problems with `0.2 ≤ p ≤ 0.8`; for each, the first passing and the first failing draw are the trajectories to search |
| 2. search | `pts_search.py [shard nshard]` | GPU | `roll_*.parquet` (every rollout), `logp_*.json`, `done_*` |
| 3. table | `pts_table.py` | CPU | `tables/pts_tokens.{md,csv}`, `tables/pts_problems.csv` |

### How the search follows the paper's pseudocode

```
Subdivide(prefix, T):
    if |T| <= 1 or |p(prefix) - p(prefix + T)| < p_gap:   return [T]
    T_left, T_right = Split(T)            # cumulative midpoint of the tokens' -log p
    return Subdivide(prefix, T_left) ∪ Subdivide(prefix + T_left, T_right)
pivots = single-token segments T with |p(prefix) - p(prefix + T)| >= p_gap
```

Implementation choices the pseudocode leaves open:

* **p(success | prefix)** = pass rate of `N_SEARCH` (32) sampled continuations (T = 0.8, top-p 0.95,
  768-token response budget). p(empty prefix) is sampled too. p(full trajectory) is that trajectory's own grade (0 or 1).
* **Split weights** are the trajectory's token log-probs at temperature 1, from one teacher-forced pass.
* **Memoised**: each prefix length is estimated once and reused by every recursion. All trajectories in a
  shard advance together, round by round, so the GPU always gets full batches.
* **Early stop**: generation stops as soon as the first ```` ``` ```` block is closed. The grader reads only that
  block, so this gives exactly the same outcome as generating to the end.
* **Optional fresh check** (`PTS_CONFIRM=1`): `N_CONFIRM` (64) new rollouts just before and just after each pivot
  re-measure Δp free of the search's selection (Fisher exact test). These rollouts also give the alternative
  next tokens and their pass rates. With only 32 samples per estimate the standard error is about 0.09 at p ≈ 0.5,
  so a Δp just above 0.2 is about 2 SE; the check tells real pivots from noise.

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, torch 2.6, transformers 4.57, datasets 3.2. The grader runs model-written code in a
subprocess with a timeout and a 4 GB memory cap, in a temporary directory (Linux; uses `resource`).
It does not run in a container, so use a machine where that is acceptable.

## Run

Single GPU, everything in one go:

```bash
export PTS_MODEL=allenai/OLMoE-1B-7B-0125-Instruct    # any chat model on the HF hub or a local path
export PTS_WORK=work/olmoe
python pts_screen.py gen
python pts_screen.py select
python pts_search.py
python pts_table.py
```

Several GPUs or cluster jobs: run shards `0..n-1` in parallel with the same `n`, for example
`python pts_search.py 3 12`. Every job saves after each batch and can be rerun to resume. Set `PTS_TLIM`
(seconds) to make a job stop cleanly before a wall-clock limit. A search shard is finished when `done_<s>of<n>` exists.
`pts_table.py` reads all shards.

Quick smoke test (a few minutes on CPU):

```bash
PTS_MODEL=ibm-granite/granite-3.1-3b-a800m-instruct PTS_DTYPE=float32 PTS_LIMIT=4 PTS_N_SCREEN=6 \
PTS_SEL_LO=0 PTS_SEL_HI=1 PTS_N_SEARCH=4 PTS_BUDGET=200 PTS_WORK=work/smoke \
  sh -c 'python pts_screen.py gen && python pts_screen.py select && python pts_search.py && python pts_table.py'
```

## Settings (environment variables)

| variable | default | meaning |
|---|---|---|
| `PTS_MODEL` | `allenai/OLMoE-1B-7B-0125-Instruct` | HF model id or local path |
| `PTS_REVISION` | pinned commit for the default model, else latest | HF revision |
| `PTS_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `PTS_DEVICE` | `cuda:0` if available, else `cpu` | torch device |
| `PTS_TRUST_REMOTE_CODE` | `0` | `1` for models that need it |
| `PTS_SYSTEM` | none | optional system prompt |
| `PTS_WORK` | `work/<model name>` | output directory |
| `PTS_BUDGET` | `768` | max response tokens |
| `PTS_TEMP`, `PTS_TOP_P` | `0.8`, `0.95` | sampler for all rollouts |
| `PTS_N_SCREEN` | `20` | draws per problem in the screen |
| `PTS_N_SEARCH` | `32` | rollouts per prefix in the search |
| `PTS_P_GAP` | `0.2` | Δp threshold |
| `PTS_SEL_LO`, `PTS_SEL_HI` | `0.2`, `0.8` | keep problems with p in this range |
| `PTS_CONFIRM`, `PTS_N_CONFIRM` | off, `64` | fresh check of each pivot |
| `PTS_LIMIT` | all 164 | use only the first N problems |
| `PTS_B` | `128` | rollouts per generation batch (lower it if you run out of GPU memory) |
| `PTS_TLIM` | none | stop a job after this many seconds |

The prompt asks the model, through its own chat template, for the complete function in one ```` ```python ````
block. The program that runs is the prompt's imports and helpers, the model's code, the official
test, and `check(entry_point)`.

## Output

`tables/pts_tokens.md` has one row per pivotal token:

| problem | traj | k | pivotal token | text before | p before → after (search) | fresh | Fisher p | ok | alt token (share, pass) |
|---|---|---|---|---|---|---|---|---|---|
| HumanEval/27 | P | 62 | `swap` | `ELLO'\n    """\n    return string.` | 0.38 → 1.00 | | | | |
| HumanEval/27 | F | 62 | `upper` | `ELLO'\n    """\n    return string.` | 0.31 → 0.00 | | | | |

(`traj` P / F = the passing / failing draw; `k` = position in the response.)
Each row of `roll_*.parquet` is one rollout: `tid` (trajectory), `k` (prefix length), `phase`
(`s` search, `cb` / `ca` fresh check before / after), `first` (first sampled token), `passed`.
