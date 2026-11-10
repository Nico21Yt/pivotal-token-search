# Pivotal Token Search on HumanEval and HumanEval-X

A small implementation of **Pivotal Token Search (PTS)** from the
[Phi-4 technical report](https://arxiv.org/abs/2412.08905) (sec. 4.3), applied to HumanEval and
HumanEval-X (Python, C++, Java, JavaScript) with the official unit tests. It works with any Hugging Face
causal LM that has a chat template. `data/` holds 20 recorded draws per problem from
`allenai/OLMoE-1B-7B-0125-Instruct`, so the search can start right away for that model.

A *pivotal token* is a single token in a model's own response where the probability that the response ends
up correct, p(success | prefix), jumps by at least `p_gap`. p(success | prefix) is estimated by sampling
continuations of `prompt + prefix` and running the tests on each one.

## Pipeline

| step | script | device | output (in `$PTS_WORK`) |
|---|---|---|---|
| 1. select | `pts_screen.py from-samples` (recorded draws, `$PTS_DATA`) or `pts_screen.py gen` + `pts_screen.py select` (new model) | CPU / GPU | `select.json`: problems with `0.2 ≤ p ≤ 0.8`; for each, the first passing and the first failing draw are the trajectories to search |
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
  768-token response budget; the recorded runs' own sampler). p(empty prefix) is sampled too.
  p(full trajectory) is that trajectory's own grade (0 or 1).
* **Split weights** are the trajectory's token log-probs at temperature 1, from one teacher-forced pass.
* **Memoised**: each prefix length is estimated once and reused by every recursion. All trajectories in a
  shard advance together, round by round, so the GPU always gets full batches.
* **Early stop**: generation stops as soon as the first fenced block is closed. The grader reads only that
  block, so this gives exactly the same outcome as generating to the end. Trajectories are cut at the same point.
* **Optional fresh check** (`PTS_CONFIRM=1`): `N_CONFIRM` (64) new rollouts just before and just after each pivot
  re-measure Δp free of the search's selection (Fisher exact test). These rollouts also give the alternative
  next tokens and their pass rates. With only 32 samples per estimate the standard error is about 0.09 at p ≈ 0.5,
  so a Δp just above 0.2 is about 2 SE; the check tells real pivots from noise.

## Data

`data/<dataset>/[<language>/]olmoe/` has `samples.parquet` (one row per draw: `task_id`, `draw`, `passed`,
`status`, `prompt_ids`, `generated_ids`, `completion`) and `config.json` (model, revision, sampler).
20 draws per problem, T = 0.8, top-p 0.95, 768 new tokens, bf16, transformers 4.48.

| run (`PTS_DATA`) | pass@1 | problems with 0.2 ≤ p ≤ 0.8 | trajectories |
|---|---|---|---|
| `data/humaneval/olmoe` | 0.352 | 37 | 74 |
| `data/humanevalx/python/olmoe` | 0.344 | 36 | 72 |
| `data/humanevalx/cpp/olmoe` | 0.204 | 26 | 52 |
| `data/humanevalx/java/olmoe` | 0.227 | 27 | 54 |
| `data/humanevalx/js/olmoe` | 0.240 | 30 | 60 |

HumanEval and HumanEval-X Python use the same prompts and tests but are separate samples.

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, torch 2.6, transformers 4.57, datasets 3.2. The grader runs model-written code in a
subprocess with a timeout and a memory cap, in a temporary directory (Linux). It does not run in a
container, so use a machine where that is acceptable.

HumanEval-X needs the toolchains (set the variables if they are not on `PATH`):

| language | needs | variables |
|---|---|---|
| C++ | g++ with C++17 (the recorded runs used 9.3), boost headers (CPP/22, 137), openssl `libcrypto` (CPP/162) | `HEX_CXX`, `HEX_CXX_INCLUDE` and `HEX_CXX_LIB` (`:`-separated dirs) |
| Java | a JDK (the recorded runs used OpenJDK 23) | `HEX_JAVA_HOME` (else `$JAVA_HOME`, else `java` on `PATH`) |
| JavaScript | Node (the recorded runs used 20.18.1), plus `npm install -g js-md5` (JS/162) | `HEX_NODE`, `HEX_NODE_PATH` |

Check them before a run. Every canonical solution should pass:

```bash
python humanevalx.py cpp java js python
```

## Run

### From the recorded OLMoE draws

```bash
export PTS_DATA=data/humanevalx/cpp/olmoe    # or data/humaneval/olmoe, data/humanevalx/{python,java,js}/olmoe
python pts_screen.py from-samples             # CPU: select.json (regrades every chosen trajectory here)
python pts_search.py                          # GPU
python pts_table.py                           # CPU
```

`PTS_DATA` sets the task, model, revision, system prompt and sampler from the run's `config.json`.
Results go to `work/<task>-<model>/` (or `$PTS_WORK`). If a recorded grade does not reproduce on your
toolchain, `from-samples` skips that draw and takes the next one with the same outcome, and it reports how many.

### Any other model

```bash
export PTS_MODEL=<HF model id or local path> PTS_TASK=humanevalx-java    # or humaneval, humanevalx-{python,cpp,js}
python pts_screen.py gen          # GPU: PTS_N_SCREEN (20) draws per problem, graded
python pts_screen.py select       # CPU
python pts_search.py
python pts_table.py
```

### Several GPUs or cluster jobs

Run shards `0..n-1` in parallel with the same `n`, for example `python pts_search.py 3 12` (also
`pts_screen.py gen 3 12`). Every job saves after each batch and can be rerun to resume. Set `PTS_TLIM`
(seconds) so that a job stops cleanly before its wall-clock limit. A search shard is finished when
`done_<s>of<n>` exists. `pts_table.py` reads all shards.

## Settings (environment variables)

| variable | default | meaning |
|---|---|---|
| `PTS_DATA` | none | recorded run dir; sets the defaults of the next rows |
| `PTS_TASK` | `humaneval` | `humaneval`, `humanevalx-python`, `humanevalx-cpp`, `humanevalx-java`, `humanevalx-js` |
| `PTS_MODEL` | `allenai/OLMoE-1B-7B-0125-Instruct` | HF model id or local path |
| `PTS_REVISION` | pinned commit for the default model, else latest | HF revision |
| `PTS_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `PTS_DEVICE` | `cuda:0` if available, else `cpu` | torch device |
| `PTS_TRUST_REMOTE_CODE` | `0` | `1` for models that need it |
| `PTS_SYSTEM` | none | optional system prompt |
| `PTS_WORK` | `work/<task>-<model name>` | output directory |
| `PTS_BUDGET` | `768` | max response tokens |
| `PTS_TEMP`, `PTS_TOP_P` | `0.8`, `0.95` | sampler for all rollouts |
| `PTS_N_SCREEN` | `20` | draws per problem in `gen` |
| `PTS_N_SEARCH` | `32` | rollouts per prefix in the search |
| `PTS_P_GAP` | `0.2` | Δp threshold |
| `PTS_SEL_LO`, `PTS_SEL_HI` | `0.2`, `0.8` | keep problems with p in this range |
| `PTS_CONFIRM`, `PTS_N_CONFIRM` | off, `64` | fresh check of each pivot |
| `PTS_LIMIT` | all 164 | use only the first N problems |
| `PTS_B` | `128` | rollouts per generation batch (lower it if you run out of GPU memory) |
| `PTS_GRADE_WORKERS` | `8` | parallel test processes (C++ / Java compile, so give the job enough CPUs) |
| `PTS_TLIM` | none | stop a job after this many seconds |

The prompt asks the model, through its own chat template, for the complete function in one fenced block
(```` ```python ````, ```` ```cpp ````, ```` ```java ```` or ```` ```javascript ````). What runs is the dataset's
imports and helpers, the model's code, and the dataset's own test.

## Output

`tables/pts_tokens.md` has one row per pivotal token, for example from OLMoE on HumanEval:

| problem | traj | k | pivotal token | text before | p before → after (search) | fresh | Fisher p | ok | alt token (share, pass) |
|---|---|---|---|---|---|---|---|---|---|
| HumanEval/27 | P | 62 | `swap` | `ELLO'\n    """\n    return string.` | 0.38 → 1.00 | | | | |
| HumanEval/27 | F | 62 | `upper` | `ELLO'\n    """\n    return string.` | 0.31 → 0.00 | | | | |

(`traj` P / F = the passing / failing draw; `k` = position in the response.)
Each row of `roll_*.parquet` is one rollout: `tid` (trajectory), `k` (prefix length), `phase`
(`s` search, `cb` / `ca` fresh check before / after), `first` (first sampled token), `passed`.
