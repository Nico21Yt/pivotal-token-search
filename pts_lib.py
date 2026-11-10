"""Pivotal Token Search (phi-4 technical report, arXiv 2412.08905, sec. 4.3) on HumanEval -- shared pieces.

p(success | prefix) is estimated by sampling continuations of  chat prompt + response prefix  and grading
each with the official HumanEval tests.  Generation stops at a stop token or as soon as the response's first
``` block is closed: the grader only reads the first fenced block, so nothing after it can change the
outcome (exactly equivalent, just shorter).

Everything is configured with environment variables (defaults in brackets):
    PTS_DATA        a capture run dir (data/<dataset>/[<lang>/]<model>, with samples.parquet + config.json):
                    task, model, revision, system prompt and sampler then default to that run's settings
    PTS_TASK        humaneval | humanevalx-python | humanevalx-cpp | humanevalx-java | humanevalx-js  [humaneval]
    PTS_MODEL       HF model id or local path              [allenai/OLMoE-1B-7B-0125-Instruct]
    PTS_REVISION    HF revision                             [pinned commit for the default model, else latest]
    PTS_DTYPE       bfloat16 | float16 | float32            [bfloat16]
    PTS_DEVICE      torch device                            [cuda:0 if available, else cpu]
    PTS_TRUST_REMOTE_CODE  1 to pass trust_remote_code      [0]
    PTS_SYSTEM      optional system prompt                  [none]
    PTS_WORK        output directory                        [work/<model name>]
    PTS_BUDGET      max response tokens                     [768]
    PTS_TEMP, PTS_TOP_P   sampler for all rollouts          [0.8, 0.95]
    PTS_N_SCREEN    draws per problem in the screen         [20]
    PTS_N_SEARCH    rollouts per prefix during the search   [32]
    PTS_N_CONFIRM   fresh rollouts before/after each pivot  [64]  (only with PTS_CONFIRM=1)
    PTS_P_GAP       |delta p| threshold                     [0.2]
    PTS_SEL_LO, PTS_SEL_HI  keep problems with SEL_LO <= p(success) <= SEL_HI   [0.2, 0.8]
    PTS_LIMIT       use only the first N problems (smoke tests)                  [all 164]
    PTS_B           rollouts per generation batch           [128]
    PTS_GRADE_WORKERS  parallel test processes             [8]
    PTS_TLIM        stop a job after this many seconds (resume later); 0 = none  [0]
"""
import hashlib
import json
import os
import re
import zlib
from concurrent.futures import ThreadPoolExecutor

import humaneval as he
import humanevalx as hx

HERE = os.path.dirname(os.path.abspath(__file__))
_E = os.environ.get
DATA = _E("PTS_DATA")
CFG = json.load(open(os.path.join(DATA, "config.json"))) if DATA else {}
TASK = _E("PTS_TASK") or ((CFG["dataset"] if CFG["dataset"] == "humaneval" else f"humanevalx-{CFG['language']}")
                          if CFG else "humaneval")
LANG = "python" if TASK == "humaneval" else TASK.split("-", 1)[1]
assert TASK == "humaneval" or (TASK.startswith("humanevalx-") and LANG in hx.LANGUAGES), f"unknown PTS_TASK {TASK}"
DEFAULT_MODEL = ("allenai/OLMoE-1B-7B-0125-Instruct", "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e")
MODEL = _E("PTS_MODEL") or CFG.get("model_id") or DEFAULT_MODEL[0]
REVISION = _E("PTS_REVISION") or (CFG.get("revision") if CFG.get("model_id") == MODEL else None) or \
    (DEFAULT_MODEL[1] if MODEL == DEFAULT_MODEL[0] else None)
DTYPE = _E("PTS_DTYPE") or CFG.get("dtype") or "bfloat16"
TRUST = _E("PTS_TRUST_REMOTE_CODE") == "1"
SYSTEM = _E("PTS_SYSTEM") or CFG.get("system_prompt") or None
WORK = _E("PTS_WORK") or os.path.join("work", f"{TASK}-{MODEL.rstrip('/').split('/')[-1]}")
BUDGET = int(_E("PTS_BUDGET") or CFG.get("max_new_tokens") or 768)
TEMP = float(_E("PTS_TEMP") or CFG.get("temperature") or 0.8)
TOP_P = float(_E("PTS_TOP_P") or CFG.get("top_p") or 0.95)
N_SCREEN = int(_E("PTS_N_SCREEN", "20"))
N_SEARCH = int(_E("PTS_N_SEARCH", "32"))
N_CONFIRM = int(_E("PTS_N_CONFIRM", "64"))
P_GAP = float(_E("PTS_P_GAP", "0.2"))
SEL_LO, SEL_HI = float(_E("PTS_SEL_LO", "0.2")), float(_E("PTS_SEL_HI", "0.8"))
LIMIT = int(_E("PTS_LIMIT", "0"))
B = int(_E("PTS_B", "128"))
TLIM = float(_E("PTS_TLIM", "0")) or float("inf")
GRADE_WORKERS = int(_E("PTS_GRADE_WORKERS", "8"))
# the reply's first fenced block is closed (humaneval's extractor takes python/py tags only)
CLOSED = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S) if TASK == "humaneval" else hx.closed_block(LANG)
os.makedirs(WORK, exist_ok=True)


# ------------------------------------------------------------------ model

def device():
    import torch
    return _E("PTS_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=TRUST)


def load_model(dev=None):
    import torch
    from transformers import AutoModelForCausalLM
    dev = dev or device()
    m = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=getattr(torch, DTYPE),
                                             trust_remote_code=TRUST, device_map={"": dev})
    return m.eval(), tokenizer()


def stops(tok, m=None):
    """Stop token ids; uses the checkpoint's generation_config (from the model, else from the hub/path)."""
    gc = getattr(m, "generation_config", None)
    if gc is None:
        try:
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=TRUST)
        except Exception:
            gc = None
    return he.stop_token_ids(tok, gc)


# ------------------------------------------------------------------ problems / prompts

def problems():
    return he.problems() if TASK == "humaneval" else hx.problems(LANG)


def items():
    P = list(problems())
    return P[:LIMIT] if LIMIT else P


def prompt_ids(tok, item):
    if TASK == "humaneval":
        return he.prompt_ids(tok, he.problems()[item], SYSTEM)
    return hx.prompt_ids(tok, hx.problems(LANG)[item], SYSTEM)


# ------------------------------------------------------------------ grading (memoised by program text)

_gcache = {}


def grade_many(items, texts, workers=16):
    """([passed], [code hash]) for (item, response text); identical extracted programs are run once."""
    H = problems()
    if TASK == "humaneval":
        ext, build, run = he.extract_code, he.build_program, (lambda progs: he.evaluate(progs, 10.0, workers))
    else:
        ext, build, run = (lambda t: hx.extract_code(t, LANG)), hx.build_program, \
            (lambda progs: hx.evaluate(LANG, progs, 10.0, workers))
    keys = [(i, ext(t)) for i, t in zip(items, texts)]
    todo = sorted({k for k in keys if k not in _gcache})
    if todo:
        for k, ok in zip(todo, run([build(H[i], c) for i, c in todo])):
            _gcache[k] = bool(ok)
    return [_gcache[k] for k in keys], [hashlib.md5(k[1].encode()).hexdigest()[:10] for k in keys]


class Grader:
    """Grade batches in a background thread while the GPU generates the next one."""

    def __init__(self, workers=None):
        workers = workers or GRADE_WORKERS
        self.pool = ThreadPoolExecutor(1)
        self.workers = workers
        self.pending = []

    def submit(self, meta, items, texts):
        self.pending.append((meta, self.pool.submit(grade_many, items, texts, self.workers)))

    def collect(self, wait=False):
        done, keep = [], []
        for meta, f in self.pending:
            if wait or f.done():
                passed, h = f.result()
                done.append((meta, passed, h))
            else:
                keep.append((meta, f))
        self.pending = keep
        return done


# ------------------------------------------------------------------ sampling

def backtick_ids(tok):
    return {i for i in range(len(tok)) if "`" in tok.decode([i])}


def _left_pad(seqs, pad_id):
    import torch
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s, dtype=torch.long)
        att[i, n - len(s):] = 1
    return ids, att


def _sample(logits, temperature, top_p, gen):
    import torch
    if not temperature:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_p is not None and top_p < 1.0:     # same rule as HF TopPLogitsWarper
        sl, si = torch.sort(logits, descending=False)
        remove = sl.softmax(-1).cumsum(-1) <= (1 - top_p)
        remove[..., -1:] = False
        logits = logits.masked_fill(remove.scatter(1, si, remove), float("-inf"))
    return torch.multinomial(logits.softmax(-1), 1, generator=gen).squeeze(1)


class _Stepper:
    """KV-cached forward over a left-padded batch."""

    def __init__(self, model):
        from transformers import DynamicCache
        self.m = model
        self.dev = next(model.parameters()).device
        self.cache = DynamicCache()

    def prefill(self, ids, att):
        import torch
        ids, att = ids.to(self.dev), att.to(self.dev)
        self.att = att
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        self.pos = pos[:, -1:]
        kw = dict(input_ids=ids, attention_mask=att, position_ids=pos, past_key_values=self.cache, use_cache=True)
        with torch.no_grad():
            try:
                out = self.m(**kw, logits_to_keep=1)      # only the last position's logits (saves memory)
            except TypeError:                             # older / remote-code models without logits_to_keep
                out = self.m(**kw)
        return out.logits[:, -1]

    def step(self, tok):
        import torch
        self.att = torch.cat([self.att, torch.ones_like(self.att[:, :1])], 1)
        self.pos = self.pos + 1
        with torch.no_grad():
            out = self.m(input_ids=tok.to(self.dev)[:, None], attention_mask=self.att, position_ids=self.pos,
                         past_key_values=self.cache, use_cache=True)
        return out.logits[:, -1]


def rollout(m, tok, prompt_ids, resp_prefix, max_new, seed, stops, bt):
    """One sampled continuation per row of  prompt_ids[r] + resp_prefix[r].  Returns [new token ids].

    A row stops at a stop token (kept), when its response (prefix + new) has a closed ``` block, or after
    max_new[r] tokens.  Finished rows are dropped from the KV cache once a quarter of the batch is done."""
    import torch
    seqs = [p + q for p, q in zip(prompt_ids, resp_prefix)]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids, att = _left_pad(seqs, pad)
    dev = next(m.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    T = _Stepper(m)
    logits = T.prefill(ids, att)
    out = [[] for _ in seqs]
    alive = list(range(len(seqs)))         # batch row -> original row
    fin = [False] * len(seqs)
    stops = set(stops)
    for _ in range(max(max_new)):
        nxt = _sample(logits, TEMP, TOP_P, gen)
        nl = nxt.tolist()
        for j, r in enumerate(alive):
            if fin[r]:
                continue
            t = nl[j]
            out[r].append(t)
            if t in stops or len(out[r]) >= max_new[r]:
                fin[r] = True
            elif t in bt and CLOSED.search(tok.decode(resp_prefix[r] + out[r])):
                fin[r] = True
        n_fin = sum(fin[r] for r in alive)
        if n_fin == len(alive):
            break
        if n_fin >= max(1, len(alive) // 4):
            keep = [j for j, r in enumerate(alive) if not fin[r]]
            idx = torch.tensor(keep, device=dev)
            T.cache.batch_select_indices(idx)
            T.att, T.pos = T.att[idx], T.pos[idx]
            nxt = nxt[idx]
            alive = [alive[j] for j in keep]
        else:
            nxt = torch.where(torch.tensor([fin[r] for r in alive], device=dev), torch.full_like(nxt, pad), nxt)
        logits = T.step(nxt)
    return out


def token_logprobs(m, prompt_ids, resp):
    """log p(resp[i] | prompt + resp[:i]) at temperature 1 (fp32 softmax)."""
    import torch
    x = torch.tensor([prompt_ids + resp], device=next(m.parameters()).device)
    with torch.no_grad():
        lg = m(input_ids=x).logits[0, len(prompt_ids) - 1:-1].float()
    return lg.log_softmax(-1).gather(1, x[0, len(prompt_ids):, None])[:, 0].tolist()


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def load_json(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default
