"""HumanEval: problems, chat prompt, code extraction and sandboxed grading (official `check()` tests).

What runs is  prompt imports/helpers + model code + the dataset's own test + check(entry_point), in a
subprocess with a timeout and a 4 GB address-space cap, inside a throwaway directory.  The program prints a
sentinel after check() returns, so a solution that calls sys.exit(0) is not scored as passing.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

DATASET = "openai/openai_humaneval"

# Asks for one fenced block and nothing else, so extraction is a single regexp.
INSTRUCTION = (
    "Complete the following Python function. Reply with the complete function "
    "implementation inside a single ```python code block, and nothing else.\n\n"
    "```python\n{prompt}```"
)

# Names that mean "the turn is over" across common instruct checkpoints (looked up in the added vocab).
END_OF_TURN_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>")

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.S)
_FENCE_MARKER = re.compile(r"^[ \t]*```(?:python|py)?[ \t]*", re.M)
SENTINEL = "__HUMANEVAL_OK__"
_PRELUDE = """import resource as _r
_r.setrlimit(_r.RLIMIT_AS, (4 * 1024**3,) * 2); _r.setrlimit(_r.RLIMIT_CORE, (0, 0))
"""


@dataclasses.dataclass
class Problem:
    index: int
    task_id: str        # "HumanEval/17"
    prompt: str         # imports, optional helpers, signature, docstring
    entry_point: str
    test: str           # official unit tests, defining check()

    @property
    def preamble(self) -> str:
        """Everything in the prompt before the `def entry_point` line (imports, helpers)."""
        m = re.search(rf"^def\s+{re.escape(self.entry_point)}\b", self.prompt, re.M)
        return self.prompt[: m.start()] if m else self.prompt


_problems = None


def problems() -> dict[str, Problem]:
    """{task_id: Problem} for all 164 problems, in dataset order."""
    global _problems
    if _problems is None:
        from datasets import load_dataset
        ds = load_dataset(DATASET, split="test")
        _problems = {r["task_id"]: Problem(i, r["task_id"], r["prompt"], r["entry_point"], r["test"])
                     for i, r in enumerate(ds)}
    return _problems


# ---------------------------------------------------------------- prompting

def prompt_ids(tok, problem: Problem, system_prompt: str | None = None) -> list[int]:
    """Chat-template prompt ids (add_special_tokens=False: the template already has them)."""
    messages = [{"role": "user", "content": INSTRUCTION.format(prompt=problem.prompt)}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def stop_token_ids(tok, generation_config=None) -> list[int]:
    """Every token that ends a turn: eos, known end-of-turn tokens, and the checkpoint's own eos list."""
    added = tok.get_added_vocab()
    ids = {tok.eos_token_id} | {added[n] for n in END_OF_TURN_TOKENS if n in added}
    ck = set()
    if generation_config is not None and generation_config.eos_token_id is not None:
        e = generation_config.eos_token_id
        ck = set(e) if isinstance(e, (list, tuple)) else {e}
    ids |= ck
    if tok.pad_token_id != tok.eos_token_id and tok.pad_token_id not in ck:
        ids.discard(tok.pad_token_id)
    return sorted(i for i in ids if i is not None)


# ---------------------------------------------------------------- extraction / grading

def extract_code(completion: str) -> str:
    """The first fenced block (an unterminated one counts), else the reply with fence markers cut out."""
    m = _FENCE.search(completion)
    text = m.group(1) if m else _FENCE_MARKER.sub("", completion)
    return text.strip("\n")


def build_program(problem: Problem, code: str) -> str:
    """Model code (with the prompt's imports if it redefines the function, else appended to the prompt)
    + the official test + check(entry_point) + sentinel."""
    defines = re.search(rf"^\s*def\s+{re.escape(problem.entry_point)}\b", code, re.M)
    body = problem.preamble + code if defines else problem.prompt + code
    return f"{body}\n\n{problem.test}\n\ncheck({problem.entry_point})\nprint(\"{SENTINEL}\")\n"


def check_correctness(program: str, timeout: float) -> bool:
    with tempfile.TemporaryDirectory(prefix="humaneval-") as tmp:
        path = pathlib.Path(tmp) / "program.py"
        path.write_text(_PRELUDE + program)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "TMPDIR": tmp,
               "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0 and SENTINEL in proc.stdout


def evaluate(programs: list[str], timeout: float = 10.0, workers: int = 16) -> list[bool]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda p: check_correctness(p, timeout), programs))
