"""HumanEval-X (Python, C++, Java, JavaScript): the 164 HumanEval problems in four languages.

Dataset: THUDM/humaneval-x at a pinned revision, read straight from its jsonl files.  Same chat instruction
as humaneval.py (word for word for Python), with the language name and fence tag swapped.

What runs is  dataset header + model code + the dataset's own test, in a temporary directory, in a
subprocess, with a timeout:
  * header: the import / #include lines of the dataset's `declaration`, plus any helper function the
    declaration defines that the model's code does not define itself.
  * model code: the first fenced block of the reply.  A `main` the model wrote (C++ `int main`, Java
    `class Main`) is removed, since the test brings its own; `public class Solution` loses `public`.
    A reply that does not define the entry point is a bare body and is appended to the dataset prompt.
  * test: unmodified, except that it cannot pass silently: JavaScript's console.assert is made to throw,
    JavaScript/32,119,151 get the missing call of their test function, and every program prints a sentinel
    after the test returns, so a program that exits early with status 0 does not count as passing.

Toolchains (environment variables, defaults in brackets):
    HEX_CXX            C++ compiler                       [g++ on PATH]  (built with -std=c++17 -O2)
    HEX_CXX_INCLUDE    extra -isystem dirs, ':'-separated  [none]  boost (CPP/22, CPP/137) and openssl
    HEX_CXX_LIB        extra -L dirs, ':'-separated        [none]  libcrypto (CPP/162)
    HEX_JAVA_HOME      JDK                                 [$JAVA_HOME, else the java on PATH]
    HEX_NODE           node binary                         [node on PATH]  (JavaScript/162 needs js-md5)
    HEX_NODE_PATH      NODE_PATH for require()             [<node prefix>/lib/node_modules]
The original runs used g++ 9.3 + boost 1.73 + openssl 1.1.1k, OpenJDK 23 and Node 20.18.1.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

DATASET = "THUDM/humaneval-x"
REVISION = "62c78627f3072a1454fa0cb0184737cafe5e4198"
LANGUAGES = ("python", "cpp", "java", "js")
LANG_NAME = {"python": "Python", "cpp": "C++", "java": "Java", "js": "JavaScript"}
FENCE_TAG = {"python": "python", "cpp": "cpp", "java": "java", "js": "javascript"}
FENCE_ALIASES = {                     # tags a reply may open its block with
    "python": r"python3?|py",
    "cpp": r"cpp|c\+\+|cc|cxx|c",
    "java": r"java",
    "js": r"javascript|js|node|jsx",
}
INSTRUCTION = (
    "Complete the following {lang} function. Reply with the complete function "
    "implementation inside a single ```{tag} code block, and nothing else.\n\n"
    "```{tag}\n{prompt}```"
)
SENTINEL = "__HUMANEVAL_OK__"
_PRELUDE = """import resource as _r
_r.setrlimit(_r.RLIMIT_AS, (4 * 1024**3,) * 2); _r.setrlimit(_r.RLIMIT_CORE, (0, 0))
"""


def _java_home():
    if os.environ.get("HEX_JAVA_HOME"):
        return os.environ["HEX_JAVA_HOME"]
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    j = shutil.which("java")
    return str(pathlib.Path(j).resolve().parents[1]) if j else "/usr"


def _split(v):
    return [x for x in (os.environ.get(v) or "").split(":") if x]


TOOLS = {
    "cxx": os.environ.get("HEX_CXX") or shutil.which("g++") or "g++",
    "java_home": _java_home(),
    "node": os.environ.get("HEX_NODE") or shutil.which("node") or "node",
    "python": os.environ.get("HEX_PYTHON") or sys.executable,
}
JVM_FLAGS = ("-Xmx1g", "-Xss16m", "-XX:+UseSerialGC", "-XX:TieredStopAtLevel=1")
COMPILE_TIMEOUT = 120.0


def cxx_args(program: str) -> list[str]:
    args = ["-std=c++17", "-O2", "-w"]
    for d in _split("HEX_CXX_INCLUDE"):
        args += ["-isystem", d]
    args += [f"-L{d}" for d in _split("HEX_CXX_LIB")]
    if "openssl" in program:          # CPP/162 (md5); linking libcrypto everywhere would fail without it
        args.append("-lcrypto")
    return args


@dataclasses.dataclass
class Problem:
    index: int              # 0..163, same numbering as HumanEval
    task_id: str            # "CPP/17"
    language: str
    prompt: str
    declaration: str
    entry_point: str
    test: str

    @property
    def header(self) -> str:
        return "".join(ln for ln in self.declaration.splitlines(keepends=True) if _is_header_line(self.language, ln))

    @property
    def helpers(self) -> list[tuple[str, str]]:
        return _helpers(self)


_cache = {}


def problems(language: str) -> dict[str, Problem]:
    if language not in _cache:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET, f"data/{language}/data/humaneval.jsonl", repo_type="dataset", revision=REVISION)
        rows = [json.loads(line) for line in open(path)]
        _cache[language] = {r["task_id"]: Problem(i, r["task_id"], language, r["prompt"], r["declaration"],
                                                  entry_point(language, r["declaration"]), r["test"])
                            for i, r in enumerate(rows)}
    return _cache[language]


# ---------------------------------------------------------------- parsing the declaration

_DEF = {
    "python": re.compile(r"^def\s+(\w+)\s*\(", re.M),
    "cpp": re.compile(r"^[A-Za-z_][\w:<>,\s\*&]*?[\s\*&](\w+)\s*\([^;{]*\)\s*\{", re.M),
    "java": re.compile(r"^[ \t]*(?:(?:public|private|protected|static|final|synchronized)\s+)*"
                       r"[\w<>\[\],.? ]+?\s(\w+)\s*\([^;{]*\)\s*(?:throws\s+[\w.,\s]+)?\{", re.M),
    # top level only: a body may shadow the entry point (JavaScript/106 declares `let f`)
    "js": re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=|^(?:export\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*\(", re.M),
}


def _names(language, text):
    return [(next(g for g in m.groups() if g), m.start()) for m in _DEF[language].finditer(text)]


def entry_point(language, declaration):
    names = _names(language, declaration)
    if not names:
        raise ValueError(f"no function in {language} declaration:\n{declaration}")
    return names[-1][0]


def _is_header_line(language, line):
    s = line.strip()
    if language == "cpp":
        return s.startswith(("#", "using ", "typedef "))      # CPP/22: typedef std::list<boost::any>
    if language == "java":
        return s.startswith("import ")
    if language == "python":
        return s.startswith("import ") or s.startswith("from ")
    return s.startswith("const ") and "require(" in s


def _helpers(p):
    names = _names(p.language, p.declaration)
    out = []
    for (name, start), (_, end) in zip(names, names[1:]):
        src = p.declaration[start:end]
        if p.language == "cpp":
            src = "".join(ln for ln in src.splitlines(keepends=True) if not _is_header_line("cpp", ln))
        out.append((name, src.rstrip() + "\n"))
    return out


def defines(language, code, name):
    return any(n == name for n, _ in _names(language, code))


# ---------------------------------------------------------------- prompting / extraction

def prompt_ids(tok, problem: Problem, system_prompt=None) -> list[int]:
    text = INSTRUCTION.format(lang=LANG_NAME[problem.language], tag=FENCE_TAG[problem.language], prompt=problem.prompt)
    messages = [{"role": "user", "content": text}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False)["input_ids"]


def closed_block(language):
    """Regexp that matches once the reply's first fenced block is closed."""
    return re.compile(rf"```(?:{FENCE_ALIASES[language]})?[ \t]*\n(.*?)```", re.S | re.I)


def extract_code(completion: str, language: str) -> str:
    """First fenced block (an unterminated one counts), else the reply without fence markers."""
    tags = FENCE_ALIASES[language]
    m = re.search(rf"```(?:{tags})?[ \t]*\n(.*?)(?:```|\Z)", completion, re.S | re.I)
    if m:
        return m.group(1).strip("\n")
    return re.sub(rf"^[ \t]*```(?:{tags})?[ \t]*", "", completion, flags=re.M | re.I).strip("\n")


# ---------------------------------------------------------------- program assembly

def _skip_literal(src, i):
    if src.startswith("//", i):
        j = src.find("\n", i)
        return len(src) if j < 0 else j
    if src.startswith("/*", i):
        j = src.find("*/", i + 2)
        return len(src) if j < 0 else j + 2
    if src[i] in "\"'`":
        q, j = src[i], i + 1
        while j < len(src) and src[j] != q:
            j += 2 if src[j] == "\\" else 1
        return j + 1
    return i


def _block_end(src, open_brace):
    depth, i = 0, open_brace
    while i < len(src):
        j = _skip_literal(src, i)
        if j != i:
            i = j
            continue
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(src)


def _remove_blocks(src, pattern):
    while True:
        m = re.search(pattern, src, re.M)
        if not m:
            return src
        brace = src.find("{", m.end())
        if brace < 0:
            return src[: m.start()]
        src = src[: m.start()] + src[_block_end(src, brace):]


def _balance_braces(src):
    depth, i = 0, 0
    while i < len(src):
        j = _skip_literal(src, i)
        if j != i:
            i = j
            continue
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src + "\n}" * max(depth, 0)


def build_program(problem: Problem, code: str) -> str:
    lang, entry = problem.language, problem.entry_point
    if lang == "python":
        m = re.search(rf"^def\s+{re.escape(entry)}\b", problem.prompt, re.M)
        defines_entry = re.search(rf"^\s*def\s+{re.escape(entry)}\b", code, re.M)
        return _with_test(problem, (problem.prompt[: m.start()] if defines_entry else problem.prompt) + code)
    if lang == "java":
        has_entry = re.search(r"\bclass\s+Solution\b", code) is not None
    else:
        has_entry = defines(lang, code, entry)
    if not has_entry:
        body = problem.header + _balance_braces(problem.prompt + code)
    else:
        if lang == "cpp":
            code = _remove_blocks(code, r"^[ \t]*(?:int|void|signed)\s+main\s*\([^)]*\)\s*")
        elif lang == "java":
            code = _remove_blocks(code, r"^[ \t]*(?:public\s+)?(?:final\s+)?class\s+Main\b[^{]*")
            code = re.sub(r"\bpublic\s+(?=(?:final\s+)?class\s+Solution\b)", "", code)
        missing = [(n, s) for n, s in problem.helpers if not defines(lang, code, n)]
        if lang == "java" and missing:
            m = re.search(r"\bclass\s+Solution\b[^{]*\{", code)
            code = code[: m.end()] + "\n" + "".join(s for _, s in missing) + code[m.end():]
            missing = []
        if lang == "java":
            imports = "".join(ln for ln in code.splitlines(keepends=True) if ln.strip().startswith("import "))
            rest = "".join(ln for ln in code.splitlines(keepends=True) if not ln.strip().startswith("import "))
            code = imports + rest
        body = problem.header + "".join(s for _, s in missing) + "\n" + code
    return _with_test(problem, body)


def _with_test(problem, body):
    lang, test = problem.language, problem.test
    if lang == "python":
        return f"{body}\n\n{test}\n\nprint(\"{SENTINEL}\")\n"
    if lang == "js":
        guard = ("console.assert = (cond, ...msg) => { if (!cond) throw new Error("
                 "'Assertion failed' + (msg.length ? ': ' + msg.join(' ') : '')); };\n")
        m = re.search(r"^const\s+(test\w*)\s*=", test, re.M)
        if m and not re.search(rf"^\s*{m.group(1)}\s*\(\s*\)", test, re.M):
            test += f"\n{m.group(1)}()\n"
        return f"{guard}{body}\n\n{test}\n\nconsole.log(\"{SENTINEL}\");\n"
    if lang == "cpp":
        test, n = re.subn(r"\bint\s+main\s*\(\s*\)", "int __humanevalx_test_main()", test)
        if n != 1:
            raise ValueError(f"{problem.task_id}: expected one main() in the test")
        brace = test.find("{", test.find("__humanevalx_test_main()"))
        end = _block_end(test, brace)
        test = test[: end - 1] + "    return 0;\n}" + test[end:]
        return (f"{body}\n\n{test}\n\n#include<cstdio>\n"
                f"int main(){{ __humanevalx_test_main(); std::puts(\"{SENTINEL}\"); return 0; }}\n")
    test, n = re.subn(r"\bpublic\s+class\s+Main\b", "class HumanEvalXTest", test)
    if n != 1:
        raise ValueError(f"{problem.task_id}: expected one public class Main in the test")
    imports = "".join(ln for ln in body.splitlines(keepends=True) if ln.strip().startswith("import "))
    rest = "".join(ln for ln in body.splitlines(keepends=True) if not ln.strip().startswith("import "))
    return (f"{imports}{rest}\n\n{test}\n\n"
            "public class Main {\n    public static void main(String[] args) throws Exception {\n"
            f"        HumanEvalXTest.main(args);\n        System.out.println(\"{SENTINEL}\");\n    }}\n}}\n")


# ---------------------------------------------------------------- execution

def _run(cmd, cwd, timeout):
    node_path = os.environ.get("HEX_NODE_PATH") or str(
        pathlib.Path(shutil.which(TOOLS["node"]) or TOOLS["node"]).resolve().parents[1] / "lib" / "node_modules")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": cwd, "TMPDIR": cwd,
           "OMP_NUM_THREADS": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "NODE_PATH": node_path}
    if os.environ.get("LD_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, errors="replace")
    except subprocess.TimeoutExpired:
        return None


def check_correctness(language: str, program: str, timeout: float) -> bool:
    with tempfile.TemporaryDirectory(prefix=f"humanevalx-{language}-", ignore_cleanup_errors=True) as tmp:
        if language == "python":
            (pathlib.Path(tmp) / "program.py").write_text(_PRELUDE + program)
            run = [TOOLS["python"], "-I", "program.py"]
        elif language == "js":
            (pathlib.Path(tmp) / "program.js").write_text(program)
            run = [TOOLS["node"], "--max-old-space-size=2048", "--stack-size=8000", "program.js"]
        elif language == "cpp":
            (pathlib.Path(tmp) / "program.cpp").write_text(program)
            c = _run([TOOLS["cxx"], "program.cpp", "-o", "program", *cxx_args(program)], tmp, COMPILE_TIMEOUT)
            if c is None or c.returncode != 0:
                return False
            run = ["/bin/sh", "-c", "ulimit -v 4194304; ulimit -s 65536; exec ./program"]
        elif language == "java":
            (pathlib.Path(tmp) / "Main.java").write_text(program)
            jh = TOOLS["java_home"]
            c = _run([f"{jh}/bin/javac", "-J-XX:+UseSerialGC", "-J-XX:TieredStopAtLevel=1", "-nowarn",
                      "-encoding", "UTF-8", "Main.java"], tmp, COMPILE_TIMEOUT)
            if c is None or c.returncode != 0:
                return False
            run = [f"{jh}/bin/java", *JVM_FLAGS, "-cp", ".", "Main"]
        else:
            raise ValueError(language)
        proc = _run(run, tmp, timeout)
    return proc is not None and proc.returncode == 0 and SENTINEL in proc.stdout


def evaluate(language: str, programs: list[str], timeout: float = 10.0, workers: int = 16) -> list[bool]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda p: check_correctness(language, p, timeout), programs))


def check_toolchain(language: str) -> list[str]:
    """Problems whose canonical solution fails here (should be empty)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET, f"data/{language}/data/humaneval.jsonl", repo_type="dataset", revision=REVISION)
    rows = [json.loads(line) for line in open(path)]
    P = problems(language)
    # prompt + canonical (declaration + canonical breaks Python/142 and CPP/38: their declarations are cut mid-body)
    progs = [build_program(P[r["task_id"]], r["prompt"] + r["canonical_solution"]) for r in rows]
    ok = evaluate(language, progs)
    return [r["task_id"] for r, o in zip(rows, ok) if not o]


if __name__ == "__main__":      # python humanevalx.py cpp java js  -> canonical solutions must all pass
    for lang in sys.argv[1:] or LANGUAGES:
        bad = check_toolchain(lang)
        print(f"{lang}: {164 - len(bad)}/164 canonical solutions pass" + (f"; failing: {bad}" if bad else ""))
