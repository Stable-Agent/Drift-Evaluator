#!/usr/bin/env python
"""Multi-seed SWE-bench agent runner + frontier selection (pilot).

Implements the pilot from docs/spec_multiseed_detector_v1.md: one agent
(minimal bash-loop), one model (qwen2.5-coder:14b via ollama, temp 0.7,
seed-controlled), K seeds over a difficulty-biased candidate pool. Produces,
per (instance, seed), a model_patch and a per-step cumulative-diff sidecar,
then evaluates with the official swebench harness and reports the per-problem
pass-rate distribution so we can keep the [0.25, 0.75] frontier band.

Generation runs in a LOCAL checkout (cheap, instrumentable); evaluation runs
in the swebench Docker harness (correct). The agent's env need not match the
eval env — it only emits a text patch, which is judged correctly in Docker.

Usage:
  # 1. generate (resumable): K seeds x N easy instances
  python gen_multiseed.py --n 120 --seeds 4 --max-steps 15
  # 2. evaluate all seeds via swebench harness + frontier report
  python gen_multiseed.py --eval
  # 3. just re-print the frontier analysis from existing reports
  python gen_multiseed.py --frontier-only
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, shutil, subprocess, sys, time
from collections import defaultdict

import requests

MODEL = os.environ.get("MULTISEED_MODEL", "qwen2.5-coder:14b")
# Backend via the OpenAI-compatible /v1/chat/completions API — works with BOTH
# ollama (local default) and vLLM (GPU host, the fast path: continuous batching
# serves concurrent rollouts in parallel). Point at vLLM with:
#   MULTISEED_BASE_URL=http://<host>:8000  MULTISEED_MODEL=SWE-bench/SWE-agent-LM-32B
BASE_URL = os.environ.get("MULTISEED_BASE_URL", "http://localhost:11434").rstrip("/")
API_KEY = os.environ.get("MULTISEED_API_KEY", "x")
GEN_TIMEOUT = int(os.environ.get("MULTISEED_GEN_TIMEOUT", "240"))
ROOT = pathlib.Path("Drift-Evaluator/datasets/multiseed_v1")
RUNS = ROOT / "runs.jsonl"                 # one row per (instance, seed)
STEPS_DIR = ROOT / "steps"                 # <inst>__s<seed>.json checkpoint sidecars
PREDS_DIR = ROOT / "preds"                 # per-seed predictions for the harness
REPORTS_DIR = ROOT / "reports"
WORK_LOGS = ROOT / "eval_logs"             # native eval logs per instance
CMD_TIMEOUT = 60
OUT_TRUNC = 2500


SYSTEM = """You are a software engineer fixing a bug in a checked-out git repo.
Each turn, reply with EXACTLY ONE action of one of these three forms:

(1) Explore with a read-only shell command:
```bash
grep -rn "something" path/    # or cat / ls / sed -n '10,40p' file
```

(2) Make an edit (this is how you change code — do NOT edit via bash):
EDIT <relative/path/to/file.py>
<<<SEARCH
<exact lines currently in the file, copied verbatim incl. indentation>
===
<the replacement lines>
>>>REPLACE
The SEARCH text must match the file EXACTLY (whitespace included) or the edit
is rejected. Use `cat`/`sed -n` first to copy the exact lines.

(3) Finish: reply with the single token TASK_COMPLETE (only after your edit
applied and `git diff` shows the intended fix).

Rules: explore to find the exact buggy lines, then EDIT them. Keep the fix
minimal. Do NOT run the test suite (hidden). One action per turn."""

USER0 = """Repository: {repo} (checked out at the buggy commit; cwd is repo root).

Bug report:
{problem}

Find the root cause and make the minimal source-code edit that fixes it.
Begin."""


def chat(messages, seed, temp=0.7, max_tokens=512):
    """OpenAI-compatible chat call. Backend-agnostic: ollama (/v1) or vLLM."""
    url = f"{BASE_URL}/v1/chat/completions"
    body = {"model": MODEL, "messages": messages, "temperature": temp,
            "seed": seed, "max_tokens": max_tokens, "stream": False}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    for attempt in range(3):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=GEN_TIMEOUT)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        time.sleep(3 + attempt * 3)
    return None

ollama_chat = chat   # back-compat alias


_CMD_RE = re.compile(r"```(?:bash|sh)?\s*\n?(.*?)```", re.DOTALL)
_FENCE_LINE = re.compile(r"^\s*```(?:bash|sh|python|py)?\s*$")

def extract_cmd(text: str) -> str | None:
    """Return the bash command to run, or None to stop. Robust to qwen's mess:
    a single fence, a stray empty fence before the real one, or a leftover
    ```bash line captured inside the block (which made bash exit 2). Also
    treats a bare/echoed TASK_COMPLETE as 'stop' so the loop doesn't spin."""
    # take the first fenced block whose content is non-empty (qwen sometimes
    # emits a stray empty ``` before the real one)
    cmd = ""
    for m in _CMD_RE.finditer(text or ""):
        lines = [ln for ln in m.group(1).splitlines() if not _FENCE_LINE.match(ln)]
        cand = "\n".join(lines).strip().strip("`").strip()
        if cand:
            cmd = cand
            break
    if not cmd:                       # no usable fenced block -> try raw text
        lines = [ln for ln in (text or "").splitlines() if not _FENCE_LINE.match(ln)]
        cmd = "\n".join(lines).strip().strip("`").strip()
    if not cmd:
        return None
    if re.fullmatch(r'(?is)\s*(echo\s+["\']?)?task[_ ]complete["\']?\.?\s*', cmd) \
            or cmd.upper().startswith("TASK_COMPLETE"):
        return None
    return cmd


def sh(args, cwd=None, timeout=120, inp=None):
    r = subprocess.run(args, cwd=cwd, input=inp, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


# ---------- Docker-backed agent environment (Linux toolchain == eval env) ----------
# The agent runs INSIDE the swebench instance container (/testbed = repo at
# base_commit, deps installed). This fixes the macOS/BSD tool mismatch the
# local-checkout pilot exposed (GNU `sed -i` etc.) and makes the agent's env
# identical to the eval env.
TESTBED = "/testbed"

# Image source, in priority:
#   ARCH=arm64  -> BUILD NATIVE arm64 with the Miniforge fix (Apple Silicon;
#                  no emulation, no GPU host — see reference-arm64-swebench-docker)
#   NAMESPACE set (x86 Linux) -> PULL prebuilt images (fast)
#   else -> local x86 build
# Default ARCH: arm64 on Apple Silicon, else x86_64.
import platform as _platform
ARCH = os.environ.get("MULTISEED_ARCH",
                      "arm64" if _platform.machine() in ("arm64", "aarch64") else "x86_64")
NAMESPACE = os.environ.get("MULTISEED_NAMESPACE", "swebench")
_MINIFORGE = ("https://github.com/conda-forge/miniforge/releases/latest/"
              "download/Miniforge3-Linux-aarch64.sh")

def _spec(inst):
    from swebench.harness.test_spec.test_spec import make_test_spec
    if ARCH == "arm64":
        return make_test_spec(inst, namespace=None, arch="arm64")
    ns = None if NAMESPACE in ("", "none", "None") else NAMESPACE
    return make_test_spec(inst, namespace=ns)

def docker_image_for(inst) -> str:
    return _spec(inst).instance_image_key

def _patch_miniforge(base_df: str) -> str:
    """Swap the broken Miniconda aarch64 installer for Miniforge (real conda
    that works native aarch64). See reference-arm64-swebench-docker."""
    out, ok = [], False
    for line in base_df.splitlines():
        if "miniconda.sh" in line and "wget" in line.lower():
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}RUN wget '{_MINIFORGE}' -O miniconda.sh \\")
            ok = True
        else:
            out.append(line)
    if not ok:
        raise RuntimeError("miniforge patch: no miniconda wget line found")
    return "\n".join(out)

def ensure_instance_image(inst):
    """Make the instance image available for the active ARCH."""
    import docker, pathlib as _pl
    cl = docker.from_env()
    s = _spec(inst)
    key = s.instance_image_key
    try:
        cl.images.get(key); return cl
    except Exception:
        pass
    if ARCH == "arm64":
        from swebench.harness.docker_build import build_image
        bd = _pl.Path("/tmp/multiseed/arm_build") / inst["instance_id"]
        bd.mkdir(parents=True, exist_ok=True)
        def have(k):
            try: cl.images.get(k); return True
            except Exception: return False
        if not have(s.base_image_key):
            build_image(s.base_image_key, {}, _patch_miniforge(s.base_dockerfile),
                        s.platform, cl, bd / "base")
        if not have(s.env_image_key):
            build_image(s.env_image_key, {"setup_env.sh": s.setup_env_script},
                        s.env_dockerfile, s.platform, cl, bd / "env")
        build_image(key, {"setup_repo.sh": s.install_repo_script},
                    s.instance_dockerfile, s.platform, cl, bd / "inst")
    elif NAMESPACE not in ("", "none", "None"):
        repo, _, tag = key.rpartition(":")
        cl.images.pull(repo, tag=tag or "latest")
    else:
        from swebench.harness.docker_build import build_instance_images
        build_instance_images(cl, [inst], force_rebuild=False, max_workers=4,
                              namespace=None, tag="latest", env_image_tag="latest")
    try:
        cl.images.get(key)
    except Exception:
        raise RuntimeError(f"instance image {key} unavailable after build/pull "
                           f"(arch={ARCH}). See reference-arm64-swebench-docker.")
    return cl

def start_container(cl, inst):
    name = "ms_" + inst["instance_id"].replace("__", "_")[:50]
    try:
        old = cl.containers.get(name); old.remove(force=True)
    except Exception:
        pass
    c = cl.containers.run(docker_image_for(inst), command="sleep infinity",
                          name=name, detach=True, working_dir=TESTBED)
    # baseline: discard anything but a clean base checkout
    c.exec_run(["bash", "-lc", "git reset --hard -q && git clean -fdxq"], workdir=TESTBED)
    return c

def dexec(container, cmd, timeout=CMD_TIMEOUT):
    """Run one shell command in the container; return (rc, output). `timeout`
    via the shell so a hung command can't wedge the run."""
    res = container.exec_run(
        ["bash", "-lc", f"timeout {timeout} bash -lc {shlex_quote(cmd)}"],
        workdir=TESTBED, demux=False)
    out = res.output.decode("utf-8", "replace") if res.output else ""
    return res.exit_code, out

def container_diff(container) -> str:
    res = container.exec_run(["bash", "-lc", "git diff"], workdir=TESTBED)
    return res.output.decode("utf-8", "replace") if res.output else ""

def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


# ---- structured SEARCH/REPLACE edit tool (robust apply via put_archive) ----
_EDIT_RE = re.compile(
    r"EDIT\s+(?P<path>\S+).*?<<<+\s*SEARCH\s*\n(?P<search>.*?)\n===+\s*\n"
    r"(?P<replace>.*?)\n>>>+\s*REPLACE", re.DOTALL)

def _strip_fences(s: str) -> str:
    """qwen often wraps SEARCH/REPLACE bodies in ```python fences — drop any
    fence-marker lines so the SEARCH text matches the real file verbatim."""
    return "\n".join(ln for ln in s.splitlines() if not _FENCE_LINE.match(ln))

def parse_edit(text: str):
    m = _EDIT_RE.search(text or "")
    if not m:
        return None
    return (m.group("path").strip().lstrip("./"),
            _strip_fences(m.group("search")), _strip_fences(m.group("replace")))

def read_file(container, path: str) -> str | None:
    res = container.exec_run(["bash", "-lc", f"cat {shlex_quote(path)}"], workdir=TESTBED)
    if res.exit_code != 0:
        return None
    return res.output.decode("utf-8", "replace")

def write_file(container, path: str, content: str):
    """Write via put_archive (a tar stream) — no shell quoting/length limits."""
    import io, tarfile, os as _os
    full = path if path.startswith("/") else f"{TESTBED}/{path}"
    data = content.encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        ti = tarfile.TarInfo(name=_os.path.basename(full))
        ti.size = len(data); ti.mode = 0o644
        tar.addfile(ti, io.BytesIO(data))
    buf.seek(0)
    container.put_archive(_os.path.dirname(full) or "/", buf.getvalue())

def apply_edit(container, path, search, replace) -> tuple[bool, str]:
    cur = read_file(container, path)
    if cur is None:
        return False, f"file not found: {path}"
    if search not in cur:
        return False, "SEARCH text not found verbatim (check exact whitespace)"
    write_file(container, path, cur.replace(search, replace, 1))
    return True, "edit applied"


def run_agent(repo, problem, container, seed, max_steps):
    """Minimal bash-loop agent running INSIDE the instance container. Returns
    (patch, steps) where steps[i] holds the cumulative diff after step i (the
    in-flight checkpoint trail)."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER0.format(repo=repo, problem=problem[:4000])}]
    steps = []
    nudged = False
    repeat_nudged = False
    recent: list[str] = []
    for i in range(max_steps):
        reply = ollama_chat(messages, seed=seed)
        if reply is None:
            steps.append({"step": i, "error": "model_call_failed"})
            break
        messages.append({"role": "assistant", "content": reply})

        # Action dispatch: EDIT block takes priority, then bash, then done.
        edit = parse_edit(reply)
        if edit is not None:
            path, search, replace = edit
            ok, msg = apply_edit(container, path, search, replace)
            diff_now = container_diff(container)
            steps.append({"step": i, "action": "edit", "path": path,
                          "applied": ok, "msg": msg, "cum_diff": diff_now})
            messages.append({"role": "user", "content":
                f"[edit {path}: {msg}]\n"
                f"git diff is now {'NON-EMPTY' if diff_now.strip() else 'EMPTY'}.\n"
                f"{'Explore and try again with exact lines.' if not ok else 'If the fix is complete, reply TASK_COMPLETE; else continue.'}"})
            continue

        cmd = extract_cmd(reply)
        diff_now = container_diff(container)
        if cmd is None:                       # claims done
            if not diff_now.strip() and not nudged:
                nudged = True
                messages.append({"role": "user", "content":
                    "`git diff` is EMPTY — you have not changed any file yet. "
                    "Do NOT say TASK_COMPLETE. Locate the buggy lines and make an "
                    "EDIT block now."})
                continue
            steps.append({"step": i, "done": True, "cum_diff": diff_now})
            break
        # loop-breaker: on a repeated command, first NUDGE toward editing;
        # only stop if it keeps repeating after the nudge.
        recent.append(cmd)
        if len(recent) >= 2 and recent[-1] == recent[-2]:
            if not repeat_nudged:
                repeat_nudged = True
                messages.append({"role": "user", "content":
                    "You are repeating the same command. You have located the "
                    "relevant code. STOP exploring and output an EDIT block now "
                    "(EDIT <path> / <<<SEARCH ... === ... >>>REPLACE) with the "
                    "exact lines to change."})
                continue
            if recent[-1] == recent[-2] == recent[-3]:
                steps.append({"step": i, "stuck": True, "cmd": cmd[:300], "cum_diff": diff_now})
                break
        rc, out = dexec(container, cmd)
        out = out[:OUT_TRUNC]
        diff_now = container_diff(container)
        steps.append({"step": i, "action": "bash", "cmd": cmd[:300], "rc": rc,
                      "cum_diff": diff_now})
        messages.append({"role": "user",
                         "content": f"[exit {rc}]\n{out}\n\n"
                         f"git diff is currently {'EMPTY' if not diff_now.strip() else 'non-empty'}. "
                         f"Make an EDIT block to fix the bug, explore more, or TASK_COMPLETE."})
    return container_diff(container), steps


# Light, pip-based env images that build under a modest Docker memory limit
# even via x86 emulation. Heavy conda/BLAS repos (astropy, matplotlib,
# scikit-learn, xarray, scipy, numpy) OOM-kill (exit 137) at ~5GB and are
# excluded by default — opt back in with --repos once Docker mem is raised.
LIGHT_REPOS = ("django/django", "sympy/sympy", "sphinx-doc/sphinx",
               "pytest-dev/pytest", "psf/requests", "pylint-dev/pylint",
               "pallets/flask", "mwaskom/seaborn", "psf/black")

def load_pool(n: int, repos: tuple[str, ...] | None = LIGHT_REPOS):
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    # Bias toward easy fixes so the agent has a chance to land in-band.
    easy = [r for r in ds if str(r.get("difficulty", "")).startswith("<15")]
    pool = easy if len(easy) >= n else list(ds)
    if repos:
        pool = [r for r in pool if r["repo"] in repos]
    pool.sort(key=lambda r: r["instance_id"])
    return pool[:n]


def generate(args):
    ROOT.mkdir(parents=True, exist_ok=True); STEPS_DIR.mkdir(exist_ok=True)
    done = set()
    if RUNS.exists():
        for l in open(RUNS):
            if l.strip():
                r = json.loads(l); done.add((r["instance_id"], r["seed"]))
    # On Linux+prebuilt images, no builds -> use the full easy pool (repos=None).
    # LIGHT_REPOS only matters for local arm64 builds (MULTISEED_NAMESPACE="").
    if args.repos:
        repos = tuple(s.strip() for s in args.repos.split(","))
    elif NAMESPACE in ("", "none", "None"):
        repos = LIGHT_REPOS              # local-build path: dodge OOM-heavy repos
    else:
        repos = None                     # prebuilt path: all repos
    pool = load_pool(args.n, repos)
    print(f"pool={len(pool)} instances x {args.seeds} seeds; {len(done)} done "
          f"(namespace={NAMESPACE or 'local-build'}, "
          f"repos={'custom' if args.repos else ('light' if repos else 'all')})", flush=True)
    # Phase 1: ensure all images (sequential — concurrent builds would race on
    # shared base/env layers). Skips instances whose image can't be built/pulled.
    import docker, threading
    from concurrent.futures import ThreadPoolExecutor
    cl = docker.from_env()
    ready = []
    for inst in pool:
        iid = inst["instance_id"]
        if all((iid, s) in done for s in range(args.seeds)):
            continue
        try:
            ensure_instance_image(inst); ready.append(inst)
        except Exception as e:
            print(f"  {iid}: image unavailable ({str(e)[:80]}); skip", file=sys.stderr)
    tasks = [(inst, s) for inst in ready for s in range(args.seeds)
             if (inst["instance_id"], s) not in done]
    print(f"images ready: {len(ready)}; rollouts to run: {len(tasks)} "
          f"(workers={args.workers}, backend={BASE_URL})", flush=True)

    # Phase 2: run rollouts concurrently — vLLM batches the parallel model calls.
    lock = threading.Lock()
    rf = RUNS.open("a")

    def one_rollout(inst, seed):
        iid, repo = inst["instance_id"], inst["repo"]
        cname = f"ms_{iid.replace('__','_')[:44]}_s{seed}"
        try:
            try: cl.containers.get(cname).remove(force=True)
            except Exception: pass
            c = cl.containers.run(_spec(inst).instance_image_key, command="sleep infinity",
                                  name=cname, detach=True, working_dir=TESTBED)
        except Exception as e:
            sys.stderr.write(f"  {iid} s{seed}: container start failed: {e}\n"); return
        try:
            c.exec_run(["bash", "-lc", "git reset --hard -q && git clean -fdxq"], workdir=TESTBED)
            t0 = time.time()
            patch, steps = run_agent(repo, inst["problem_statement"], c, seed, args.max_steps)
            with lock:
                (STEPS_DIR / f"{iid}__s{seed}.json").write_text(json.dumps(steps))
                rf.write(json.dumps({
                    "instance_id": iid, "repo": repo, "seed": seed,
                    "model": MODEL, "temperature": 0.7, "max_steps": args.max_steps,
                    "n_steps": len(steps), "patch": patch,
                    "empty_patch": not patch.strip(),
                    "secs": round(time.time() - t0, 1)}) + "\n")
                rf.flush()
                print(f"  {iid} s{seed}: {len(steps)} steps, "
                      f"{'EMPTY' if not patch.strip() else str(patch.count(chr(10)))+'L'} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        finally:
            try: c.remove(force=True)
            except Exception: pass

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda t: one_rollout(*t), tasks))
    rf.close()
    print("generation done.", flush=True)


EVAL_OUT = REPORTS_DIR / "eval_native.jsonl"   # {instance_id, seed, resolved}

def native_eval_one(cl, inst, patch: str) -> bool | None:
    """Grade one patch in a NATIVE container (works on arm64 and x86; bypasses
    the x86-defaulting run_evaluation CLI). Returns resolved bool, or None if
    the harness errored. Empty patch -> False (a real failed attempt)."""
    from swebench.harness.grading import get_eval_report
    if not (patch or "").strip():
        return False
    s = _spec(inst)
    c = cl.containers.run(s.instance_image_key, command="sleep infinity",
                          detach=True, working_dir=TESTBED)
    try:
        c.exec_run(["bash", "-lc", "git reset --hard -q && git clean -fdxq"], workdir=TESTBED)
        write_file(c, "/tmp/patch.diff", patch)
        ap = c.exec_run(["bash", "-lc", "cd /testbed && git apply -v /tmp/patch.diff"])
        if ap.exit_code != 0:                      # candidate doesn't apply = fail
            return False
        write_file(c, "/eval.sh", s.eval_script)
        res = c.exec_run(["bash", "-lc", "chmod +x /eval.sh && /eval.sh"], workdir=TESTBED)
        log = res.output.decode("utf-8", "replace") if res.output else ""
        lp = WORK_LOGS / f"{inst['instance_id']}.log"; lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(log)
        pred = {"instance_id": inst["instance_id"], "model_name_or_path": "ms",
                "model_patch": patch}
        report = get_eval_report(s, pred, str(lp), include_tests_status=True)
        return bool(report.get(inst["instance_id"], {}).get("resolved", False))
    except Exception as e:
        sys.stderr.write(f"  native_eval err {inst['instance_id']}: {e}\n")
        return None
    finally:
        try: c.remove(force=True)
        except Exception: pass


def evaluate(args):
    """Native in-container grading per (instance, seed). Resumable."""
    import docker
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(RUNS) if l.strip()]
    by_iid = {}
    pool = load_pool(10**6, None)              # instance metadata for specs
    meta = {r["instance_id"]: r for r in pool}
    done = set()
    if EVAL_OUT.exists():
        done = {(r["instance_id"], r["seed"]) for r in
                (json.loads(l) for l in open(EVAL_OUT) if l.strip())
                if r.get("resolved") is not None}
    import threading
    from concurrent.futures import ThreadPoolExecutor
    cl = docker.from_env()
    todo = [r for r in rows if (r["instance_id"], r["seed"]) not in done
            and r["instance_id"] in meta]
    # ensure images once (sequential), then grade concurrently
    for iid in {r["instance_id"] for r in todo}:
        try: ensure_instance_image(meta[iid])
        except Exception as e: print(f"  {iid}: image unavailable ({str(e)[:60]})", file=sys.stderr)
    lock = threading.Lock(); of = EVAL_OUT.open("a")
    print(f"native eval (arch={ARCH}, workers={args.max_workers}): "
          f"{len(todo)} to grade, {len(done)} done", flush=True)

    def grade(r):
        resolved = native_eval_one(cl, meta[r["instance_id"]], r["patch"])
        with lock:
            of.write(json.dumps({"instance_id": r["instance_id"], "seed": r["seed"],
                                 "resolved": resolved}) + "\n")
            of.flush()
            print(f"  {r['instance_id']} s{r['seed']}: "
                  f"{'RESOLVED' if resolved else ('error' if resolved is None else 'fail')}", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(grade, todo))
    of.close()
    frontier()


def frontier():
    """Aggregate per-seed harness reports -> per-instance pass rate -> band.

    DENOMINATOR = all ATTEMPTED seeds (from runs.jsonl), NOT just harness-scored
    ones. An empty patch is a real failed attempt; the harness simply doesn't
    score it, so dividing by scored-seeds would inflate any instance whose only
    non-empty seed happened to pass (e.g. 1 pass + 3 empty would read 1.0
    all-pass instead of the honest 0.25)."""
    resolved = defaultdict(set)
    attempted = defaultdict(set)       # ALL seeds we ran the agent on
    for r in (json.loads(l) for l in open(RUNS) if l.strip()):
        attempted[r["instance_id"]].add(r["seed"])
    if EVAL_OUT.exists():              # native grading results
        for r in (json.loads(l) for l in open(EVAL_OUT) if l.strip()):
            if r.get("resolved"):
                resolved[r["instance_id"]].add(r["seed"])
    if not attempted:
        print("no runs found (generate first)."); return
    rates = {iid: len(resolved[iid]) / len(attempted[iid]) for iid in attempted}
    band = sorted(iid for iid, p in rates.items() if 0.25 <= p <= 0.75)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "frontier_v1.json").write_text(json.dumps(
        {"rates": rates, "frontier_band": band,
         "n_attempted": {i: len(s) for i, s in attempted.items()},
         "n_resolved": {i: len(s) for i, s in resolved.items()}}, indent=2))
    hist = defaultdict(int)
    for p in rates.values():
        hist[round(p, 2)] += 1
    print(f"\n=== frontier selection (n={len(rates)} instances; denom=attempted seeds) ===")
    print("pass-rate histogram:")
    for p in sorted(hist):
        print(f"  {p:.2f}: {'#'*hist[p]} ({hist[p]})")
    print(f"\nall-fail (0.0): {sum(1 for p in rates.values() if p==0)}")
    print(f"all-pass (1.0): {sum(1 for p in rates.values() if p==1)}")
    print(f"FRONTIER BAND [0.25,0.75]: {len(band)} of {len(rates)} -> {band}")
    print("Gate (spec §7): need >=30 in-band to proceed to the full 8-seed run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="candidate instances")
    ap.add_argument("--seeds", type=int, default=4, help="K seeds (pilot=4)")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("MULTISEED_WORKERS", "1")),
                    help="concurrent rollouts (set high on a GPU host w/ vLLM batching)")
    ap.add_argument("--max-workers", type=int, default=4, help="eval workers")
    ap.add_argument("--repos", default="", help="comma list to override the "
                    "default light-repo allowlist (e.g. 'django/django')")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--frontier-only", action="store_true")
    args = ap.parse_args()
    if args.frontier_only:
        frontier()
    elif args.eval:
        evaluate(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
