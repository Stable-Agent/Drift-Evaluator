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

MODEL = "qwen2.5-coder:14b"
OLLAMA = "http://localhost:11434/api/chat"
ROOT = pathlib.Path("Drift-Evaluator/datasets/multiseed_v1")
RUNS = ROOT / "runs.jsonl"                 # one row per (instance, seed)
STEPS_DIR = ROOT / "steps"                 # <inst>__s<seed>.json checkpoint sidecars
PREDS_DIR = ROOT / "preds"                 # per-seed predictions for the harness
REPORTS_DIR = ROOT / "reports"
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


def ollama_chat(messages, seed, temp=0.7, num_predict=512):
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA, json={
                "model": MODEL, "messages": messages, "stream": False,
                "options": {"temperature": temp, "seed": seed,
                            "num_predict": num_predict}}, timeout=180)
            if r.status_code == 200:
                return r.json()["message"]["content"]
        except Exception:
            pass
        time.sleep(3 + attempt * 3)
    return None


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

# namespace: "swebench" pulls PREBUILT x86 images from Docker Hub (Linux host,
# the fast path — no building). None builds locally (the arm64-Mac path that
# OOMs under emulation; see reference-arm64-swebench-docker).
NAMESPACE = os.environ.get("MULTISEED_NAMESPACE", "swebench")

def docker_image_for(inst) -> str:
    from swebench.harness.test_spec.test_spec import make_test_spec
    ns = None if NAMESPACE in ("", "none", "None") else NAMESPACE
    return make_test_spec(inst, namespace=ns).instance_image_key

def ensure_instance_image(inst):
    """Make the instance image available. With NAMESPACE set (Linux x86),
    PULL the prebuilt image (fast, no build). With NAMESPACE empty, build
    locally (slow; OOMs under arm emulation — tag defaults are None in
    swebench 4.1.0 but make_test_spec requires them, so pass 'latest')."""
    import docker
    cl = docker.from_env()
    key = docker_image_for(inst)
    if NAMESPACE not in ("", "none", "None"):
        try:
            cl.images.get(key)
        except Exception:
            repo, _, tag = key.rpartition(":")
            cl.images.pull(repo, tag=tag or "latest")
    else:
        from swebench.harness.docker_build import build_instance_images
        build_instance_images(cl, [inst], force_rebuild=False, max_workers=4,
                              namespace=None, tag="latest", env_image_tag="latest")
    # swebench LOGS build failures without raising — verify the image exists,
    # else generate() would proceed on a phantom image.
    try:
        cl.images.get(key)
    except Exception:
        raise RuntimeError(
            f"instance image {key} unavailable (pull failed, or local build "
            f"OOM/exit-137 under emulation). See reference-arm64-swebench-docker.")
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
    rf = RUNS.open("a")
    for inst in pool:
        iid, repo = inst["instance_id"], inst["repo"]
        if all((iid, s) in done for s in range(args.seeds)):
            continue
        # Build instance image (once) + start one container per instance,
        # reset between seeds. Linux toolchain == eval env.
        try:
            cl = ensure_instance_image(inst)
        except Exception as e:
            print(f"  {iid}: image build failed ({e}); skip", file=sys.stderr)
            continue
        container = None
        try:
            container = start_container(cl, inst)
            for seed in range(args.seeds):
                if (iid, seed) in done:
                    continue
                container.exec_run(["bash", "-lc", "git reset --hard -q && git clean -fdxq"],
                                   workdir=TESTBED)
                t0 = time.time()
                patch, steps = run_agent(repo, inst["problem_statement"], container,
                                         seed, args.max_steps)
                (STEPS_DIR / f"{iid}__s{seed}.json").write_text(json.dumps(steps))
                rf.write(json.dumps({
                    "instance_id": iid, "repo": repo, "seed": seed,
                    "model": MODEL, "temperature": 0.7, "max_steps": args.max_steps,
                    "n_steps": len(steps), "patch": patch,
                    "empty_patch": not patch.strip(),
                    "secs": round(time.time() - t0, 1)}) + "\n")
                rf.flush()
                print(f"  {iid} s{seed}: {len(steps)} steps, "
                      f"{'EMPTY' if not patch.strip() else str(patch.count(chr(10)))+' diff lines'} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
    rf.close()
    print("generation done.", flush=True)


def evaluate(args):
    PREDS_DIR.mkdir(parents=True, exist_ok=True); REPORTS_DIR.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(RUNS) if l.strip()]
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    for seed, rs in sorted(by_seed.items()):
        pf = PREDS_DIR / f"preds_s{seed}.jsonl"
        with pf.open("w") as f:
            for r in rs:
                f.write(json.dumps({
                    "instance_id": r["instance_id"],
                    "model_name_or_path": f"multiseed_s{seed}",
                    "model_patch": r["patch"]}) + "\n")
        run_id = f"multiseed_s{seed}"
        print(f"=== eval seed {seed}: {len(rs)} preds -> harness run_id={run_id} ===", flush=True)
        cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
               "--dataset_name", "princeton-nlp/SWE-bench_Verified",
               "--predictions_path", str(pf), "--run_id", run_id,
               "--max_workers", str(args.max_workers), "--cache_level", "env"]
        if NAMESPACE not in ("", "none", "None"):
            cmd += ["--namespace", NAMESPACE]      # pull prebuilt x86 images
        subprocess.run(cmd)
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
    for rep in pathlib.Path(".").glob("multiseed_s*.multiseed_s*.json"):
        data = json.loads(rep.read_text())
        seed = int(re.search(r"s(\d+)", rep.name).group(1))
        for iid in data.get("resolved_ids", []):
            resolved[iid].add(seed)
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
    ap.add_argument("--max-workers", type=int, default=4, help="harness workers")
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
