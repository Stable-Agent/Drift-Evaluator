#!/usr/bin/env python
"""Validate arm64-native swebench eval: build one instance image natively with
the Miniforge fix, then confirm the GOLD patch RESOLVES via the official
harness grading. If gold resolves on arm64, the local fast path is trustworthy.

Fix: swebench's base Dockerfile installs Miniconda, whose aarch64 `constructor`
installer bootstrap (`_conda`) is broken -> exit 127. Miniforge provides a real
conda that works natively on aarch64; swap the installer line.

Strategy: build arm64 base->env->instance, tag the instance image as BOTH the
arm64 key AND the x86_64 key the harness computes, so run_evaluation reuses it
(skips its broken x86 build) and runs the container natively on arm64.
"""
from __future__ import annotations
import json, pathlib, subprocess, sys, time
import docker
from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_image

IID = sys.argv[1] if len(sys.argv) > 1 else "django__django-10914"
MINIFORGE = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"


def patch_miniforge(base_df: str) -> str:
    """Replace the broken Miniconda aarch64 installer with Miniforge."""
    import re
    out, replaced = [], False
    for line in base_df.splitlines():
        if "miniconda.sh" in line and "wget" in line.lower():
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}RUN wget '{MINIFORGE}' -O miniconda.sh \\")
            replaced = True
        else:
            out.append(line)
    assert replaced, "no miniconda wget line found to patch"
    return "\n".join(out)


def main():
    cl = docker.from_env()
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    inst = [r for r in ds if r["instance_id"] == IID][0]
    s = make_test_spec(inst, namespace=None, arch="arm64")
    bd = pathlib.Path(f"/tmp/arm_val/{IID}"); bd.mkdir(parents=True, exist_ok=True)

    def have(k):
        try: cl.images.get(k); return True
        except Exception: return False

    t0 = time.time()
    base_df = patch_miniforge(s.base_dockerfile)
    if not have(s.base_image_key):
        build_image(s.base_image_key, {}, base_df, s.platform, cl, bd / "base")
    print(f"base OK {time.time()-t0:.0f}s", flush=True)
    if not have(s.env_image_key):
        build_image(s.env_image_key, {"setup_env.sh": s.setup_env_script},
                    s.env_dockerfile, s.platform, cl, bd / "env")
    print(f"env OK {time.time()-t0:.0f}s", flush=True)
    if not have(s.instance_image_key):
        build_image(s.instance_image_key, {"setup_repo.sh": s.install_repo_script},
                    s.instance_dockerfile, s.platform, cl, bd / "inst")
    print(f"instance OK {time.time()-t0:.0f}s -> {s.instance_image_key}", flush=True)

    # NATIVE grading in the arm64 container (bypass the x86-defaulting CLI).
    from swebench.harness.grading import get_eval_report
    pred = {"instance_id": IID, "model_name_or_path": "gold",
            "model_patch": inst["patch"]}
    c = cl.containers.run(s.instance_image_key, command="sleep infinity",
                          detach=True, working_dir="/testbed")
    try:
        c.exec_run(["bash", "-lc", "git reset --hard -q && git clean -fdxq"], workdir="/testbed")
        # apply the GOLD model patch
        import io, tarfile
        def put(path, content):
            data = content.encode(); buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as t:
                ti = tarfile.TarInfo(pathlib.Path(path).name); ti.size = len(data); ti.mode = 0o644
                t.addfile(ti, io.BytesIO(data))
            buf.seek(0); c.put_archive(str(pathlib.Path(path).parent), buf.getvalue())
        put("/tmp/patch.diff", inst["patch"])
        ap = c.exec_run(["bash", "-lc", "cd /testbed && git apply -v /tmp/patch.diff"])
        print(f"gold patch apply rc={ap.exit_code}", flush=True)
        # run the official eval script (applies test_patch + runs F2P/P2P)
        put("/eval.sh", s.eval_script)
        res = c.exec_run(["bash", "-lc", "chmod +x /eval.sh && /eval.sh"], workdir="/testbed")
        log = res.output.decode("utf-8", "replace")
        log_fp = bd / "eval.log"; log_fp.write_text(log)
        report = get_eval_report(s, pred, str(log_fp), include_tests_status=True)
        r = report.get(IID, {})
        resolved = r.get("resolved", False)
        print(f"\n=== ARM64 GOLD CHECK: {'RESOLVED ✓' if resolved else 'NOT resolved ✗'} ===")
        ts = r.get("tests_status", {})
        f2p = ts.get("FAIL_TO_PASS", {}); p2p = ts.get("PASS_TO_PASS", {})
        print(f"FAIL_TO_PASS: {len(f2p.get('success',[]))} pass / {len(f2p.get('failure',[]))} fail")
        print(f"PASS_TO_PASS: {len(p2p.get('success',[]))} pass / {len(p2p.get('failure',[]))} fail")
    finally:
        c.remove(force=True)


if __name__ == "__main__":
    main()
