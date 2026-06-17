#!/usr/bin/env python
"""Precompute the NN reference embeddings + labels for shipping.

Output: Drift-Detector/src/drift_detector/data/nn_reference_v0.npz
Contains:
  - embeddings: (N, 384) float32  (normalized for cosine via dot product)
  - labels:     (N,)   int8     (1 = failed, 0 = resolved)
  - sources:    (N,)   object   (for diagnostics / per-substrate analysis)
"""

from __future__ import annotations
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sentence_transformers import SentenceTransformer

TRAJ_ROOT = pathlib.Path("Drift-Evaluator/datasets/swebench_trajs")
SOURCES = ["livesweagent_opus45", "livesweagent_gemini3",
           "sonar_opus45", "sonar_sonnet45"]
OUT_PATH = pathlib.Path("Drift-Detector/src/drift_detector/data/nn_reference_v0.npz")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def msg_text(m):
    if "blocks" in m:
        parts = []
        for b in m.get("blocks") or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("block_type") or b.get("type")
            if bt == "text":
                parts.append(b.get("text") or "")
            elif bt in ("tool_use", "tool"):
                nm = b.get("name") or b.get("tool_name") or ""
                inp = b.get("input") or b.get("arguments") or ""
                parts.append(f"[{nm} {json.dumps(inp)[:300] if not isinstance(inp, str) else inp[:300]}]")
            elif bt in ("tool_result", "tool_output"):
                tr = b.get("content") or b.get("output") or ""
                if isinstance(tr, list):
                    tr = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in tr)
                parts.append(str(tr)[:1000])
        return "\n".join(parts)
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return json.dumps(c)
    return ""


def load_traj(path):
    d = json.load(path.open())
    if isinstance(d, dict) and "messages" in d:
        return d["messages"]
    if isinstance(d, list):
        return d
    return []


def trajectory_text(messages, max_chars=8000):
    lines = []
    for m in messages:
        text = msg_text(m)
        if not text:
            continue
        lines.append(f"{m.get('role', '?')}: {text[:1500]}")
    full = "\n".join(lines)
    return full[-max_chars:] if len(full) > max_chars else full


def main():
    rows = []
    for src in SOURCES:
        for suffix, label in [("_held_out", 0), ("_held_out_failed", 1)]:
            d = TRAJ_ROOT / f"{src}{suffix}"
            if not d.exists():
                continue
            for f in d.glob("*.json"):
                try:
                    msgs = load_traj(f)
                    if not msgs:
                        continue
                    text = trajectory_text(msgs)
                    if len(text) < 200:
                        continue
                    rows.append({"source": src, "y": label, "text": text})
                except Exception as e:
                    print(f"skip {f.name}: {e}", file=sys.stderr)

    print(f"loaded {len(rows)} trajectories")
    print(f"encoding with {MODEL_NAME}...")
    embedder = SentenceTransformer(MODEL_NAME)
    embeddings = embedder.encode([r["text"] for r in rows],
                                  batch_size=64, show_progress_bar=True,
                                  normalize_embeddings=True).astype(np.float32)
    labels = np.array([r["y"] for r in rows], dtype=np.int8)
    sources = np.array([r["source"] for r in rows], dtype=object)
    print(f"embeddings: {embeddings.shape}  labels: {labels.shape}  "
          f"failed: {int(labels.sum())} / resolved: {int((labels == 0).sum())}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH,
                        embeddings=embeddings, labels=labels, sources=sources,
                        model_name=MODEL_NAME)
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"saved: {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
