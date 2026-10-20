#!/usr/bin/env python3
"""
confusion_matrix_figure.py
==========================
Standalone version of FigureFactory.fig05_confusion_matrix() from
analyze_results.py: one row-normalised confusion matrix (predicted step vs
ground-truth step) per model, built from trial logs of any experiment.

Usage
-----
    python confusion_matrix_figure.py --log-dir sample_confusion
    python confusion_matrix_figure.py --log-dir my_logs --memory my_steps.json \
        --labels "hf_api=Llama 4 Scout,local=Qwen3 4B INT8" --vlm-label "Gemini" \
        --output my_figs/cm.pdf

Two input modes
---------------
  --log-dir D        raw per-trial logs (below); counts are computed here.
  --counts-json F    ONE json file with already-aggregated counts, no logs and
                     no memory file needed. A list with one entry per panel;
                     "matrix"[i][j] = calls with ground-truth step i+1 predicted
                     as step j+1 (rows = ground truth). "type" is "llm" (blue)
                     or "vlm" (purple) and only picks the colour map:
                       [{"title": "LLM — Llama 4 Scout", "type": "llm",
                         "matrix": [[5,2,0,0],[1,4,2,1],[0,0,7,0],[0,2,1,5]]},
                        {"title": "VLM — Gemini", "type": "vlm", "matrix": [...]}]

Input for --log-dir (same layout trial_logger.py writes, one pair per trial)
-----
    <tag>_summary.json       needs only "llm_backend" (the model that served
                             the LLM calls of that trial)
    <tag>_predictions.jsonl  one record per LLM/VLM call; fields used:
                               type                  "llm" | "vlm"
                               output.stage_of_assembly   free text
                               ground_truth_step     1..N | "none" | null
                               prediction_correct    true | false | "unclear" | null
                             Lines with an "event_type" key are ignored.

    <memory>.json            list of {"step number": int, "step description": str}.
                             Used to map stage_of_assembly text -> step number
                             by word overlap (as in analyze_results.py, but
                             punctuation is ignored and ties are dropped, see
                             --tie-break). The matrix is NxN, N = highest step
                             number in this file.
                             Default lookup: --memory, then learned_memory.json /
                             memory.json in --log-dir and its parent.

Rows are kept only if ground truth is a step number, the predicted text maps
to a step, and prediction_correct is true/false ("unclear" rows are dropped),
exactly as in the original figure.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

DPI = 300
AMBIGUOUS = -1          # resolve_step() result when several steps tie

DEFAULT_LABELS = {"hf_api": "Llama 4 Scout", "local": "Qwen3 4B INT8"}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "on", "in", "at", "with",
    "for", "by", "onto", "into", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "from",
}


# ── loading ──────────────────────────────────────────────────────────────────

def load_memory(explicit: str | None, log_dir: Path) -> list[dict]:
    candidates = [Path(explicit)] if explicit else []
    for name in ("learned_memory.json", "memory.json"):
        candidates += [log_dir / name, log_dir.parent / name]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                print(f"  memory: {p}")
                return json.load(f)
    raise FileNotFoundError("No memory file found; pass --memory <steps.json>")


def _tokens(text: str) -> set[str]:
    """Lower-cased words without punctuation ('motherboard.' == 'motherboard');
    underscores are kept so 'carburetor_body' stays one word."""
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower())
            if w not in _STOPWORDS}


def resolve_step(text: str, memory: list[dict], tie_break: str = "drop"):
    """Match stage_of_assembly text to a step number via word overlap.

    Returns None when nothing overlaps, or on a tie between steps when
    tie_break == "drop" (with "first" the lowest step number wins, which is
    what analyze_results.py does).
    """
    if not text:
        return None
    words = _tokens(text)
    best_steps, best_score = [], 0
    for step in memory:
        score = len(words & _tokens(step["step description"]))
        if score > best_score:
            best_score, best_steps = score, [step["step number"]]
        elif score == best_score and score > 0:
            best_steps.append(step["step number"])
    if not best_steps:
        return None
    if len(best_steps) > 1 and tie_break == "drop":
        return AMBIGUOUS
    return best_steps[0]


def to_step_int(v):
    if v is None or v == "none":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def to_bool(v):
    if v is True or v == "True":
        return True
    if v is False or v == "False":
        return False
    return None            # null / "unclear"


def load_predictions(log_dir: Path, memory: list[dict],
                     tie_break: str = "drop") -> list[dict]:
    """One dict per usable call: model key, gt step, predicted step."""
    rows = []
    n_total = n_unresolved = n_unannotated = n_ambiguous = 0
    jsonl_files = sorted(log_dir.glob("*_predictions.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No *_predictions.jsonl files in {log_dir}")

    for jf in jsonl_files:
        sf = Path(str(jf).replace("_predictions.jsonl", "_summary.json"))
        backend = "unknown"
        if sf.exists():
            with open(sf, encoding="utf-8") as f:
                backend = json.load(f).get("llm_backend", "hf_api")
        else:
            print(f"  WARNING: no summary for {jf.name}; LLM backend = 'unknown'")

        with open(jf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event_type") is not None:
                    continue
                n_total += 1
                gt = to_step_int(r.get("ground_truth_step"))
                ok = to_bool(r.get("prediction_correct"))
                if gt is None or ok is None:
                    n_unannotated += 1
                    continue
                pred = resolve_step(r.get("output", {}).get("stage_of_assembly", ""),
                                    memory, tie_break)
                if pred == AMBIGUOUS:
                    n_ambiguous += 1
                    continue
                if pred is None:
                    n_unresolved += 1
                    continue
                rows.append({"type": r.get("type"), "backend": backend,
                             "gt": gt, "pred": int(pred)})

    print(f"  calls: {n_total} total, {len(rows)} used, "
          f"{n_unannotated} not annotated / none / unclear, "
          f"{n_unresolved} text not mapped to a step, "
          f"{n_ambiguous} text tied between steps")
    return rows


# ── figure ───────────────────────────────────────────────────────────────────

def rows_to_matrix(rows, n_steps) -> np.ndarray:
    mat = np.zeros((n_steps, n_steps), dtype=int)
    for r in rows:
        gi, pi = r["gt"] - 1, r["pred"] - 1
        if 0 <= gi < n_steps and 0 <= pi < n_steps:
            mat[gi, pi] += 1
    return mat


def load_counts(path: Path) -> list[tuple]:
    """Panels from one aggregated JSON file (see module docstring)."""
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    panels = []
    for e in entries:
        mat = np.array(e["matrix"], dtype=int)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"'{e.get('title')}': matrix must be square, got {mat.shape}")
        cmap = "Purples" if e.get("type", "llm") == "vlm" else "Blues"
        panels.append((mat, cmap, e["title"]))
    sizes = {m.shape[0] for m, _, _ in panels}
    if len(sizes) > 1:
        raise ValueError(f"all matrices must have the same size, got {sorted(sizes)}")
    return panels


def build_panels(rows, labels, vlm_label, n_steps):
    panels = []
    llm = [r for r in rows if r["type"] == "llm"]
    for backend in sorted({r["backend"] for r in llm}):
        sub = [r for r in llm if r["backend"] == backend]
        panels.append((rows_to_matrix(sub, n_steps), "Blues",
                       f"LLM — {labels.get(backend, backend)}"))
    vlm = [r for r in rows if r["type"] == "vlm"]
    if vlm:
        panels.append((rows_to_matrix(vlm, n_steps), "Purples", f"VLM — {vlm_label}"))
    return panels


def plot_confusion(panels, note, out_path: Path):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.dpi": DPI, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    panel_w = 3.8
    fig_w = panel_w * len(panels) + 0.6 * len(panels)       # + colorbar space
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_w, panel_w + 0.6),
                             squeeze=False)
    axes = axes[0]

    for ax, (mat, cmap_name, title) in zip(axes, panels):
        n_steps = mat.shape[0]
        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(row_sums > 0, mat / row_sums, 0.0)

        im = ax.imshow(norm, cmap=cmap_name, vmin=0, vmax=1, aspect="equal")
        for i in range(n_steps):
            for j in range(n_steps):
                cnt, total = mat[i, j], int(row_sums[i, 0])
                pct = f"\n({cnt / total:.0%})" if total > 0 and cnt > 0 else ""
                ax.text(j, i, f"{cnt}{pct}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if norm[i, j] > 0.55 else "black")

        lbls = [f"S{i + 1}" for i in range(n_steps)]
        ax.set_xticks(range(n_steps)); ax.set_xticklabels(lbls)
        ax.set_yticks(range(n_steps)); ax.set_yticklabels(lbls)
        ax.set_xlabel("Predicted Step", fontsize=8)
        ax.set_ylabel("Ground Truth Step", fontsize=8)
        ax.set_title(f"{title}  (n={int(mat.sum())})")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Row-normalised", fontsize=5.5)
        cb.ax.tick_params(labelsize=5.5)

    fig.suptitle("Confusion Matrix: Predicted vs Ground Truth Step", fontsize=10)
    if note:
        axes[0].annotate(note, xy=(0.0, -0.20), xycoords="axes fraction",
                         fontsize=5.5, color="#888888", ha="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def parse_labels(spec: str | None) -> dict:
    labels = dict(DEFAULT_LABELS)
    if spec:
        for pair in spec.split(","):
            k, _, v = pair.partition("=")
            labels[k.strip()] = v.strip()
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--log-dir",
                     help="folder with *_summary.json / *_predictions.jsonl")
    src.add_argument("--counts-json",
                     help="one JSON file with aggregated confusion-matrix counts")
    ap.add_argument("--memory", default=None, help="step list JSON")
    ap.add_argument("--output", default="pc_motherboard_llm_confusion_matrix.pdf",
                    help="output file; extension picks the format (pdf/png/svg)")
    ap.add_argument("--labels", default=None,
                    help='backend=name pairs, e.g. "hf_api=Llama 4 Scout,local=Qwen3"')
    ap.add_argument("--vlm-label", default="Gemini")
    ap.add_argument("--tie-break", choices=("drop", "first"), default="drop",
                    help="stage text matching several steps equally: drop the call "
                         "(default) or take the lowest step like analyze_results.py")
    ap.add_argument("--note", default="", help="small grey footnote under panel 1")
    args = ap.parse_args()

    if args.counts_json:
        panels = load_counts(Path(args.counts_json))
    else:
        log_dir = Path(args.log_dir)
        memory = load_memory(args.memory, log_dir)
        n_steps = max(s["step number"] for s in memory)
        rows = load_predictions(log_dir, memory, args.tie_break)
        panels = build_panels(rows, parse_labels(args.labels), args.vlm_label, n_steps)
    if not panels:
        print("  [skip] no annotated predictions")
        return
    plot_confusion(panels, args.note, Path(args.output))


if __name__ == "__main__":
    main()
