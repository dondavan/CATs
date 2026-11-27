#!/usr/bin/env python
"""Visualise CATs early-exit training.

Reads `trainer_state.json` (per-step/epoch log history) and `all_results.json`
(final metrics) from one or more output dirs and writes PNG figures:

  1. loss.png             train loss per step + eval loss per epoch
  2. metric_by_epoch.png  per-layer eval metric (pearson/accuracy) over epochs
  3. final_by_layer.png   final per-layer metric / consistency / meta accuracy
                          (multiple runs overlaid -> e.g. cls vs mean pooling)

Usage:
  python plot_training.py RUN_DIR [RUN_DIR2 ...] [--labels cls mean] [--out plots/]
  python plot_training.py ../das_results/cat_model_stsb_cls ../das_results/cat_model_stsb_mean --labels cls mean --out plots/
  python plot_training.py --all ../das_results/smoke --out plots/smoke   # one subfolder per cat_model_* run
  python plot_training.py --all ../das_results/ --out plots/
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def layer_series(d, prefix):
    """Collect {layer_idx: value} for keys like f'{prefix}_{i}'."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    out = {}
    for k, v in d.items():
        m = pat.match(k)
        if m and isinstance(v, (int, float)):
            out[int(m.group(1))] = v
    return dict(sorted(out.items()))


def pick_metric(d):
    for name in ("eval_pearson", "eval_accuracy", "eval_matthews_correlation", "eval_f1"):
        if layer_series(d, name):
            return name
    return None


def plot_loss(state, label, ax):
    hist = state.get("log_history", [])
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h and "step" in h]
    ev = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    if tr:
        ax[0].plot(*zip(*tr), label=label)
    if ev:
        ax[1].plot(*zip(*ev), marker="o", label=label)


def plot_by_epoch(state, label, out_dir):
    evals = [h for h in state.get("log_history", []) if "eval_loss" in h]
    if not evals:
        return
    metric = pick_metric(evals[-1])
    if metric is None:
        return
    epochs = [h["epoch"] for h in evals]
    n_layers = len(layer_series(evals[-1], metric))
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis", n_layers + 1)
    for i in range(n_layers):
        ys = [h.get(f"{metric}_{i}", float("nan")) for h in evals]
        ax.plot(epochs, ys, color=cmap(i), label=f"layer {i + 1}")
    if metric in evals[-1]:
        ax.plot(epochs, [h.get(metric, float("nan")) for h in evals], "k--", label="top (layer 12)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric.replace("eval_", ""))
    ax.set_title(f"Early-exit {metric.replace('eval_', '')} per epoch ({label})")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"metric_by_epoch_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", help="output dirs of train_cat.py")
    p.add_argument("--labels", nargs="*", help="legend labels (default: dir names)")
    p.add_argument("--out", default="plots")
    p.add_argument("--all", metavar="DIR",
                   help="plot every cat_model_* run in DIR separately, into OUT/<run_name>/")
    args = p.parse_args()
    if args.all:
        runs = sorted(d for d in os.listdir(args.all)
                      if d.startswith("cat_model_") and os.path.isdir(os.path.join(args.all, d)))
        if not runs:
            print(f"no cat_model_* dirs in {args.all}")
        for name in runs:
            print(f"== {name}")
            plot_runs([os.path.join(args.all, name)], [name], os.path.join(args.out, name))
        return
    if not args.runs:
        p.error("give RUN_DIR(s) or --all DIR")
    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    plot_runs(args.runs, labels, args.out)


def plot_runs(runs, labels, out):
    os.makedirs(out, exist_ok=True)
    for r in runs:
        if not os.path.isdir(r):
            print(f"WARNING: run dir not found: {r}")
        elif not os.path.exists(os.path.join(r, "all_results.json")):
            print(f"WARNING: no all_results.json in {r} (training not finished?)")

    # 1. Loss curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    any_loss = False
    for run, label in zip(runs, labels):
        state = load_json(os.path.join(run, "trainer_state.json"))
        if state:
            any_loss = True
            plot_loss(state, label, ax)
            plot_by_epoch(state, label, out)
    if any_loss:
        ax[0].set(title="Train loss (avg over early heads)", xlabel="step", ylabel="MSE / CE")
        ax[1].set(title="Eval loss", xlabel="epoch")
        for a in ax:
            a.grid(alpha=0.3)
            if a.get_legend_handles_labels()[0]:
                a.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "loss.png"), dpi=150)
        print("wrote", os.path.join(out, "loss.png"))
    plt.close(fig)

    # 2. Final per-layer comparison
    finals = [(load_json(os.path.join(r, "all_results.json")), l) for r, l in zip(runs, labels)]
    finals = [(d, l) for d, l in finals if d]
    if not finals:
        print("no all_results.json found")
        return
    # Each run uses its own task metric (pearson / accuracy / matthews ...).
    panels = [("metric", "task metric"),
              ("eval_consistency", "consistency with top layer"),
              ("eval_meta_accuracy", "meta accuracy")]
    def run_key(d, key):
        return pick_metric(d) if key == "metric" else key

    panels = [(k, t) for k, t in panels
              if any(run_key(d, k) and layer_series(d, run_key(d, k)) for d, _ in finals)]
    if not panels:
        print("no per-layer eval_* metrics in all_results.json (eval did not run?)")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, (key, title) in zip(axes[0], panels):
        for d, label in finals:
            k = run_key(d, key)
            if not k:
                continue
            s = layer_series(d, k)
            if not s:
                continue
            name = f"{label} ({k.replace('eval_', '')})" if key == "metric" else label
            ax.plot([i + 1 for i in s], list(s.values()), marker="o", label=name)
            if key == "metric" and k in d:
                ax.axhline(d[k], ls="--", alpha=0.5, label=f"{label} top")
        ax.set(title=title, xlabel="exit layer")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, "final_by_layer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
