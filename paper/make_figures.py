# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Figures 2 and 3 of the paper, from paper/results.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "fig"), exist_ok=True)
d = json.load(open(os.path.join(HERE, "results.json")))
R = d["results"]
SETS = [("benchmark", "Benchmark (n=125)", "#9AA5B1"), ("A (dev)", "Challenge A, dev (n=50)", "#5B8DB8"),
        ("B (held-out)", "Challenge B, held-out (n=50)", "#C0504D")]
NAMES = [("Always supported", "Always\nsupported"), ("TF-IDF similarity >= 0.86 (tuned on benchmark)", "Similarity\nthreshold"),
         ("Content-word overlap >= 0.91 (tuned on benchmark)", "Overlap\nthreshold"), ("v1 detector (original)", "v1"),
         ("v2 detector (full)", "v2 (ours)")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

fig, ax = plt.subplots(figsize=(7.2, 3.5))
w = 0.26
for j, (s, lab, col) in enumerate(SETS):
    xs = [i + (j - 1) * w for i in range(len(NAMES))]
    acc = [R[s][n]["acc"] * 100 for n, _ in NAMES]
    lo = [(R[s][n]["acc"] - R[s][n]["ci"][0]) * 100 for n, _ in NAMES]
    hi = [(R[s][n]["ci"][1] - R[s][n]["acc"]) * 100 for n, _ in NAMES]
    ax.bar(xs, acc, w, label=lab, color=col, yerr=[lo, hi], capsize=2.5, error_kw={"elinewidth": 0.8})
ax.axhline(50, color="#444", lw=0.8, ls="--")
ax.set_xticks(range(len(NAMES))); ax.set_xticklabels([b for _, b in NAMES])
ax.set_ylabel("Binary accuracy (%)"); ax.set_ylim(0, 102)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig", "accuracy.png"), dpi=220)

sw = d["threshold_sweep"]
fig, ax = plt.subplots(figsize=(5.2, 2.7))
for s, lab, col in SETS:
    ax.plot([r["t"] for r in sw], [r[s] * 100 for r in sw], marker="o", ms=3.5, label=lab, color=col)
ax.axvline(0.28, color="#444", lw=0.8, ls=":"); ax.text(0.287, 100.5, "default τ = 0.28", fontsize=8, color="#444")
ax.set_xlabel("Support threshold τ (TF-IDF cosine)"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(60, 103)
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig", "sweep.png"), dpi=220)
print("wrote paper/fig/accuracy.png and paper/fig/sweep.png")
