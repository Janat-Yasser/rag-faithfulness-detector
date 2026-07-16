"""
report.py
---------
Turns the raw metrics from evaluate.py into a human-readable report:
a confusion matrix plot and a markdown summary, framed against the
evaluation dimensions described in Section 6 (Table 1) of the survey.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")


def plot_confusion_matrix(cm, labels, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title("Claim-level faithfulness detection\n(confusion matrix)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_faithfulness_by_type(avg_by_type, path):
    types = list(avg_by_type.keys())
    scores = [avg_by_type[t] for t in types]
    colors = ["#2ca02c" if t == "grounded" else "#d62728" for t in types]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(types, scores, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean predicted faithfulness score")
    ax.set_title("Predicted faithfulness by injected hallucination type")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    for i, s in enumerate(scores):
        ax.text(i, s + 0.02, f"{s:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown_report(metrics, cm_path, bar_path, md_path):
    binary = metrics["claim_level_binary_report"]
    report3 = metrics["claim_level_3class_report"]
    retrieval = metrics["retrieval"]

    lines = []
    lines.append("# RAG Faithfulness Detector — Evaluation Report\n")
    lines.append(
        "This report evaluates a RAGAS-style, offline faithfulness/groundedness "
        "detector (see Section 6.3 and 7.5.4 of *A Systematic Review of Key "
        "Retrieval-Augmented Generation (RAG) Systems*) against a synthetic "
        "benchmark with known, claim-level hallucination labels.\n"
    )

    lines.append("## 1. Retrieval quality (Section 6.1/6.2 metrics)\n")
    lines.append(f"- Recall@k: **{retrieval[list(retrieval.keys())[0]]}**")
    lines.append(f"- MRR: **{retrieval['mrr']}**\n")

    lines.append("## 2. Claim-level hallucination detection (binary: supported vs hallucinated)\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for cls in ["supported", "hallucinated"]:
        r = binary[cls]
        lines.append(f"| {cls} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1-score']:.2f} | {int(r['support'])} |")
    lines.append(f"\nOverall accuracy: **{binary['accuracy']:.2f}**\n")

    lines.append("## 3. Claim-level detection, 3-class (supported / unsupported / contradicted)\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for cls in ["supported", "unsupported", "contradicted"]:
        r = report3[cls]
        lines.append(f"| {cls} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1-score']:.2f} | {int(r['support'])} |")

    lines.append(f"\n![Confusion matrix]({os.path.basename(cm_path)})\n")

    lines.append("## 4. Answer-level faithfulness score by injected hallucination type\n")
    for t, s in metrics["avg_faithfulness_score_by_answer_type"].items():
        lines.append(f"- **{t}**: {s}")
    lines.append(f"\n![Faithfulness by type]({os.path.basename(bar_path)})\n")

    lines.append("## 5. Notes and limitations\n")
    lines.append(
        "- The detector uses TF-IDF similarity + numeric/polarity heuristics "
        "as a transparent, offline proxy for the NLI/entailment model that "
        "RAGAS normally relies on (per Section 6.3, RAGAS uses an LLM judge). "
        "Real deployments should validate against human judgments, as the "
        "survey notes in Section 7.3.4 (Accountability and Transparency).\n"
        "- Fabricated claims that reuse a lot of the source's vocabulary, or "
        "contradictions that don't hinge on a number or a known superlative "
        "pair, are the hardest cases for this heuristic — consistent with "
        "Section 7.2.3's point that no current approach fully eliminates "
        "hallucination.\n"
    )

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def build_report():
    with open(os.path.join(OUT_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    bar_path = os.path.join(OUT_DIR, "faithfulness_by_type.png")
    md_path = os.path.join(OUT_DIR, "REPORT.md")

    plot_confusion_matrix(metrics["claim_level_3class_confusion_matrix"], metrics["confusion_matrix_labels"], cm_path)
    plot_faithfulness_by_type(metrics["avg_faithfulness_score_by_answer_type"], bar_path)
    write_markdown_report(metrics, cm_path, bar_path, md_path)
    print(f"Report written to {md_path}")


if __name__ == "__main__":
    build_report()
