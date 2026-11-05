# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
evaluate.py
-----------
Runs the full pipeline (retrieve -> score faithfulness) over the labeled
answer bank and reports how well the heuristic detector recovers the known
ground-truth hallucination labels. This is the same spirit as the
ARAGOG / RAGAS validation described in Section 6.3 and Table 1 of the
survey: correlate an automatic groundedness metric with ground truth.
"""
import json
import os

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .retriever import TfidfRetriever
from .faithfulness import FaithfulnessScorer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")


def load_data():
    with open(os.path.join(DATA_DIR, "corpus.json")) as f:
        corpus = json.load(f)
    with open(os.path.join(DATA_DIR, "generated_answers.json")) as f:
        answers = json.load(f)
    with open(os.path.join(DATA_DIR, "qa_dataset.json")) as f:
        qa = json.load(f)
    return corpus, answers, qa


def run(k: int = 3):
    os.makedirs(OUT_DIR, exist_ok=True)
    corpus, answers, qa = load_data()
    retriever = TfidfRetriever(corpus)
    scorer = FaithfulnessScorer()

    # --- Section 6.1-style retrieval quality check: Recall@k and MRR ---
    recall_hits, reciprocal_ranks = 0, []
    for item in qa:
        passages = retriever.retrieve(item["question"], k=k)
        doc_ids = [p.doc_id for p in passages]
        if item["gold_doc_id"] in doc_ids:
            recall_hits += 1
            reciprocal_ranks.append(1.0 / (doc_ids.index(item["gold_doc_id"]) + 1))
        else:
            reciprocal_ranks.append(0.0)
    retrieval_metrics = {
        f"recall_at_{k}": round(recall_hits / len(qa), 3),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
    }

    claim_rows = []
    answer_rows = []

    for ans in answers:
        passages = retriever.retrieve(ans["question"], k=k)
        context_sentences = retriever.context_sentences(passages)
        retrieved_doc_ids = [p.doc_id for p in passages]

        claims = [c["text"] for c in ans["claims"]]
        result = scorer.score_answer(ans["answer_id"], claims, context_sentences)

        for gold_c, verdict in zip(ans["claims"], result.claim_verdicts):
            claim_rows.append({
                "answer_id": ans["answer_id"],
                "qid": ans["qid"],
                "answer_type": ans["answer_type"],
                "claim": gold_c["text"],
                "gold_label": gold_c["gold_label"],
                "predicted_label": verdict.predicted_label,
                "similarity": round(verdict.similarity, 3),
                "best_evidence": verdict.best_evidence,
            })

        answer_rows.append({
            "answer_id": ans["answer_id"],
            "qid": ans["qid"],
            "answer_type": ans["answer_type"],
            "predicted_faithfulness": round(result.faithfulness_score, 3),
            "gold_all_supported": all(c["gold_label"] == "supported" for c in ans["claims"]),
            "top_retrieved_doc": retrieved_doc_ids[0] if retrieved_doc_ids else None,
        })

    claim_df = pd.DataFrame(claim_rows)
    answer_df = pd.DataFrame(answer_rows)

    claim_df.to_csv(os.path.join(OUT_DIR, "claim_level_results.csv"), index=False)
    answer_df.to_csv(os.path.join(OUT_DIR, "answer_level_results.csv"), index=False)

    metrics = {}

    # --- 3-class claim-level report ---
    labels3 = ["supported", "unsupported", "contradicted"]
    report3 = classification_report(
        claim_df["gold_label"], claim_df["predicted_label"], labels=labels3,
        output_dict=True, zero_division=0,
    )
    metrics["claim_level_3class_report"] = report3
    metrics["claim_level_3class_confusion_matrix"] = confusion_matrix(
        claim_df["gold_label"], claim_df["predicted_label"], labels=labels3
    ).tolist()
    metrics["confusion_matrix_labels"] = labels3

    # --- binary hallucination-detection view (supported vs hallucinated) ---
    def binarize(x):
        return "hallucinated" if x in ("unsupported", "contradicted") else "supported"

    gold_bin = claim_df["gold_label"].apply(binarize)
    pred_bin = claim_df["predicted_label"].apply(binarize)
    report_bin = classification_report(gold_bin, pred_bin, output_dict=True, zero_division=0)
    metrics["claim_level_binary_report"] = report_bin

    # --- answer-level faithfulness score by hallucination type ---
    avg_score_by_type = answer_df.groupby("answer_type")["predicted_faithfulness"].mean().round(3).to_dict()
    metrics["avg_faithfulness_score_by_answer_type"] = avg_score_by_type

    metrics["retrieval"] = retrieval_metrics

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return claim_df, answer_df, metrics


if __name__ == "__main__":
    claim_df, answer_df, metrics = run()
    print("Claim-level binary hallucination-detection report:")
    print(json.dumps(metrics["claim_level_binary_report"], indent=2))
    print("\nAverage predicted faithfulness score by answer type (1.0 = fully grounded):")
    print(json.dumps(metrics["avg_faithfulness_score_by_answer_type"], indent=2))
