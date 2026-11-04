"""
challenge.py
------------
Held-out stress test for the faithfulness detector: faithful PARAPHRASES (should be
"supported") and ENTITY SWAPS (should be flagged). Nothing here was used to tune
the detector. Writes outputs/challenge_metrics.json and prints a table.

Usage: python -m src.challenge
"""
import json
import os

from .faithfulness import FaithfulnessScorer
from .retriever import TfidfRetriever

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(k: int = 3, verbose: bool = True, set_file: str = "challenge_set.json", out_name: str = "challenge_metrics.json"):
    with open(os.path.join(HERE, "data", "corpus.json")) as f:
        corpus = json.load(f)
    with open(os.path.join(HERE, "data", "qa_dataset.json")) as f:
        questions = {q["qid"]: q["question"] for q in json.load(f)}
    with open(os.path.join(HERE, "data", set_file)) as f:
        items = json.load(f)["items"]

    retriever, scorer = TfidfRetriever(corpus), FaithfulnessScorer()
    rows = []
    for it in items:
        ctx = retriever.context_sentences(retriever.retrieve(questions[it["qid"]], k=k))
        for kind, claim, should_flag in (("paraphrase", it["paraphrase"], False),
                                          ("entity_swap", it["entity_swap"], True)):
            v = scorer.score_answer(f'{it["qid"]}_{kind}', [claim], ctx).claim_verdicts[0]
            flagged = v.predicted_label != "supported"
            rows.append({"qid": it["qid"], "kind": kind, "claim": claim, "predicted": v.predicted_label,
                         "similarity": round(v.similarity, 3), "correct": flagged == should_flag})

    def acc(kind):
        r = [x for x in rows if x["kind"] == kind]
        return round(sum(x["correct"] for x in r) / len(r), 3)

    metrics = {
        "paraphrase_accepted": acc("paraphrase"),       # true claims correctly NOT flagged
        "entity_swap_caught": acc("entity_swap"),       # false claims correctly flagged
        "overall_accuracy": round(sum(x["correct"] for x in rows) / len(rows), 3),
        "n": len(rows),
    }
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    with open(os.path.join(HERE, "outputs", out_name), "w") as f:
        json.dump({"metrics": metrics, "rows": rows}, f, indent=2)
    if verbose:
        print(json.dumps(metrics, indent=2))
        for x in rows:
            if not x["correct"]:
                print(f"  MISS [{x['kind']}] -> {x['predicted']:12s} sim={x['similarity']}  {x['claim'][:90]}")
    return metrics, rows


if __name__ == "__main__":
    import sys
    if "--b" in sys.argv:
        run(set_file="challenge_set_b.json", out_name="challenge_b_metrics.json")
    else:
        run()
