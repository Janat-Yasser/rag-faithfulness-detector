"""
main.py
-------
End-to-end demo: retrieve context for each question, run the offline
RAGAS-style faithfulness scorer on grounded / hallucinated answer variants,
evaluate against ground truth, and produce a report.

Usage: python main.py
"""
from src.evaluate import run
from src.report import build_report

if __name__ == "__main__":
    claim_df, answer_df, metrics = run(k=3)
    build_report()

    print("\n=== Sample claim-level verdicts ===")
    print(claim_df.sample(8, random_state=1).to_string(index=False))
