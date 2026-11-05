# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
main.py
-------
End-to-end demo: retrieve context for each question, run the offline
RAGAS-style faithfulness scorer on grounded / hallucinated answer variants,
evaluate against ground truth, and produce a report.

Usage: python main.py
"""
from src.challenge import run as run_challenge
from src.evaluate import run
from src.report import build_report

if __name__ == "__main__":
    claim_df, answer_df, metrics = run(k=3)
    run_challenge(verbose=False)
    run_challenge(verbose=False, set_file="challenge_set_b.json", out_name="challenge_b_metrics.json")
    build_report()

    print("\n=== Sample claim-level verdicts ===")
    print(claim_df.sample(8, random_state=1).to_string(index=False))
