# RAG Faithfulness Detector — Evaluation Report

This report evaluates a RAGAS-style, offline faithfulness/groundedness detector (see Section 6.3 and 7.5.4 of *A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems*) against a synthetic benchmark with known, claim-level hallucination labels.

## 1. Retrieval quality (Section 6.1/6.2 metrics)

- Recall@k: **1.0**
- MRR: **0.98**

## 2. Claim-level hallucination detection (binary: supported vs hallucinated)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 1.00 | 1.00 | 1.00 | 50 |
| hallucinated | 1.00 | 1.00 | 1.00 | 75 |

Overall accuracy: **1.00**

## 3. Claim-level detection, 3-class (supported / unsupported / contradicted)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 1.00 | 1.00 | 1.00 | 50 |
| unsupported | 0.61 | 1.00 | 0.76 | 25 |
| contradicted | 1.00 | 0.68 | 0.81 | 50 |

![Confusion matrix](confusion_matrix.png)

## 4. Answer-level faithfulness score by injected hallucination type

- **contradiction**: 0.0
- **fabrication**: 0.5
- **grounded**: 1.0
- **numeric**: 0.0

![Faithfulness by type](faithfulness_by_type.png)

## 5. Notes and limitations

- The detector uses TF-IDF similarity + numeric/polarity heuristics as a transparent, offline proxy for the NLI/entailment model that RAGAS normally relies on (per Section 6.3, RAGAS uses an LLM judge). Real deployments should validate against human judgments, as the survey notes in Section 7.3.4 (Accountability and Transparency).
- Fabricated claims that reuse a lot of the source's vocabulary, or contradictions that don't hinge on a number or a known superlative pair, are the hardest cases for this heuristic — consistent with Section 7.2.3's point that no current approach fully eliminates hallucination.
