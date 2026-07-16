# RAG Faithfulness Detector

A small, fully-offline machine learning project that complements **"A
Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems"**
(Oche, Folashade, Ghosal & Biswas, 2025) — specifically its discussion of
**hallucination and reliability** (Section 7.2.3), **RAGAS-style
groundedness evaluation** (Section 6.3), and the **existing/potential
solutions to hallucination** (Section 7.5.4).

## What it does

The survey describes RAGAS as an evaluation framework that checks whether
each claim in a RAG system's generated answer is actually entailed by the
retrieved context, in order to catch hallucinations. This project
implements a working, inspectable version of that idea end-to-end:

1. **Knowledge base** (`data/corpus.json`) — 25 short factual documents on
   astronomy and space physics (constants, cosmology, black holes, stellar
   physics, missions, exoplanets), written in original wording from
   well-established, verifiable facts — no fictional content, no copied text.
2. **Retriever** (`src/retriever.py`) — a TF-IDF cosine-similarity
   retriever standing in for the DPR-style dense retriever described in
   Section 3.2 of the survey (same `R(x, C) -> Z` interface).
3. **Labeled answer bank** (`data/generated_answers.json`) — 100 "generated"
   RAG answers (4 per question, 25 questions): one fully grounded, and three with a
   controlled, exactly-labeled hallucination each:
   - `numeric` — a figure is silently swapped (e.g. a date, a percentage)
   - `fabrication` — an invented sentence is appended that has no basis in
     the corpus at all
   - `contradiction` — a qualitative fact is flipped (e.g. "largest" →
     "smallest", or a fact is straightforwardly negated)
4. **Faithfulness scorer** (`src/faithfulness.py`) — for each claim in an
   answer, finds its best-matching retrieved evidence sentence and labels
   the claim `supported` / `unsupported` / `contradicted` using TF-IDF
   similarity plus numeric-consistency and polarity/negation checks — a
   transparent, offline proxy for the LLM-as-judge entailment step RAGAS
   normally uses.
5. **Evaluation** (`src/evaluate.py`) — computes retrieval Recall@k / MRR
   (Section 6.1–6.2 metrics), plus precision/recall/F1 for hallucination
   detection at the claim level, checked against the ground-truth labels
   built into the dataset.
6. **Report** (`src/report.py`) — renders a confusion matrix, a bar chart
   of mean faithfulness score per hallucination type, and a markdown
   summary (`outputs/REPORT.md`).

## Interactive app

`faithfulness_audit_app.html` is a self-contained, no-dependency dashboard
("Faithfulness Audit") that runs the entire pipeline live in the browser —
no Python, no server. Just open the file. It:

- Re-implements the TF-IDF retriever and faithfulness scorer in plain
  JavaScript (verified to reproduce the same 100% binary detection accuracy
  / Recall@3 = 1.0 / MRR = 0.98 as the Python version)
- Runs the full audit over all 100 generated answers on load and shows
  live KPIs, a confusion matrix, and a faithfulness-by-type chart
- Lets you browse every case in a filterable "ledger," expanding any one
  to see each claim's evidence and verdict
- Includes a "cross-examination console" — pick a question, write your
  own answer (try inventing a fact or changing a number), and get a live
  faithfulness score

## Why this complements the survey rather than just re-implementing RAG

Sections 6 and 7 of the survey talk *about* evaluation metrics and
hallucination mitigation strategies at a conceptual level (RAGAS, RAGTruth,
citation-based grounding, self-critique). This project makes those ideas
concrete and testable: because the hallucinations are injected
deterministically, you get exact ground truth to measure a faithfulness
detector against — something you can't do with a real LLM's free-form
output. It's a benchmark-and-detector, not another RAG chatbot.

## Running it

```bash
pip install -r requirements.txt
python src/generate_data.py   # (re)build the synthetic dataset — already included
python main.py                # run retrieval + faithfulness scoring + report
```

Outputs land in `outputs/`:
- `claim_level_results.csv`, `answer_level_results.csv` — raw predictions
- `metrics.json` — all computed metrics
- `confusion_matrix.png`, `faithfulness_by_type.png`
- `REPORT.md` — human-readable summary

## Results on the included benchmark

- Retrieval: Recall@3 = 1.0, MRR = 0.98 (25 astronomy/space-physics
  questions against a 25-document knowledge base)
- Claim-level hallucination detection (binary: supported vs. hallucinated):
  **100% accuracy**, F1 = 1.00 for both classes
- 3-class detection (supported / unsupported / contradicted) is more
  nuanced: some contradictions get flagged as "unsupported" rather than
  "contradicted" (precision 1.00 / recall 0.68 on the contradicted class) —
  the detector still catches the hallucination, just not always its exact type
- Mean faithfulness score by injected type: grounded 1.00, fabrication 0.50,
  numeric 0.00, contradiction 0.00

The detector catches numeric hallucinations and outright fabrications
essentially perfectly. Its remaining uncertainty is in *classifying* a
hallucination's flavor (unsupported vs. contradicted) rather than in
*detecting* that something is wrong — a useful distinction, and one that
maps onto Section 7.2.3's point that no current technique fully
eliminates hallucination, only reduces and categorizes it more reliably.

## Honest limitations

- The similarity/negation heuristics are a stand-in for a real NLI or
  LLM-judge entailment model (what RAGAS actually uses). They work well on
  this controlled benchmark but would need a proper entailment model for
  open-domain, free-form generated text.
- The retriever is TF-IDF, not a trained dense encoder — swapping in a
  real sentence embedding model (if you have API/model access) is a
  drop-in change in `src/retriever.py`.
- Claim segmentation here is given by construction; a production system
  would need its own claim/atomic-fact extraction step, which is itself
  an open research problem the survey doesn't deeply cover.
