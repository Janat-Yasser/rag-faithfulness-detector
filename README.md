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
4. **Faithfulness scorer** (`src/faithfulness.py`): for each claim in an
   answer, finds its best-matching retrieved evidence sentence and labels
   the claim `supported` / `unsupported` / `contradicted`. It uses TF-IDF
   similarity, number checks (exact for years, tolerant for hedged figures),
   whole-word negation, a **substitution check** (word alignment spots
   "speed of *sound*" standing where the evidence says "speed of *light*"), and
   **number-anchored support** for paraphrases. Every verdict carries a reason.
   It's a transparent, offline proxy for the LLM-as-judge entailment step RAGAS
   normally uses.
5. **Evaluation** (`src/evaluate.py`) — computes retrieval Recall@k / MRR
   (Section 6.1–6.2 metrics), plus precision/recall/F1 for hallucination
   detection at the claim level, checked against the ground-truth labels
   built into the dataset.
6. **Report** (`src/report.py`) — renders a confusion matrix, a bar chart
   of mean faithfulness score per hallucination type, and a markdown
   summary (`outputs/REPORT.md`).

## Interactive app

**▶ Live demo: [janat-yasser.github.io/rag-faithfulness-detector](https://janat-yasser.github.io/rag-faithfulness-detector/)**

`faithfulness_audit_app.html` is a self-contained, no-dependency dashboard
("Faithfulness Audit") that runs the entire pipeline live in the browser —
no Python, no server. Just open the file. It:

- Re-implements the TF-IDF retriever and faithfulness scorer in plain
  JavaScript, matching scikit-learn's tokenisation. A test checks that the
  browser and Python versions give identical labels (225/225 claims).
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
python src/generate_data.py   # (re)build the synthetic dataset (already included)
python main.py                # retrieval + scoring + challenge sets + report
pytest                        # tests, including Python/browser agreement
```

Outputs land in `outputs/`: per-claim and per-answer CSVs, `metrics.json`,
`challenge_metrics.json` / `challenge_b_metrics.json`, two charts, and
`REPORT.md`.

## Results (honest version)

**The included benchmark is easy.** It was generated from the corpus itself:
17 of the 25 "grounded" claims are copied word for word, and each hallucination
is a one-token edit of a corpus sentence. The first version of the detector
scored 100% on it, but its thresholds were written with that data in view.

So there are two **challenge sets** (`data/challenge_set*.json`) of faithful
**paraphrases** (true, reworded) and **entity swaps** (false: "Hubble" for
"James Webb", "sound" for "light", numbers untouched). These are the two cases
real hallucination detection lives or dies on. Set B was written *before* the
v2 rules were implemented and wasn't looked at while developing them.

| | v1 (original) | v2 (current) |
|---|---|---|
| Benchmark: binary accuracy | 100% | 98.4% |
| Challenge set A: overall | 24% | 76% |
| **Challenge set B (held out): overall** | **46%** | **86%** |
| ↳ paraphrases accepted | 80% | 76% |
| ↳ entity swaps caught | 12% | 96% |

v1 was essentially a word-overlap test: it rejected paraphrases and accepted
almost any entity swap. v2's biggest gain is catching substitutions. It also
fixes two bugs that made the number check pass false claims: years matched
within 2% (1989 accepted for 2012), and numbers in names ("Voyager **1**",
"**51** Pegasi b") were treated as facts. A third bug, substring negation
(the "no" in "kno*wn*" flipped a claim to "contradicted"), is fixed too.

Retrieval on the benchmark: Recall@3 = 1.0, MRR = 0.98. Like the claims,
the questions were written from the documents, so this is optimistic as well.

## Honest limitations

- Where v2 still fails: synonym paraphrases ("little moons" for "small moons")
  look like substitutions; meaning carried by negation ("*no* star is closer")
  looks like a polarity flip; contradictions without a number, negation or
  word swap ("at the outer edge" vs "at the center") get through. These are
  exactly the cases an NLI/entailment model handles. The next step is to plug
  one in behind the same `judge()` interface and compare it on set B.
- The retriever is TF-IDF, not a trained dense encoder — swapping in a
  real sentence embedding model (if you have API/model access) is a
  drop-in change in `src/retriever.py`.
- Claim segmentation here is given by construction; a production system
  would need its own claim/atomic-fact extraction step, which is itself
  an open research problem the survey doesn't deeply cover.
