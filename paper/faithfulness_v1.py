# Copyright (c) 2026 Janat Yasser. All rights reserved.
# The original (v1) detector, kept for the comparison in the paper.
"""
faithfulness.py
---------------
A RAGAS-inspired faithfulness / groundedness scorer (Section 6.3 of the
survey). RAGAS itself relies on an LLM to judge whether each atomic claim
in a generated answer is entailed by the retrieved context. Since this
project runs fully offline, we approximate that entailment judgment with
three cheap, interpretable signals computed against the best-matching
context sentence for each claim:

  1. Lexical/semantic similarity   (TF-IDF cosine)   -> is there ANY support?
  2. Numeric consistency           (regex number extraction) -> do figures match?
  3. Polarity/contrast cues        (negation & superlative flips) -> contradiction?

Each claim is labeled SUPPORTED, UNSUPPORTED (no matching evidence -
fabrication), or CONTRADICTED (evidence exists but disagrees with the claim).
This mirrors the "supported / hallucinated" distinction discussed in
Section 7.2.3 and 7.5.4 of the survey, while being fully transparent about
being a heuristic proxy for a real NLI/entailment model.
"""
import re
from dataclasses import dataclass, field
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retriever import split_sentences

NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")

NEGATION_WORDS = {"not", "no", "never", "none", "cancelled", "banned", "unable", "without", "no longer"}
SUPERLATIVE_PAIRS = [
    ("largest", "smallest"), ("smallest", "largest"),
    ("tallest", "shortest"), ("shortest", "tallest"),
    ("warmest", "coolest"), ("primary", "banned"),
    ("fastest", "slowest"), ("slowest", "fastest"),
    ("nearest", "farthest"), ("farthest", "nearest"),
    ("closest", "farthest"), ("first", "last"), ("last", "first"),
    ("highest", "lowest"), ("lowest", "highest"),
]

SUPPORT_THRESHOLD = 0.28   # below this similarity to best evidence -> "no support found"


def _numbers(text: str) -> set:
    return set(NUMBER_RE.findall(text.replace(",", "")))


def _has_polarity_flip(claim: str, evidence: str) -> bool:
    c, e = claim.lower(), evidence.lower()
    for word, opposite in SUPERLATIVE_PAIRS:
        if word in e and opposite in c:
            return True
        if opposite in e and word in c:
            return True
    neg_in_claim = any(w in c for w in NEGATION_WORDS)
    neg_in_evidence = any(w in e for w in NEGATION_WORDS)
    return neg_in_claim != neg_in_evidence


@dataclass
class ClaimVerdict:
    claim: str
    predicted_label: str
    best_evidence: str
    similarity: float
    gold_label: str = None


@dataclass
class FaithfulnessResult:
    answer_id: str
    claim_verdicts: List[ClaimVerdict] = field(default_factory=list)

    @property
    def faithfulness_score(self) -> float:
        if not self.claim_verdicts:
            return 1.0
        supported = sum(1 for c in self.claim_verdicts if c.predicted_label == "supported")
        return supported / len(self.claim_verdicts)


class FaithfulnessScorer:
    """Scores generated answers for groundedness against retrieved context."""

    def __init__(self):
        # A fresh small TF-IDF space per (claims + context) call keeps this
        # retriever-agnostic: it only needs the context actually retrieved
        # for a given question, not the whole corpus.
        pass

    def score_answer(self, answer_id: str, claims: List[str], context_sentences: List[str]) -> FaithfulnessResult:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = claims + context_sentences
        matrix = vectorizer.fit_transform(corpus)
        claim_vecs = matrix[: len(claims)]
        ctx_vecs = matrix[len(claims):]

        result = FaithfulnessResult(answer_id=answer_id)
        sims_matrix = cosine_similarity(claim_vecs, ctx_vecs) if context_sentences else None

        for i, claim in enumerate(claims):
            if not context_sentences:
                result.claim_verdicts.append(ClaimVerdict(claim, "unsupported", "", 0.0))
                continue

            sims = sims_matrix[i]
            best_idx = sims.argmax()
            best_sim = float(sims[best_idx])
            best_evidence = context_sentences[best_idx]

            if best_sim < SUPPORT_THRESHOLD:
                label = "unsupported"
            else:
                claim_nums = _numbers(claim)
                evidence_nums = _numbers(best_evidence)
                numeric_conflict = bool(claim_nums and evidence_nums and claim_nums != evidence_nums)
                polarity_conflict = _has_polarity_flip(claim, best_evidence)

                if numeric_conflict or polarity_conflict:
                    label = "contradicted"
                else:
                    label = "supported"

            result.claim_verdicts.append(ClaimVerdict(claim, label, best_evidence, best_sim))

        return result
