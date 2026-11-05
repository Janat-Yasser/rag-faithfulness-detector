# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
faithfulness.py
---------------
A RAGAS-inspired faithfulness / groundedness scorer (Section 6.3 of the
survey). RAGAS uses an LLM to judge whether each atomic claim in an answer is
entailed by the retrieved context. This project runs fully offline, so it
approximates that judgment with transparent signals computed against the
best-matching context sentence for each claim:

  1. Lexical similarity (TF-IDF cosine)            -> is there any support?
  2. Numeric consistency (tolerant number matching) -> do the figures agree?
  3. Polarity cues (whole-word negation, opposite superlatives)
  4. Substitution check (v2): word alignment finds a content word in the claim
     that REPLACES a different content word in the evidence ("speed of SOUND"
     vs "speed of LIGHT"). This is how most real hallucinations look, and pure
     similarity can't see it.
  5. Number-anchored support (v2): a paraphrase reuses few words but keeps the
     facts, so a claim whose distinctive numbers all match the evidence and
     that shares some content words counts as supported even at low similarity.

Each claim is labelled SUPPORTED, UNSUPPORTED (no matching evidence) or
CONTRADICTED (evidence exists but disagrees). It's a heuristic stand-in for a
real NLI/entailment model; see README "Honest results" for where it fails.
"""
import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# a number not glued to letters (skips "M87*", "J1748-2446ad")
NUMBER_RE = re.compile(r"(?<![A-Za-z\d.])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z\d])")
WORD_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")

NEGATIONS = {"not", "no", "never", "none", "cannot", "neither", "nor", "without", "unable", "cancelled", "banned"}
OPPOSITES = [
    ("largest", "smallest"), ("tallest", "shortest"), ("warmest", "coolest"), ("fastest", "slowest"),
    ("nearest", "farthest"), ("closest", "farthest"), ("first", "last"), ("highest", "lowest"),
    ("maximum", "minimum"), ("primary", "banned"),
]
HEDGES = {"about", "around", "roughly", "approximately", "nearly", "almost", "some", "close", "over", "near"}
STOP = set(ENGLISH_STOP_WORDS) | HEDGES

SUPPORT_THRESHOLD = 0.28    # similarity above this = the evidence is about the same fact
COPY_THRESHOLD = 0.45       # above this the claim largely copies the evidence; check substitutions
ANCHOR_OVERLAP = 0.20       # share of claim content words needed for number-anchored support


# ── text helpers ─────────────────────────────────────────────────────────────

def _stem(w: str) -> str:
    """Tiny suffix stripper, enough to match gravity/gravitational, orbit/orbits/orbiting."""
    for suf in ("ational", "ation", "ional", "ing", "ies", "ied", "ed", "es", "al", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower().replace(",", ""))


def _content(words: List[str]) -> List[str]:
    return [_stem(w) for w in words if w not in STOP and not w[0].isdigit()]


def _is_year(x: float) -> bool:
    return x.is_integer() and 1000 <= x <= 2100


def _numbers(text: str) -> List[float]:
    """Factual numbers in a sentence. Numbers that are part of a name ("Voyager 1",
    "51 Pegasi b") are skipped: they identify a thing, they aren't a claim about it."""
    out = []
    for m in NUMBER_RE.finditer(text):
        x = float(m.group().replace(",", ""))
        before = text[: m.start()].rstrip().split(" ")[-1:] or [""]
        after = text[m.end():].lstrip().split(" ")[:1] or [""]
        prev_cap = (bool(before[0]) and before[0][:1].isupper() and before[0][-1:] not in ",;:"
                    and m.start() > 0 and text[m.start() - 1] == " ")
        next_cap = bool(after[0]) and after[0][:1].isupper()
        if not _is_year(x) and (prev_cap or next_cap):
            continue
        out.append(x)
    return out


def _num_match(a: float, b: float, hedged: bool) -> bool:
    """Exact match, except hedged approximate figures ("about 300,000") may be 5% off.
    Years always have to match exactly."""
    if a == b:
        return True
    if not hedged or _is_year(a) or _is_year(b):
        return False
    return abs(a - b) <= 0.05 * max(abs(a), abs(b))


def _numeric_status(claim: str, evidence: str) -> Optional[str]:
    """'match' if every claim number appears in the evidence (within tolerance),
    'conflict' if one doesn't, None if the claim has no numbers to check."""
    cn, en = _numbers(claim), _numbers(evidence)
    if not cn or not en:
        return None
    hedged = bool(HEDGES & set(_words(claim)))
    return "match" if all(any(_num_match(c, e, hedged) for e in en) for c in cn) else "conflict"


def _polarity_flip(claim: str, evidence: str) -> bool:
    cw, ew = set(_words(claim)), set(_words(evidence))
    for a, b in OPPOSITES:
        if (a in ew and b in cw) or (b in ew and a in cw):
            return True
    return bool(NEGATIONS & cw) != bool(NEGATIONS & ew)


def _substitution(claim: str, evidence: str) -> Optional[str]:
    """Find a content word in the claim that stands where a DIFFERENT content word
    stands in the evidence. Returns 'claim-word / evidence-word' or None."""
    cw, ew = _words(claim), _words(evidence)
    c_all, e_all = set(_content(cw)), set(_content(ew))
    sm = difflib.SequenceMatcher(a=ew, b=cw, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace":
            continue
        new = [w for w in _content(cw[j1:j2]) if w not in e_all]
        old = [w for w in _content(ew[i1:i2]) if w not in c_all]
        if new and old:
            return f"{' '.join(new)} / {' '.join(old)}"
    return None


# ── result types ─────────────────────────────────────────────────────────────

@dataclass
class ClaimVerdict:
    claim: str
    predicted_label: str
    best_evidence: str
    similarity: float
    gold_label: str = None
    reason: str = ""


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


def judge(claim: str, evidence: str, sim: float):
    """Label one claim against its best evidence sentence. Returns (label, reason)."""
    nums = _numeric_status(claim, evidence)
    cc = _content(_words(claim))
    overlap = len(set(cc) & set(_content(_words(evidence)))) / max(1, len(set(cc)))

    if sim >= SUPPORT_THRESHOLD:
        if nums == "conflict":
            return "contradicted", "numbers disagree"
        if _polarity_flip(claim, evidence):
            return "contradicted", "negation / opposite"
        if sim >= COPY_THRESHOLD:
            sub = _substitution(claim, evidence)
            if sub:
                return "contradicted", f"substituted: {sub}"
        return "supported", "matches evidence"

    if nums == "match" and overlap >= ANCHOR_OVERLAP and not _polarity_flip(claim, evidence):
        return "supported", "paraphrase: numbers and key words match"
    if nums == "conflict" and overlap >= ANCHOR_OVERLAP:
        return "contradicted", "numbers disagree"
    return "unsupported", "no matching evidence"


class FaithfulnessScorer:
    """Scores generated answers for groundedness against retrieved context."""

    def score_answer(self, answer_id: str, claims: List[str], context_sentences: List[str]) -> FaithfulnessResult:
        result = FaithfulnessResult(answer_id=answer_id)
        if not context_sentences:
            result.claim_verdicts = [ClaimVerdict(c, "unsupported", "", 0.0, reason="no context") for c in claims]
            return result

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(claims + context_sentences)
        sims_matrix = cosine_similarity(matrix[: len(claims)], matrix[len(claims):])

        for i, claim in enumerate(claims):
            sims = sims_matrix[i]
            best = int(sims.argmax())
            label, reason = judge(claim, context_sentences[best], float(sims[best]))
            result.claim_verdicts.append(ClaimVerdict(claim, label, context_sentences[best], float(sims[best]), reason=reason))
        return result
