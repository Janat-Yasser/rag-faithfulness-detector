"""
retriever.py
------------
A lightweight stand-in for the DPR-style dense retriever described in
Section 3.2 of the survey (bi-encoder + inner-product search). We use a
TF-IDF vector space instead of a learned neural encoder so the whole
pipeline runs offline with no model downloads, while preserving the same
interface (query -> top-k ranked passages) that a real dense retriever
would expose.
"""
import re
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_sentences(text: str) -> List[str]:
    """Simple rule-based sentence splitter (avoids needing nltk downloads)."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class RetrievedPassage:
    doc_id: str
    text: str
    score: float


class TfidfRetriever:
    """Retriever module R(x, C) -> Z, as defined in Section 3.1 of the survey."""

    def __init__(self, corpus: List[dict]):
        self.corpus = corpus
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.texts = [d["text"] for d in corpus]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)

        # also index at sentence granularity for the faithfulness checker
        self.sentences = []  # list of (doc_id, sentence)
        for d in corpus:
            for s in split_sentences(d["text"]):
                self.sentences.append((d["doc_id"], s))
        self.sentence_texts = [s for _, s in self.sentences]
        self.sentence_matrix = self.vectorizer.transform(self.sentence_texts)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedPassage]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [
            RetrievedPassage(doc_id=self.doc_ids[i], text=self.texts[i], score=float(sims[i]))
            for i in top_idx
        ]

    def context_sentences(self, passages: List[RetrievedPassage]) -> List[str]:
        """Flatten retrieved passages down to sentence-level context units,
        which is the granularity the faithfulness checker operates on."""
        sents = []
        for p in passages:
            sents.extend(split_sentences(p.text))
        return sents
