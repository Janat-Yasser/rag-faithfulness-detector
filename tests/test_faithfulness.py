import json
import os

import pytest

from src.challenge import run as run_challenge
from src.faithfulness import FaithfulnessScorer, _numbers, judge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = "The James Webb Space Telescope launched in December 2021 and operates near the Sun-Earth L2 point, roughly 1.5 million kilometers from Earth."


def label(claim, evidence=EV):
    return FaithfulnessScorer().score_answer("t", [claim], [evidence]).claim_verdicts[0].predicted_label


def test_verbatim_claim_is_supported():
    assert label(EV) == "supported"


def test_entity_swap_is_contradicted():
    assert label(EV.replace("James Webb", "Hubble")) == "contradicted"


def test_changed_year_is_contradicted_even_if_close():
    # v2 bug fixed: years used to match within 2% (2021 vs 2019)
    assert label(EV.replace("2021", "2019")) == "contradicted"


def test_paraphrase_with_matching_numbers_is_supported():
    assert label("After its 2021 launch, Webb settled at L2, roughly 1.5 million km away.") == "supported"


def test_numbers_in_names_are_not_facts():
    assert _numbers("Voyager 1 was launched in 1977") == [1977.0]
    assert _numbers("51 Pegasi b, discovered in 1995") == [1995.0]
    assert _numbers("M87*, estimated at 6.5 billion") == [6.5]


def test_whole_word_negation_only():
    # "known" contains "no"; v1 treated that as a negation
    assert judge("Proxima Centauri is the nearest known star.", "Proxima Centauri is the nearest star.", 0.9)[0] == "supported"


def test_challenge_sets_meet_reported_numbers():
    a, _ = run_challenge(verbose=False, out_name="_test_a.json")
    b, _ = run_challenge(verbose=False, set_file="challenge_set_b.json", out_name="_test_b.json")
    for f in ("_test_a.json", "_test_b.json"):
        os.remove(os.path.join(ROOT, "outputs", f))
    assert a["overall_accuracy"] >= 0.75
    assert b["overall_accuracy"] >= 0.85


def test_browser_app_agrees_with_python():
    pw = pytest.importorskip("playwright.sync_api")
    from src.retriever import TfidfRetriever

    with open(os.path.join(ROOT, "data", "corpus.json")) as f:
        corpus = json.load(f)
    with open(os.path.join(ROOT, "data", "challenge_set_b.json")) as f:
        items = json.load(f)["items"]
    with open(os.path.join(ROOT, "data", "qa_dataset.json")) as f:
        qa = {q["qid"]: q["question"] for q in json.load(f)}
    retr, scorer = TfidfRetriever(corpus), FaithfulnessScorer()
    cases = [(qa[it["qid"]], c) for it in items for c in (it["paraphrase"], it["entity_swap"])]
    py = [scorer.score_answer("x", [c], retr.context_sentences(retr.retrieve(q, k=3))).claim_verdicts[0].predicted_label
          for q, c in cases]
    with pw.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("file://" + os.path.join(ROOT, "faithfulness_audit_app.html"))
        js = page.evaluate("cases => cases.map(([q, c]) => scoreClaims([c], contextSentences(retrieve(q, 3)))[0].label)", cases)
        browser.close()
    assert js == py


def test_built_app_is_up_to_date_with_template_and_data():
    import build_app
    with open(os.path.join(ROOT, "faithfulness_audit_app.html"), encoding="utf-8") as f:
        current = f.read()
    build_app.build()
    with open(os.path.join(ROOT, "faithfulness_audit_app.html"), encoding="utf-8") as f:
        rebuilt = f.read()
    assert "__CORPUS_JSON__" not in rebuilt and current == rebuilt, "run `python build_app.py`"
