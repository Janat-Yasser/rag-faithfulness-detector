# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Experiments for the paper: baselines, ablations, confidence intervals, McNemar tests,
threshold sweep and the list of held-out errors.

Run from the repository root:  python paper/experiments.py   (writes paper/results.json)
Then:                           python paper/make_figures.py  (writes paper/fig/*.png)
"""
import importlib.util, json, math, os, sys
from collections import Counter

sys.path.insert(0, os.getcwd())
import src.faithfulness as F
from src.retriever import TfidfRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT = os.path.dirname(os.path.abspath(__file__))
corpus = json.load(open("data/corpus.json"))
questions = {q["qid"]: q["question"] for q in json.load(open("data/qa_dataset.json"))}
answers = json.load(open("data/generated_answers.json"))
retr = TfidfRetriever(corpus)


def ctx_for(question):
    return retr.context_sentences(retr.retrieve(question, k=3))


# ---- evaluation sets as (claim, context, gold_binary, gold3, kind) ----------------------
bench = []
for a in answers:
    ctx = ctx_for(a["question"])
    for c in a["claims"]:
        bench.append(dict(claim=c["text"], ctx=ctx, gold3=c["gold_label"], kind=a["answer_type"],
                          hall=c["gold_label"] != "supported"))


def challenge(fname):
    rows = []
    for it in json.load(open(f"data/{fname}"))["items"]:
        ctx = ctx_for(questions[it["qid"]])
        rows.append(dict(claim=it["paraphrase"], ctx=ctx, kind="paraphrase", hall=False, gold3="supported"))
        rows.append(dict(claim=it["entity_swap"], ctx=ctx, kind="entity_swap", hall=True, gold3="contradicted"))
    return rows


setA, setB = challenge("challenge_set.json"), challenge("challenge_set_b.json")
SETS = {"benchmark": bench, "A (dev)": setA, "B (held-out)": setB}


def best_evidence(claim, ctx):
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    m = v.fit_transform([claim] + ctx)
    s = cosine_similarity(m[0], m[1:])[0]
    i = int(s.argmax())
    return ctx[i], float(s[i])


for S in SETS.values():
    for r in S:
        r["ev"], r["sim"] = best_evidence(r["claim"], r["ctx"])


# ---- systems --------------------------------------------------------------------------
def v2_variant(numbers=True, polarity=True, substitution=True, anchored=True):
    def j(claim, ev, sim):
        nums = F._numeric_status(claim, ev) if numbers else None
        cc = F._content(F._words(claim))
        overlap = len(set(cc) & set(F._content(F._words(ev)))) / max(1, len(set(cc)))
        flip = F._polarity_flip(claim, ev) if polarity else False
        if sim >= F.SUPPORT_THRESHOLD:
            if nums == "conflict":
                return "contradicted"
            if flip:
                return "contradicted"
            if substitution and sim >= F.COPY_THRESHOLD and F._substitution(claim, ev):
                return "contradicted"
            return "supported"
        if anchored and nums == "match" and overlap >= F.ANCHOR_OVERLAP and not flip:
            return "supported"
        if nums == "conflict" and overlap >= F.ANCHOR_OVERLAP:
            return "contradicted"
        return "unsupported"
    return j


spec = importlib.util.spec_from_file_location("fv1", f"{OUT}/faithfulness_v1.py")
V1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V1)
v1_scorer = V1.FaithfulnessScorer()


def v1_label(r):
    return v1_scorer.score_answer("x", [r["claim"]], r["ctx"]).claim_verdicts[0].predicted_label


def overlap(r):
    cc = set(F._content(F._words(r["claim"])))
    return len(cc & set(F._content(F._words(r["ev"])))) / max(1, len(cc))


def tune(score_fn, rows):
    """Threshold on a score (higher = supported) that maximises accuracy on rows."""
    cands = sorted({round(score_fn(r), 4) for r in rows})
    best = max(cands, key=lambda t: (sum((score_fn(r) >= t) == (not r["hall"]) for r in rows), -t))
    return best


t_sim = tune(lambda r: r["sim"], bench)
t_ovl = tune(overlap, bench)

SYSTEMS = {
    "Always supported": lambda r: "supported",
    f"TF-IDF similarity >= {t_sim:.2f} (tuned on benchmark)": lambda r: "supported" if r["sim"] >= t_sim else "unsupported",
    f"Content-word overlap >= {t_ovl:.2f} (tuned on benchmark)": lambda r: "supported" if overlap(r) >= t_ovl else "unsupported",
    "v1 detector (original)": v1_label,
    "v2 detector (full)": lambda r, j=v2_variant(): j(r["claim"], r["ev"], r["sim"]),
    "v2 without number checks": lambda r, j=v2_variant(numbers=False): j(r["claim"], r["ev"], r["sim"]),
    "v2 without negation/opposites": lambda r, j=v2_variant(polarity=False): j(r["claim"], r["ev"], r["sim"]),
    "v2 without substitution check": lambda r, j=v2_variant(substitution=False): j(r["claim"], r["ev"], r["sim"]),
    "v2 without number-anchored support": lambda r, j=v2_variant(anchored=False): j(r["claim"], r["ev"], r["sim"]),
}

# sanity: our v2 re-implementation must equal the shipped judge
for S in SETS.values():
    for r in S:
        assert v2_variant()(r["claim"], r["ev"], r["sim"]) == F.judge(r["claim"], r["ev"], r["sim"])[0]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def evaluate(sysfn, rows):
    preds = [sysfn(r) for r in rows]
    corr = [(p != "supported") == r["hall"] for p, r in zip(preds, rows)]
    tp = sum(p != "supported" and r["hall"] for p, r in zip(preds, rows))
    fp = sum(p != "supported" and not r["hall"] for p, r in zip(preds, rows))
    fn = sum(p == "supported" and r["hall"] for p, r in zip(preds, rows))
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    k, n = sum(corr), len(corr)
    by = {}
    for kind in sorted({r["kind"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["kind"] == kind]
        by[kind] = sum(corr[i] for i in idx) / len(idx)
    return dict(acc=k / n, k=k, n=n, ci=wilson(k, n), f1=f1, prec=prec, rec=rec, by_kind=by, preds=preds, corr=corr)


results = {s: {name: evaluate(fn, rows) for name, fn in SYSTEMS.items()} for s, rows in SETS.items()}


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired correctness."""
    b01 = sum(x and not y for x, y in zip(a, b))
    b10 = sum(y and not x for x, y in zip(a, b))
    n = b01 + b10
    if n == 0:
        return 1.0, b01, b10
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p), b01, b10


tests = {}
for s in SETS:
    p, a_only, b_only = mcnemar(results[s]["v2 detector (full)"]["corr"], results[s]["v1 detector (original)"]["corr"])
    tests[s] = dict(p=p, v2_only_correct=a_only, v1_only_correct=b_only)

# 3-class report for v1 and v2 on the benchmark
from sklearn.metrics import classification_report, confusion_matrix
labels3 = ["supported", "unsupported", "contradicted"]
three = {}
for name in ("v1 detector (original)", "v2 detector (full)"):
    preds = results["benchmark"][name]["preds"]
    gold = [r["gold3"] for r in bench]
    rep = classification_report(gold, preds, labels=labels3, output_dict=True, zero_division=0)
    three[name] = dict(macro_f1=rep["macro avg"]["f1-score"], per_class={l: rep[l] for l in labels3},
                       confusion=confusion_matrix(gold, preds, labels=labels3).tolist())

# threshold sensitivity of v2 on A and B
sweep = []
orig = F.SUPPORT_THRESHOLD
for t in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.36, 0.40, 0.45]:
    F.SUPPORT_THRESHOLD = t
    row = {"t": t}
    for s, rows in SETS.items():
        row[s] = evaluate(lambda r, j=v2_variant(): j(r["claim"], r["ev"], r["sim"]), rows)["acc"]
    sweep.append(row)
F.SUPPORT_THRESHOLD = orig

# held-out misses with reasons
misses = []
for r, p, c in zip(setB, results["B (held-out)"]["v2 detector (full)"]["preds"], results["B (held-out)"]["v2 detector (full)"]["corr"]):
    if not c:
        misses.append(dict(kind=r["kind"], claim=r["claim"], evidence=r["ev"], sim=round(r["sim"], 3), predicted=p,
                           reason=F.judge(r["claim"], r["ev"], r["sim"])[1]))

summary = {
    "data": {"benchmark_claims": len(bench), "benchmark_gold": dict(Counter(r["gold3"] for r in bench)),
             "A": len(setA), "B": len(setB), "docs": len(corpus)},
    "tuned_thresholds": {"sim": t_sim, "overlap": t_ovl},
    "results": {s: {n: {k: v for k, v in d.items() if k not in ("preds", "corr")} for n, d in res.items()} for s, res in results.items()},
    "mcnemar_v2_vs_v1": tests, "three_class_benchmark": three, "threshold_sweep": sweep, "heldout_misses": misses,
}
json.dump(summary, open(f"{OUT}/results.json", "w"), indent=2)

for s in SETS:
    print(f"\n== {s}")
    for n, d in results[s].items():
        print(f"  {n:45s} acc {d['acc']:.3f} [{d['ci'][0]:.2f}-{d['ci'][1]:.2f}]  F1 {d['f1']:.3f}  " + "  ".join(f"{k} {v:.2f}" for k, v in d["by_kind"].items()))
print("\nMcNemar v2 vs v1:", json.dumps(tests))
print("3-class macro F1:", {k: round(v["macro_f1"], 3) for k, v in three.items()})
print("sweep:", json.dumps(sweep))
print("misses on B:", len(misses))
for m in misses:
    print(" ", m["kind"], m["predicted"], m["sim"], "|", m["claim"][:80], "| EV:", m["evidence"][:70])
