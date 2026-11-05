# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Build the single-file browser app (faithfulness_audit_app.html) from
app_template.html and the data files. Run after changing either:

    python build_app.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def load(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as f:
        return json.load(f)


def build():
    with open(os.path.join(ROOT, "app_template.html"), encoding="utf-8") as f:
        html = f.read()
    challenge = {"A": load("challenge_set.json")["items"], "B": load("challenge_set_b.json")["items"]}
    for key, value in {
        "__CORPUS_JSON__": load("corpus.min.json"),
        "__QA_JSON__": load("qa.min.json"),
        "__ANSWERS_JSON__": load("answers.min.json"),
        "__CHALLENGE_JSON__": challenge,
    }.items():
        assert key in html, key
        html = html.replace(key, compact(value))
    out = os.path.join(ROOT, "faithfulness_audit_app.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


if __name__ == "__main__":
    print("wrote", build())
