"""Span rules for the final submission (v11): learned from the 2,067 CV original/curated sentence pairs
with src/textnorm.learn_span_rules and applied on top of the v2 normaliser.

Why: learn_rules builds one candidate rule per edit operation, so a spelling change that needs several
edits at once (qillqa -> qhilqa: insert "h" and drop one "l") never yields a rule that reaches the
curated form by itself; the word map only covers word forms seen in the pairs. Unseen inflections such
as qillqasaq therefore stayed in the original spelling. Span rules cover the whole differing span.
"""
import glob
import os

import pandas as pd

from textnorm import basic_norm, learn_span_rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cv_pairs():
    val = pd.read_csv(glob.glob(os.path.join(ROOT, "dataset common voice ", "*cv-corpus-26.0*", "**", "validated.tsv"), recursive=True)[0],
                      sep="\t", quoting=3, dtype=str, keep_default_na=False)
    cur = pd.read_csv(os.path.join(ROOT, "asr-puno-quechua", "data", "shared-task", "validated_sentences_curated.tsv"),
                      sep="\t", quoting=3, dtype=str)
    pairs = val.drop_duplicates("sentence_id").merge(cur, on="sentence_id")
    return [(basic_norm(o), basic_norm(c)) for o, c in zip(pairs.sentence, pairs.sentence_curated)]


def span_word_fn(norm):
    """Returns word -> word that applies the span rules to words outside the curated vocabulary and the
    word map of the v2 normaliser `norm` (both are left unchanged)."""
    rules = learn_span_rules(cv_pairs(), norm.vocab)

    def word(w):
        if w in norm.vocab or w in norm.wmap:
            return w
        x = "^" + w + "$"
        for a, b in rules:
            if a in x:
                x = x.replace(a, b)
        return x.strip("^$")
    word.rules = rules
    return word
