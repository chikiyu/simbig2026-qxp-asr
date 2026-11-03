"""Local CTC decoding over log-probs dumped by kaggle/qxp-logits.

Vocab handling: model tokens are folded into a clean label set *at the log-prob level*
(logsumexp of columns mapping to the same symbol): '<unk>' and '|' -> word boundary ' ',
accented vowels -> plain vowels, fairseq specials dropped. This makes models trained with
'<unk>' as separator (QuechuaBase) and with '|' (ours) decodable by the same decoder.
"""
import json
import os

import numpy as np

from textnorm import strip_accents

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGITS = os.path.join(ROOT, "results", "logits")


def fold_token(tok):
    if tok in ("<unk>", "|"):
        return " "
    if tok in ("<pad>", "</s>"):
        return None
    if tok == "<s>":
        return ""  # CTC blank
    return strip_accents(tok.lower())


class Folded:
    """Maps a model's vocabulary onto clean labels; labels[0] == '' (blank)."""

    def __init__(self, vocab, blank):
        groups = {}
        for i, t in enumerate(vocab):
            f = "" if i == blank else fold_token(t)
            if f is None:
                continue
            groups.setdefault(f, []).append(i)
        self.labels = [""] + sorted(k for k in groups if k != "")
        self.cols = [groups[k] for k in self.labels]

    def __call__(self, lp):
        lp = lp.astype(np.float32)
        out = np.stack([np.logaddexp.reduce(lp[:, c], axis=1) if len(c) > 1 else lp[:, c[0]]
                        for c in self.cols], axis=1)
        return out


def load(model, split):
    d = os.path.join(LOGITS, model)
    info = json.load(open(os.path.join(d, "vocab.json")))
    z = np.load(os.path.join(d, f"{split}.npz"))
    fold = Folded(info["vocab"], info["blank"])
    return fold, {k: z[k] for k in z.files}


def greedy(lp, labels):
    ids = lp.argmax(-1)
    out, prev = [], -1
    for t in ids:
        if t != prev and t != 0:
            out.append(labels[t])
        prev = t
    return " ".join("".join(out).split())


def sets_meta():
    return json.load(open(os.path.join(LOGITS, "sets.json")))
