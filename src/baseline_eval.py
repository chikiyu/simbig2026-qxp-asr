"""Official-style baseline scoring: greedy CTC over the model's raw vocabulary (as in the
organisers' inference/transcribe.py: '<unk>' = word boundary), no LM, no normaliser,
scored with the official normalisation (NFC, lowercase, no punctuation, keep ' and ñ)
against references in the curated orthography.

Usage: python src/baseline_eval.py MODEL [MODEL ...] [--sets ood sps_devtest scr_test scr_heldout]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from decode import LOGITS, sets_meta  # noqa: E402
from evaluate import NORM, score  # noqa: E402
from textnorm import basic_norm, official_norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def raw_greedy(lp, vocab, blank):
    ids = lp.argmax(-1)
    out, prev = [], -1
    for t in ids:
        if t != prev and t != blank:
            out.append(vocab[t])
        prev = t
    return official_norm("".join(out).replace("<unk>", " ").replace("|", " ")
                         .replace("<pad>", "").replace("</s>", ""))


def heldout_ids():
    return set(json.load(open(os.path.join(ROOT, "results", "scr_test_heldout_ids.json"))))


def load_raw(model, split):
    d = os.path.join(LOGITS, model)
    info = json.load(open(os.path.join(d, "vocab.json")))
    z = np.load(os.path.join(d, f"{split}.npz"))
    return info, z


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--sets", nargs="+", default=["ood", "sps_devtest", "scr_test", "scr_heldout"])
    a = ap.parse_args()
    meta = sets_meta()
    H = heldout_ids()
    for s in a.sets:
        src = "scr_test" if s == "scr_heldout" else s
        refs = {i: NORM(basic_norm(r)) for i, r in meta[src] if r}
        if s == "scr_heldout":
            refs = {i: r for i, r in refs.items() if i in H}
        for m in a.models:
            try:
                info, z = load_raw(m, src)
            except FileNotFoundError:
                continue
            ids = [i for i in refs if i in z.files]
            hyp = [raw_greedy(z[i].astype(np.float32), info["vocab"], info["blank"]) for i in ids]
            w, c = score([refs[i] for i in ids], hyp)
            print(f"{s:12s} {m:16s} official-greedy WER {w:6.2f} CER {c:5.2f} (n={len(ids)})", flush=True)
