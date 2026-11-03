"""Evaluate decoding strategies over dumped log-probs (tune on one half, report on the other).

Usage: python src/evaluate.py MODEL [MODEL ...] [--sets sps_devtest ood scr_test] [--beam] [--lm all_clean] [--morph]
Halves: sps_devtest -> SPS5 'dev' tunes / 'test' reports; other sets -> md5(id) parity.
WER/CER are reported against the raw reference (basic_norm) and against the curated-normalised reference.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys

import jiwer

sys.path.insert(0, os.path.dirname(__file__))
from decode import LOGITS, greedy, load, sets_meta  # noqa: E402
from textnorm import Normalizer, basic_norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_n = json.load(open(os.path.join(ROOT, "results", "lm", "normalizer.json")))
NORM = Normalizer(_n["wmap"], [tuple(r) for r in _n["rules"]], _n.get("vocab", []))
LN10 = 2.302585


def score(refs, hyps):
    return 100 * jiwer.wer(refs, hyps), 100 * jiwer.cer(refs, hyps)


def halves(set_name, ids):
    if set_name == "sps_devtest":
        sp = json.load(open(os.path.join(LOGITS, "sps_meta.json")))["sps_split"]
        return [i for i in ids if sp.get(i) == "dev"], [i for i in ids if sp.get(i) == "test"]
    h = lambda i: int(hashlib.md5(i.encode()).hexdigest(), 16) % 2
    return [i for i in ids if h(i) == 0], [i for i in ids if h(i) == 1]


_UNI = {}


def unigrams_for(lm_path):
    """Vocabulary of the LM training corpus (binary KenLM files do not expose it to pyctcdecode)."""
    txt = lm_path.split(".4g.")[0] + ".txt"
    if txt not in _UNI:
        _UNI[txt] = sorted({w for line in open(txt) for w in line.split()})
    return _UNI[txt]


def make_decoder(labels, arpa=None, alpha=0.5, beta=1.0, unk=-10.0):
    from pyctcdecode import build_ctcdecoder
    if arpa is None:
        return build_ctcdecoder(labels)
    return build_ctcdecoder(labels, kenlm_model_path=arpa, unigrams=unigrams_for(arpa),
                            alpha=alpha, beta=beta, unk_score_offset=unk)


def tune_grid(fn, grid, refs, ids):
    best = None
    for params in grid:
        w = score([refs[i] for i in ids], [fn(i, *params) for i in ids])[0]
        if best is None or w < best[0]:
            best = (w, params)
    return best[1]


def run(model, sets, use_beam, lm, morph):
    meta = sets_meta()
    for sname in sets:
        if sname not in meta:
            continue
        refs = {i: basic_norm(r) for i, r in meta[sname]}
        nrefs = {i: NORM(r) for i, r in refs.items()}
        try:
            fold, lps = load(model, sname)
        except FileNotFoundError:
            continue
        ids = [i for i in refs if i in lps and refs[i]]
        tune, rep = halves(sname, ids)
        flp = {i: fold(lps[i]) for i in ids}
        g = {i: greedy(flp[i], fold.labels) for i in ids}
        rows = {"greedy": g, "greedy+norm": {i: NORM(g[i]) for i in ids}}
        if use_beam:
            arpa = os.path.join(ROOT, "results", "lm", f"{lm}.4g.bin")
            cache = {}

            def beam_word(i, a, b, u):
                key = (a, b, u)
                if key not in cache:
                    cache.clear(); import gc; gc.collect()
                    cache[key] = make_decoder(fold.labels, arpa, a, b, u)
                return NORM(cache[key].decode(flp[i], beam_width=32))
            grid = list(itertools.product([0.3, 0.6, 1.0], [0.0, 1.0, 2.0], [-10.0, -4.0, -1.0]))
            a, b, u = tune_grid(beam_word, grid, nrefs, tune)
            cache.clear()
            dec = make_decoder(fold.labels, arpa, a, b, u)
            rows[f"beam+{lm}(a={a},b={b},unk={u})+norm"] = {i: NORM(dec.decode(flp[i], beam_width=64)) for i in ids}
            if morph:
                from morph_lm import MorphScorer
                ms = MorphScorer(lm)
                for tag, d in [("nolm", make_decoder(fold.labels)), ("wordlm", dec)]:
                    cand = {}
                    for i in ids:
                        seen = {}
                        for bm in d.decode_beams(flp[i], beam_width=64)[:24]:
                            t = NORM(bm[0])
                            if t not in seen:
                                seen[t] = bm[4]  # combined score (acoustic + word LM, if any)
                        cand[i] = [(t, s, ms(t) * LN10, len(t.split())) for t, s in seen.items()]

                    def pick(i, lam, mu):
                        return max(cand[i], key=lambda c: c[1] + lam * c[2] + mu * c[3])[0]
                    lam, mu = tune_grid(pick, list(itertools.product([0.0, 0.2, 0.4, 0.7, 1.0, 1.5], [-1.0, 0.0, 1.0, 2.0, 3.0])), nrefs, tune)
                    rows[f"beam-{tag}+morphLM(l={lam},m={mu})"] = {i: pick(i, lam, mu) for i in ids}
        for k, hy in rows.items():
            w, c = score([refs[i] for i in rep], [hy[i] for i in rep])
            wn, cn = score([nrefs[i] for i in rep], [hy[i] for i in rep])
            print(f"{model:15s} {sname:11s} {k:44s} WER {w:6.2f} CER {c:5.2f} | normRef WER {wn:6.2f} CER {cn:5.2f} (n={len(rep)})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--sets", nargs="+", default=["sps_devtest", "ood", "scr_test"])
    ap.add_argument("--beam", action="store_true")
    ap.add_argument("--lm", default="all_clean")
    ap.add_argument("--morph", action="store_true")
    a = ap.parse_args()
    for m in a.models:
        run(m, a.sets, a.beam, a.lm, a.morph)
