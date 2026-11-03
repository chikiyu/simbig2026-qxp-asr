"""Learned system combination for spontaneous speech: rerank the MBR candidate pool.

Candidates per utterance: union of the top-4 beams of 5 XLS-R systems + 1-best of 2 omniASR systems
(results/rerank/<set>.json, built by src/rerank_features.py).
Features per candidate (all reference-free):
  cons     MBR consensus cost: mean word-edit distance to the 7 system 1-bests (/ mean length)
  cons_w   same, weighting each system by its utterance confidence
  nsys     fraction of systems whose top-4 contains the candidate
  ntop     fraction of systems whose 1-best is the candidate
  rank     best beam rank among XLS-R systems (/4)
  ac       best per-frame acoustic score relative to that system's best beam (x100)
  lmw      word 4-gram log10 prob per word;  lmc  char 6-gram log10 prob per char
  oov      fraction of words outside the LM vocabulary;  lendev  |#words - median| / median
Methods (evaluated with leave-one-recording-out cross-validation on OOD + paired bootstrap):
  mbr      argmin cons (= MBR over N-best, the current system)
  mbr_conf argmin cons_w (no parameters)
  mbr_lmc  argmin cons - lam * lmc (lam tuned inside each training fold)
  ranker   pairwise logistic-regression ranker over all features
"""
import json
import os
import re
import sys

import kenlm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from combine import _wdist  # noqa: E402
from evaluate import score  # noqa: E402
from final_eval import refs_for  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated", "omni_llm3b", "omni_ctc3b"]
XLSR = SYS[:5]
FEATS = ["cons", "cons_w", "nsys", "ntop", "rank", "ac", "lmw", "lmc", "oov", "lendev"]


class LMs:
    def __init__(self, name):
        d = os.path.join(ROOT, "results", "lm")
        self.word = kenlm.Model(os.path.join(d, f"{name}.4g.bin"))
        self.char = kenlm.Model(os.path.join(d, f"{name}.char6.bin"))
        self.vocab = {w for line in open(os.path.join(d, f"{name}.txt")) for w in line.split()}


def featurize(entry, lms):
    one = [entry["onebest"].get(s, "") for s in SYS]
    lens = [len(o.split()) for o in one]
    L, med = max(1.0, float(np.mean(lens))), float(np.median(lens))
    conf = entry["conf"]
    mconf = float(np.mean(list(conf.values()))) if conf else 1.0
    cw = [conf.get(s, mconf) for s in SYS]
    best_logit = {}
    for c in entry["cands"].values():
        for s, v in c["sys"].items():
            if "logit" in v:
                best_logit[s] = max(best_logit.get(s, -1e9), v["logit"])
    cands, X = [], []
    for c, info in entry["cands"].items():
        d = [_wdist(c, o) for o in one]
        acs = [(v["logit"] - best_logit[s]) / entry["nframes"] * 100 for s, v in info["sys"].items() if "logit" in v]
        w = c.split()
        X.append([sum(d) / L / len(SYS),
                  sum(a * b for a, b in zip(cw, d)) / sum(cw) / L,
                  len(info["sys"]) / len(SYS),
                  sum(1 for o in one if o == c) / len(SYS),
                  min([v["rank"] for s, v in info["sys"].items() if s in XLSR] or [4]) / 4,
                  max(acs) if acs else np.nan,
                  lms.word.score(c, bos=True, eos=True) / (len(w) + 1) if w else -10.0,
                  lms.char.score(" ".join("_" if ch == " " else ch for ch in c), bos=True, eos=True) / (len(c) + 1) if c else -10.0,
                  float(np.mean([x not in lms.vocab for x in w])) if w else 1.0,
                  abs(len(w) - med) / max(1.0, med)])
        cands.append(c)
    X = np.array(X, dtype=float)
    col = X[:, FEATS.index("ac")]
    col[np.isnan(col)] = (np.nanmin(col) - 1.0) if np.isfinite(col).any() else -5.0
    return cands, X


def load(set_name, lm):
    data = json.load(open(os.path.join(ROOT, "results", "rerank", f"{set_name}.json")))
    lms = LMs(lm)
    return {u: featurize(e, lms) for u, e in data.items()}


def group(u):
    return re.sub(r"_\d+$", "", u)


def pick(cands, X, method, model=None):
    if method == "mbr":
        return cands[int(np.argmin(X[:, 0]))]
    if method == "mbr_conf":
        return cands[int(np.argmin(X[:, 1]))]
    if method == "mbr_lmc":
        return cands[int(np.argmin(X[:, 0] - model * X[:, FEATS.index("lmc")]))]
    if method == "ranker":
        scaler, clf = model
        return cands[int(np.argmax(clf.decision_function(scaler.transform(X))))]
    raise ValueError(method)


def fit(train, refs, method):
    """train: {utt: (cands, X)}; returns the fitted parameter(s) for `method`."""
    if method in ("mbr", "mbr_conf"):
        return None
    if method == "mbr_lmc":
        best = None
        for lam in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
            e = sum(_wdist(refs[u], pick(c, X, "mbr_lmc", lam)) for u, (c, X) in train.items())
            if best is None or e < best[0]:
                best = (e, lam)
        return best[1]
    # pairwise ranking data
    D, y, wts = [], [], []
    for u, (c, X) in train.items():
        err = np.array([_wdist(refs[u], ci) for ci in c], dtype=float)
        for i in range(len(c)):
            for j in range(len(c)):
                if err[i] < err[j]:
                    D.append(X[i] - X[j]); y.append(1); wts.append(err[j] - err[i])
                    D.append(X[j] - X[i]); y.append(0); wts.append(err[j] - err[i])
    D = np.array(D)
    scaler = StandardScaler(with_mean=False).fit(D)
    clf = LogisticRegression(C=0.1, fit_intercept=False, max_iter=2000).fit(scaler.transform(D), y, sample_weight=wts)
    return scaler, clf


def cv_ood(methods):
    feats = load("ood", "all_clean")
    refs = refs_for("ood")
    utts = [u for u in feats if u in refs]
    groups = sorted({group(u) for u in utts})
    sel = {m: {} for m in methods}
    for g in groups:
        tr = {u: feats[u] for u in utts if group(u) != g}
        te = [u for u in utts if group(u) == g]
        for m in methods:
            prm = fit(tr, refs, m)
            for u in te:
                sel[m][u] = pick(*feats[u], m, prm)
    return utts, refs, sel


def bootstrap(utts, refs, a, b, n=2000, seed=0):
    """P(method b has lower total word errors than a) under utterance resampling."""
    rng = np.random.default_rng(seed)
    ea = np.array([_wdist(refs[u], a[u]) for u in utts]); eb = np.array([_wdist(refs[u], b[u]) for u in utts])
    idx = rng.integers(0, len(utts), size=(n, len(utts)))
    return float(((ea[idx].sum(1) - eb[idx].sum(1)) > 0).mean())


if __name__ == "__main__":
    methods = ["mbr", "mbr_conf", "mbr_lmc", "ranker"]
    utts, refs, sel = cv_ood(methods)
    R = [refs[u] for u in utts]
    print("OOD leave-one-recording-out CV (13 recordings):")
    for m in methods:
        w, c = score(R, [sel[m][u] for u in utts])
        p = bootstrap(utts, refs, sel["mbr"], sel[m]) if m != "mbr" else float("nan")
        print(f"  {m:9s} WER {w:6.2f} CER {c:5.2f}   P(better than mbr) = {p:.3f}")
    # secondary check on SPS 5.0 dev+test: parameters fit on all of OOD
    feats_o = load("ood", "all_clean"); refs_o = refs_for("ood")
    tr = {u: feats_o[u] for u in feats_o if u in refs_o}
    feats_s = load("sps_devtest", "all_clean"); refs_s = refs_for("sps_devtest")
    us = [u for u in feats_s if u in refs_s]
    print("SPS 5.0 dev+test (parameters fit on all OOD):")
    for m in methods:
        prm = fit(tr, refs_o, m)
        w, c = score([refs_s[u] for u in us], [pick(*feats_s[u], m, prm) for u in us])
        print(f"  {m:9s} WER {w:6.2f} CER {c:5.2f}" + (f"   lam={prm}" if m == "mbr_lmc" else ""))
    _, clf = fit(tr, refs_o, "ranker")
    print("ranker weights (standardised):", dict(zip(FEATS, np.round(clf.coef_[0], 3))))
