"""Normaliser 5-fold cross-validation (md5 folds) and behavioural leakage test on SPS 5.0 dev.

Usage: .venv/bin/python tools/verify_paper_extra.py
"""
import os, sys, json, hashlib, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import jiwer, pandas as pd
from textnorm import Normalizer, basic_norm
import textnorm_v1_backup as v1
def P(k, v): print(f"{k:72s} {v}")
val = pd.read_csv(sorted(__import__('glob').glob(ROOT + "/dataset common voice /*cv-corpus-26.0*/**/validated.tsv", recursive=True))[0], sep="\t", quoting=3, dtype=str, keep_default_na=False)
cur = pd.read_csv("asr-puno-quechua/data/shared-task/validated_sentences_curated.tsv", sep="\t", quoting=3, dtype=str)
pairs = val.drop_duplicates("sentence_id").merge(cur, on="sentence_id")[["sentence_id", "sentence", "sentence_curated"]]
P("pairs", len(pairs))
fold = pairs.sentence_id.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 5)
R, H0, H2, H1 = [], [], [], []
ch2 = ch1 = tot = 0
for k in range(5):
    tr, te = pairs[fold != k], pairs[fold == k]
    n2 = Normalizer.from_pairs(list(zip(tr.sentence, tr.sentence_curated)))
    has_v1 = hasattr(v1, "Normalizer") and hasattr(v1.Normalizer, "from_pairs")
    n1 = v1.Normalizer.from_pairs(list(zip(tr.sentence, tr.sentence_curated))) if has_v1 else None
    for o, c in zip(te.sentence, te.sentence_curated):
        R.append(basic_norm(c)); H0.append(basic_norm(o)); H2.append(n2(o))
        if n1: H1.append(n1(o))
        cw = basic_norm(c).split(); tot += len(cw)
        ch2 += sum(a != b for a, b in zip(cw, n2(c).split()))
        if n1: ch1 += sum(a != b for a, b in zip(cw, n1(c).split()))
P("original vs curated WER", f"{100*jiwer.wer(R, H0):.2f}")
P("v2 normalised vs curated WER (5-fold by md5)", f"{100*jiwer.wer(R, H2):.2f}")
P("v2 curated words altered (%)", f"{100*ch2/tot:.2f}")
if H1:
    P("v1 (no idempotence filter) WER / curated words altered (%)", f"{100*jiwer.wer(R, H1):.2f} / {100*ch1/tot:.2f}")
# v2 + span rules (as applied to the submission: on top of the v2 output), same md5 folds
from textnorm import learn_span_rules
HS, chs = [], 0
for k in range(5):
    tr, te = pairs[fold != k], pairs[fold == k]
    n2 = Normalizer.from_pairs(list(zip(tr.sentence, tr.sentence_curated)))
    rules = learn_span_rules([(basic_norm(o), basic_norm(c)) for o, c in zip(tr.sentence, tr.sentence_curated)], n2.vocab)
    def sw(w):
        if w in n2.vocab or w in n2.wmap:
            return w
        x = "^" + w + "$"
        for a_, b_ in rules:
            if a_ in x:
                x = x.replace(a_, b_)
        return x.strip("^$")
    for o, c in zip(te.sentence, te.sentence_curated):
        HS.append(" ".join(sw(w) for w in n2(o).split()))
        cw = basic_norm(c).split()
        chs += sum(x != y for x, y in zip(cw, [sw(w) for w in n2(c).split()]))
P("v2 + span rules (submission) WER / curated words altered (%)", f"{100*jiwer.wer(R, HS):.2f} / {100*chs/tot:.2f}")
# behavioural test on SPS5 dev: seen (text in V / V+S training manifest) vs unseen
spec = importlib.util.spec_from_file_location("vd", os.path.join(ROOT, "tools/verify_paper_data.py"))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    vd = importlib.util.module_from_spec(spec); spec.loader.exec_module(vd)
from final_eval import decode_system, refs_for
refs = refs_for("sps_devtest")
dev = vd.dt[vd.dt.split == "dev"]
txt = {a.replace(".mp3", ""): basic_norm(t) for a, t in zip(dev.audio_file, dev.transcription)}
for s, S in [("cpt_validated", vd.Vn), ("cpt_silver", vd.VSn)]:
    h = decode_system(s, "sps_devtest")["official"].to_dict()
    seen = [i for i in txt if i in refs and i in h and txt[i] in S]; uns = [i for i in txt if i in refs and i in h and txt[i] not in S]
    w = lambda ids: 100 * jiwer.wer([refs[i] for i in ids], [h[i] for i in ids])
    P(f"behavioural SPS5 dev {s}: seen (n) / unseen (n)", f"{w(seen):.2f} ({len(seen)}) / {w(uns):.2f} ({len(uns)})")
