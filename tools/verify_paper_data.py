"""Recompute, from the raw corpora and manifests, every data-derived number quoted in the paper.

Usage: .venv/bin/python tools/verify_paper_data.py
Prints one line per fact; compare with main.tex of the paper repo.
"""
import glob
import hashlib
import json
import os
import statistics
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from textnorm import basic_norm  # noqa: E402

DS = os.path.join(ROOT, "dataset common voice ")
BL = os.path.join(ROOT, "asr-puno-quechua", "data")
MAN = os.path.join(BL, "manifests", "finetune", "qxp_v2")


def tsv(p, **kw):
    return pd.read_csv(p, sep="\t", quoting=3, dtype=str, keep_default_na=False, **kw)


def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def ltr_texts(name):
    return [l.rstrip("\n").replace(" ", "").replace("|", " ").strip() for l in open(os.path.join(MAN, name), encoding="utf-8")]


def P(key, value):
    print(f"{key:70s} {value}")


# ---------------------------------------------------------------- Common Voice scripted
cvdir = os.path.dirname(one(os.path.join(DS, "*cv-corpus-26.0*", "**", "validated.tsv")))
val = tsv(os.path.join(cvdir, "validated.tsv"))
dur = tsv(os.path.join(cvdir, "clip_durations.tsv"))
dur = dict(zip(dur["clip"], dur["duration[ms]"].astype(int)))
cur = tsv(os.path.join(BL, "shared-task", "validated_sentences_curated.tsv"))
P("CV26 validated clips", len(val))
P("CV26 validated distinct sentence_id", val.sentence_id.nunique())
P("CV26 validated speakers", val.client_id.nunique())
P("CV26 validated hours", round(sum(dur[p] for p in val.path) / 3.6e6, 2))
cov = val[val.sentence_id.isin(set(cur.sentence_id))]
P("curated sentences (rows)", len(cur))
P("curated distinct texts", cur.sentence_curated.nunique())
P("validated clips covered by curated sentences", len(cov))
P("  speakers of those clips", cov.client_id.nunique())
P("  hours of those clips", round(sum(dur[p] for p in cov.path) / 3.6e6, 2))
P("  readings per sentence (clips / sentences)", round(len(cov) / len(cur), 2))
toks = [w for s in cur.sentence_curated for w in basic_norm(s).split()]
L = [len(w) for w in toks]
P("curated tokens / types", f"{len(toks)} / {len(set(toks))}")
P("curated tokens with >=15 chars (%)", round(100 * sum(x >= 15 for x in L) / len(L), 1))
P("curated tokens with >=10 chars (%)", round(100 * sum(x >= 10 for x in L) / len(L), 1))
P("curated types with >=15 chars (%)", round(100 * sum(len(w) >= 15 for w in set(toks)) / len(set(toks)), 1))
P("longest curated word (chars)", max(L))

# ---------------------------------------------------------------- Spontaneous Speech 5.0
sps = tsv(one(os.path.join(DS, "*sps-corpus-5.0*", "**", "ss-corpus-qxp.tsv")))
sps["ms"] = pd.to_numeric(sps.duration_ms, errors="coerce").fillna(0)
tr = sps[sps.transcription.str.strip() != ""]
un = sps[sps.transcription.str.strip() == ""]
P("SPS5 rows", len(sps))
P("SPS5 transcribed clips / hours", f"{len(tr)} / {tr.ms.sum() / 3.6e6:.2f}")
P("SPS5 untranscribed clips / hours", f"{len(un)} / {un.ms.sum() / 3.6e6:.2f}")
P("SPS5 transcribed split counts", dict(Counter(tr.split)))
P("SPS5 transcribed age counts", dict(Counter(tr.age)))
old = {"fifties", "sixties", "seventies", "eighties", "nineties"}
P("SPS5 transcribed clips aged 50+", int(tr.age.isin(old).sum()))
P("SPS5 transcribed clips aged 60+", int(tr.age.isin(old - {"fifties"}).sum()))
P("SPS5 all clips aged 50+ / hours", f"{int(sps.age.isin(old).sum())} / {sps[sps.age.isin(old)].ms.sum() / 3.6e6:.2f}")
dt = tr[tr.split.isin(["dev", "test"])]
P("SPS5 dev+test transcribed clips", len(dt))

# silver as released by the baseline repo
sil = pd.concat([tsv(os.path.join(BL, "splits", "silver_spontaneous", f"{s}.tsv")) for s in ["train", "dev", "test"]])
sms = dict(zip(sps.audio_file, sps.ms))
P("baseline silver clips (train+dev+test)", len(sil))
P("  of which have SPS5 duration / hours", f"{sum(p in sms for p in sil.path)} / {sum(sms.get(p, 0) for p in sil.path) / 3.6e6:.2f}")
P("  of which are SPS5-transcribed clips", int(sil.path.isin(set(tr.audio_file)).sum()))
P("  of which are SPS5-untranscribed clips", int(sil.path.isin(set(un.audio_file)).sum()))

# ---------------------------------------------------------------- OOD and blind test
ood = tsv(os.path.join(BL, "additional_data", "additional_data_qxp.tsv"))
P("OOD clips", len(ood))
bl = tsv(os.path.join(ROOT, "qxp_test_dataset", "qxp_test_dataset", "qxp_test_dataset.tsv"))
P("blind clips by type", dict(Counter(bl.type)))
P("blind (type, age)", dict(Counter(zip(bl.type, bl.age))))
try:
    import soundfile as sf
    adir = os.path.join(ROOT, "qxp_test_dataset", "qxp_test_dataset", "audios")
    d = {f: sf.info(os.path.join(adir, f)).duration for f in bl.file_name}
    for t in ["scripted", "spontaneous"]:
        P(f"blind {t} minutes", round(sum(d[f] for f in bl[bl.type == t].file_name) / 60, 1))
except Exception as e:  # noqa: BLE001
    P("blind durations", f"not computed ({e})")

# ---------------------------------------------------------------- audit of the baseline benchmark
V, VS = ltr_texts("train.ltr"), ltr_texts("train_silver.ltr")
n = lambda s: basic_norm(s)  # noqa: E731
Vn, VSn = {n(t) for t in V}, {n(t) for t in VS}
spont, test = ltr_texts("test_spont.ltr"), ltr_texts("test.ltr")
P("manifest lines train(V) / train_silver(V+S) / test / test_spont", f"{len(V)} / {len(VS)} / {len(test)} / {len(spont)}")
P("(1) test_spont refs verbatim in V+S manifest", f"{sum(t in set(VS) for t in spont)} / {len(spont)}")
P("(1) test_spont refs verbatim in V manifest", f"{sum(t in set(V) for t in spont)} / {len(spont)}")
P("(1) same after basic_norm (V+S / V)", f"{sum(n(t) in VSn for t in spont)} / {sum(n(t) in Vn for t in spont)}")
allsps = Counter(n(t) for t in tr.transcription)
P("(1) test_spont texts unique in SPS5 (count == 1)", f"{sum(allsps[n(t)] == 1 for t in spont)} / {len(spont)}")
P("(1) test_spont distinct texts", len(set(spont)))
P("(2) scripted test texts found in V manifest", f"{sum(n(t) in Vn for t in test)} / {len(test)}")
dtn = [n(t) for t in dt.transcription]
P("(3) SPS5 dev+test texts in V manifest", f"{sum(t in Vn for t in dtn)} / {len(dtn)}")
P("(3) SPS5 dev+test texts in V+S manifest", f"{sum(t in VSn for t in dtn)} / {len(dtn)}")
clean = [f.replace('.mp3', '') for f, t in zip(dt.audio_file, dtn) if t not in Vn and t not in VSn]
ref190 = set(json.load(open(os.path.join(ROOT, "results", "sps_clean_for_paper_models_ids.json"))))
P("(3) SPS5 dev+test clips clean for V and V+S", f"{len(clean)} (json: {len(ref190)}, identical: {set(clean) == ref190})")
P("(3) clean clips by split", dict(Counter(dt.set_index(dt.audio_file.str.replace('.mp3', '')).loc[clean].split)))
c26s = tsv(os.path.join(BL, "splits_cv26", "scripted", "train.tsv"))
c26d = tsv(os.path.join(BL, "splits_cv26", "scripted", "dev.tsv"))
c26p = tsv(os.path.join(BL, "splits_cv26", "spontaneous", "train.tsv"))
tclips = [l.split("\t")[0].split("/")[-1].replace(".wav", ".mp3") for l in open(os.path.join(MAN, "test.tsv"), encoding="utf-8").read().splitlines()[1:]]
P("(4) scripted test clips in cv26 scripted train", f"{sum(c in set(c26s.path) for c in tclips)} / {len(tclips)}")
P("(4) scripted test clips in cv26 scripted dev", f"{sum(c in set(c26d.path) for c in tclips)} / {len(tclips)}")
P("(4) SPS5 dev+test clips in cv26 spontaneous train", f"{int(dt.audio_file.isin(set(c26p.audio_file)).sum())} / {len(dt)}")
cd_ids = set(json.load(open(os.path.join(ROOT, "results", "scr_test_cv26dev_ids.json"))))
P("(b) clip-disjoint ids json / recomputed", f"{len(cd_ids)} / {sum(c in set(c26d.path) for c in tclips)}")

# ---------------------------------------------------------------- held-out sentences
sid = dict(zip(val.path, val.sentence_id))
held = {s for s in cur.sentence_id if int(hashlib.md5(s.encode()).hexdigest(), 16) % 10 == 0}
P("(c) held-out sentences (md5 % 10 == 0) among curated", len(held))
hclips = [c for c in tclips if sid.get(c) in held]
hj = set(json.load(open(os.path.join(ROOT, "results", "scr_test_heldout_ids.json"))))
P("(c) scripted test clips with held-out sentences", f"{len(hclips)} (json: {len(hj)}); distinct sentences: {len({sid[c] for c in hclips})}")

# ------------------------------------------------------------ numbers added in the revised paper
P("SPS5 dev+test ages", dict(Counter(dt.age)))
P("SPS clean-190 ages", dict(Counter(dt[dt.audio_file.str.replace('.mp3', '').isin(ref190)].age)))
c26p_ids = {a.replace(".mp3", "") for a in c26p.audio_file}
P("SPS clean-190 clips in the cv26 spontaneous train split", f"{len(ref190 & c26p_ids)} / {len(ref190)}")
spk_cv = dict(zip(val.path.str.replace(".mp3", ""), val.client_id))
cds = {spk_cv[i] for i in cd_ids if i in spk_cv}
otr = tsv(os.path.join(BL, "splits", "validated_scripted", "train.tsv"))
P("CV clip-disjoint: speakers / of them in cv26 train / in original train split", f"{len(cds)} / {len(cds & set(c26s.client_id))} / {len(cds & set(otr.client_id))}")
