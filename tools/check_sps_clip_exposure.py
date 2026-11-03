"""Check by clip ID whether the 190 'SPS 5.0 clean' clips were in the training data of the original models.

Usage: python3 tools/check_sps_clip_exposure.py
The clean set is defined by text (its SPS 5.0 transcriptions are in neither manifest). This script also
uses the clip IDs of the task repository's splits: a clip whose earlier transcription (SPS 3.0 split or
silver list) is in the V / V+S training manifests was trained on even if its SPS 5.0 text changed.
Clips longer than 30 s never appear in the manifests (length filter), which the script also prints.
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from textnorm import basic_norm as n  # noqa: E402

BL = os.path.join(ROOT, "asr-puno-quechua", "data")
MAN = os.path.join(BL, "manifests", "finetune", "qxp_v2")


def ltr(name):
    return {n(l.rstrip("\n").replace(" ", "").replace("|", " ").strip()) for l in open(os.path.join(MAN, name), encoding="utf-8")}


def tsv(p):
    d = pd.read_csv(p, sep="\t", quoting=3, dtype=str, keep_default_na=False)
    d["cid"] = d.path.str.replace(".mp3", "", regex=False)
    return d


V, VS = ltr("train.ltr"), ltr("train_silver.ltr")
ids = set(json.load(open(os.path.join(ROOT, "results", "sps_clean_for_paper_models_ids.json"))))
hum = tsv(os.path.join(BL, "splits", "validated_spontaneous", "train_spontaneous.tsv"))
sil = tsv(os.path.join(BL, "splits", "silver_spontaneous", "train.tsv"))
hum["ms"] = hum.duration_ms.astype(int)
hum["inV"] = [n(t) in V for t in hum.sentence]
print("human SPS train split: in V manifest by duration > 30 s:", hum.groupby(hum.ms > 30000).inV.agg(["sum", "count"]).to_dict())
sil["inVS"] = [n(t) in VS for t in sil.sentence]
exposed = set(hum[hum.cid.isin(ids) & hum.inV].cid) | set(sil[sil.cid.isin(ids) & sil.inVS].cid)
print("clean clips in human SPS train split:", int(hum.cid.isin(ids).sum()), "| in silver train split:", int(sil.cid.isin(ids).sum()))
print("clean clips whose earlier transcription is in a training manifest:", len(exposed), sorted(exposed))
