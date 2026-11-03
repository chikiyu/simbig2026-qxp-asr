"""Build text corpora and KenLM word n-gram models.

Corpora (all normalised to the curated orthography: basic_norm + learned CV->curated word map):
  scr_pool   : unique curated CV sentences (2,051 texts)
  sps_gold   : SPS validated transcriptions
  silver     : omniASR silver transcriptions of SPS (from the official repo)
  ood        : radio/social OOD transcriptions
"clean" variants drop every SPS5 dev/test clip (eval sets) from sps_gold/silver.
"""
import glob
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from textnorm import Normalizer, basic_norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "lm")
KB = os.environ.get("KENLM_BIN", os.path.join(ROOT, ".kenlm-bin"))  # KenLM build: bin/lmplz, bin/build_binary, lib/
CV = glob.glob(os.path.join(ROOT, "dataset common voice ", "*cv-corpus-27*", "*", "qxp", "validated.tsv"))[0]
SPS5 = glob.glob(os.path.join(ROOT, "dataset common voice ", "*sps-corpus-5.0*", "*", "ss-corpus-qxp.tsv"))[0]
REPO = os.path.join(ROOT, "asr-puno-quechua", "data")


def corpora():
    v = pd.read_csv(CV, sep="\t", quoting=3, dtype=str)
    cur = pd.read_csv(os.path.join(REPO, "shared-task", "validated_sentences_curated.tsv"), sep="\t", quoting=3, dtype=str)
    s = v.drop_duplicates("sentence_id")[["sentence_id", "sentence"]].merge(cur, on="sentence_id")
    sn = Normalizer.from_pairs(zip(s.sentence, s.sentence_curated))
    wmap = sn.wmap
    scr_pool = sorted(set(s.sentence_curated.map(basic_norm)))
    sps = pd.read_csv(SPS5, sep="\t", quoting=3, dtype=str)
    held = set(sps[sps.split.isin(["dev", "test"])].audio_file)
    gold = sps[sps.transcription.notna()]
    sil = pd.concat([pd.read_csv(f, sep="\t", quoting=3, dtype=str) for f in glob.glob(os.path.join(REPO, "splits", "silver_spontaneous", "*.tsv"))])
    sil = sil[sil.sentence.notna() & ~sil.path.isin(set(gold.audio_file))]  # silver only for unannotated clips
    ood = pd.read_csv(os.path.join(REPO, "additional_data", "additional_data_qxp.tsv"), sep="\t", quoting=3, dtype=str)
    c = {
        "scr_pool": scr_pool,
        "sps_gold_full": [sn(t) for t in gold.transcription],
        "sps_gold_clean": [sn(t) for t in gold[~gold.audio_file.isin(held)].transcription],
        "silver_full": [sn(t) for t in sil.sentence],
        "silver_clean": [sn(t) for t in sil[~sil.path.isin(held)].sentence],
        "ood": [sn(t) for t in ood.sentence.dropna()],
    }
    return {k: [x for x in val if x] for k, val in c.items()}, wmap


def kenlm(name, lines, order=4):
    txt = os.path.join(OUT, f"{name}.txt")
    open(txt, "w").write("\n".join(lines) + "\n")
    arpa = os.path.join(OUT, f"{name}.{order}g.arpa")
    env = {**os.environ, "LD_LIBRARY_PATH": f"{KB}/lib"}
    with open(txt) as fi, open(arpa, "w") as fo:
        subprocess.run([f"{KB}/bin/lmplz", "-o", str(order), "--discount_fallback", "-S", "20%"],
                       stdin=fi, stdout=fo, stderr=subprocess.DEVNULL, check=True, env=env)
    return arpa


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    c, wmap = corpora()
    json.dump(wmap, open(os.path.join(OUT, "wordmap.json"), "w"), ensure_ascii=False, indent=0)
    for k, val in c.items():
        print(f"{k:16s} lines {len(val):6d} words {sum(len(x.split()) for x in val):7d}")
    combos = {
        "scr": c["scr_pool"],
        "spont_clean": c["sps_gold_clean"] + c["silver_clean"],
        "all_clean": c["scr_pool"] + c["sps_gold_clean"] + c["silver_clean"],
        "all_full": c["scr_pool"] + c["sps_gold_full"] + c["silver_full"] + c["ood"],
    }
    for k, val in combos.items():
        print("built", kenlm(k, val))
