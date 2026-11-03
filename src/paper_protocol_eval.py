"""Evaluate under the baseline paper's own protocol (Huaman et al., 2026) and under the curated
protocol of the shared task, on the paper's three test sets:

  scripted : qxp_v2 test (2,296 clips, CV v25 split; refs = paper's test.ltr, original CV orthography)
  spont    : qxp_v2 test_spont (360 clips; refs = paper's test_spont.ltr) — 353 of them are in our logits
  ood      : Add_data radio/social media (109 clips; refs = native-speaker transcriptions)

protocol 'paper'  : hypothesis and reference only lower-cased / punctuation-stripped (official_norm)
protocol 'curated': reference converted to the curated norm (NORM), as for the blind test
"""
import json
import os
import sys

import jiwer
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from final_eval import decode_system, raw_greedy  # noqa: E402
from decode import LOGITS  # noqa: E402
from evaluate import NORM  # noqa: E402
from textnorm import official_norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, "asr-puno-quechua", "data", "manifests", "finetune", "qxp_v2")


def ltr_refs(split):
    stems = [os.path.splitext(os.path.basename(l.split("\t")[0]))[0]
             for l in open(os.path.join(MAN, f"{split}.tsv")).read().split("\n")[1:] if l.strip()]
    ltr = [" ".join(l.strip().replace(" ", "").replace("|", " ").split()) for l in open(os.path.join(MAN, f"{split}.ltr")) if l.strip()]
    return {s: official_norm(r) for s, r in zip(stems, ltr)}


def ood_refs():
    d = pd.read_csv(os.path.join(ROOT, "asr-puno-quechua", "data", "additional_data", "additional_data_qxp.tsv"), sep="\t", quoting=3, dtype=str)
    return {p.replace(".wav", ""): official_norm(s) for p, s in zip(d.path, d.sentence) if isinstance(s, str)}


def sc(refs, hyps):
    ids = [i for i in refs if i in hyps]
    R, H = [refs[i] for i in ids], [hyps[i] for i in ids]
    return 100 * jiwer.wer(R, H), 100 * jiwer.cer(R, H), len(ids)


SETS = {"scripted (paper test, 2,296)": ("scr_test", lambda: ltr_refs("test")),
        "spontaneous (paper test_spont, 360)": ("sps_devtest", lambda: ltr_refs("test_spont")),
        "OOD (paper Add_data, 109)": ("ood", ood_refs)}

if __name__ == "__main__":
    systems = sys.argv[1:] or ["cpt_validated", "cpt_silver", "cpt_cv26_silver", "ft_D"]
    rows = []
    for sname, (logit_set, rf) in SETS.items():
        refs = rf()
        for s in systems:
            d = os.path.join(LOGITS, s)
            if not os.path.exists(os.path.join(d, f"{logit_set}.npz")):
                continue
            info = json.load(open(os.path.join(d, "vocab.json")))
            z = np.load(os.path.join(d, f"{logit_set}.npz"))
            off = {k: raw_greedy(z[k].astype(np.float32), info["vocab"], info["blank"]) for k in z.files if k in refs}
            w1, c1, n = sc(refs, off)
            w2, c2, _ = sc({i: NORM(r) for i, r in refs.items()}, off)
            rows.append((sname, s, n, w1, c1, w2, c2))
            print(f"{sname:38s} {s:16s} n={n:4d} | paper protocol {w1:6.2f}/{c1:5.2f} | curated refs {w2:6.2f}/{c2:5.2f}", flush=True)
    pd.DataFrame(rows, columns=["set", "system", "n", "paper_WER", "paper_CER", "curated_WER", "curated_CER"]).to_csv(
        os.path.join(ROOT, "results", "paper_protocol.tsv"), sep="\t", index=False)
