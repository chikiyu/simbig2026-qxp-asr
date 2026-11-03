"""SPS 5.0 clean scores with and without the 3 clips found by tools/check_sps_clip_exposure.py.

Usage: .venv/bin/python tools/sps_clean_without_exposed.py
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from evaluate import score
from final_eval import decode_system, mbr_nbest_outputs, refs_for

def sc(out, ids):
    r = refs_for("sps_devtest")
    k = [i for i in r if i in out and i in ids]
    return score([r[i] for i in k], [out[i] for i in k]), len(k), sum(len(r[i].split()) for i in k)

col = lambda s, c: decode_system(s, "sps_devtest")[c].to_dict()
ids190 = set(json.load(open("results/sps_clean_for_paper_models_ids.json")))
bad = {"spontaneous-speech-qxp-67261", "spontaneous-speech-qxp-68628", "spontaneous-speech-qxp-70083"}
print("bad in ids190:", len(bad & ids190))
ids187 = ids190 - bad
MBR7 = ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated", "omni_llm3b", "omni_ctc3b"]
rows = [
    ("cpt_silver official (T2 as released)", col("cpt_silver", "official")),
    ("cpt_cv26_silver official (2.2: 12.8)", col("cpt_cv26_silver", "official")),
    ("cpt_silver greedy+norm (T2 +norm)", col("cpt_silver", "greedy")),
    ("cpt_silver beam", col("cpt_silver", "beam")),
    ("ft_C beam", col("ft_C", "beam")),
    ("MBR6 N-best K=4 (T2 ours)", mbr_nbest_outputs([s for s in MBR7 if s != "ft_D"], "sps_devtest", 4)),
    ("MBR4 clean XLS-R K=4 (13.52)", mbr_nbest_outputs(["ft_C", "cpt_silver", "ft_curated", "cpt_validated"], "sps_devtest", 4)),
]
for name, out in rows:
    (w1, c1), n1, wd1 = sc(out, ids190)
    (w2, c2), n2, wd2 = sc(out, ids187)
    print(f"{name:42s} 190: {w1:.2f}/{c1:.2f} (n={n1}, words={wd1})   187: {w2:.2f}/{c2:.2f} (n={n2}, words={wd2})")
