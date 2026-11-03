"""Paired bootstrap over the 13 OOD recordings for the WER differences quoted in the paper.

Usage: .venv/bin/python tools/bootstrap_ood.py
Compares the submitted MBR configuration with (i) the best released checkpoint plus our normaliser
and (ii) the best single system with the LM. Resamples recordings, not utterances (10,000 draws, seed 0).
"""
import os, sys, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import jiwer
from final_eval import decode_system, mbr_nbest_outputs, refs_for
R = refs_for("ood")
col = lambda s, c: decode_system(s, "ood")[c].to_dict()
MBR7 = ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated", "omni_llm3b", "omni_ctc3b"]
sysd = {
    "base_norm(cv26 greedy+norm)": col("cpt_cv26_silver", "greedy"),
    "best_single(cpt_silver beam)": col("cpt_silver", "beam"),
    "mbr7_k4": mbr_nbest_outputs(MBR7, "ood", 4),
}
utts = sorted(R)
def errs(hyp):
    out = {}
    for u in utts:
        o = jiwer.process_words(R[u], hyp[u])
        out[u] = (o.substitutions + o.deletions + o.insertions, len(R[u].split()))
    return out
E = {k: errs(v) for k, v in sysd.items()}
groups = {}
for u in utts: groups.setdefault(u.rsplit("_", 1)[0], []).append(u)
G = list(groups)
print("utts", len(utts), "recordings", len(G), "words", sum(E["mbr7_k4"][u][1] for u in utts))
for k in E: print(k, "errors", sum(E[k][u][0] for u in utts))
rng = random.Random(0)
def wer(k, us): return 100 * sum(E[k][u][0] for u in us) / sum(E[k][u][1] for u in us)
for a, b in [("base_norm(cv26 greedy+norm)", "mbr7_k4"), ("best_single(cpt_silver beam)", "mbr7_k4")]:
    d = []
    for _ in range(10000):
        us = [u for g in (rng.choice(G) for _ in G) for u in groups[g]]
        d.append(wer(a, us) - wer(b, us))
    d.sort()
    print(f"{a} - {b}: point {wer(a, utts) - wer(b, utts):.2f}, 95% CI [{d[250]:.2f}, {d[9749]:.2f}], P(diff<=0) {sum(x <= 0 for x in d) / len(d):.4f}")
    # per-recording wins
    w = sum(wer(a, groups[g]) > wer(b, groups[g]) for g in G); l = sum(wer(a, groups[g]) < wer(b, groups[g]) for g in G)
    print(f"   recordings where {b} is better / worse: {w} / {l}")
