"""Re-score, from the cached decodes, every result quoted in the paper (Tables 3-5 and text).

Usage: .venv/bin/python tools/verify_paper_results.py
"""
import json
import os
import statistics
import sys
from collections import Counter

import jiwer
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from combine import combine, combine_rover  # noqa: E402
from evaluate import NORM, score  # noqa: E402
from final_eval import decode_system, mbr_nbest_outputs, refs_for  # noqa: E402
from textnorm import basic_norm  # noqa: E402


def P(key, value):
    print(f"{key:72s} {value}")


def f(w_c):
    return f"{w_c[0]:.2f} / {w_c[1]:.2f}"


def sc(set_name, out, ids=None):
    r = refs_for(set_name)
    k = [i for i in r if i in out and (ids is None or i in ids)]
    return score([r[i] for i in k], [out[i] for i in k])


def col(system, set_name, c):
    return decode_system(system, set_name)[c].to_dict()


MBR5 = ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated"]
MBR7 = MBR5 + ["omni_llm3b", "omni_ctc3b"]

# ------------------------------------------------------------ OOD (Table 3 row 1, Table 4, text)
P("OOD n", len(refs_for("ood")))
P("OOD best baseline cpt_cv26_silver official", f(sc("ood", col("cpt_cv26_silver", "ood", "official"))))
for c in ["official", "greedy", "beam"]:
    P(f"OOD cpt_silver {c}", f(sc("ood", col("cpt_silver", "ood", c))))
P("OOD cpt_validated official", f(sc("ood", col("cpt_validated", "ood", "official"))))
o5 = [col(s, "ood", "beam") for s in MBR5]
o7 = [col(s, "ood", "beam") for s in MBR7]
P("OOD MBR5 1-best", f(sc("ood", combine(o5))))
P("OOD MBR7 1-best", f(sc("ood", combine(o7))))
P("OOD ROVER7", f(sc("ood", combine_rover(o7))))
nb = {K: mbr_nbest_outputs(MBR7, "ood", K) for K in [2, 4, 8]}
for K in [2, 4, 8]:
    P(f"OOD MBR7 N-best K={K}", f(sc("ood", nb[K])))
w0, c0 = sc("ood", col("cpt_cv26_silver", "ood", "official"))
w1, c1 = sc("ood", nb[4])
P("OOD relative reduction WER / CER (%)", f"{100 * (w0 - w1) / w0:.1f} / {100 * (c0 - c1) / c0:.1f}")
P("OOD omni_llm3b greedy+norm / official", f"{f(sc('ood', col('omni_llm3b', 'ood', 'greedy')))}  |  {f(sc('ood', col('omni_llm3b', 'ood', 'official')))}")
P("OOD omni_ctc3b greedy+norm", f(sc("ood", col("omni_ctc3b", "ood", "greedy"))))
for s in ["ft_A", "ft_C", "ft_D", "ft_curated", "cpt_validated"]:
    P(f"OOD {s} beam", f(sc("ood", col(s, "ood", "beam"))))
P("OOD MBR6 (MBR7 without ft_D) N-best K=4", f(sc("ood", mbr_nbest_outputs([s for s in MBR7 if s != "ft_D"], "ood", 4))))

# ------------------------------------------------------------ CV clip-disjoint (100) and held-out (234)
for s in ["cpt_validated", "cpt_silver", "cpt_cv26_validated", "cpt_cv26_silver"]:
    P(f"clip-disjoint {s} official", f(sc("scr_clipclean", col(s, "scr_clipclean", "official"))))
P("clip-disjoint cpt_validated greedy+norm", f(sc("scr_clipclean", col("cpt_validated", "scr_clipclean", "greedy"))))
for s in ["ft_A", "ft_C", "ft_D", "ft_A26"]:
    P(f"clip-disjoint {s} beam", f(sc("scr_clipclean", col(s, "scr_clipclean", "beam"))))
P("clip-disjoint MBR(A,A26,D) 1-best / N-best", f"{f(sc('scr_clipclean', combine([col(s, 'scr_clipclean', 'beam') for s in ['ft_A', 'ft_A26', 'ft_D']])))}  |  {f(sc('scr_clipclean', mbr_nbest_outputs(['ft_A', 'ft_A26', 'ft_D'], 'scr_clipclean', 4)))}")
for s in ["cpt_validated", "cpt_silver"]:
    P(f"held-out {s} official", f(sc("scr_heldout", col(s, "scr_heldout", "official"))))
for s in ["ft_A", "ft_S3", "ft_S2"]:
    P(f"held-out {s} beam", f(sc("scr_heldout", col(s, "scr_heldout", "beam"))))
P("held-out n", len(refs_for("scr_heldout")))
P("clip-disjoint n", len(refs_for("scr_clipclean")))

# ------------------------------------------------------------ SPS 5.0 clean 190 + behavioural test
ids190 = set(json.load(open("results/sps_clean_for_paper_models_ids.json")))
for s in ["cpt_validated", "cpt_silver", "cpt_cv26_silver"]:
    P(f"SPS190 {s} official", f(sc("sps_devtest", col(s, "sps_devtest", "official"), ids190)))
for s in ["cpt_validated", "cpt_silver"]:
    P(f"SPS190 {s} beam", f(sc("sps_devtest", col(s, "sps_devtest", "beam"), ids190)))
P("SPS190 MBR6 (no ft_D) N-best K=4", f(sc("sps_devtest", mbr_nbest_outputs([s for s in MBR7 if s != "ft_D"], "sps_devtest", 4), ids190)))
P("SPS190 MBR4 clean XLS-R N-best K=4", f(sc("sps_devtest", mbr_nbest_outputs(["ft_C", "cpt_silver", "ft_curated", "cpt_validated"], "sps_devtest", 4), ids190)))
P("SPS190 ft_C beam", f(sc("sps_devtest", col("ft_C", "sps_devtest", "beam"), ids190)))
P("SPS full 596 MBR7 N-best K=4 (cv26 member: contaminated)", f(sc("sps_devtest", nb4 := mbr_nbest_outputs(MBR7, "sps_devtest", 4))))
P("SPS full 596 MBR7 1-best / ROVER7", f"{f(sc('sps_devtest', combine([col(s, 'sps_devtest', 'beam') for s in MBR7])))}  |  {f(sc('sps_devtest', combine_rover([col(s, 'sps_devtest', 'beam') for s in MBR7])))}")

# ------------------------------------------------------------ oracle (pool = top-4 of 5 XLS-R + 1-best of 2 omni)
rr = json.load(open("results/rerank/ood.json"))
R = refs_for("ood")


def werr(ref, hyp):
    o = jiwer.process_words(ref, hyp)
    return o.substitutions + o.deletions + o.insertions


utts = [u for u in rr if u in R]
N = sum(len(R[u].split()) for u in utts)
P("oracle: utts / mean candidates per utt", f"{len(utts)} / {statistics.mean(len(rr[u]['cands']) for u in utts):.1f}")
P("oracle WER, N-best pool", f"{100 * sum(min(werr(R[u], c) for c in rr[u]['cands']) for u in utts) / N:.2f}")
P("oracle WER, the seven 1-best", f"{100 * sum(min(werr(R[u], h) for h in rr[u]['onebest'].values()) for u in utts) / N:.2f}")
P("oracle: OOD recordings (groups)", len({u.rsplit('_', 1)[0] for u in utts}))

# ------------------------------------------------------------ error analysis of the submitted spontaneous system
vocab = set(NORM.vocab) if hasattr(NORM, "vocab") else set()
P("curated vocabulary size used for the analysis", len(vocab))
for set_name, out in [("ood", nb[4]), ("sps_devtest", nb4)]:
    r = refs_for(set_name)
    k = [i for i in r if i in out]
    tot = seg = cur_sub = n = 0
    for i in k:
        o = jiwer.process_words(r[i], out[i])
        rw, hw = r[i].split(), out[i].split()
        n += len(rw)
        tot += o.substitutions + o.deletions + o.insertions
        chunks = o.alignments[0]
        j = 0
        while j < len(chunks):
            if chunks[j].type == "equal":
                j += 1
                continue
            a = j
            while j < len(chunks) and chunks[j].type != "equal":
                j += 1
            rs = rw[chunks[a].ref_start_idx:chunks[j - 1].ref_end_idx]
            hs = hw[chunks[a].hyp_start_idx:chunks[j - 1].hyp_end_idx]
            e = sum(max(c.ref_end_idx - c.ref_start_idx, c.hyp_end_idx - c.hyp_start_idx) for c in chunks[a:j])
            if rs and hs and "".join(rs) == "".join(hs):
                seg += e
        for c in chunks:
            if c.type == "substitute":
                for x, y in zip(rw[c.ref_start_idx:c.ref_end_idx], hw[c.hyp_start_idx:c.hyp_end_idx]):
                    if y in vocab and x not in vocab:
                        cur_sub += 1
    P(f"{set_name}: errors / ref words / WER", f"{tot} / {n} / {100 * tot / n:.2f}")
    P(f"{set_name}: compound-segmentation errors (% of errors)", f"{seg} ({100 * seg / tot:.1f}%)")
    P(f"{set_name}: subs hyp-in-curated & ref-not (% of errors) -> WER w/o them", f"{cur_sub} ({100 * cur_sub / tot:.1f}%) -> {100 * (tot - cur_sub) / n:.2f}")

# ------------------------------------------------------------ utterance lengths
for set_name in ["ood", "sps_devtest", "scr_heldout", "scr_clipclean"]:
    r = refs_for(set_name)
    P(f"mean words per reference: {set_name}", f"{statistics.mean(len(x.split()) for x in r.values()):.1f}")
sub = [l.rstrip("\n").split("\t") for l in open("submissions/qxp.tsv", encoding="utf-8")]
import pandas as pd  # noqa: E402
meta = pd.read_csv("qxp_test_dataset/qxp_test_dataset/qxp_test_dataset.tsv", sep="\t", dtype=str)
typ = dict(zip(meta.file_name, meta.type))
for t in ["scripted", "spontaneous"]:
    hs = [h for fn, h in sub if typ[fn] == t]
    P(f"blind hypotheses {t}: n / mean words", f"{len(hs)} / {statistics.mean(len(h.split()) for h in hs):.1f}")

# ------------------------------------------------------------ LM size, OOV, blind sentences vs CV pool
for lm in ["all_clean", "all_full"]:
    w = [x for l in open(f"results/lm/{lm}.txt", encoding="utf-8") for x in l.split()]
    P(f"LM text {lm}: lines / words / vocabulary", f"{sum(1 for _ in open(f'results/lm/{lm}.txt', encoding='utf-8'))} / {len(w)} / {len(set(w))}")
voc = {x for l in open("results/lm/all_full.txt", encoding="utf-8") for x in l.split()}
toks = [w for _, h in sub for w in h.split()]
P("blind hypothesis tokens OOV wrt all_full LM (%)", f"{100 * sum(w not in voc for w in toks) / len(toks):.1f}")
P("blind hypothesis types OOV wrt all_full LM (%)", f"{100 * sum(w not in voc for w in set(toks)) / len(set(toks)):.1f}")
for t in ["scripted", "spontaneous"]:
    tk = [w for fn, h in sub if typ[fn] == t for w in h.split()]
    P(f"  {t}: tokens OOV (%)", f"{100 * sum(w not in voc for w in tk) / len(tk):.1f}")
cur = pd.read_csv("asr-puno-quechua/data/shared-task/validated_sentences_curated.tsv", sep="\t", dtype=str)
pool = [basic_norm(s) for s in cur.sentence_curated]
scr = [h for fn, h in sub if typ[fn] == "scripted"]
d = [process.extractOne(h, pool, scorer=Levenshtein.normalized_distance)[1] for h in scr]
P("blind scripted hyps: median normalised char edit distance to closest CV sentence", f"{statistics.median(d):.2f}")
P("blind scripted hyps: exact matches with a CV sentence", f"{sum(x == 0 for x in d)} / {len(d)}")

# ------------------------------------------------------------ normaliser
nj = json.load(open("results/lm/normalizer.json"))
P("normaliser word maps / rules", f"{len(nj['wmap'])} / {len(nj['rules'])}")

# ------------------------------------------------------------ numbers added in the revised paper (Table 2 columns, text)
P("OOD best baseline + our normaliser (cpt_cv26_silver greedy+norm)", f(sc("ood", col("cpt_cv26_silver", "ood", "greedy"))))
P("clip-disjoint best baseline + our normaliser (cpt_validated greedy+norm)", f(sc("scr_clipclean", col("cpt_validated", "scr_clipclean", "greedy"))))
P("SPS190 best baseline + our normaliser (cpt_silver greedy+norm)", f(sc("sps_devtest", col("cpt_silver", "sps_devtest", "greedy"), ids190)))
from decode import sets_meta  # noqa: E402
voc_clean = {w for l in open("results/lm/all_clean.txt", encoding="utf-8") for w in l.split()}
m = sets_meta()
for s in ["ood", "sps_devtest"]:
    rt = [w for r in refs_for(s).values() for w in r.split()]
    P(f"{s}: reference tokens not in the local LM vocabulary (%)", f"{100 * sum(w not in voc_clean for w in rt) / len(rt):.1f}")
    a = b = 0
    for i, r in m[s]:
        if r:
            x, y = basic_norm(r).split(), NORM(basic_norm(r)).split()
            a += len(x); b += sum(p != q for p, q in zip(x, y))
    P(f"{s}: reference words changed by the normaliser (%)", f"{100 * b / a:.1f}")
P("MBR candidates per utterance: upper bound (5 x 4 + 2)", 5 * 4 + 2)
