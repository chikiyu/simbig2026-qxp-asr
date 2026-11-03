"""Consistent final evaluation of every system on every leakage-free set, with one pipeline.

For each (system, set) it caches three outputs in results/final/<set>/<system>.tsv:
  official : greedy CTC on the raw vocabulary, official normalisation only (= released baseline style)
  greedy   : greedy CTC (folded vocabulary) + learned normaliser
  beam     : beam 64 + KenLM + learned normaliser
omniASR systems (zero-shot) are text-only: official = official_norm(hyp); greedy = beam = NORM(hyp).

Sets (references in the curated orthography, as confirmed for the blind test):
  ood          109 radio/interview clips, never seen by any model          LM all_clean
  sps_devtest  596 SPS 5.0 dev+test clips (young speakers)                 LM all_clean
  scr_heldout  234 CV test clips whose sentences were held out of S* models LM all_clean_sd
  blind        659 blind-test clips (no references; decoded for the submission) LM all_full

Usage: python src/final_eval.py [--systems ...] [--tables]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from combine import combine, combine_rover  # noqa: E402
from decode import LOGITS, Folded, greedy, sets_meta  # noqa: E402
from evaluate import NORM, make_decoder, score  # noqa: E402
from textnorm import basic_norm, official_norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "final")
SETS = {"ood": ("ood", "all_clean"), "sps_devtest": ("sps_devtest", "all_clean"),
        "scr_heldout": ("scr_test", "all_clean_sd"), "scr_clipclean": ("scr_test", "all_clean_sd"),
        "blind": ("blind", "all_full")}
XLSR = ["cpt_validated", "cpt_silver", "cpt_cv26_silver", "ft_curated", "ft_A", "ft_C", "ft_D",
        "ft_S", "ft_S2", "ft_S3", "ft_G"]
OMNI = {"omni_llm3b": "omniASR_LLM_3B_v2", "omni_ctc3b": "omniASR_CTC_3B_v2", "omni_llm1b": "omniASR_LLM_1B_v2",
        "omni_ctc1b": "omniASR_CTC_1B_v2", "omni_llm300m": "omniASR_LLM_300M_v2"}
P = dict(alpha=0.3, beta=2.0, unk=-4.0, beam=64)
HELD = set(json.load(open(os.path.join(ROOT, "results", "scr_test_heldout_ids.json"))))
# 100 CV-test clips that lie in the cv26 *dev* split: no released or fine-tuned model was trained on these clips
CLIPCLEAN = set(json.load(open(os.path.join(ROOT, "results", "scr_test_cv26dev_ids.json"))))
# models whose starting checkpoint was trained on splits_cv26/train (contains CV-test and SPS5 dev/test clips)
CV26_BASED = {"cpt_cv26_validated", "cpt_cv26_silver", "ft_D", "ft_S", "ft_G", "ft_S4", "ft_A26"}
SUBSET = {"scr_heldout": HELD, "scr_clipclean": CLIPCLEAN}


def raw_greedy(lp, vocab, blank):
    ids = lp.argmax(-1)
    out, prev = [], -1
    for t in ids:
        if t != prev and t != blank:
            out.append(vocab[t])
        prev = t
    return official_norm("".join(out).replace("<unk>", " ").replace("|", " ").replace("<pad>", "").replace("</s>", ""))


def decode_system(system, set_name):
    src, lm = SETS[set_name]
    os.makedirs(os.path.join(OUT, set_name), exist_ok=True)
    f = os.path.join(OUT, set_name, f"{system}.tsv")
    if os.path.exists(f):
        return pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False).set_index("id")
    rows = []
    if system in OMNI:
        p = os.path.join(ROOT, "results", "omni", OMNI[system], f"{set_name if set_name != 'scr_heldout' else 'scr_heldout'}.tsv")
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
        rows = [(i, official_norm(h), NORM(h), NORM(h)) for i, h in zip(d.id, d.hyp)]
    else:
        d = os.path.join(LOGITS, system)
        npz = os.path.join(d, f"{src}.npz")
        if not os.path.exists(npz):
            return None
        info = json.load(open(os.path.join(d, "vocab.json")))
        z = np.load(npz)
        fold = Folded(info["vocab"], info["blank"])
        dec = make_decoder(fold.labels, os.path.join(ROOT, "results", "lm", f"{lm}.4g.bin"), P["alpha"], P["beta"], P["unk"])
        keys = [k for k in z.files if set_name not in SUBSET or k in SUBSET[set_name]]
        for k in keys:
            lp = z[k].astype(np.float32)
            flp = fold(lp)
            rows.append((k, raw_greedy(lp, info["vocab"], info["blank"]), NORM(greedy(flp, fold.labels)),
                         NORM(dec.decode(flp, beam_width=P["beam"]))))
    df = pd.DataFrame(rows, columns=["id", "official", "greedy", "beam"])
    df.to_csv(f, sep="\t", index=False)
    print(f"decoded {system:16s} {set_name:12s} n={len(df)}", flush=True)
    return df.set_index("id")


def decode_nbest(system, set_name, K=4):
    """Top-K distinct normalised beam hypotheses per utterance (cached JSON). omniASR: its 1-best."""
    src, lm = SETS[set_name]
    f = os.path.join(OUT, set_name, f"{system}.nbest{K}.json")
    if os.path.exists(f):
        return json.load(open(f))
    if system in OMNI:
        d = decode_system(system, set_name)
        res = {i: [h] for i, h in d["beam"].items()} if d is not None else None
    else:
        d = os.path.join(LOGITS, system)
        npz = os.path.join(d, f"{src}.npz")
        if not os.path.exists(npz):
            return None
        info = json.load(open(os.path.join(d, "vocab.json")))
        z = np.load(npz)
        fold = Folded(info["vocab"], info["blank"])
        dec = make_decoder(fold.labels, os.path.join(ROOT, "results", "lm", f"{lm}.4g.bin"), P["alpha"], P["beta"], P["unk"])
        res = {}
        for k in z.files:
            if set_name in SUBSET and k not in SUBSET[set_name]:
                continue
            seen = []
            for bm in dec.decode_beams(fold(z[k].astype(np.float32)), beam_width=P["beam"]):
                t = NORM(bm[0])
                if t not in seen:
                    seen.append(t)
                if len(seen) >= K:
                    break
            res[k] = seen
    if res is not None:
        json.dump(res, open(f, "w"), ensure_ascii=False)
    return res


def mbr_nbest_outputs(systems, set_name, K=4):
    from combine import mbr_nbest
    one = {s: decode_system(s, set_name)["beam"].to_dict() for s in systems}
    nb = {s: decode_nbest(s, set_name, K) for s in systems}
    ids = [i for i in one[systems[0]] if all(i in one[s] for s in systems)]
    return {i: mbr_nbest([c for s in systems for c in nb[s][i]], [one[s][i] for s in systems]) for i in ids}


def refs_for(set_name):
    src, _ = SETS[set_name]
    refs = {i: NORM(basic_norm(r)) for i, r in sets_meta()[src] if r}
    if set_name in SUBSET:
        refs = {i: r for i, r in refs.items() if i in SUBSET[set_name]}
    return refs


def fmt(w, c):
    return f"{w:.2f} / {c:.2f}"


def tables(systems):
    lines = ["# Final evaluation tables (auto-generated by src/final_eval.py)", "",
             "Models based on the released *-cv26-* checkpoints were trained (via splits_cv26/train) on 2,199/2,299 CV-test clips "
             "and 589/596 SPS 5.0 dev+test clips: their CV/SPS numbers are marked invalid; OOD and `scr_clipclean` (100 clips "
             "in the cv26 dev split) are clean for every model.", "",
             "WER / CER (%) against references in the curated orthography.",
             "`official` = released-baseline style (greedy, no LM, no normaliser); `greedy` = greedy + learned normaliser; "
             "`beam` = beam 64 + KenLM + learned normaliser.", ""]
    groups = {
        "MBR5 (C,D,cpt_silver,ft_curated,cpt_validated)": ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated"],
        "MBR7 (MBR5 + omni LLM/CTC 3B)  [pre-registered spontaneous system]": ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated", "omni_llm3b", "omni_ctc3b"],
        "MBR7+G (MBR7 + ft_G)": ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated", "omni_llm3b", "omni_ctc3b", "ft_G"],
        "MBR all omni (MBR7 + omni 1B/300M)": ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated", "omni_llm3b", "omni_ctc3b", "omni_llm1b", "omni_ctc1b", "omni_llm300m"],
        "MBR scripted analogue (A,C,D)": ["ft_A", "ft_C", "ft_D"],
        "MBR scripted (A,A26,D)": ["ft_A", "ft_A26", "ft_D"],
        "MBR S-family (S,S2,S3)": ["ft_S", "ft_S2", "ft_S3"],
        "MBR S-family + omni 3B": ["ft_S", "ft_S2", "ft_S3", "omni_llm3b", "omni_ctc3b"],
    }
    for set_name in ["ood", "sps_devtest", "scr_heldout", "scr_clipclean"]:
        refs = refs_for(set_name)
        H = {s: decode_system(s, set_name) for s in systems}
        H = {s: h for s, h in H.items() if h is not None}
        lines += [f"## {set_name} (n={len(refs)})", "", "| system | official | greedy+norm | beam+LM+norm |", "|---|---|---|---|"]
        for s, h in H.items():
            ids = [i for i in refs if i in h.index]
            if len(ids) < len(refs) * 0.95:
                continue
            R = [refs[i] for i in ids]
            cells = [fmt(*score(R, [h.loc[i, col] for i in ids])) for col in ["official", "greedy", "beam"]]
            note = ""
            if s in CV26_BASED and set_name in ("sps_devtest", "scr_heldout"):
                note = " **(base trained on these clips: invalid)**"
            elif set_name == "scr_heldout" and not s.startswith("ft_S") and not s.startswith("omni"):
                note = " *(saw these sentences in training)*"
            lines.append(f"| {s}{note} | " + " | ".join(cells) + " |")
        for gname, members in groups.items():
            if not all(m in H for m in members):
                continue
            ids = [i for i in refs if all(i in H[m].index for m in members)]
            R = [refs[i] for i in ids]
            outs = [H[m]["beam"].to_dict() for m in members]
            mb, rv = combine(outs), combine_rover(outs)
            nb = mbr_nbest_outputs(members, set_name, 4)
            lines.append(f"| **{gname}** — MBR | | | {fmt(*score(R, [mb[i] for i in ids]))} |")
            lines.append(f"| {gname} — MBR N-best K=4 | | | {fmt(*score(R, [nb[i] for i in ids]))} |")
            lines.append(f"| {gname} — ROVER | | | {fmt(*score(R, [rv[i] for i in ids]))} |")
        lines.append("")
    open(os.path.join(ROOT, "results", "final_tables.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=XLSR + list(OMNI))
    ap.add_argument("--decode-only", action="store_true")
    a = ap.parse_args()
    if a.decode_only:
        for s in a.systems:
            for set_name in SETS:
                decode_system(s, set_name)
    else:
        tables(a.systems)
