"""Build the final submission reproducibly from dumped log-probs + zero-shot omniASR outputs.

    python src/build_final.py --scripted ft_A --spontaneous ft_C ft_D cpt_silver ft_curated cpt_validated omni:omniASR_LLM_3B_v2 omni:omniASR_CTC_3B_v2 --nbest 4 --span-rules --out submissions/qxp.tsv

(final submission v11; without --span-rules it builds v10)

Each acoustic model is decoded with beam search (64) + KenLM (`--lm`, default all_full) and the
learned orthographic normaliser; `omni:<card>` entries read zero-shot omniASR transcriptions from
results/omni/<card>/blind.tsv (normalised). When a domain lists several systems, the hypotheses are
combined with MBR selection (src/combine.py). Per-system hypotheses are cached in results/blind_hyps/.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from combine import combine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "qxp_test_dataset", "qxp_test_dataset", "qxp_test_dataset.tsv")
TEMPLATE = os.path.join(ROOT, "qxp_test_dataset", "qxp_test_dataset", "qxp.tsv")


def system_hyps(name, lm):
    """Blind-test hypotheses (beam + KenLM + normaliser) from the single consistent cache of
    src/final_eval.py (results/final/blind/<system>.tsv); decoded on demand if missing."""
    from final_eval import OMNI, decode_system
    key = name[5:] if name.startswith("omni:") else name
    if name.startswith("omni:"):
        key = {v: k for k, v in OMNI.items()}.get(key, key)
    df = decode_system(key, "blind")
    if df is None:
        raise SystemExit(f"no blind outputs for {name}")
    return df["beam"].to_dict()


def _key(name):
    from final_eval import OMNI
    if name.startswith("omni:"):
        return {v: k for k, v in OMNI.items()}.get(name[5:], name[5:])
    return name


def domain_output(systems, lm, nbest=0):
    if len(systems) > 1 and nbest > 0:
        from final_eval import mbr_nbest_outputs
        return mbr_nbest_outputs([_key(s) for s in systems], "blind", nbest)
    outs = [system_hyps(s, lm) for s in systems]
    return outs[0] if len(outs) == 1 else combine(outs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripted", nargs="+", required=True)
    ap.add_argument("--spontaneous", nargs="+", required=True)
    ap.add_argument("--lm", default="all_full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nbest", type=int, default=4, help="N-best MBR (0 = 1-best MBR)")
    ap.add_argument("--span-rules", action="store_true", help="apply the span rules of src/span_rules.py (v11)")
    a = ap.parse_args()
    meta = pd.read_csv(META, sep="\t").set_index("file_name")
    scr, spo = domain_output(a.scripted, a.lm, a.nbest), domain_output(a.spontaneous, a.lm, a.nbest)
    order = pd.read_csv(TEMPLATE, sep="\t", header=None)[0]
    rows = [(f, scr[f] if meta.type[f] == "scripted" else spo[f]) for f in order]
    if a.span_rules:
        from evaluate import NORM
        from span_rules import span_word_fn
        word = span_word_fn(NORM)
        rows = [(f, " ".join(word(w) for w in h.split())) for f, h in rows]
    df = pd.DataFrame(rows)
    assert len(df) == 659 and df[0].is_unique and set(df[0]) == set(meta.index)
    assert not df[1].str.contains(r"[áéíóú.,;:¿?¡!]").any()
    df.to_csv(a.out, sep="\t", header=False, index=False)
    print(f"wrote {a.out}: {len(df)} lines, empty={int((df[1].str.strip() == '').sum())}")
