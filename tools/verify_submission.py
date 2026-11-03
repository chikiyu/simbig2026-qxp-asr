"""Checks on the blind-test submission (submissions/qxp.tsv): format, characters, content, agreement with
other systems; writes results/submission_review.tsv with the utterances worth listening to.

Usage: .venv/bin/python tools/verify_submission.py [submissions/qxp.tsv]
"""
import collections
import os
import re
import statistics
import sys
import unicodedata

import jiwer
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from final_eval import decode_system  # noqa: E402
from textnorm import official_norm  # noqa: E402

SUB = sys.argv[1] if len(sys.argv) > 1 else "submissions/qxp.tsv"
D = os.path.join("qxp_test_dataset", "qxp_test_dataset")


def P(k, v):
    print(f"{k:64s} {v}")


raw = open(SUB, "rb").read()
P("file is valid UTF-8", raw.decode("utf-8", errors="strict") is not None)
lines = raw.decode("utf-8").split("\n")
P("ends with a newline / trailing empty lines", f"{raw.endswith(b'\n')} / {sum(1 for l in lines[:-1] if l == '')}")
lines = [l for l in lines if l != ""]
P("lines", len(lines))
P("lines with exactly one TAB", sum(l.count("\t") == 1 for l in lines))
P("Windows line endings (\\r)", sum("\r" in l for l in lines))
rows = [l.split("\t") for l in lines]
tmpl = [l.split("\t")[0] for l in open(os.path.join(D, "qxp.tsv"), encoding="utf-8").read().splitlines() if l.strip()]
P("same files and same order as the official template", [r[0] for r in rows] == tmpl)
meta = pd.read_csv(os.path.join(D, "qxp_test_dataset.tsv"), sep="\t", dtype=str).set_index("file_name")
P("every file exists in the metadata / audio folder", f"{all(r[0] in meta.index for r in rows)} / {all(os.path.exists(os.path.join(D, 'audios', r[0])) for r in rows)}")
hyp = {r[0]: r[1] for r in rows}
P("empty transcriptions", sum(not h.strip() for h in hyp.values()))
P("leading/trailing or double spaces", sum(h != h.strip() or "  " in h for h in hyp.values()))
P("already NFC", all(unicodedata.normalize("NFC", h) == h for h in hyp.values()))
P("unchanged by the official scoring normalisation", sum(official_norm(h) == h for h in hyp.values()))
chars = collections.Counter(c for h in hyp.values() for c in h)
P("character inventory", "".join(sorted(chars)))
cur = pd.read_csv("asr-puno-quechua/data/shared-task/validated_sentences_curated.tsv", sep="\t", dtype=str)
cchars = set("".join(cur.sentence_curated.str.lower()))
P("characters not used in the curated sentences", {c: n for c, n in chars.items() if c not in cchars})
P("non-ASCII apostrophes (’ ‘ ` ´ ʼ)", sum(h.count(c) for h in hyp.values() for c in "’‘`´ʼ"))
P("digits / '<unk>' / '|' tokens", f"{sum(bool(re.search(r'[0-9]', h)) for h in hyp.values())} / {sum('<unk>' in h for h in hyp.values())} / {sum('|' in h for h in hyp.values())}")
P("'quqi' (old normaliser bug)", sum(" quqi" in " " + h for h in hyp.values()))
P("same word 3+ times in a row", sum(bool(re.search(r"\b(\w+)( \1){2,}\b", h)) for h in hyp.values()))

typ = meta.type.to_dict()
for t in ["scripted", "spontaneous"]:
    hs = [h for f, h in hyp.items() if typ[f] == t]
    n = [len(h.split()) for h in hs]
    P(f"{t}: n / mean words / min / max / one-word outputs", f"{len(hs)} / {statistics.mean(n):.1f} / {min(n)} / {max(n)} / {sum(x == 1 for x in n)}")
voc = {w for s in cur.sentence_curated for w in official_norm(s).split()}
voc |= {w for l in open("results/lm/all_full.txt", encoding="utf-8") for w in l.split()}
oov = collections.Counter(w for h in hyp.values() for w in h.split() if w not in voc)
toks = sum(len(h.split()) for h in hyp.values())
P("tokens not in curated sentences or LM text (%)", f"{100 * sum(oov.values()) / toks:.1f}")
P("most frequent such words", oov.most_common(15))

# agreement with other systems (no references: disagreement = WER between two outputs)
alts = {"scripted": ["ft_A26", "ft_D", "ft_C", "cpt_validated"], "spontaneous": ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated", "omni_llm3b", "omni_ctc3b"]}
H = {s: decode_system(s, "blind")["beam"].to_dict() for s in set(alts["scripted"]) | set(alts["spontaneous"]) | {"ft_A"}}
for t, ss in alts.items():
    fs = [f for f in hyp if typ[f] == t]
    for s in ss:
        P(f"{t}: word disagreement with {s} (%)", f"{100 * jiwer.wer([hyp[f] for f in fs], [H[s][f] for f in fs]):.1f}")
fs = [f for f in hyp if typ[f] == "scripted"]
P("scripted: submission identical to ft_A beam output", f"{sum(hyp[f] == H['ft_A'][f] for f in fs)} / {len(fs)}")

# review list: utterances where the strongest alternatives disagree most with the submission
rows = []
for f, h in hyp.items():
    t = typ[f]
    ref_alts = ["ft_A26", "ft_D"] if t == "scripted" else ["ft_C", "ft_D", "omni_llm3b"]
    d = statistics.mean(jiwer.wer(h, H[s][f]) if h.strip() else 1.0 for s in ref_alts)
    rows.append({"file_name": f, "type": t, "disagreement": round(d, 3), "submitted": h,
                 **{f"alt_{s}": H[s][f] for s in ref_alts}})
rv = pd.DataFrame(rows).sort_values(["type", "disagreement"], ascending=[True, False])
out = pd.concat([rv[rv.type == t].head(25) for t in ["scripted", "spontaneous"]])
out.to_csv("results/submission_review.tsv", sep="\t", index=False)
for t in ["scripted", "spontaneous"]:
    r = rv[rv.type == t]
    P(f"{t}: utterances with any disagreement / mean disagreement", f"{int((r.disagreement > 0).sum())} / {r.disagreement.mean():.3f}")
P("review file (25 most disputed per type)", "results/submission_review.tsv")
