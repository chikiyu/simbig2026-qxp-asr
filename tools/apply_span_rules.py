"""Apply the span rules (src/span_rules.py) to a submission file that already went through the v2 normaliser.

Usage: .venv/bin/python tools/apply_span_rules.py submissions/v10_scrA_spoNbestMBR7.tsv out.tsv
(src/build_final.py --span-rules does the same in one step.)
"""
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from evaluate import NORM  # noqa: E402
from span_rules import span_word_fn  # noqa: E402

word = span_word_fn(NORM)
print(f"{len(word.rules)} span rules")
src, dst = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n").split("\t") for l in open(src, encoding="utf-8") if l.strip()]
out, ch = [], collections.Counter()
for f, h in rows:
    new = " ".join(word(w) for w in h.split())
    for a, b in zip(h.split(), new.split()):
        if a != b:
            ch[(a, b)] += 1
    out.append(f"{f}\t{new}")
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"wrote {dst}: {len(out)} lines; {sum(ch.values())} words changed: {dict(ch)}")
