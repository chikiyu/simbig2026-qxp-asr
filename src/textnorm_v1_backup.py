"""Text normalisation for Puno Quechua ASR.

- official_norm: the shared-task scoring normalisation (NFC, lowercase, no punctuation, keep ' and ñ).
- learn_word_map / apply_word_map: data-driven CV-orthography -> curated-orthography word mapping,
  learned from aligned (original, curated) sentence pairs.
"""
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

APOS = r"[’‘`´ʻʼ]"


def official_norm(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(APOS, "'", s)
    s = re.sub(r"[^\w\s']|_", " ", s)
    s = re.sub(r"(?<!\w)'|'(?!\w)", " ", s)  # stray apostrophes not inside a word
    return re.sub(r"\s+", " ", s).strip()


def strip_accents(s):
    return s.translate(str.maketrans("áéíóúàèìòùü", "aeiouaeiouu"))


def basic_norm(s):
    """official_norm + accent stripping (accents never occur in the curated norm)."""
    return strip_accents(official_norm(s))


def aligned_word_pairs(src, tgt):
    """Yield (src_word, tgt_word) for 1:1 aligned positions (equal or 1:1 replaced)."""
    a, b = src.split(), tgt.split()
    for op, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for x, y in zip(a[i1:i2], b[j1:j2]):
                yield x, y


def learn_word_map(pairs, min_count=1, min_ratio=0.6):
    """pairs: iterable of (original_sentence, curated_sentence), both already basic_norm'ed.
    Returns {src_word: tgt_word} for words whose majority curated form differs from them."""
    c = defaultdict(Counter)
    curated_vocab = Counter()
    for s, t in pairs:
        curated_vocab.update(t.split())
        for x, y in aligned_word_pairs(s, t):
            c[x][y] += 1
    m = {}
    for x, cnt in c.items():
        y, n = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if y != x and n >= min_count and n / tot >= min_ratio:
            m[x] = y
    return m


def apply_word_map(text, m):
    return " ".join(m.get(w, w) for w in text.split())


def learn_rules(pairs, ctx=2, min_prec=0.9, min_n=2):
    """Learn substring rewrite rules (with up to `ctx` chars of context, '^'/'$' word anchors)
    from aligned word pairs. A rule is kept if, over all aligned training words containing
    its source pattern, applying it yields the curated form in >= min_prec of cases."""
    cand = Counter()
    words = Counter()
    for o, c in pairs:
        for x, y in aligned_word_pairs(o, c):
            words[(x, y)] += 1
            if x == y:
                continue
            X, Y = "^" + x + "$", "^" + y + "$"
            for op, i1, i2, j1, j2 in SequenceMatcher(a=X, b=Y, autojunk=False).get_opcodes():
                if op == "equal":
                    continue
                for L in range(ctx + 1):
                    for R in range(ctx + 1):
                        if L + R == 0 and i2 - i1 < 2:
                            continue
                        src = X[max(0, i1 - L):i2 + R]
                        tgt = X[max(0, i1 - L):i1] + Y[j1:j2] + X[i2:i2 + R]
                        if src:
                            cand[(src, tgt)] += 1
    rules = []
    for (src, tgt), n in cand.items():
        if n < min_n:
            continue
        good = bad = 0
        for (x, y), k in words.items():
            X = "^" + x + "$"
            if src in X:
                if X.replace(src, tgt).strip("^$") == y:
                    good += k
                else:
                    bad += k
        if good / (good + bad) >= min_prec:
            rules.append((good, src, tgt))
    rules.sort(key=lambda r: (-len(r[1]), -r[0]))
    return [(a, b) for _, a, b in rules]


def apply_rules(w, rules):
    X = "^" + w + "$"
    for a, b in rules:
        if a in X:
            X = X.replace(a, b)
    return X.strip("^$")


class Normalizer:
    """basic_norm -> learned word map (exact) -> learned substring rules (unseen words)."""

    def __init__(self, wmap, rules):
        self.wmap, self.rules = wmap, rules

    @classmethod
    def from_pairs(cls, pairs):
        pairs = [(basic_norm(o), basic_norm(c)) for o, c in pairs]
        return cls(learn_word_map(pairs), learn_rules(pairs))

    def __call__(self, text):
        return " ".join(self.wmap[w] if w in self.wmap else apply_rules(w, self.rules)
                        for w in basic_norm(text).split())
