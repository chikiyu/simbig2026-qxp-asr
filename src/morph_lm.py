"""Morfessor-segmented KenLM for N-best rescoring.

Words are segmented with Morfessor Baseline into root + '+suffix' units, so an n-gram LM over
morphs can score unseen (OOV) agglutinated words through their known suffix sequences.
"""
import os
import subprocess
import sys

import kenlm
import morfessor

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LMDIR = os.path.join(ROOT, "results", "lm")
KB = os.environ.get("KENLM_BIN", os.path.join(ROOT, ".kenlm-bin"))  # KenLM build: bin/lmplz, bin/build_binary, lib/


def train_morfessor(corpus_txt, out_bin, count_damp=True):
    io = morfessor.MorfessorIO()
    words = [w for line in open(corpus_txt) for w in line.split()]
    from collections import Counter
    cnt = Counter(words)
    m = morfessor.BaselineModel(corpusweight=0.2)
    data = [(1 if count_damp else c, w) for w, c in cnt.items()]  # type-based training is better for morphology
    m.load_data(data)
    m.train_batch()
    io.write_binary_model_file(out_bin, m)
    return m


class MorphSeg:
    def __init__(self, model):
        self.m = model
        self.cache = {}

    def word(self, w):
        if w not in self.cache:
            segs = self.m.viterbi_segment(w)[0]
            self.cache[w] = [segs[0]] + ["+" + s for s in segs[1:]]
        return self.cache[w]

    def sent(self, s):
        return " ".join(u for w in s.split() for u in self.word(w))


def build_morph_lm(name, corpus_txt, order=6):
    mb = os.path.join(LMDIR, f"{name}.morfessor.bin")
    m = train_morfessor(corpus_txt, mb)
    seg = MorphSeg(m)
    mtxt = os.path.join(LMDIR, f"{name}.morph.txt")
    with open(mtxt, "w") as f:
        for line in open(corpus_txt):
            if line.strip():
                f.write(seg.sent(line.strip()) + "\n")
    arpa = os.path.join(LMDIR, f"{name}.morph.{order}g.arpa")
    env = {**os.environ, "LD_LIBRARY_PATH": f"{KB}/lib"}
    with open(mtxt) as fi, open(arpa, "w") as fo:
        subprocess.run([f"{KB}/bin/lmplz", "-o", str(order), "--discount_fallback", "-S", "20%"],
                       stdin=fi, stdout=fo, stderr=subprocess.DEVNULL, check=True, env=env)
    return mb, arpa


class MorphScorer:
    def __init__(self, name, order=6):
        io = morfessor.MorfessorIO()
        self.seg = MorphSeg(io.read_binary_model_file(os.path.join(LMDIR, f"{name}.morfessor.bin")))
        self.lm = kenlm.Model(os.path.join(LMDIR, f"{name}.morph.{order}g.bin"))

    def __call__(self, sentence):
        """log10 prob of the morph sequence (with sentence boundaries)."""
        return self.lm.score(self.seg.sent(sentence), bos=True, eos=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["all_clean", "all_full"]:
        print(name, build_morph_lm(name, os.path.join(LMDIR, f"{name}.txt")))
        sc = MorphScorer(name)
        seg = sc.seg
        for w in ["allqukunapi", "wasinpi", "llank'ashani", "yachachiqkunaqa", "rikuykuchkan"]:
            print("  ", w, "->", seg.word(w))
