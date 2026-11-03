"""Collect the candidate pool + per-candidate evidence for learned system combination.

For each set (ood, sps_devtest, blind) and each XLS-R system, the top-K distinct normalised beam
hypotheses are stored with their acoustic (logit) score, combined (acoustic+LM) score and rank; each
system's utterance-level confidence (mean max posterior over non-blank frames) is stored too.
zero-shot omniASR systems contribute their 1-best text.
Output: results/rerank/<set>.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from decode import LOGITS, Folded  # noqa: E402
from evaluate import NORM, make_decoder  # noqa: E402
from final_eval import OMNI, P, SETS, decode_system  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSR = ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated"]
OMN = ["omni_llm3b", "omni_ctc3b"]
K = 4


def collect(set_name):
    src, lm = SETS[set_name]
    out = {}
    for s in XLSR:
        info = json.load(open(os.path.join(LOGITS, s, "vocab.json")))
        z = np.load(os.path.join(LOGITS, s, f"{src}.npz"))
        fold = Folded(info["vocab"], info["blank"])
        dec = make_decoder(fold.labels, os.path.join(ROOT, "results", "lm", f"{lm}.4g.bin"), P["alpha"], P["beta"], P["unk"])
        for u in z.files:
            lp = fold(z[u].astype(np.float32))
            post = np.exp(lp)
            nb = post.argmax(-1) != 0
            conf = float(post.max(-1)[nb].mean()) if nb.any() else 0.0
            e = out.setdefault(u, {"cands": {}, "conf": {}, "nframes": int(lp.shape[0]), "onebest": {}})
            e["conf"][s] = conf
            seen = []
            for bm in dec.decode_beams(lp, beam_width=P["beam"]):
                t = NORM(bm[0])
                if t in seen:
                    continue
                seen.append(t)
                c = e["cands"].setdefault(t, {"sys": {}})
                c["sys"][s] = {"rank": len(seen) - 1, "logit": float(bm[3]), "comb": float(bm[4])}
                if len(seen) >= K:
                    break
            e["onebest"][s] = seen[0] if seen else ""
        print(f"{set_name}: {s} done", flush=True)
    for s in OMN:
        d = decode_system(s, set_name)
        for u, h in d["beam"].items():
            if u in out:
                c = out[u]["cands"].setdefault(h, {"sys": {}})
                c["sys"][s] = {"rank": 0}
                out[u]["onebest"][s] = h
    os.makedirs(os.path.join(ROOT, "results", "rerank"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "rerank", f"{set_name}.json"), "w"), ensure_ascii=False)
    print(f"{set_name}: {len(out)} utterances, avg candidates {np.mean([len(e['cands']) for e in out.values()]):.1f}", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["ood", "sps_devtest", "blind"]:
        collect(s)
