"""Pre-registered test of new spontaneous-MBR members (qxp-final kernel).

Rule: a candidate member (or group) is adopted only if MBR7 N-best (K=4) + candidate improves OOD WER by
>= 0.3 points, OOD CER does not get worse, and SPS 5.0 WER does not get worse by more than 0.2.
Usage: python tools/eval_new_members.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from evaluate import score  # noqa: E402
from final_eval import decode_system, mbr_nbest_outputs, refs_for  # noqa: E402

BASE = ["ft_C", "ft_D", "cpt_silver", "ft_curated", "cpt_validated", "omni_llm3b", "omni_ctc3b"]
CANDIDATES = {"+B": ["ft_B"], "+noCPT(XS,XV)": ["xlsr_silver", "xlsr_validated"], "+omni1B": ["omni_llm1b", "omni_ctc1b"]}


def available(systems, set_name):
    return all(decode_system(s, set_name) is not None for s in systems)


def run(members, set_name):
    r = refs_for(set_name)
    out = mbr_nbest_outputs(members, set_name, 4)
    ids = [i for i in r if i in out]
    return score([r[i] for i in ids], [out[i] for i in ids])


if __name__ == "__main__":
    res = {}
    groups = {"MBR7 (current)": BASE}
    for name, extra in CANDIDATES.items():
        if available(extra, "ood") and available(extra, "sps_devtest"):
            groups["MBR7 " + name] = BASE + extra
    allx = [x for e in CANDIDATES.values() for x in e]
    if available(allx, "ood") and available(allx, "sps_devtest"):
        groups["MBR7 + all new"] = BASE + allx
    for g, members in groups.items():
        res[g] = (run(members, "ood"), run(members, "sps_devtest"))
        (wo, co), (ws, cs) = res[g]
        print(f"{g:24s} OOD {wo:6.2f}/{co:5.2f}  SPS {ws:6.2f}/{cs:5.2f}")
    (bwo, bco), (bws, _) = res["MBR7 (current)"]
    print("\nPre-registered decision:")
    for g, ((wo, co), (ws, _)) in res.items():
        if g == "MBR7 (current)":
            continue
        ok = (bwo - wo) >= 0.3 and co <= bco and (ws - bws) <= 0.2
        print(f"  {g:24s} dWER_OOD {wo - bwo:+.2f}  dCER_OOD {co - bco:+.2f}  dWER_SPS {ws - bws:+.2f}  -> {'ADOPT' if ok else 'reject'}")
