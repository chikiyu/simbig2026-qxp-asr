"""Hypothesis-level system combination (MBR selection).

For each utterance, among the 1-best hypotheses of N systems, pick the one with minimum weighted
expected word-edit distance to the others (a robust alternative to ROVER for few systems).
"""
import Levenshtein


def _wdist(a, b):
    wa, wb = a.split(), b.split()
    vocab = {w: chr(0x100 + i) for i, w in enumerate(set(wa) | set(wb))}
    return Levenshtein.distance("".join(vocab[w] for w in wa), "".join(vocab[w] for w in wb))


def mbr_select(hyps, weights=None):
    """hyps: list of hypothesis strings (one per system). Returns the selected string."""
    n = len(hyps)
    weights = weights or [1.0] * n
    best, best_cost = hyps[0], None
    for i, h in enumerate(hyps):
        cost = sum(weights[j] * _wdist(h, hyps[j]) for j in range(n) if j != i) - 1e-6 * weights[i]
        if best_cost is None or cost < best_cost:
            best, best_cost = h, cost
    return best


def combine(system_outputs, weights=None):
    """system_outputs: list of {id: hyp}; returns {id: combined hyp}."""
    ids = system_outputs[0].keys()
    return {i: mbr_select([s[i] for s in system_outputs], weights) for i in ids}


def _align(ref, hyp):
    """Word-level Levenshtein alignment. Returns list of (ref_word|None, hyp_word|None)."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out.append((ref[i - 1], None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]


def rover(hyps, weights=None):
    """ROVER-style word voting: align every hypothesis to the MBR pivot, build slots
    (pivot words + insertion slots between them), and take the weighted majority per slot
    (empty = deletion). Ties go to the pivot."""
    n = len(hyps)
    weights = weights or [1.0] * n
    pivot = mbr_select(hyps, weights).split()
    P = len(pivot)
    # slot 2k+1 = pivot word k ; slot 2k = insertions before pivot word k (2P = after last)
    slots = [dict() for _ in range(2 * P + 1)]
    for h, w in zip(hyps, weights):
        k, ins = 0, {}
        for rw, hw in _align(pivot, h.split()):
            if rw is not None:
                slots[2 * k + 1][hw] = slots[2 * k + 1].get(hw, 0.0) + w
                k += 1
            else:
                ins.setdefault(2 * k, []).append(hw)
        for s, words in ins.items():
            key = " ".join(words)
            slots[s][key] = slots[s].get(key, 0.0) + w
    total = sum(weights)
    out = []
    for idx, sl in enumerate(slots):
        if idx % 2 == 1:
            piv = pivot[idx // 2]
            best = max(sl.items(), key=lambda kv: (kv[1], kv[0] == piv))
            if best[0] is not None:
                out.append(best[0])
        else:
            if sl:
                best = max(sl.items(), key=lambda kv: kv[1])
                if best[1] > total - sum(sl.values()) and best[1] > total / 2:
                    out.append(best[0])
    return " ".join(out)


def combine_rover(system_outputs, weights=None):
    ids = system_outputs[0].keys()
    return {i: rover([s[i] for s in system_outputs], weights) for i in ids}


def mbr_nbest(candidates, evidence, weights=None):
    """MBR selection over an enlarged hypothesis space: pick, among `candidates` (e.g. the union of
    the top-K beams of every system), the string with minimum weighted word-edit distance to the
    `evidence` hypotheses (the 1-best of every system)."""
    weights = weights or [1.0] * len(evidence)
    return min(sorted(set(candidates)), key=lambda c: sum(w * _wdist(c, e) for w, e in zip(weights, evidence)))
