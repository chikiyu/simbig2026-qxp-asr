"""Compare every entry of the paper's references.bib with its official record, field by field.

Sources: Crossref (entries with a DOI), the ACL Anthology BibTeX (ACL papers without DOI), the NeurIPS
proceedings page and the arXiv abstract page. Needs internet.
Usage: python3 tools/check_references.py /path/to/simbig2026-paper/references.bib
"""
import html
import json
import re
import sys
import unicodedata
import urllib.request

BIB = sys.argv[1] if len(sys.argv) > 1 else "../simbig2026-paper/references.bib"
EXTRA = {  # official records for entries without a DOI
    "ardila2020commonvoice": ("acl", "2020.lrec-1.520"),
    "heafield2011kenlm": ("acl", "W11-2123"),
    "baevski2020wav2vec": ("neurips", "https://papers.nips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html"),
    "omnilingual2025": ("arxiv", "2511.09690"),
    "fiscus1997rover": ("skip", "not cited in the paper"),
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "reference-check"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


def norm(s):
    s = unicodedata.normalize("NFKD", html.unescape(str(s or ""))).encode("ascii", "ignore").decode()
    s = re.sub(r"\\[a-zA-Z]+|[{}\\'\"`^~]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", m.group(3) + "\n", re.S))
        entries[m.group(2).strip()] = fields
    return entries


def families(author_field):
    out = []
    for a in re.split(r"\s+and\s+", author_field):
        a = a.strip()
        if a.lower() == "others":
            continue
        if a.startswith("{") and a.endswith("}"):  # group author, e.g. {Omnilingual ASR Team}
            out.append(norm(a))
            continue
        out.append(norm(a.split(",")[0] if "," in a else a.split()[-1]))
    return out


def official(key, f):
    if "doi" in f:
        m = json.loads(get("https://api.crossref.org/works/" + f["doi"]))["message"]
        title = (m.get("title") or [""])[0] + ((": " + m["subtitle"][0]) if m.get("subtitle") else "")
        return {"title": title, "venue": (m.get("container-title") or [""])[0],
                "volume": m.get("volume", ""), "number": m.get("issue", ""), "pages": m.get("page", ""),
                "year": str((m.get("issued", {}).get("date-parts") or [[""]])[0][0]),
                "authors": [norm(a.get("family", a.get("name", ""))) for a in m.get("author", [])],
                "source": "https://doi.org/" + f["doi"]}
    kind, ref = EXTRA.get(key, ("none", ""))
    if kind == "acl":
        b = get(f"https://aclanthology.org/{ref}.bib")
        g = lambda k: (re.search(k + r'\s*=\s*"(.*?)",?\n', b, re.S) or [None, ""])[1]  # noqa: E731
        return {"title": g("title"), "venue": g("booktitle"), "pages": g("pages").replace("--", "-"), "year": g("year"),
                "authors": [norm(a.split(",")[0]) for a in re.split(r"\s+and\s+", g("author"))], "source": f"https://aclanthology.org/{ref}/"}
    if kind == "neurips":
        p = get(ref)
        meta = lambda k: re.findall(rf'name="{k}" content="([^"]*)"', p)  # noqa: E731
        return {"title": (meta("citation_title") or [""])[0], "volume": (meta("citation_volume") or [""])[0],
                "pages": f'{(meta("citation_firstpage") or [""])[0]}-{(meta("citation_lastpage") or [""])[0]}',
                "year": (meta("citation_publication_date") or meta("citation_date") or [""])[0][:4],
                "authors": [norm(a.split(",")[0] if "," in a else a.split()[-1]) for a in meta("citation_author")], "source": ref}
    if kind == "arxiv":
        p = get(f"https://arxiv.org/abs/{ref}")
        meta = lambda k: re.findall(rf'name="{k}" content="([^"]*)"', p)  # noqa: E731
        return {"title": (meta("citation_title") or [""])[0], "year": (meta("citation_date") or [""])[0][:4],
                "authors": [norm(a.split(",")[0]) for a in meta("citation_author")], "source": f"https://arxiv.org/abs/{ref}"}
    return None


entries = parse_bib(open(BIB, encoding="utf-8").read())
problems = 0
for key, f in entries.items():
    kind = EXTRA.get(key, ("", ""))[0]
    if kind == "skip":
        print(f"\n== {key}: skipped ({EXTRA[key][1]})")
        continue
    try:
        o = official(key, f)
    except Exception as e:  # noqa: BLE001
        print(f"\n== {key}: could not fetch the official record ({e})"); problems += 1; continue
    print(f"\n== {key}   [{o['source']}]")
    checks = [("title", norm(f.get("title")), norm(o.get("title")))]
    if o.get("year"):
        checks.append(("year", f.get("year", ""), o["year"]))
    if o.get("pages"):
        checks.append(("pages", f.get("pages", "").replace("--", "-"), o["pages"]))
    for k in ("volume", "number"):
        if o.get(k):
            checks.append((k, f.get(k, ""), str(o[k])))
    ours_a, off_a = families(f.get("author", "")), o.get("authors", [])
    n = len(ours_a)
    checks.append(("authors (in order)", " ; ".join(ours_a), " ; ".join(off_a[:n]) + (" ; ..." if len(off_a) > n else "")))
    for name, a, b in checks:
        ok = (a == b) if name != "authors (in order)" else (ours_a == off_a[:n])
        problems += not ok
        print(f"   {'OK ' if ok else 'XX '} {name:20s} ours: {a}")
        if not ok:
            print(f"   {'':24s} official: {b}")
    if o.get("venue"):
        print(f"   ..  venue (compare by eye)  ours: {f.get('booktitle') or f.get('journal')}\n   {'':24s} official: {o['venue']}")
print(f"\n{problems} mismatching field(s)")
