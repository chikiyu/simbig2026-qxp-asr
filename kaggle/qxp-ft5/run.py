# Kaggle kernel: clean fine-tuning of XLS-R CPT (validated) on curated scripted + SPS (+ normalised silver).
# Two configs run in parallel, one per T4:  A = validated only,  B = validated + silver.
# Held out (never trained on): qxp_v2 valid/test scripted speakers, SPS5 dev/test clips AND their speakers.
# Outputs: /kaggle/working/out/ft_<cfg>/{best_slim.pt, log.txt, vocab.json, <set>.npz, <set>_greedy.tsv}
import os, subprocess, glob, json

W = "/kaggle/working"
ENV_PY = f"{W}/env/bin/python"
HOURS = float(os.environ.get("FT_HOURS", "5.0"))   # wall-clock training budget per config


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


if not os.path.exists(ENV_PY):
    sh(f"cd {W} && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba")
    sh(f"{W}/bin/micromamba create -y -q -p {W}/env python=3.10 -c conda-forge")
    sh(f"{ENV_PY} -m pip install -q 'pip<24.1' 'setuptools<70' wheel")
    sh(f"{ENV_PY} -m pip install -q torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
    sh(f"{ENV_PY} -m pip install -q omegaconf==2.0.6 hydra-core==1.0.7 bitarray soundfile scipy pandas tqdm jiwer huggingface_hub sacrebleu scikit-learn tensorboardX")
if not os.path.exists(f"{W}/fairseq"):
    sh(f"git clone -q --depth 1 https://github.com/facebookresearch/fairseq.git {W}/fairseq")
if not os.path.exists(f"{W}/asr-puno-quechua"):
    sh(f"git clone -q --depth 1 https://github.com/QuechuaBase/asr-puno-quechua.git {W}/asr-puno-quechua")

RAW = (glob.glob("/kaggle/input/**/puno-quechua-raw", recursive=True) or ["/kaggle/input/puno-quechua-raw"])[0]
AUX = (glob.glob("/kaggle/input/**/qxp-aux", recursive=True) or ["/kaggle/input/qxp-aux"])[0]
if not os.path.exists(f"{W}/data/scripted"):
    os.makedirs(f"{W}/data/scripted", exist_ok=True)
    sh(f"tar -xf '{RAW}/scripted_puno' -C {W}/data/scripted")

# ------------------------------------------------------------------ data prep
prep = r'''
import os, sys, glob, json, re, unicodedata
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from multiprocessing import Pool
from collections import Counter, defaultdict
from difflib import SequenceMatcher
W="/kaggle/working"; RAW, AUX = sys.argv[1], sys.argv[2]
OUT="/tmp/prep"; os.makedirs(OUT, exist_ok=True)

def official_norm(s):
    s = unicodedata.normalize("NFC", str(s)).lower(); s = re.sub(r"[’‘`´ʻʼ]", "'", s)
    s = re.sub(r"[^\w\s']|_", " ", s); s = re.sub(r"(?<!\w)'|'(?!\w)", " ", s)
    return re.sub(r"\s+", " ", s).strip()
def basic_norm(s): return official_norm(s).translate(str.maketrans("áéíóúàèìòùü", "aeiouaeiouu"))
def pairs(a, b):
    a, b = a.split(), b.split()
    for op, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal" or (op == "replace" and i2-i1 == j2-j1):
            yield from zip(a[i1:i2], b[j1:j2])

cvdir = glob.glob(f"{W}/data/scripted/**/qxp/clips", recursive=True)[0]
val = pd.read_csv(os.path.join(os.path.dirname(cvdir), "validated.tsv"), sep="\t", quoting=3, dtype=str)
cur = pd.read_csv(f"{W}/asr-puno-quechua/data/shared-task/validated_sentences_curated.tsv", sep="\t", quoting=3, dtype=str)
val = val.merge(cur, on="sentence_id")
# learned CV->curated word map (applied to spontaneous/silver text)
c = defaultdict(Counter)
for s, t in val.drop_duplicates("sentence_id")[["sentence", "sentence_curated"]].itertuples(index=False):
    for x, y in pairs(basic_norm(s), basic_norm(t)): c[x][y] += 1
wmap = {x: k.most_common(1)[0][0] for x, k in c.items() if k.most_common(1)[0][0] != x and k.most_common(1)[0][1]/sum(k.values()) >= 0.6}
json.dump(wmap, open(f"{OUT}/wordmap.json", "w"), ensure_ascii=False)
spont_norm = lambda s: " ".join(wmap.get(w, w) for w in basic_norm(s).split())

M = f"{W}/asr-puno-quechua/data/manifests/finetune/qxp_v2"
stems = lambda f: {os.path.splitext(os.path.basename(l.split("\t")[0]))[0] for l in open(f"{M}/{f}").read().split("\n")[1:] if l.strip()}
scr_test, scr_valid = stems("test.tsv"), stems("valid.tsv")
val["stem"] = val.path.str.replace(".mp3", "", regex=False)
held_spk = set(val[val.stem.isin(scr_test | scr_valid)].client_id)
rows = []
for r in val.itertuples():
    if r.stem in scr_test: continue
    split = "dev_scr" if r.stem in scr_valid else ("train" if r.client_id not in held_spk else None)
    if split: rows.append((r.stem, f"{cvdir}/{r.path}", basic_norm(r.sentence_curated), "scr", r.client_id, str(r.age), split, r.sentence_id))
# dev_scr: subsample 400 clips for fast validation
sps = pd.read_csv(f"{AUX}/sps5/ss-corpus-qxp.tsv", sep="\t", quoting=3, dtype=str)
files = {os.path.basename(p): p for p in glob.glob(f"{RAW}/spontaneous_puno/**/*.mp3", recursive=True)}
held_sps_spk = set(sps[sps.split.isin(["dev", "test"])].client_id)
silver = pd.concat([pd.read_csv(f, sep="\t", quoting=3, dtype=str) for f in glob.glob(f"{W}/asr-puno-quechua/data/splits/silver_spontaneous/*.tsv")])
silver = dict(zip(silver.path, silver.sentence))
import glob as _g
_pl = []
if _pl:
    _p = pd.read_csv(_pl[0], sep="\t", quoting=3, dtype=str, keep_default_na=False)
    for _i, _t in zip(_p["id"], _p["hyp"]):
        if _t.strip(): silver[_i + ".mp3"] = _t
    print("pseudo labels loaded", len(_p), flush=True)
for r in sps.itertuples():
    if r.audio_file not in files: continue
    sid = r.audio_file.replace(".mp3", "")
    if r.split == "dev": rows.append((sid, files[r.audio_file], basic_norm(r.transcription), "sps", r.client_id, str(r.age), "dev_sps", "")); continue
    if r.split == "test" or r.client_id in held_sps_spk: continue
    if isinstance(r.transcription, str) and r.transcription.strip():
        rows.append((sid, files[r.audio_file], spont_norm(r.transcription), "sps", r.client_id, str(r.age), "train", ""))
    elif r.audio_file in silver and isinstance(silver[r.audio_file], str):
        rows.append((sid, files[r.audio_file], spont_norm(silver[r.audio_file]), "silver", r.client_id, str(r.age), "train", ""))
df = pd.DataFrame(rows, columns=["id", "path", "text", "domain", "spk", "age", "split", "sentence_id"])
d = df[df.split == "dev_scr"].sample(n=min(400, (df.split == "dev_scr").sum()), random_state=0).index
df = df[(df.split != "dev_scr") | df.index.isin(d)]

def load(p):
    try:
        a, sr = sf.read(p, dtype="float32", always_2d=True); a = a.mean(1)
        if sr != 16000: g = gcd(sr, 16000); a = resample_poly(a, 16000 // g, sr // g)
        return np.clip(a * 32767, -32768, 32767).astype(np.int16)
    except Exception as e:
        return None
with Pool(os.cpu_count()) as pool:
    wavs = pool.map(load, list(df.path), chunksize=64)
keep = [w is not None and 0.5 * 16000 <= len(w) <= 30 * 16000 for w in wavs]
df = df[keep].reset_index(drop=True); wavs = [w for w, k in zip(wavs, keep) if k]
lens = np.array([len(w) for w in wavs]); offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
mm = np.lib.format.open_memmap(f"{OUT}/audio.npy", mode="w+", dtype=np.int16, shape=(int(lens.sum()),))
for o, w in zip(offs, wavs): mm[o:o + len(w)] = w
mm.flush(); df["off"] = offs; df["len"] = lens
df.drop(columns=["path"]).to_csv(f"{OUT}/manifest.tsv", sep="\t", index=False)
print(df.groupby(["split", "domain"]).len.agg(["count", lambda x: round(x.sum() / 16000 / 3600, 2)]))
'''
open(f"{W}/prep.py", "w").write(prep)
sh(f"{ENV_PY} {W}/prep.py '{RAW}' '{AUX}'")

# ------------------------------------------------------------------ training
train = r'''
import os, sys, time, json, math, random, re
import numpy as np, pandas as pd, torch, torchaudio, torch.nn.functional as F, jiwer
CFG, HOURS = sys.argv[1], float(sys.argv[2])
W = "/kaggle/working"; OD = f"{W}/out/ft_{CFG}"; os.makedirs(OD, exist_ok=True)
log = open(f"{OD}/log.txt", "a")
def P(*a):
    s = " ".join(map(str, a)); print(f"[{CFG}] {s}", flush=True); log.write(s + "\n"); log.flush()
sys.path.insert(0, f"{W}/fairseq")
_l = torch.load; torch.load = lambda *a, **k: _l(*a, **{**k, "weights_only": False})
import fairseq.utils, fairseq.models.wav2vec.wav2vec2 as w2v
def safe_index_put(t, idx, v):
    if isinstance(v, torch.Tensor) and v.dtype != t.dtype: v = v.to(t.dtype)
    t[idx] = v; return t
fairseq.utils.index_put = safe_index_put; w2v.index_put = safe_index_put
from fairseq import checkpoint_utils
from huggingface_hub import hf_hub_download
random.seed(0); torch.manual_seed(0)
dev = torch.device("cuda")

BASE = {"A": "QuechuaBase/xls-r-cpt-qxp-validated", "B": "QuechuaBase/xls-r-cpt-qxp-validated", "C": "QuechuaBase/xls-r-cpt-qxp-silver", "D": "QuechuaBase/xls-r-cpt-qxp-cv26-silver", "P": "file:ft_D", "Q": "file:ft_D", "S": "QuechuaBase/xls-r-cpt-qxp-cv26-silver", "G": "file:ft_D"}[CFG]
import glob as _g
ck = _g.glob("/kaggle/input/**/ft_D/best_slim.pt", recursive=True)[0] if BASE.startswith("file:") else f"/tmp/ckpt/{BASE.split('/')[1]}/checkpoint_best.pt"
M = f"{W}/asr-puno-quechua/data/manifests/finetune/qxp_v2"
state = checkpoint_utils.load_checkpoint_to_cpu(ck, {"data": M}); state.pop("last_optimizer_state", None)
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ck], arg_overrides={"data": M}, state=state)
from omegaconf import OmegaConf
SAVE_CFG = OmegaConf.to_container(state["cfg"], resolve=True, enum_to_str=True) if not isinstance(state["cfg"], dict) else state["cfg"]
model = models[0].to(dev); td = task.target_dictionary
vocab = [td[i] for i in range(len(td))]; BLANK = td.bos()
normalize = bool(getattr(task.cfg, "normalize", False))
json.dump({"vocab": vocab, "blank": BLANK, "normalize": normalize}, open(f"{OD}/vocab.json", "w"), ensure_ascii=False)
for p in model.w2v_encoder.w2v_model.feature_extractor.parameters(): p.requires_grad = False
enc = model.w2v_encoder
enc.w2v_model.mask_prob = 0.5; enc.w2v_model.mask_channel_prob = 0.25 if hasattr(enc.w2v_model, "mask_channel_prob") else 0
P("normalize", normalize, "mask_prob", enc.w2v_model.mask_prob, "boundary=<unk> idx", td.unk())

char2id = {c: td.index(c) for c in vocab if len(c) == 1}
def encode(t):
    ids = []
    for ch in t:
        if ch == " ": ids.append(td.unk())
        elif ch in char2id: ids.append(char2id[ch])
        else: return None
    return ids

PRE = "/tmp/prep"; audio = np.load(f"{PRE}/audio.npy", mmap_mode="r")
df = pd.read_csv(f"{PRE}/manifest.tsv", sep="\t", dtype={"text": str, "sentence_id": str}, keep_default_na=False)
tr = df[df.split == "train"].copy()
if CFG == "S":  # sentence-disjoint validation: hold out 10% of CV sentences (all their recordings)
    import hashlib
    held = tr.sentence_id.fillna("").map(lambda x: bool(x) and int(hashlib.md5(x.encode()).hexdigest(), 16) % 10 == 0)
    P("held-out sentence clips removed from train:", int(held.sum())); tr = tr[~held]
if CFG == "A": tr = tr[tr.domain != "silver"]
tr["tgt"] = tr.text.map(encode); tr = tr[tr.tgt.notna() & (tr.text.str.len() > 0)]
# upsampling: validated spontaneous x3 (as in the paper); in B, older speakers (50+) x2
rep = np.where(tr.domain == "sps", 3, 1)
if CFG != "A": rep = rep * np.where(tr.age.isin(["fifties", "sixties", "seventies", "eighties", "nineties"]), 2, 1)
tr = tr.loc[tr.index.repeat(rep)].reset_index(drop=True)
P("train utts", len(tr), "hours", round(tr.len.sum() / 16000 / 3600, 1), tr.domain.value_counts().to_dict())
devsets = {k: df[df.split == k] for k in ["dev_scr", "dev_sps"]}

def wav(r):
    x = torch.from_numpy(audio[r.off:r.off + r.len].astype(np.float32) / 32768.0)
    return F.layer_norm(x, x.shape) if normalize else x
MAXS = 16000 * 75  # samples per batch (padded)
def batches():
    idx = tr.sort_values("len").index.tolist(); bs = []; cur = []; mx = 0
    for i in idx:
        L = tr.len[i]
        if cur and max(mx, L) * (len(cur) + 1) > MAXS: bs.append(cur); cur = []; mx = 0
        cur.append(i); mx = max(mx, L)
    if cur: bs.append(cur)
    random.shuffle(bs); return bs
AUG = CFG == "G"
def augment_gpu(x):  # x: 1-D float tensor on GPU
    if random.random() < 0.6:
        f = random.choice([0.9, 1.1])
        x = torch.nn.functional.interpolate(x.view(1, 1, -1), scale_factor=1.0 / f, mode="linear", align_corners=False).view(-1)
    if random.random() < 0.5:
        snr = random.uniform(10, 30); p = x.pow(2).mean().clamp(min=1e-8)
        x = x + torch.randn_like(x) * torch.sqrt(p / (10 ** (snr / 10)))
    if random.random() < 0.5:
        x = x * random.uniform(0.5, 1.5)
    return F.layer_norm(x, x.shape) if normalize else x
def collate(rows):
    ws = [wav(r) for r in rows]
    if AUG:
        ws = [augment_gpu(w.to(dev)).cpu() for w in ws]
    L = max(len(w) for w in ws)
    x = torch.zeros(len(ws), L); pm = torch.ones(len(ws), L, dtype=torch.bool)
    for i, w in enumerate(ws): x[i, :len(w)] = w; pm[i, :len(w)] = False
    return x, pm

def greedy_eval(name):
    model.eval(); refs, hyps = [], []
    with torch.no_grad():
        for r in devsets[name].itertuples():
            x = wav(r).to(dev).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(source=x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
            ids = out["encoder_out"][:, 0].argmax(-1).tolist(); prev = None; t = []
            for i in ids:
                if i != prev and i != BLANK: t.append(vocab[i])
                prev = i
            hyps.append(" ".join("".join(t).replace("<unk>", " ").replace("|", " ").split())); refs.append(r.text)
    model.train(); enc.w2v_model.feature_extractor.eval()
    return 100 * jiwer.wer(refs, hyps), 100 * jiwer.cer(refs, hyps)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0)
scaler = torch.cuda.amp.GradScaler()
ACC, PEAK, WARM = 4, 3e-5, 0.05
def lr_at(f): return PEAK * (f / WARM if f < WARM else max(0.05, 0.5 * (1 + math.cos(math.pi * (f - WARM) / (1 - WARM)))))
w0 = greedy_eval("dev_scr"); s0 = greedy_eval("dev_sps"); P("init dev_scr", w0, "dev_sps", s0)
best = (w0[0] + s0[0]) / 2; P("init score", best)
torch.save({**state, "cfg": SAVE_CFG, "model": {k: v.cpu() for k, v in model.state_dict().items()}}, f"{OD}/best_slim.pt")
model.train(); enc.w2v_model.feature_extractor.eval()
t0 = time.time(); upd = 0; step = 0; last_eval = time.time(); run_loss = 0.0; done = False
while not done:
    for b in batches():
        rows = [tr.loc[i] for i in b]
        x, pm = collate(rows); x, pm = x.to(dev), pm.to(dev)
        tg = [torch.tensor(r.tgt) for r in rows]; tl = torch.tensor([len(t) for t in tg])
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(source=x, padding_mask=pm)
            lp = out["encoder_out"].float().log_softmax(-1)
            ol = (~out["padding_mask"]).sum(-1) if out.get("padding_mask") is not None else torch.full((len(rows),), lp.shape[0], device=dev)
            loss = F.ctc_loss(lp, torch.cat(tg).to(dev), ol, tl.to(dev), blank=BLANK, reduction="sum", zero_infinity=True) / tl.sum().clamp(min=1)
            scaler.scale(loss / ACC).backward()
        except torch.cuda.OutOfMemoryError:
            P("OOM, skipping batch of", len(rows)); del x, pm; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
        step += 1; run_loss += loss.item()
        if step % ACC == 0:
            f = (time.time() - t0) / (HOURS * 3600)
            for g in opt.param_groups: g["lr"] = lr_at(min(f, 1.0))
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); upd += 1
            if upd % 100 == 0: P(f"upd {upd} loss {run_loss / (100 * ACC):.3f} lr {opt.param_groups[0]['lr']:.2e} elapsed {(time.time() - t0) / 3600:.2f}h"); run_loss = 0.0
            if time.time() - last_eval > 1800 or f >= 1.0:
                ws, ss = greedy_eval("dev_scr"), greedy_eval("dev_sps"); sc = (ws[0] + ss[0]) / 2
                P(f"EVAL upd {upd} dev_scr WER/CER {ws[0]:.2f}/{ws[1]:.2f} dev_sps {ss[0]:.2f}/{ss[1]:.2f} score {sc:.2f}")
                if sc < best:
                    best = sc; torch.save({**state, "cfg": SAVE_CFG, "model": {k: v.cpu() for k, v in model.state_dict().items()}}, f"{OD}/best_slim.pt"); P("  saved best")
                last_eval = time.time()
            if f >= 1.0: done = True; break
P("TRAINING DONE best", best)
'''
open(f"{W}/train.py", "w").write(train)
BASES = {"A": "QuechuaBase/xls-r-cpt-qxp-validated", "B": "QuechuaBase/xls-r-cpt-qxp-validated", "C": "QuechuaBase/xls-r-cpt-qxp-silver", "D": "QuechuaBase/xls-r-cpt-qxp-cv26-silver", "E": "QuechuaBase/xls-r-cpt-qxp-validated", "Q": "QuechuaBase/xls-r-cpt-qxp-cv26-silver"}
for repo in ['QuechuaBase/xls-r-cpt-qxp-cv26-silver']:
    sh(f"{ENV_PY} -c \"from huggingface_hub import hf_hub_download; hf_hub_download('{repo}', 'checkpoint_best.pt', local_dir='/tmp/ckpt/{repo.split('/')[1]}')\"")
sh("df -h /kaggle/working /tmp | cat")
procs = [subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} {ENV_PY} {W}/train.py {c} {HOURS} > {W}/train_{c}.log 2>&1", shell=True)
         for g, c in [(0, "S"), (1, "G")]]
rcs = [p.wait() for p in procs]
print("train exit codes", rcs, flush=True)
sh(f"tail -n 40 {W}/train_S.log {W}/train_G.log || true")

# ------------------------------------------------------------------ dump logits of best checkpoints
dump = r'''
import os, sys, glob, json, re, unicodedata, numpy as np, pandas as pd, torch, torchaudio
from tqdm import tqdm
W = "/kaggle/working"; AUX, RAW, CFG = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, f"{W}/fairseq")
_l = torch.load; torch.load = lambda *a, **k: _l(*a, **{**k, "weights_only": False})
from fairseq import checkpoint_utils
OD = f"{W}/out/ft_{CFG}"; M = f"{W}/asr-puno-quechua/data/manifests/finetune/qxp_v2"
models, _, task = checkpoint_utils.load_model_ensemble_and_task([f"{OD}/best_slim.pt"], arg_overrides={"data": M})
model = models[0].eval().cuda(); normalize = bool(getattr(task.cfg, "normalize", False))
cvdir = glob.glob(f"{W}/data/scripted/**/qxp/clips", recursive=True)[0]
spsfiles = {os.path.basename(p): p for p in glob.glob(f"{RAW}/spontaneous_puno/**/*.mp3", recursive=True)}
te = [os.path.splitext(os.path.basename(l.split("\t")[0]))[0] for l in open(f"{M}/test.tsv").read().split("\n")[1:] if l.strip()]
sps = pd.read_csv(f"{AUX}/sps5/ss-corpus-qxp.tsv", sep="\t", quoting=3, dtype=str)
sets = {"blind": [(f, f"{AUX}/blind_test/audios/{f}") for f in pd.read_csv(f"{AUX}/blind_test/qxp_test_dataset.tsv", sep="\t").file_name],
        "scr_test": [(s, f"{cvdir}/{s}.mp3") for s in te],
        "sps_devtest": [(a.replace(".mp3", ""), spsfiles[a]) for a in sps[sps.split.isin(["dev", "test"])].audio_file if a in spsfiles]}
for sname, items in sets.items():
    arrs = {}
    with torch.no_grad():
        for uid, p in tqdm(items, desc=sname, mininterval=60):
            w, sr = torchaudio.load(p); w = w.mean(0)
            if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
            if normalize: w = torch.nn.functional.layer_norm(w, w.shape)
            x = w.cuda().unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(source=x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
            arrs[uid] = torch.log_softmax(out["encoder_out"][:, 0].float(), -1).cpu().numpy().astype(np.float16)
    np.savez_compressed(f"{OD}/{sname}.npz", **arrs)
print("dumped", CFG)
'''
open(f"{W}/dump.py", "w").write(dump)
for c in ["S", "G"]:
    if os.path.exists(f"{W}/out/ft_{c}/best_slim.pt"):
        sh(f"{ENV_PY} {W}/dump.py '{AUX}' '{RAW}' {c}")
sh(f"rm -rf {W}/data {W}/env {W}/bin {W}/fairseq /tmp/ckpt /tmp/prep {W}/asr-puno-quechua")
assert all(r == 0 for r in rcs), f"training failed: {rcs}"
