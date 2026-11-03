# Kaggle kernel: pseudo-label the unannotated SPS clips with 5 acoustic models
# (beam + KenLM + learned normaliser). Output: /kaggle/working/out/<model>/pseudo.tsv  (id, hyp)
import os, subprocess, glob
W = "/kaggle/working"; ENV_PY = f"{W}/env/bin/python"
def sh(c):
    print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
if not os.path.exists(ENV_PY):
    sh(f"cd {W} && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba")
    sh(f"{W}/bin/micromamba create -y -q -p {W}/env python=3.10 -c conda-forge")
    sh(f"{ENV_PY} -m pip install -q 'pip<24.1' 'setuptools<70' wheel")
    sh(f"{ENV_PY} -m pip install -q torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
    sh(f"{ENV_PY} -m pip install -q omegaconf==2.0.6 hydra-core==1.0.7 bitarray soundfile scipy pandas tqdm jiwer huggingface_hub sacrebleu scikit-learn tensorboardX pyctcdecode https://github.com/kpu/kenlm/archive/master.zip")
if not os.path.exists(f"{W}/fairseq"):
    sh(f"git clone -q --depth 1 https://github.com/facebookresearch/fairseq.git {W}/fairseq")
if not os.path.exists(f"{W}/asr-puno-quechua"):
    sh(f"git clone -q --depth 1 https://github.com/QuechuaBase/asr-puno-quechua.git {W}/asr-puno-quechua")
F = lambda n: (glob.glob(f"/kaggle/input/**/{n}", recursive=True) or [f"/kaggle/input/{n}"])[0]
RAW, AUX, LM = F("puno-quechua-raw"), F("qxp-aux"), F("qxp-lm")
FT2 = glob.glob("/kaggle/input/**/ft_D/best_slim.pt", recursive=True)
print("RAW", RAW, "AUX", AUX, "LM", LM, "FT2", FT2, flush=True)
FT2 = os.path.dirname(os.path.dirname(FT2[0]))
worker = r'''
import os, sys, glob, json, gc, time, numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import resample_poly
from math import gcd
from multiprocessing import Pool
RAW, AUX, LM, FT2, MODELS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(",")
W = "/kaggle/working"; sys.path.insert(0, f"{W}/fairseq"); sys.path.insert(0, LM)
_l = torch.load; torch.load = lambda *a, **k: _l(*a, **{**k, "weights_only": False})
from fairseq import checkpoint_utils
from huggingface_hub import hf_hub_download
M = f"{W}/asr-puno-quechua/data/manifests/finetune/qxp_v2"
sps = pd.read_csv(f"{AUX}/sps5/ss-corpus-qxp.tsv", sep="\t", quoting=3, dtype=str)
files = {os.path.basename(p): p for p in glob.glob(f"{RAW}/spontaneous_puno/**/*.mp3", recursive=True)}
un = sps[sps.transcription.isna() & sps.split.isna() & sps.audio_file.isin(files)]
un = un[un.duration_ms.astype(float) <= 30000]
items = [(a.replace(".mp3", ""), files[a]) for a in un.audio_file]
def ld(p):
    a, sr = sf.read(p, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != 16000: g = gcd(sr, 16000); a = resample_poly(a, 16000 // g, sr // g)
    return np.clip(a * 32767, -32768, 32767).astype(np.int16)
t0 = time.time()
with Pool(2) as p: wavs = p.map(ld, [q for _, q in items], chunksize=32)
print("loaded", len(wavs), "clips", round(sum(len(w) for w in wavs) / 16000 / 3600, 1), "h in", round(time.time() - t0), "s", flush=True)
SRC = {"cpt_silver": ("hf", "QuechuaBase/xls-r-cpt-qxp-silver"), "cpt_validated": ("hf", "QuechuaBase/xls-r-cpt-qxp-validated"),
       "ft_curated": ("f", f"{AUX}/ckpt/ft_curated_ayayon_slim.pt"), "ft_C": ("f", f"{FT2}/ft_C/best_slim.pt"), "ft_D": ("f", f"{FT2}/ft_D/best_slim.pt")}
DEC = None
def init(labels):
    global DEC
    from pyctcdecode import build_ctcdecoder
    DEC = build_ctcdecoder(labels, kenlm_model_path=f"{LM}/all_full.4g.bin", alpha=0.3, beta=2.0, unk_score_offset=-4.0)
def dec(lp):
    from textnorm import Normalizer
    global NORM
    if "NORM" not in globals():
        n = json.load(open(f"{LM}/normalizer.json")); NORM = Normalizer(n["wmap"], [tuple(r) for r in n["rules"]])
    return NORM(DEC.decode(lp, beam_width=32))
from decode import Folded
for name in MODELS:
    kind, src = SRC[name]
    ck = hf_hub_download(src, "checkpoint_best.pt", local_dir=f"/tmp/ck_{name}") if kind == "hf" else src
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([ck], arg_overrides={"data": M})
    model = models[0].eval().cuda(); td = task.target_dictionary; normalize = bool(getattr(task.cfg, "normalize", False))
    fold = Folded([td[i] for i in range(len(td))], td.bos())
    pool = Pool(2, initializer=init, initargs=(fold.labels,)); jobs = []; t0 = time.time()
    with torch.no_grad():
        for w in wavs:
            x = torch.from_numpy(w.astype(np.float32) / 32768.0)
            if normalize: x = torch.nn.functional.layer_norm(x, x.shape)
            x = x.cuda().unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(source=x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
            lp = fold(torch.log_softmax(o["encoder_out"][:, 0].float(), -1).cpu().numpy())
            jobs.append(pool.apply_async(dec, (lp,)))
    print(name, "acoustic done", round(time.time() - t0), "s", flush=True)
    hyps = [j.get() for j in jobs]; pool.close(); pool.join()
    os.makedirs(f"{W}/out/{name}", exist_ok=True)
    pd.DataFrame({"id": [u for u, _ in items], "hyp": hyps}).to_csv(f"{W}/out/{name}/pseudo.tsv", sep="\t", index=False)
    print(name, "decoded", len(hyps), "total", round(time.time() - t0), "s sample", hyps[:2], flush=True)
    del model, models, task; gc.collect(); torch.cuda.empty_cache(); os.system(f"rm -rf /tmp/ck_{name}")
'''
open(f"{W}/pworker.py", "w").write(worker)
procs = [subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} {ENV_PY} {W}/pworker.py '{RAW}' '{AUX}' '{LM}' '{FT2}' {ms} > {W}/pseudo_{g}.log 2>&1", shell=True)
         for g, ms in [(0, "ft_D,cpt_silver,cpt_validated"), (1, "ft_C,ft_curated")]]
rcs = [p.wait() for p in procs]
sh(f"tail -n 20 {W}/pseudo_0.log {W}/pseudo_1.log || true")
sh(f"rm -rf {W}/env {W}/bin {W}/fairseq {W}/asr-puno-quechua")
assert all(r == 0 for r in rcs), rcs
