# Kaggle kernel: zero-shot omniASR (Meta Omnilingual ASR) transcriptions for blind / OOD / SPS5 dev+test.
# GPU0: omniASR_LLM_3B_v2, GPU1: omniASR_CTC_3B_v2. Output: /kaggle/working/out/<card>/<set>.tsv (id, hyp)
import os, subprocess, glob
W = "/kaggle/working"; PY = f"{W}/oenv/bin/python"
def sh(c):
    print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
if not os.path.exists(PY):
    sh(f"cd {W} && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba")
    sh(f"{W}/bin/micromamba create -y -q -p {W}/oenv python=3.11 libsndfile -c conda-forge")
    sh(f"{PY} -m pip install -q omnilingual-asr pandas soundfile scipy")
    sh(f"{PY} -m pip install -q --force-reinstall --no-deps torchaudio==$({PY} -c 'import torch;print(torch.__version__.split(\"+\")[0])') --index-url https://download.pytorch.org/whl/$({PY} -c 'import torch;print(\"cu\"+torch.version.cuda.replace(\".\",\"\"))')")
    sh(f"{PY} -m pip install -q 'pyarrow==19.0.1'")
    sh(f"{PY} -c 'import torch, torchaudio, numpy, pyarrow; print(torch.__version__, torchaudio.__version__, numpy.__version__, pyarrow.__version__); from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; print(\"import ok\")'")
    sh(f"{PY} -c 'import torch;print(torch.__version__, torch.cuda.is_available())'")
if not os.path.exists(f"{W}/asr-puno-quechua"):
    sh(f"git clone -q --depth 1 https://github.com/QuechuaBase/asr-puno-quechua.git {W}/asr-puno-quechua")
RAW = (glob.glob("/kaggle/input/**/puno-quechua-raw", recursive=True) or ["/kaggle/input/puno-quechua-raw"])[0]
if not os.path.exists(f"{W}/data/scripted"):
    os.makedirs(f"{W}/data/scripted", exist_ok=True)
    sh(f"tar -xf '{RAW}/scripted_puno' -C {W}/data/scripted")
AUX = (glob.glob("/kaggle/input/**/qxp-aux", recursive=True) or ["/kaggle/input/qxp-aux"])[0]
worker = r'''
import os, sys, glob, time, numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import resample_poly
from math import gcd
card, RAW, AUX, WANT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
W = "/kaggle/working"; OD = f"{W}/out/{card}"; os.makedirs(OD, exist_ok=True); TMP = f"/tmp/wav16/{card}"; os.makedirs(TMP, exist_ok=True)
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
try:
    from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
    print("qxp_Latn supported:", "qxp_Latn" in supported_langs, "quz_Latn:", "quz_Latn" in supported_langs, flush=True)
except Exception as e: print("lang check failed", e)
kw = {}
try:
    pipe = ASRInferencePipeline(model_card=card, dtype=torch.float16)
except TypeError:
    pipe = ASRInferencePipeline(model_card=card)
sps = pd.read_csv(f"{AUX}/sps5/ss-corpus-qxp.tsv", sep="\t", quoting=3, dtype=str)
spsf = {os.path.basename(p): p for p in glob.glob(f"{RAW}/spontaneous_puno/**/*.mp3", recursive=True)}
ood = pd.read_csv(f"{W}/asr-puno-quechua/data/additional_data/additional_data_qxp.tsv", sep="\t", quoting=3, dtype=str)
bt = pd.read_csv(f"{AUX}/blind_test/qxp_test_dataset.tsv", sep="\t")
import hashlib
cvdir = glob.glob(f"{W}/data/scripted/**/qxp/clips", recursive=True)[0]
val = pd.read_csv(os.path.join(os.path.dirname(cvdir), "validated.tsv"), sep="\t", quoting=3, dtype=str)
sid = dict(zip(val.path.str.replace(".mp3", "", regex=False), val.sentence_id))
M = f"{W}/asr-puno-quechua/data/manifests/finetune/qxp_v2"
te = [os.path.splitext(os.path.basename(l.split("\t")[0]))[0] for l in open(f"{M}/test.tsv").read().split("\n")[1:] if l.strip()]
held = [t for t in te if t in sid and int(hashlib.md5(sid[t].encode()).hexdigest(), 16) % 10 == 0]
sets = {"scr_heldout": [(t, f"{cvdir}/{t}.mp3") for t in held],
        "blind": [(f, f"{AUX}/blind_test/audios/{f}") for f in bt.file_name],
        "ood": [(r.path.replace(".wav", ""), f"{W}/asr-puno-quechua/data/additional_data/wav/{r.path}") for r in ood.itertuples() if isinstance(r.sentence, str)],
        "sps_devtest": [(a.replace(".mp3", ""), spsf[a]) for a in sps[sps.split.isin(["dev", "test"])].audio_file if a in spsf]}
def to16(uid, p):
    a, sr = sf.read(p, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != 16000: g = gcd(sr, 16000); a = resample_poly(a, 16000 // g, sr // g)
    a = a[: int(39.5 * 16000)]
    q = f"{TMP}/{uid}.wav"; sf.write(q, a, 16000); return q
for sname, items in sets.items():
    if sname not in WANT: continue
    t0 = time.time(); paths = [to16(u, p) for u, p in items]; hyps = []
    B = 4
    for i in range(0, len(paths), B):
        chunk = paths[i:i + B]
        try:
            hyps += pipe.transcribe(chunk, lang=["qxp_Latn"] * len(chunk), batch_size=len(chunk))
        except Exception as e:
            print("batch error", e, flush=True)
            for c in chunk:
                try: hyps += pipe.transcribe([c], lang=["qxp_Latn"], batch_size=1)
                except Exception as e2: print("err", c, e2, flush=True); hyps.append("")
    pd.DataFrame({"id": [u for u, _ in items], "hyp": hyps}).to_csv(f"{OD}/{sname}.tsv", sep="\t", index=False)
    print(card, sname, len(hyps), "done in", round(time.time() - t0), "s; sample:", hyps[:2], flush=True)
'''
open(f"{W}/oworker.py", "w").write(worker)
JOBS = {0: [("omniASR_LLM_1B_v2", "blind,ood,scr_heldout"), ("omniASR_LLM_3B_v2", "scr_heldout")],
        1: [("omniASR_CTC_1B_v2", "blind,ood,scr_heldout"), ("omniASR_LLM_300M_v2", "blind,ood,scr_heldout"), ("omniASR_CTC_3B_v2", "scr_heldout")]}
procs = []
for g, jobs in JOBS.items():
    cmd = " ; ".join(f"CUDA_VISIBLE_DEVICES={g} {PY} {W}/oworker.py {c} '{RAW}' '{AUX}' {ss} > {W}/omni_{c}.log 2>&1" for c, ss in jobs)
    procs.append(subprocess.Popen(cmd, shell=True))
rcs = [p.wait() for p in procs]
sh(f"tail -n 25 {W}/omni_*.log || true")
sh(f"rm -rf {W}/oenv {W}/bin {W}/asr-puno-quechua {W}/data /tmp/wav16")
print("exit codes", rcs)
