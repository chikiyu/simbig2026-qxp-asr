# Kaggle kernel: extract CTC log-probs from several fairseq checkpoints on our eval sets.
# Outputs /kaggle/working/out/<model>/<set>.npz (+ vocab.json, greedy.tsv) for local decoding.
import os, subprocess, sys, textwrap, json, glob

W = "/kaggle/working"
ENV_PY = f"{W}/env/bin/python"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# ---------- 1. environment (py3.10 + torch 2.5.1 cu121 + fairseq main) ----------
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
sh("which ffmpeg || (apt-get -qq update && apt-get -qq install -y ffmpeg)")

# ---------- 2. data ----------
RAW = glob.glob("/kaggle/input/**/puno-quechua-raw", recursive=True)
RAW = RAW[0] if RAW else "/kaggle/input/puno-quechua-raw"
AUX = glob.glob("/kaggle/input/**/qxp-aux", recursive=True)
AUX = AUX[0] if AUX else "/kaggle/input/qxp-aux"
print("RAW", RAW, "AUX", AUX, flush=True)
if not os.path.exists(f"{W}/data/scripted"):
    os.makedirs(f"{W}/data/scripted", exist_ok=True)
    sh(f"tar -xf '{RAW}/scripted_puno' -C {W}/data/scripted")

worker = r'''
import os, sys, glob, json, re, unicodedata, gc
import numpy as np, pandas as pd, torch, torchaudio, jiwer
from tqdm import tqdm
W="/kaggle/working"; RAW, AUX = sys.argv[1], sys.argv[2]
os.environ.setdefault("ONLY_SETS", "ood")
sys.path.insert(0, f"{W}/fairseq")
_l = torch.load
torch.load = lambda *a, **k: _l(*a, **{**k, "weights_only": False})
from fairseq import checkpoint_utils
from huggingface_hub import hf_hub_download

def norm(s):  # official scoring normalisation: NFC, lowercase, no punctuation, keep ' and ñ
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[’‘`´ʻʼ]", "'", s)
    s = re.sub(r"[^\w\s']|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ---- eval sets: (id, path, ref)
sets = {}
bt = pd.read_csv(f"{AUX}/blind_test/qxp_test_dataset.tsv", sep="\t")
sets["blind"] = [(r.file_name, f"{AUX}/blind_test/audios/{r.file_name}", "") for r in bt.itertuples()]
cvdir = glob.glob(f"{W}/data/scripted/**/qxp/clips", recursive=True)[0]
val = pd.read_csv(os.path.join(os.path.dirname(cvdir), "validated.tsv"), sep="\t", quoting=3, dtype=str)
cur = pd.read_csv(f"{W}/asr-puno-quechua/data/shared-task/validated_sentences_curated.tsv", sep="\t", quoting=3, dtype=str)
val = val.merge(cur, on="sentence_id")
val["stem"] = val.path.str.replace(".mp3", "", regex=False)
M = f"{W}/asr-puno-quechua/data/manifests/finetune/qxp_v2"
te = [os.path.splitext(os.path.basename(l.split("\t")[0]))[0] for l in open(f"{M}/test.tsv").read().split("\n")[1:] if l.strip()]
v = val.set_index("stem")
sets["scr_test"] = [(s, f"{cvdir}/{s}.mp3", v.sentence_curated[s]) for s in te if s in v.index]
sps = pd.read_csv(f"{AUX}/sps5/ss-corpus-qxp.tsv", sep="\t", quoting=3, dtype=str)
spsfiles = {os.path.basename(p): p for p in glob.glob(f"{RAW}/spontaneous_puno/**/*.mp3", recursive=True)}
d = sps[sps.split.isin(["dev", "test"])]
sets["sps_devtest"] = [(r.audio_file.replace(".mp3", ""), spsfiles[r.audio_file], norm(r.transcription)) for r in d.itertuples() if r.audio_file in spsfiles]
ood = pd.read_csv(f"{W}/asr-puno-quechua/data/additional_data/additional_data_qxp.tsv", sep="\t", quoting=3, dtype=str)
sets["ood"] = [(r.path.replace(".wav", ""), f"{W}/asr-puno-quechua/data/additional_data/wav/{r.path}", norm(r.sentence)) for r in ood.itertuples() if isinstance(r.sentence, str)]
ONLY = os.environ.get("ONLY_SETS")
if ONLY: sets = {k: v_ for k, v_ in sets.items() if k in ONLY.split(",")}
meta = {k: [(i, ref) for i, _, ref in v_] for k, v_ in sets.items()}
os.makedirs(f"{W}/out", exist_ok=True)
json.dump(meta, open(f"{W}/out/sets.json", "w"), ensure_ascii=False)
json.dump({"sps_split": dict(zip(d.audio_file.str.replace(".mp3", "", regex=False), d.split)),
           "sps_client": dict(zip(d.audio_file.str.replace(".mp3", "", regex=False), d.client_id))},
          open(f"{W}/out/sps_meta.json", "w"))
print({k: len(v_) for k, v_ in sets.items()}, flush=True)

def load_wav(p):
    w, sr = torchaudio.load(p)
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)
    if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
    return w[0]

# preload audio once (all sets ~7h of 16k float32 ~ 1.6GB)
audio = {k: [load_wav(p) for _, p, _ in tqdm(v_, desc=f"load {k}")] for k, v_ in sets.items()}

MODELS = {
    "cpt_silver":    ("QuechuaBase/xls-r-cpt-qxp-silver", None),
    "cpt_validated": ("QuechuaBase/xls-r-cpt-qxp-validated", None),
    "cpt_cv26_silver": ("QuechuaBase/xls-r-cpt-qxp-cv26-silver", None),
    "ft_curated":    (None, f"{AUX}/ckpt/ft_curated_ayayon_slim.pt"),
}
dev = torch.device("cuda")
for name, (repo, local) in MODELS.items():
    od = f"{W}/out/{name}"
    if os.path.exists(f"{od}/done"): continue
    os.makedirs(od, exist_ok=True)
    ck = local or hf_hub_download(repo, "checkpoint_best.pt", local_dir=f"{W}/ckpt_tmp/{name}")
    state = checkpoint_utils.load_checkpoint_to_cpu(ck, {"data": M}); state.pop("last_optimizer_state", None)
    ts = state.get("task_state") or {}
    has_dict = "target_dictionary" in ts
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ck], arg_overrides={"data": M}, state=state)
    del state; gc.collect()
    model = models[0].eval().to(dev)
    td = task.target_dictionary
    vocab = [td[i] for i in range(len(td))]
    normalize = bool(getattr(task.cfg, "normalize", False))
    info = {"vocab": vocab, "blank": td.bos(), "normalize": normalize, "dict_from_ckpt": has_dict,
            "out_dim": int(model.w2v_encoder.proj.out_features)}
    print(name, json.dumps(info, ensure_ascii=False)[:400], flush=True)
    json.dump(info, open(f"{od}/vocab.json", "w"), ensure_ascii=False)
    for sname, items in sets.items():
        arrs, hyps = {}, []
        with torch.no_grad():
            for (uid, _, ref), wav in tqdm(list(zip(items, audio[sname])), desc=f"{name}/{sname}"):
                x = wav
                if normalize: x = torch.nn.functional.layer_norm(x, x.shape)
                x = x.to(dev).unsqueeze(0)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(source=x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
                lp = torch.log_softmax(out["encoder_out"][:, 0].float(), -1).cpu().numpy()
                arrs[uid] = lp.astype(np.float16)
                ids = lp.argmax(-1); prev = None; toks = []
                for t in ids:
                    if t != prev and t != info["blank"]: toks.append(vocab[t])
                    prev = t
                txt = "".join(toks).replace("<unk>", " ").replace("|", " ")
                hyps.append((uid, norm(txt), ref))
        np.savez_compressed(f"{od}/{sname}.npz", **arrs)
        df = pd.DataFrame(hyps, columns=["id", "hyp", "ref"]); df.to_csv(f"{od}/{sname}_greedy.tsv", sep="\t", index=False)
        if sname != "blind":
            print(f"== {name}/{sname} greedy WER {100*jiwer.wer(list(df.ref), list(df.hyp)):.2f} CER {100*jiwer.cer(list(df.ref), list(df.hyp)):.2f}", flush=True)
    del model, models, task; gc.collect(); torch.cuda.empty_cache()
    if repo: os.system(f"rm -rf {W}/ckpt_tmp/{name}")
    open(f"{od}/done", "w").close()
print("ALL DONE")
'''
open(f"{W}/worker.py", "w").write(worker)
sh(f"{ENV_PY} {W}/worker.py '{RAW}' '{AUX}' 2>&1 | grep -v 'it/s]' ")
sh(f"rm -rf {W}/data {W}/env {W}/bin {W}/fairseq {W}/ckpt_tmp {W}/asr-puno-quechua")
