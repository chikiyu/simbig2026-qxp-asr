"""CPU dump of CTC log-probs on the OOD set for a fairseq checkpoint (local, no GPU).
Run with a Python 3.10 env that has fairseq installed, from a directory that contains the fairseq/ source tree:
  cd <workdir> && <fairseq-env>/bin/python "<repo>/tools/dump_ood_local.py" ft_G "<repo>/checkpoints/ft_G_slim.pt"
"""
import os, sys, json
import numpy as np, pandas as pd, soundfile as sf, torch
from math import gcd
from scipy.signal import resample_poly
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, "fairseq")
_l = torch.load; torch.load = lambda *a, **k: _l(*a, **{**k, "weights_only": False})
from fairseq import checkpoint_utils
torch.set_num_threads(7)
name, ck = sys.argv[1], sys.argv[2]
M = P + "asr-puno-quechua/data/manifests/finetune/qxp_v2"
models, _, task = checkpoint_utils.load_model_ensemble_and_task([ck], arg_overrides={"data": M})
m = models[0].eval(); norm = bool(task.cfg.normalize); td = task.target_dictionary
od = P + f"results/logits/{name}"; os.makedirs(od, exist_ok=True)
if not os.path.exists(od + "/vocab.json"):
    json.dump({"vocab": [td[i] for i in range(len(td))], "blank": td.bos(), "normalize": norm}, open(od + "/vocab.json", "w"), ensure_ascii=False)
ood = pd.read_csv(P + "asr-puno-quechua/data/additional_data/additional_data_qxp.tsv", sep="\t", quoting=3, dtype=str)
arr = {}
for r in ood.itertuples():
    if not isinstance(r.sentence, str):
        continue
    a, sr = sf.read(P + "asr-puno-quechua/data/additional_data/wav/" + r.path, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != 16000:
        g = gcd(sr, 16000); a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    x = torch.from_numpy(a)
    if norm:
        x = torch.nn.functional.layer_norm(x, x.shape)
    with torch.inference_mode():
        o = m(source=x.unsqueeze(0), padding_mask=torch.zeros(1, len(x), dtype=torch.bool))
    arr[r.path.replace(".wav", "")] = torch.log_softmax(o["encoder_out"][:, 0].float(), -1).numpy().astype(np.float16)
np.savez_compressed(od + "/ood.npz", **arr); print("done", name, len(arr))
