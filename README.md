# Puno Quechua ASR: SIMBig 2026 Shared Task 1

Code of the system described in *Fine-Tuning, Orthographic Normalisation and System Combination for Puno
Quechua Speech Recognition* (M. A. Pachari Lipa, L. Kim Negrillo, Y. P. Nieto Barrientos; SIMBig 2026).
Checkpoints, language models and the learned normaliser: https://huggingface.co/ayayon/simbig2026-qxp-asr

## Results (official blind test)
Team **🐣_en_🏡**, 4th of 10 ranked teams in Task 1
([leaderboard](http://quechuabase.github.io/SimBigASR2026/), 27 October 2026).

| | WER overall | WER scripted | WER spontaneous |
|---|---|---|---|
| This system | **16.10** | 14.90 | 16.89 |
| Best organiser baseline (omniLLM7B_v2) | 18.81 | 20.56 | 17.66 |

## System
- **Acoustic models:** XLS-R (300M) checkpoints with continued pre-training on Quechua released by
  [QuechuaBase](https://huggingface.co/QuechuaBase) (Huaman et al., 2026), fine-tuned with CTC in fairseq on
  Common Voice transcriptions in the curated orthography of the task (`kaggle/qxp-ft*`).
- **Decoding:** CTC beam search (width 64) with a word 4-gram KenLM model (LM weight 0.3, word bonus 2.0,
  unknown-word penalty −4).
- **Learned orthographic normaliser** (`src/textnorm.py`): 375 word mappings and 107 rewrite rules learned from
  the 2,067 original/curated Common Voice sentence pairs; words of the curated vocabulary are never changed.
  The submission also applies span rules for changes that need several edits (`src/span_rules.py`).
- **Read speech:** one model (`ft_A`). **Spontaneous speech:** minimum Bayes risk selection among the 4-best
  hypotheses of five XLS-R models and the outputs of omniASR-LLM-3B and omniASR-CTC-3B (zero-shot)
  (`src/combine.py`, `src/build_final.py`).

## Repository
```
src/       normaliser, LM building, decoding, evaluation, combination, submission builder,
           baseline scoring, negative results (morph LM, learned reranking)
kaggle/    Kaggle kernels used for fine-tuning and for extracting CTC log-probabilities (T4 GPUs)
tools/     checks behind every number of the paper (verify_paper_all.sh), submission and reference checks
results/   final tables, reproduction of the published protocol, evaluation subsets, normaliser,
           verification report
```
Not included: audio and corpora (Mozilla Common Voice via the Mozilla Data Collective, and the task
repository), the blind test and our predictions, and the checkpoints (on Hugging Face).

## Data layout expected by the scripts
- `dataset common voice /` (note the trailing space) with the Common Voice Scripted Speech 26.0 and
  Spontaneous Speech 5.0 releases for Puno Quechua;
- `asr-puno-quechua/`: a clone of https://github.com/QuechuaBase/asr-puno-quechua (curated sentences, splits,
  manifests);
- `qxp_test_dataset/qxp_test_dataset/`: the blind test, only needed to build a submission;
- `results/logits/<model>/<set>.npz`: CTC log-probabilities written by the Kaggle kernels.

## Reproduce
```
KENLM_BIN=/path/to/kenlm/build python src/build_lm.py   # text corpora and KenLM models (bin/lmplz, bin/build_binary)
python src/final_eval.py        # all local results -> results/final_tables.md
python src/build_final.py --scripted ft_A \
  --spontaneous ft_C ft_D cpt_silver ft_curated cpt_validated omni:omniASR_LLM_3B_v2 omni:omniASR_CTC_3B_v2 \
  --nbest 4 --span-rules --out submissions/qxp.tsv
bash tools/verify_paper_all.sh  # recomputes every number of the paper -> results/paper_verification_report.txt
```

## License
Apache-2.0 (see `LICENSE`), like the task repository. Please also cite Huaman et al. (2026),
*Building Community-Centred NLP Resources for Puno Quechua*, AmericasNLP 2026, doi:10.18653/v1/2026.americasnlp-6.13.
