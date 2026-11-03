#!/usr/bin/env bash
# Runs every check behind the numbers of the paper and writes one report (about 10 minutes on CPU).
# Usage: bash tools/verify_paper_all.sh   ->  results/paper_verification_report.txt
cd "$(dirname "$0")/.." || exit 1
OUT=results/paper_verification_report.txt
{
  echo "== Data, overlap and evaluation subsets (tools/verify_paper_data.py)"; .venv/bin/python tools/verify_paper_data.py
  echo; echo "== Results, oracle, errors, LM (tools/verify_paper_results.py)"; .venv/bin/python tools/verify_paper_results.py
  echo; echo "== Normaliser cross-validation, behavioural test (tools/verify_paper_extra.py)"; .venv/bin/python tools/verify_paper_extra.py
  echo; echo "== Learned combination (src/rerank.py)"; .venv/bin/python src/rerank.py
  echo; echo "== Reproduction of the published protocol (results/paper_protocol.tsv)"; cat results/paper_protocol.tsv
} 2>&1 | grep -viE "warning|unigram" > "$OUT"
echo "report written to $OUT"
