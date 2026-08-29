#!/bin/bash
# Test whether Muon peaked EARLY then overfit: eval chck_70M/80M/90M (ep7/8/9) on fast BLiMP + supplement.
# Reference: Phase-1 chck_70M=70.13, Phase-1 final=70.01; Muon final=65.12/61.20.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
LOG=$REPO/checkpoints/muon_ckpt_eval.log; : > "$LOG"
ts(){ date "+%F %T"; }
DIR=evaluation_data/fast_eval
for ck in chck_70M chck_80M chck_90M; do
  M=$REPO/checkpoints/gptbert_small_muon_v1/$ck
  for t in "blimp:blimp_fast" "blimp:supplement_fast"; do
    task="${t%%:*}"; data="${t##*:}"; s=$(date +%s)
    if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$M" \
         --backend mntp --task "$task" --data_path "$DIR/$data" --save_predictions >>"$LOG" 2>&1; then
      avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
      echo "$(ts) | RESULT MUON-$ck $data AVG=$avg ($(($(date +%s)-s))s)" | tee -a "$LOG"
    else
      echo "$(ts) | RESULT MUON-$ck $data FAILED" | tee -a "$LOG"
    fi
  done
done
echo "$(ts) | MUON CKPT EVAL COMPLETE" | tee -a "$LOG"
