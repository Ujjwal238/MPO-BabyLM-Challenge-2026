#!/bin/bash
# Eval adaptive-masking v2 final on fast BLiMP + supplement. Ref: Phase-1 70.01 / 65.20.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
LOG=$REPO/checkpoints/adaptive_eval.log; : > "$LOG"
ts(){ date "+%F %T"; }
DIR=evaluation_data/fast_eval
M=$REPO/checkpoints/gptbert_small_adaptive_v2/final
for t in "blimp:blimp_fast" "blimp:supplement_fast"; do
  task="${t%%:*}"; data="${t##*:}"; s=$(date +%s)
  if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$M" \
       --backend mntp --task "$task" --data_path "$DIR/$data" --save_predictions >>"$LOG" 2>&1; then
    avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
    echo "$(ts) | RESULT ADAPTIVE $data AVG=$avg ($(($(date +%s)-s))s)" | tee -a "$LOG"
  else
    echo "$(ts) | RESULT ADAPTIVE $data FAILED" | tee -a "$LOG"
  fi
done
echo "$(ts) | ADAPTIVE EVAL COMPLETE" | tee -a "$LOG"
