#!/bin/bash
# A/B: span-masking v2 final vs Phase-1 final. fast BLiMP + supplement, mntp, sequential.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
LOG=$REPO/checkpoints/span_vs_phase1_final.log
: > "$LOG"
ts(){ date "+%F %T"; }
DIR=evaluation_data/fast_eval
TASKS=("blimp:blimp_fast" "blimp:supplement_fast")

eval_model(){  # $1=model_dir  $2=label
  echo "$(ts) | ===== $2 =====" | tee -a "$LOG"
  for t in "${TASKS[@]}"; do
    task="${t%%:*}"; data="${t##*:}"
    s=$(date +%s)
    if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$1" \
         --backend mntp --task "$task" --data_path "$DIR/$data" --save_predictions >>"$LOG" 2>&1; then
      avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
      echo "$(ts) | RESULT $2 $data AVG=$avg ($(($(date +%s)-s))s)" | tee -a "$LOG"
    else
      echo "$(ts) | RESULT $2 $data FAILED" | tee -a "$LOG"
    fi
  done
}

eval_model "$REPO/checkpoints/gptbert_small_span_v2/final" "SPAN-v2-FINAL"
eval_model "$REPO/checkpoints/gptbert_small_v1/final"       "PHASE1-FINAL"
echo "$(ts) | ===== COMPARE COMPLETE =====" | tee -a "$LOG"
