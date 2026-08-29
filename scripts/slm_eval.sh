#!/bin/bash
# A/B: SLM (selective LM / token-level RHO-LOSS) final vs Phase-1 reference numbers.
# fast BLiMP + supplement, mntp. Phase-1 reference: BLiMP 70.01 / supplement 65.20.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
LOG=$REPO/checkpoints/slm_eval.log
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

eval_model "$REPO/checkpoints/gptbert_small_slm_v1/final" "SLM-FINAL"
echo "$(ts) | Phase-1 reference: BLiMP 70.01 / supplement 65.20" | tee -a "$LOG"
echo "$(ts) | ===== SLM EVAL COMPLETE =====" | tee -a "$LOG"
