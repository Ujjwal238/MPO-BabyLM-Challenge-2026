#!/bin/bash
# Weak-suite battery for the uni8k (morpheme-aligned Unigram-8192 tokenizer) run.
# Gate (NEXT_APPROACH_PLAN §4): mean(BLiMP, supp, EWoK, entity) vs Phase-1 refs,
# no suite catastrophically down. Phase-1 fast refs: BLiMP 70.01 | supp 65.20 | EWoK 52.82.
# Phase-1 entity refs: fast/causal (see gap_fill logs) + full/causal 25.14.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
MODEL=$REPO/checkpoints/gptbert_small_uni8k_v1/final
LOG=$REPO/checkpoints/uni8k_eval.log
: > "$LOG"
ts(){ date "+%F %T"; }

run_task(){  # $1=task $2=data_path $3=backend $4=label
  echo "$(ts) | >>> $4 ($3)" | tee -a "$LOG"
  local s=$(date +%s)
  if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$MODEL" \
       --backend "$3" --task "$1" --data_path "$2" --save_predictions >>"$LOG" 2>&1; then
    avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
    echo "$(ts) | RESULT $4 AVG=$avg ($(($(date +%s)-s))s)" | tee -a "$LOG"
  else
    echo "$(ts) | RESULT $4 FAILED" | tee -a "$LOG"
  fi
}

run_task blimp evaluation_data/fast_eval/blimp_fast        mntp   blimp_fast
run_task blimp evaluation_data/fast_eval/supplement_fast   mntp   supplement_fast
run_task ewok  evaluation_data/fast_eval/ewok_fast         mntp   ewok_fast
run_task entity_tracking evaluation_data/fast_eval/entity_tracking_fast causal entity_fast
run_task entity_tracking evaluation_data/full_eval/entity_tracking     causal entity_full
run_task comps evaluation_data/full_eval/comps             mntp   comps_full

echo "$(ts) | Phase-1 refs: BLiMP 70.01 | supp 65.20 | EWoK-fast 52.82 | entity_full(causal) 25.14 | COMPS 52.23" | tee -a "$LOG"
echo "$(ts) | ===== UNI8K EVAL COMPLETE =====" | tee -a "$LOG"
