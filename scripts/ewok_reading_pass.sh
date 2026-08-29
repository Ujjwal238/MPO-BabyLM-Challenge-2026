#!/bin/bash
# Cleanup pass (run after the main eval):
#   - final model: EWoK-fast + EWoK-full (ewok_filtered) + re-run reading (failed pre-statsmodels-fix)
#   - all checkpoints: EWoK-fast
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
CK=$REPO/checkpoints/gptbert_small_v1
LOG=$CK/eval_ewok_reading.log
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
ts(){ date "+%F %T"; }
log(){ echo "$(ts) | $*" | tee -a "$LOG"; }

# place fast-ewok into the path the eval expects (was staged aside during the curve)
if [ ! -d evaluation_data/fast_eval/ewok_fast ] && [ -d evaluation_data/ewok_fast_staged ]; then
  cp -r evaluation_data/ewok_fast_staged evaluation_data/fast_eval/ewok_fast
  log "placed fast-ewok -> evaluation_data/fast_eval/ewok_fast"
fi

run_ewok(){  # $1=model_dir  $2=label  $3=data_path
  local s=$(date +%s)
  log ">>> ewok $2"
  if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$1" \
       --backend mntp --task ewok --data_path "$3" --save_predictions >>"$LOG" 2>&1; then
    local avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
    log "<<< ewok $2 AVG=$avg ($(($(date +%s)-s))s)"
  else
    log "<<< FAILED ewok $2"
  fi
}

log "===== EWOK + READING PASS START ====="
# final model first (most important): fast + full ewok
run_ewok "$CK/final" "final-fast" evaluation_data/fast_eval/ewok_fast
[ -d evaluation_data/full_eval/ewok_filtered ] && run_ewok "$CK/final" "final-FULL" evaluation_data/full_eval/ewok_filtered
# re-run final reading (failed pre-statsmodels-fix)
s=$(date +%s); log ">>> reading final (re-run)"
if $PY -m evaluation_pipeline.reading.run --model_path_or_name "$CK/final" --backend mntp \
     --data_path evaluation_data/full_eval/reading/reading_data.csv >>"$LOG" 2>&1; then
  log "<<< reading final OK ($(($(date +%s)-s))s)"
else
  log "<<< FAILED reading final"
fi
# all checkpoints: fast ewok (completes the learning curve)
for n in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100; do
  [ -d "$CK/chck_${n}M" ] && run_ewok "$CK/chck_${n}M" "chck_${n}M" evaluation_data/fast_eval/ewok_fast
done
log "===== EWOK + READING PASS COMPLETE ====="