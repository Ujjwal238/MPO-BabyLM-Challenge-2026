#!/bin/bash
# Run zero-shot evaluation on a LOCAL checkpoint dir, with a live tailable log.
# Usage: eval_local.sh <model_dir> <label> <full|fast>
# Skips EWoK automatically if its data isn't downloaded yet (gated; needs HF login).
set -uo pipefail

REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
BACKEND=mntp
MODEL="$1"; LABEL="$2"; MODE="${3:-full}"
LOG="$REPO/checkpoints/gptbert_small_v1/eval_${LABEL}_${MODE}.log"
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
logmsg(){ echo "$(ts) | $*" | tee -a "$LOG"; }

if [ "$MODE" = "full" ]; then
  DIR=evaluation_data/full_eval
  TASKS=("blimp:blimp_filtered" "blimp:supplement_filtered" "ewok:ewok_filtered" "comps:comps")
else
  DIR=evaluation_data/fast_eval
  TASKS=("blimp:blimp_fast" "blimp:supplement_fast" "ewok:ewok_fast")
fi

logmsg "=== EVAL label='$LABEL' mode=$MODE backend=$BACKEND model=$MODEL ==="
overall=$(date +%s)
for t in "${TASKS[@]}"; do
  task="${t%%:*}"; data="${t##*:}"
  if [ ! -e "$DIR/$data" ]; then logmsg "SKIP $task/$data (data not present)"; continue; fi
  start=$(date +%s)
  logmsg ">>> START $task/$data"
  if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$MODEL" \
       --backend $BACKEND --task "$task" --data_path "$DIR/$data" --save_predictions >>"$LOG" 2>&1; then
    avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
    logmsg "<<< DONE  $task/$data | AVG=$avg | $(($(date +%s)-start))s"
  else
    logmsg "<<< FAILED $task/$data (see log above)"
  fi
done
# reading-time correlation
if [ -e "$DIR/reading/reading_data.csv" ]; then
  start=$(date +%s); logmsg ">>> START reading"
  $PY -m evaluation_pipeline.reading.run --model_path_or_name "$MODEL" --backend $BACKEND \
      --data_path "$DIR/reading/reading_data.csv" >>"$LOG" 2>&1 \
      && logmsg "<<< DONE reading | $(($(date +%s)-start))s" || logmsg "<<< FAILED reading"
fi
logmsg "=== EVAL '$LABEL' ($MODE) COMPLETE in $(($(date +%s)-overall))s ==="