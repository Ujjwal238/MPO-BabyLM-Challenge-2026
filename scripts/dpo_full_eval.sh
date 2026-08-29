#!/bin/bash
# Full mntp eval of the best DPO checkpoint (chck_dpo_1250) — the candidate submission model.
# Order: cheap zero-shot suites first (results early), then reading, then entity_full (causal, ~3.7h) last.
# Writes only to checkpoints/gptbert_small_dpo_v1/ and the pipeline's results/ dir. Never touches Phase-1.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
MODEL=${1:-$REPO/checkpoints/gptbert_small_dpo_v1/chck_dpo_1250}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
LOG=$REPO/checkpoints/gptbert_small_dpo_v1/eval_full.log
DIR=evaluation_data/full_eval
ts(){ date "+%F %T"; }
logmsg(){ echo "$(ts) | $*" | tee -a "$LOG"; }

logmsg "===== DPO FULL EVAL | model=$MODEL ====="

# 1) zero-shot suites via mntp (fast->slower)
for t in "blimp:supplement_filtered" "ewok:ewok_filtered" "comps:comps" "blimp:blimp_filtered"; do
  task="${t%%:*}"; data="${t##*:}"
  [ -e "$DIR/$data" ] || { logmsg "SKIP $task/$data (missing)"; continue; }
  s=$(date +%s); logmsg ">>> $task/$data (mntp)"
  if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$MODEL" \
       --backend mntp --task "$task" --data_path "$DIR/$data" --save_predictions >>"$LOG" 2>&1; then
    avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
    logmsg "<<< $task/$data AVG=$avg ($(($(date +%s)-s))s)"
  else
    logmsg "<<< $task/$data FAILED"
  fi
done

# 2) reading-time correlation (mntp)
if [ -e "$DIR/reading/reading_data.csv" ]; then
  s=$(date +%s); logmsg ">>> reading (mntp)"
  $PY -m evaluation_pipeline.reading.run --model_path_or_name "$MODEL" --backend mntp \
      --data_path "$DIR/reading/reading_data.csv" >>"$LOG" 2>&1 \
      && logmsg "<<< reading done ($(($(date +%s)-s))s)" || logmsg "<<< reading FAILED"
fi

# 3) entity tracking — causal backend, full set (~3.7h on MPS slow-path); LAST
s=$(date +%s); logmsg ">>> entity_tracking FULL (causal, slow ~3.7h)"
if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$MODEL" \
     --backend causal --task entity_tracking --data_path "$DIR/entity_tracking" --save_predictions >>"$LOG" 2>&1; then
  avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
  logmsg "<<< entity_tracking AVG=$avg ($(($(date +%s)-s))s)"
else
  logmsg "<<< entity_tracking FAILED"
fi

logmsg "Phase-1 refs (mntp): blimp_filtered 69.83 | supp 63.77 | ewok 52.18 | comps 52.23 | entity(causal) 25.14"
logmsg "===== DPO FULL EVAL COMPLETE ====="
