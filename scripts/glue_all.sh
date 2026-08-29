#!/bin/bash
# GLUE finetuning on the final model, MPS-tractable config (seq=128 — GPT-BERT's
# relative-position attention falls back to CPU at seq>128 on MPS). Small tasks first
# (quick wins + rate calibration), big tasks last. Logged to glue.log.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
CK=$REPO/checkpoints/gptbert_small_v1
MODEL=$CK/final
LOG=$CK/glue.log
SEQ=${SEQ:-128}
EP=${EP:-10}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
ts(){ date "+%F %T"; }
log(){ echo "$(ts) | $*" | tee -a "$LOG"; }
G=evaluation_data/full_eval/glue_filtered

run_task(){  # $1=task $2=num_labels $3=batch $4=epochs $5=metric_for_valid $6=metrics(optional)
  local s=$(date +%s)
  local METRICS="${6:-accuracy f1 mcc}"   # 3-class (mnli) must use only "accuracy": f1/mcc default to binary avg -> ValueError
  log ">>> GLUE $1 (labels=$2 bsz=$3 ep=$4 metric=$5 metrics='$METRICS' seq=$SEQ)"
  if $PY -m evaluation_pipeline.finetune.run --model_name_or_path "$MODEL" \
      --train_data "$G/$1.train.jsonl" --valid_data "$G/$1.valid.jsonl" --predict_data "$G/$1.valid.jsonl" \
      --task "$1" --num_labels "$2" --batch_size "$3" --learning_rate 3e-5 --num_epochs "$4" \
      --sequence_length "$SEQ" --results_dir results --save --save_dir models \
      --metrics $METRICS --metric_for_valid "$5" --seed 42 --padding_side left --take_final >>"$LOG" 2>&1; then
    local sc=$(grep -iE "accuracy:|f1:|mcc:" "$LOG" | tail -2 | tr '\n' ' ')
    log "<<< GLUE $1 done | $sc | $(($(date +%s)-s))s"
  else
    log "<<< GLUE $1 FAILED"
  fi
}

log "===== GLUE FINETUNING START (seq=$SEQ epochs=$EP) ====="
ov=$(date +%s)
run_task rte     2 32 "$EP"        accuracy
run_task wsc     2 32 "$((EP*3))"  accuracy
run_task mrpc    2 32 "$EP"        f1
run_task mnli    3 32 "$EP"        accuracy "accuracy"
run_task qqp     2 32 "$EP"        f1
run_task boolq   2 16 "$EP"        accuracy
run_task multirc 2 16 "$EP"        accuracy
log "===== GLUE FINETUNING COMPLETE in $(($(date +%s)-ov))s ====="