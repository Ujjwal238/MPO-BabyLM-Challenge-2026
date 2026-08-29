#!/bin/bash
# GLUE finetuning on the DPO best checkpoint (chck_dpo_1250) — same config as glue_all.sh
# (seq=128, 10ep, lr 3e-5, seed 42, identical task order) so numbers are comparable to Phase-1.
# Writes to the DPO run dir's log + results/chck_dpo_1250/ (Phase-1's results/final/ untouched).
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
MODEL=${MODEL:-$REPO/checkpoints/gptbert_small_dpo_v1/chck_dpo_1250}
LOG=$REPO/checkpoints/gptbert_small_dpo_v1/glue.log
SEQ=${SEQ:-128}
EP=${EP:-10}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
ts(){ date "+%F %T"; }
log(){ echo "$(ts) | $*" | tee -a "$LOG"; }
G=evaluation_data/full_eval/glue_filtered

run_task(){  # $1=task $2=num_labels $3=batch $4=epochs $5=metric_for_valid $6=metrics(optional)
  local s=$(date +%s)
  local METRICS="${6:-accuracy f1 mcc}"   # mnli (3-class) must use only "accuracy"
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

log "===== GLUE FINETUNING (DPO chck_dpo_1250) START (seq=$SEQ epochs=$EP) ====="
log "Phase-1 refs: rte 65.5 | wsc 65.4 | mrpc 83.8/f1 88.3 | mnli 57.9 | qqp 77.3/f1 71.9 | boolq 65.2 | multirc 65.4"
ov=$(date +%s)
run_task rte     2 32 "$EP"        accuracy
run_task wsc     2 32 "$((EP*3))"  accuracy
run_task mrpc    2 32 "$EP"        f1
run_task mnli    3 32 "$EP"        accuracy "accuracy"
run_task qqp     2 32 "$EP"        f1
run_task boolq   2 16 "$EP"        accuracy
run_task multirc 2 16 "$EP"        accuracy
log "===== GLUE FINETUNING (DPO) COMPLETE in $(($(date +%s)-ov))s ====="
