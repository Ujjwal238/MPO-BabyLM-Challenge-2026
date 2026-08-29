#!/bin/bash
# Fill remaining leaderboard gaps after GLUE:
#   1) mnli finetune re-run (corrected: --metrics accuracy only, 3-class)
#   2) Entity Tracking via CAUSAL backend (mntp was pathological on MPS for long seqs):
#      final model (full) + all checkpoints (fast)
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
CK=$REPO/checkpoints/gptbert_small_v1
MODEL=$CK/final
LOG=$CK/gapfill.log
SEQ=128
EP=5
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
ts(){ date "+%F %T"; }
log(){ echo "$(ts) | $*" | tee -a "$LOG"; }
G=evaluation_data/full_eval/glue_filtered

log "===== GAP-FILL START ====="

# 1) mnli re-run (3-class -> accuracy metric only)
s=$(date +%s); log ">>> mnli re-run (finetune, metrics=accuracy)"
if $PY -m evaluation_pipeline.finetune.run --model_name_or_path "$MODEL" \
    --train_data "$G/mnli.train.jsonl" --valid_data "$G/mnli.valid.jsonl" --predict_data "$G/mnli.valid.jsonl" \
    --task mnli --num_labels 3 --batch_size 32 --learning_rate 3e-5 --num_epochs "$EP" \
    --sequence_length "$SEQ" --results_dir results --save --save_dir models \
    --metrics accuracy --metric_for_valid accuracy --seed 42 --padding_side left --take_final >>"$LOG" 2>&1; then
  log "<<< mnli done | $(grep -i 'accuracy' results/final/main/finetune/mnli/results.txt 2>/dev/null | tail -1) | $(($(date +%s)-s))s"
else
  log "<<< mnli FAILED"
fi

# 2) Entity Tracking via causal backend
ent_run(){  # $1=model_dir $2=label $3=data_path
  local s=$(date +%s); log ">>> entity $2 (causal)"
  if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$1" --backend causal \
      --task entity_tracking --data_path "$3" --save_predictions >>"$LOG" 2>&1; then
    log "<<< entity $2 | $(grep -iE 'AVERAGE ACCURACY' -A1 "$LOG" | tail -1 | tr -d '[:space:]') | $(($(date +%s)-s))s"
  else
    log "<<< entity $2 FAILED"
  fi
}
ent_run "$MODEL" "final-FULL" evaluation_data/full_eval/entity_tracking
for n in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100; do
  [ -d "$CK/chck_${n}M" ] && ent_run "$CK/chck_${n}M" "chck_${n}M" evaluation_data/fast_eval/entity_tracking_fast
done

log "===== GAP-FILL COMPLETE ====="