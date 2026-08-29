#!/bin/bash
# Rerun updated/new tasks (2026-07-09 release) for both leaderboard entries:
#   - entity_tracking: bias-filtered in new code (causal backend, ~2.5h/model)
#   - global_piqa parallel+nonparallel: new task (mntp), main + all fast revisions
# Then copy into the OLD results tree, re-collate with the NEW collate_preds, and
# leave JSONs ready for resubmission. AoA/GLUE/BLiMP etc. reuse existing predictions.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
NEW=$REPO/eval_v2/strict
OLDRES=${BABYLM_EVAL:?set BABYLM_EVAL}/results
PY=${MPO_PYTHON:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$NEW" || exit 1
LOG=$REPO/checkpoints/rerun_new_tasks.log
ts(){ date "+%F %T"; }
log(){ echo "$(ts) | $*" | tee -a "$LOG"; }
: > "$LOG"

mkdir -p models
ln -sfn "$REPO/checkpoints/gptbert_small_v1/final"          models/Ujjwal-bored
ln -sfn "$REPO/checkpoints/gptbert_small_dpo_v1/chck_dpo_1250" models/chck_dpo_1250

run_zs(){ # $1=model_path $2=backend $3=task $4=data_dir $5=label
  local s=$(date +%s)
  if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$1" \
       --backend "$2" --task "$3" --data_path "$4" --save_predictions >>"$LOG" 2>&1; then
    local avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
    log "RESULT $5 AVG=$avg ($(($(date +%s)-s))s)"
  else
    log "RESULT $5 FAILED"
  fi
}

for STEM in Ujjwal-bored chck_dpo_1250; do
  log "===== MODEL $STEM ====="
  # 1) GlobalPIQA main (mntp)
  run_zs "models/$STEM" mntp global_piqa_parallel    evaluation_data/full_eval/global_piqa_parallel    "$STEM piqa_par"
  run_zs "models/$STEM" mntp global_piqa_nonparallel evaluation_data/full_eval/global_piqa_nonparallel "$STEM piqa_non"
  # 2) entity filtered (causal)
  run_zs "models/$STEM" causal entity_tracking evaluation_data/full_eval/entity_tracking "$STEM entity_filtered"
  # copy into old tree: piqa (mntp) + entity (causal AND mntp-copy, as before)
  for t in global_piqa_parallel global_piqa_nonparallel; do
    mkdir -p "$OLDRES/$STEM/main/zero_shot/mntp"
    rm -rf "$OLDRES/$STEM/main/zero_shot/mntp/$t"
    cp -R "results/$STEM/main/zero_shot/mntp/$t" "$OLDRES/$STEM/main/zero_shot/mntp/$t"
  done
  rm -rf "$OLDRES/$STEM/main/zero_shot/causal/entity_tracking" "$OLDRES/$STEM/main/zero_shot/mntp/entity_tracking"
  cp -R "results/$STEM/main/zero_shot/causal/entity_tracking" "$OLDRES/$STEM/main/zero_shot/causal/entity_tracking"
  cp -R "results/$STEM/main/zero_shot/causal/entity_tracking" "$OLDRES/$STEM/main/zero_shot/mntp/entity_tracking"
done

# 3) GlobalPIQA fast revisions for the Phase-1 lineage (shared by both entries)
for i in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100; do
  CK=$REPO/checkpoints/gptbert_small_v1/chck_${i}M
  [ -d "$CK" ] || continue
  ln -sfn "$CK" models/ckpt_tmp
  for t in global_piqa_parallel global_piqa_nonparallel; do
    run_zs models/ckpt_tmp mntp "$t" "evaluation_data/fast_eval/$t" "chck_${i}M $t"
    dst="$OLDRES/Ujjwal-bored/chck_${i}M/zero_shot/mntp/$t"
    mkdir -p "$(dirname "$dst")"; rm -rf "$dst"
    cp -R "results/ckpt_tmp/main/zero_shot/mntp/$t" "$dst"
  done
  rm -rf results/ckpt_tmp
done
# DPO lineage: 1M..70M identical to Phase-1 -> copy; 80/90/100M = chck_dpo_1250 (budget-elapsed convention)
ln -sfn "$REPO/checkpoints/gptbert_small_dpo_v1/chck_dpo_1250" models/ckpt_tmp
for t in global_piqa_parallel global_piqa_nonparallel; do
  run_zs models/ckpt_tmp mntp "$t" "evaluation_data/fast_eval/$t" "dpo_final_for_tail $t"
done
for i in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70; do
  for t in global_piqa_parallel global_piqa_nonparallel; do
    src="$OLDRES/Ujjwal-bored/chck_${i}M/zero_shot/mntp/$t"
    dst="$OLDRES/chck_dpo_1250/chck_${i}M/zero_shot/mntp/$t"
    [ -d "$src" ] && mkdir -p "$(dirname "$dst")" && rm -rf "$dst" && cp -R "$src" "$dst"
  done
done
for i in 80 90 100; do
  for t in global_piqa_parallel global_piqa_nonparallel; do
    dst="$OLDRES/chck_dpo_1250/chck_${i}M/zero_shot/mntp/$t"
    mkdir -p "$(dirname "$dst")"; rm -rf "$dst"
    cp -R "results/ckpt_tmp/main/zero_shot/mntp/$t" "$dst"
  done
done
rm -rf results/ckpt_tmp

# 4) re-collate both with the NEW collate_preds against the old tree
for STEM in Ujjwal-bored chck_dpo_1250; do
  log ">>> collate $STEM (new pipeline)"
  $PY -m evaluation_pipeline.collate_preds --model_path_or_name "$STEM" --backend mntp --fast \
      --track strict-small --results_dir "$OLDRES" --fast_eval_dir evaluation_data/fast_eval >>"$LOG" 2>&1 \
    && log "<<< collate $STEM done: $OLDRES/$STEM/all_full_preds_and_fast_scores_mntp.json" \
    || log "<<< collate $STEM FAILED"
done
log "===== RERUN NEW TASKS COMPLETE ====="
