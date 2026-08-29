#!/bin/bash
# Master zero-shot eval for the submission:
#   1) final model, FULL eval (blimp_filtered, supplement, comps, entity, reading; ewok if present)
#   2) FAST eval on every checkpoint chck_1M..100M (for the required learning-curve submission)
# Everything is mirrored into one tailable master log. (GLUE finetuning is run separately.)
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
CKDIR=$REPO/checkpoints/gptbert_small_v1
MLOG=$CKDIR/eval_all.log
ts(){ date "+%F %T"; }
log(){ echo "$(ts) | MASTER | $*" | tee -a "$MLOG"; }

log "===== MASTER EVAL START ====="
overall=$(date +%s)

log "[stage 1/2] final model -> FULL zero-shot eval"
bash "$REPO/scripts/eval_local.sh" "$CKDIR/final" final full 2>&1 | tee -a "$MLOG"

log "[stage 2/2] all checkpoints -> FAST zero-shot eval"
for n in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100; do
  ck="$CKDIR/chck_${n}M"
  if [ -d "$ck" ]; then
    bash "$REPO/scripts/eval_local.sh" "$ck" "chck_${n}M" fast 2>&1 | tee -a "$MLOG"
  fi
done

log "===== MASTER EVAL COMPLETE in $(($(date +%s)-overall))s ====="