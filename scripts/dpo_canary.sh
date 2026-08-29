#!/bin/bash
# DPO canary: fast BLiMP + supplement on a DPO checkpoint vs chck_70M baseline.
# Abort rule (lever-1 collapse guard): if the fragile long-distance cluster drops hard
# (wh_vs_that_with_gap / distractor_agreement_relational_noun > 10 abs below chck_70M),
# kill the DPO run -- it is repeating the contrastive failure and won't recover.
# Usage: scripts/dpo_canary.sh <checkpoint_dir>
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=${1:?usage: dpo_canary.sh <checkpoint_dir>}
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
LOG=$REPO/checkpoints/dpo_canary.log
ts(){ date "+%F %T"; }
echo "$(ts) | ===== canary $CK =====" | tee -a "$LOG"
for t in "blimp:blimp_fast" "blimp:supplement_fast"; do
  task="${t%%:*}"; data="${t##*:}"
  if $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$CK" \
       --backend mntp --task "$task" --data_path "evaluation_data/fast_eval/$data" >>"$LOG" 2>&1; then
    avg=$(grep -iE "AVERAGE ACCURACY" -A1 "$LOG" | tail -1 | tr -d '[:space:]')
    echo "$(ts) | RESULT $data AVG=$avg" | tee -a "$LOG"
  fi
done
echo "$(ts) | chck_70M refs: BLiMP 70.13 / supp 65.60 | ABORT if either < ~66 / < ~60" | tee -a "$LOG"
