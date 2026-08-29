#!/bin/bash
# Per-operator corruption ablation (guide review #5): four MPO runs, each using a
# SINGLE corruption operator via one-hot --hard_neg_weights (adj,dist,func,rand).
# Identical protocol to Table 3 rows: init=chck_70M, beta=0.1, seed=42, canary @1250.
# Sequential to avoid MPS contention. ~30 min per run, ~2h total.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO" || exit 1
REPORT=$REPO/DPO_OP_ABLATION.md
ts(){ date "+%F %T"; }
echo "# Per-operator corruption ablation ($(ts))" > "$REPORT"
echo "" >> "$REPORT"
echo "Reference: mixed uniform (v1) chck_dpo_1250 = 70.24 / 65.60 | baseline 70.01 / 65.20 | chck_70M 70.13 / 65.60" >> "$REPORT"
echo "" >> "$REPORT"

run_one(){  # $1=short_name  $2=one-hot weights
  local name="gptbert_small_dpo_op_$1"
  echo "$(ts) | === TRAIN $name (weights $2) ===" | tee -a "$REPORT"
  caffeinate -i "$PY" "$REPO/src/train_dpo.py" --run_name "$name" \
     --init "${INIT:-$REPO/checkpoints/gptbert_small_v1/chck_70M}" --beta 0.1 --seed 42 \
     --hard_neg --hard_neg_weights "$2" \
     > "$REPO/checkpoints/${name}_stdout.log" 2>&1
  local ck="$REPO/checkpoints/$name/chck_dpo_1250"
  [ -d "$ck" ] || ck="$REPO/checkpoints/$name/final"
  "$REPO/scripts/dpo_canary.sh" "$ck" >/dev/null 2>&1
  local res=$(grep "RESULT" "$REPO/checkpoints/dpo_canary.log" | tail -2 | tr '\n' ' ')
  local pa=$(grep -oE "pref-acc [0-9.]+" "$REPO/checkpoints/$name/train.log" | tail -1)
  echo "$(ts) | RESULT $name : $res ($pa)" | tee -a "$REPORT"
}

run_one adj  "1,0,0,0"
run_one dist "0,1,0,0"
run_one func "0,0,1,0"
run_one rand "0,0,0,1"

echo "$(ts) | === ALL OPERATOR ABLATIONS COMPLETE ===" | tee -a "$REPORT"
