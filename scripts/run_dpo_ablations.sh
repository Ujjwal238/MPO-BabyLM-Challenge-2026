#!/bin/bash
# DPO ablations for the paper (Table 3), run sequentially to avoid MPS contention.
# Each: ~22 min train + ~8 min canary (fast BLiMP + supplement, mntp) on chck_dpo_1250.
# Reference (v1, already done): init=chck_70M, beta=0.1, mixed negatives, seed 42 -> 70.24/65.60.
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO" || exit 1
REPORT=$REPO/DPO_ABLATIONS.md
ts(){ date "+%F %T"; }
echo "# DPO ablation results ($(ts))" > "$REPORT"
echo "" >> "$REPORT"
echo "Reference v1: init=chck_70M beta=0.1 mixed-neg seed=42 -> chck_dpo_1250 = 70.24 / 65.60" >> "$REPORT"
echo "" >> "$REPORT"

run_one(){  # $1=run_name  $2..=extra train_dpo args
  local name="$1"; shift
  echo "$(ts) | === TRAIN $name ($*) ===" | tee -a "$REPORT"
  caffeinate -i "$PY" "$REPO/src/train_dpo.py" --run_name "$name" "$@" \
     > "$REPO/checkpoints/${name}_stdout.log" 2>&1
  # canary the step-1250 checkpoint (our operating point) for apples-to-apples with v1
  local ck="$REPO/checkpoints/$name/chck_dpo_1250"
  [ -d "$ck" ] || ck="$REPO/checkpoints/$name/final"
  "$REPO/scripts/dpo_canary.sh" "$ck" >/dev/null 2>&1
  local res=$(grep "RESULT" "$REPO/checkpoints/dpo_canary.log" | tail -2 | tr '\n' ' ')
  echo "$(ts) | RESULT $name : $res" | tee -a "$REPORT"
  # also capture pref-acc trajectory endpoint
  local pa=$(grep -oE "pref-acc [0-9.]+" "$REPO/checkpoints/$name/train.log" | tail -1)
  echo "$(ts) |   ($name final $pa)" | tee -a "$REPORT"
}

# 1) seed robustness (same config as v1, seeds 43 and 44)
run_one gptbert_small_dpo_s43 --init "${INIT:-$REPO/checkpoints/gptbert_small_v1/chck_70M}" --beta 0.1 --seed 43
run_one gptbert_small_dpo_s44 --init "${INIT:-$REPO/checkpoints/gptbert_small_v1/chck_70M}" --beta 0.1 --seed 44
# 2) init ablation: DPO from the FINAL (99M/10ep) checkpoint instead of the 70M peak
run_one gptbert_small_dpo_fromfinal --init "$REPO/checkpoints/gptbert_small_v1/final" --beta 0.1 --seed 42
# 3) deconfound v2: hard negatives at beta=0.1 (isolates negative-hardness from the beta bump)
run_one gptbert_small_dpo_hardneg_b01 --init "${INIT:-$REPO/checkpoints/gptbert_small_v1/chck_70M}" --beta 0.1 --hard_neg --seed 42

echo "$(ts) | === ALL DPO ABLATIONS COMPLETE ===" | tee -a "$REPORT"
