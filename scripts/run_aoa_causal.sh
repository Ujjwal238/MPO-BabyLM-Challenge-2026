#!/bin/bash
# AoA with the CAUSAL backend (the eval's own default + Chang&Bergen methodology; our
# earlier mntp run scored a near-zero 4.45 vs the causal GPT-2 baseline's 35.68).
# Loads each checkpoint locally (AOA_LOCAL_CKPT_DIR) + offline. Writes to
# results/<stem>/main/zero_shot/causal/AoA_word/surprisal.json (separate from mntp).
# Usage: run_aoa_causal.sh <model_name_for_stem> <local_ckpt_dir>
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
MODELNAME=${1:-Ujjwal101/Ujjwal-bored}
CK=${2:-$REPO/checkpoints/gptbert_small_v1}
LOG=$CK/aoa_causal.log
PROG=$CK/aoa_causal_progress.log
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
export AOA_LOCAL_CKPT_DIR=$CK
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1
: > "$LOG"
START=$(date +%s)
echo "$(date '+%F %T') | AoA CAUSAL START | model=$MODELNAME ckpts=$CK" > "$PROG"

caffeinate -i "$PY" -m evaluation_pipeline.AoA_word.run \
  --model_name "$MODELNAME" --backend causal --track_name strict-small \
  --word_path evaluation_data/full_eval/aoa/cdi_childes.json \
  --output_dir results --resume >>"$LOG" 2>&1 &
PID=$!

while kill -0 "$PID" 2>/dev/null; do
  el=$(( $(date +%s) - START ))
  seen=$(grep -c "Checkpoint: chck" "$LOG" 2>/dev/null || echo 0)
  cur=$(grep "Checkpoint: chck" "$LOG" 2>/dev/null | tail -1 | sed 's/.*Checkpoint: //')
  printf "%s | elapsed %dm%02ds | checkpoints seen %s | current %s\n" \
    "$(date '+%F %T')" $((el/60)) $((el%60)) "${seen:-0}" "${cur:-?}" > "$PROG"
  sleep 20
done
el=$(( $(date +%s) - START ))
final=$(grep -c "Results saved to:" "$LOG" 2>/dev/null || echo 0)
printf "%s | AoA CAUSAL DONE in %dm%02ds | final-save-logged=%s\n" \
  "$(date '+%F %T')" $((el/60)) $((el%60)) "$final" >> "$PROG"
