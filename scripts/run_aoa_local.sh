#!/bin/bash
# AoA across all 19 checkpoints, loading each from LOCAL disk (no Hub, no network
# stall) via AOA_LOCAL_CKPT_DIR + HF_HUB_OFFLINE=1. --resume => crash-safe (interim
# results persisted per-checkpoint). Writes a detailed tqdm log (aoa.log) and a
# high-level tailable progress log (aoa_progress.log: elapsed / N-of-19 / ETA).
set -uo pipefail
REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${MPO_PYTHON:-python}
CK=$REPO/checkpoints/gptbert_small_v1
LOG=$CK/aoa.log
PROG=$CK/aoa_progress.log
TOTAL=19
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
export AOA_LOCAL_CKPT_DIR=$CK
cd "${BABYLM_EVAL:?set BABYLM_EVAL to your patched babylm-eval/strict checkout}" || exit 1

: > "$LOG"
START=$(date +%s)
echo "$(date '+%F %T') | AoA START (local+offline, mntp, 19 ckpts) | ETA ~2h" > "$PROG"

caffeinate -i "$PY" -m evaluation_pipeline.AoA_word.run \
  --model_name Ujjwal101/Ujjwal-bored --backend mntp --track_name strict-small \
  --word_path evaluation_data/full_eval/aoa/cdi_childes.json \
  --output_dir results --resume >>"$LOG" 2>&1 &
PID=$!

while kill -0 "$PID" 2>/dev/null; do
  el=$(( $(date +%s) - START ))
  seen=$(grep -c "Checkpoint: chck" "$LOG" 2>/dev/null || echo 0)
  done_ck=$(( seen > 0 ? seen - 1 : 0 ))
  cur=$(grep "Checkpoint: chck" "$LOG" 2>/dev/null | tail -1 | sed 's/.*Checkpoint: //')
  word=$(grep -oE '[0-9]+it \[' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo 0)
  if [ "$done_ck" -ge 1 ]; then
    eta=$(( (el / done_ck) * (TOTAL - done_ck) / 60 ))
  else
    eta=-1
  fi
  if [ "$eta" -ge 0 ]; then etastr="~${eta}m"; else etastr="(calibrating)"; fi
  printf "%s | elapsed %dm%02ds | checkpoints %d/%d done | current %s (word %s/504) | ETA %s\n" \
    "$(date '+%F %T')" $((el/60)) $((el%60)) "$done_ck" "$TOTAL" "${cur:-?}" "${word:-0}" "$etastr" > "$PROG"
  sleep 20
done

ec=$?
el=$(( $(date +%s) - START ))
final=$(grep -c "Results saved to:" "$LOG" 2>/dev/null || echo 0)
printf "%s | AoA FINISHED in %dm%02ds | exit=%s | final-save-logged=%s\n" \
  "$(date '+%F %T')" $((el/60)) $((el%60)) "$ec" "$final" >> "$PROG"
