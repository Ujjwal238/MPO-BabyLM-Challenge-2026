#!/bin/bash
# Clone the official BabyLM 2026 evaluation pipeline, pin it to the commit the paper
# ran against, and apply our patch.
#
# The patch adds 39 lines across 7 files and does three things, none of which touch
# scoring: it routes tensors to MPS when CUDA is absent (6 files), it lets the
# age-of-acquisition task load byte-identical checkpoints from a local directory
# instead of re-downloading each of 19 revisions from the Hub, and it adds an opt-in
# no-context AoA variant behind AOA_USE_BOS_ONLY. Every number in the paper is scored
# by the organizers' code; see patches/babylm-eval-mps.patch to read it.
#
#   bash scripts/setup_eval.sh [target_dir]      # default: ../babylm-eval
#
# Then, in every shell that runs an evaluation:
#   export BABYLM_EVAL=<target_dir>/strict
set -euo pipefail

REPO=${MPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
TARGET=${1:-$(dirname "$REPO")/babylm-eval}
PIN=68cdd160f34826307e650c484904c274692e82ce   # babylm-org/babylm-eval, 2026-06-15

if [ -d "$TARGET/.git" ]; then
  echo "==> existing checkout at $TARGET"
else
  echo "==> cloning babylm-org/babylm-eval into $TARGET"
  git clone https://github.com/babylm-org/babylm-eval "$TARGET"
fi

cd "$TARGET"
echo "==> pinning to $PIN"
git checkout --quiet "$PIN"

echo "==> applying patches/babylm-eval-mps.patch"
if git apply --check "$REPO/patches/babylm-eval-mps.patch" 2>/dev/null; then
  git apply "$REPO/patches/babylm-eval-mps.patch"
  echo "    applied"
else
  echo "    already applied (or conflicts) — verifying"
  git apply --reverse --check "$REPO/patches/babylm-eval-mps.patch" 2>/dev/null \
    && echo "    confirmed already applied" \
    || { echo "    ERROR: patch neither applies nor is applied; resolve manually"; exit 1; }
fi

cat <<EOF

==> done.

Next, download the evaluation data (the organizers host it; we do not redistribute it):

    cd $TARGET/strict
    python -m scripts.download_evals
    python -m evaluation_pipeline.global_piqa.dl
    # ewok_fast.zip is password protected: BabyLM2025

Then point this repository's scripts at it:

    export BABYLM_EVAL=$TARGET/strict
EOF
