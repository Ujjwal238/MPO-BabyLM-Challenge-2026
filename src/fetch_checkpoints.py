#!/usr/bin/env python
"""Fetch published checkpoints from the Hugging Face Hub into ./checkpoints.

Every checkpoint behind every number in the paper is public. Each is a branch of one
of two repositories, and each is a self-contained `trust_remote_code` model directory
(weights, config, modeling code, tokenizer), so a fetched directory loads directly:

    AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)

Nothing here trains anything. The two repositories are:

  Ujjwal101/Ujjwal-bored        the 10-epoch baseline. `main` is the final model at
                                99M words seen; branches chck_1M..chck_100M are the
                                word-milestone checkpoints behind the saturation
                                diagnosis (Fig. 2, Tables 5 and 6).

  Ujjwal101/ujjwal-very-bored   the MPO model. `main` is chck_dpo_1250, the submitted
                                checkpoint. Branches chck_1M..chck_70M are the shared
                                pretraining lineage; chck_dpo_250..chck_dpo_1500 are the
                                preference-phase trajectory (Fig. 5b). chck_80M/90M/100M
                                repeat the final model: this run stopped MLE at 70M and
                                never trained past it, so the milestones exist only to
                                satisfy the submission format.

  python src/fetch_checkpoints.py --what headline     # the two checkpoints in the paper
  python src/fetch_checkpoints.py --what saturation   # the 19-point diagnosis trajectory
  python src/fetch_checkpoints.py --what mpo-phase    # the preference-phase trajectory
  python src/fetch_checkpoints.py --what all
  python src/fetch_checkpoints.py --list
"""
import argparse
import os
import sys

BASE = "Ujjwal101/Ujjwal-bored"
MPO = "Ujjwal101/ujjwal-very-bored"

# milestone grid used by train.py: every 1M for the first epoch, then every 10M
MILESTONES = [f"chck_{i}M" for i in range(1, 11)] + [f"chck_{i}M" for i in range(20, 101, 10)]
DPO_STEPS = [f"chck_dpo_{s}" for s in (250, 500, 750, 1000, 1250, 1500)]

SETS = {
    # the two models in Table 1
    "headline": [
        (BASE, "main", "baseline_10ep_final"),
        (MPO, "main", "mpo_submitted_dpo1250"),
    ],
    # Fig. 2 / Table 5: fast BLiMP against words seen, 19 points
    "saturation": [(BASE, ck, f"baseline_{ck}") for ck in MILESTONES],
    # the checkpoint MPO initializes from, and its frozen reference
    "init": [(BASE, "chck_70M", "baseline_chck_70M")],
    # Fig. 5b: the evaluation plateau across preference-phase checkpoints
    "mpo-phase": [(MPO, ck, f"mpo_{ck}") for ck in DPO_STEPS],
    # Table 3 initialization ablation: too early, at, and past saturation
    "init-ablation": [
        (BASE, "chck_30M", "baseline_chck_30M"),
        (BASE, "chck_70M", "baseline_chck_70M"),
        (BASE, "main", "baseline_10ep_final"),
    ],
}
SETS["all"] = SETS["saturation"] + SETS["mpo-phase"] + SETS["headline"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--what", default="headline", choices=sorted(SETS),
                   help="which checkpoint set to fetch (default: headline)")
    p.add_argument("--out_dir", default=None,
                   help="destination (default: <repo>/checkpoints/hub)")
    p.add_argument("--list", action="store_true", help="print the set and exit")
    args = p.parse_args()

    root = os.environ.get("MPO_ROOT",
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir or os.path.join(root, "checkpoints", "hub")
    items = SETS[args.what]

    if args.list:
        for repo, rev, name in items:
            print(f"{name:<28} {repo}@{rev}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    seen, n = set(), 0
    for repo, rev, name in items:
        if (repo, rev) in seen:          # init-ablation repeats chck_70M
            continue
        seen.add((repo, rev))
        dest = os.path.join(out_dir, name)
        if os.path.isdir(dest) and os.path.exists(os.path.join(dest, "pytorch_model.bin")):
            print(f"  have {name}")
            continue
        print(f"  get  {name}  <-  {repo}@{rev}", flush=True)
        snapshot_download(repo_id=repo, revision=rev, local_dir=dest,
                          allow_patterns=["*.json", "*.bin", "*.py", "*.txt"])
        n += 1
    print(f"\n{n} fetched, {len(seen) - n} already present  ->  {out_dir}")


if __name__ == "__main__":
    main()
