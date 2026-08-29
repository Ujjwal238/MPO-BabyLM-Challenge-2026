#!/usr/bin/env python
"""Push the DPO submission model to a NEW private HF repo: Ujjwal101/ujjwal-very-bored.

Layout (honest trajectory for a post-trained model):
  main                = chck_dpo_1250 (the submission: chck_70M + DPO post-training)
  chck_1M..chck_70M   = the shared PRETRAINING lineage (Phase-1's checkpoints; DPO began at 70M)
  chck_dpo_250..1500  = the DPO POST-TRAINING trajectory
No chck_80M..100M: this model never trained MLE past 70M (it switched to DPO), so those
checkpoints are not part of its history.
"""
import os
import sys

from huggingface_hub import HfApi, create_branch, create_repo, upload_folder

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ID = "Ujjwal101/ujjwal-very-bored"
V1 = f"{REPO}/checkpoints/gptbert_small_v1"      # pretraining lineage
DPO = f"{REPO}/checkpoints/gptbert_small_dpo_v1"  # post-training
IGNORE = ["*.log", "eval_*", "*.json.lock", "runs/*", "*_state_dict*", "stdout.log"]

PRETRAIN_CKPTS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i}M" for i in (10, 20, 30, 40, 50, 60, 70)]
DPO_CKPTS = [f"chck_dpo_{s}" for s in (250, 500, 750, 1000, 1250, 1500)]


def main():
    api = HfApi()
    print("logged in as:", api.whoami().get("name"), flush=True)
    create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    print("repo ready (PRIVATE):", REPO_ID, flush=True)

    # 1) final submission model -> main
    print("\n=== main <- chck_dpo_1250 ===", flush=True)
    upload_folder(repo_id=REPO_ID, folder_path=f"{DPO}/chck_dpo_1250",
                  commit_message="submission: chck_70M + MP-DPO post-training (chck_dpo_1250)",
                  ignore_patterns=IGNORE)
    print("  main done", flush=True)

    # 2) pretraining lineage -> chck_1M..chck_70M
    print(f"\n=== {len(PRETRAIN_CKPTS)} pretraining-lineage branches ===", flush=True)
    for ck in PRETRAIN_CKPTS:
        src = f"{V1}/{ck}"
        if not os.path.isdir(src):
            print(f"  SKIP {ck} (missing)", flush=True); continue
        create_branch(REPO_ID, branch=ck, exist_ok=True)
        upload_folder(repo_id=REPO_ID, folder_path=src, revision=ck,
                      commit_message=f"pretraining checkpoint {ck} (shared Phase-1 lineage)",
                      ignore_patterns=IGNORE)
        print(f"  {ck} done", flush=True)

    # 3) DPO post-training trajectory -> chck_dpo_*
    print(f"\n=== {len(DPO_CKPTS)} DPO post-training branches ===", flush=True)
    for ck in DPO_CKPTS:
        src = f"{DPO}/{ck}"
        if not os.path.isdir(src):
            print(f"  SKIP {ck} (missing)", flush=True); continue
        create_branch(REPO_ID, branch=ck, exist_ok=True)
        upload_folder(repo_id=REPO_ID, folder_path=src, revision=ck,
                      commit_message=f"DPO post-training {ck}", ignore_patterns=IGNORE)
        print(f"  {ck} done", flush=True)

    print("\nPUSH DONE ->", f"https://huggingface.co/{REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
