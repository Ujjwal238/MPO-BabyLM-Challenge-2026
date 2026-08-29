#!/usr/bin/env python
"""Push the trained model + all intermediate checkpoints to the HF Hub.

Layout (as required by BabyLM): final model on `main`, each checkpoint as a branch
chck_1M..chck_100M. Uses the cached huggingface-cli login token.
"""
import os
import sys

from huggingface_hub import HfApi, create_branch, create_repo, upload_folder

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ID = "Ujjwal101/Ujjwal-bored"
CKDIR = f"{REPO}/checkpoints/gptbert_small_v1"
IGNORE = ["*.log", "eval_*", "*.json.lock", "runs/*", "*_state_dict*"]


def ck_key(d):
    return int(d.split("_")[1].rstrip("M"))


def main():
    api = HfApi()
    who = api.whoami()
    print("logged in as:", who.get("name"))
    create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    print("repo ready:", REPO_ID)

    # 1) final -> main
    print("\n=== uploading final -> main ===", flush=True)
    upload_folder(repo_id=REPO_ID, folder_path=f"{CKDIR}/final",
                  commit_message="final model (10 epochs / ~99M words seen)",
                  ignore_patterns=IGNORE)
    print("  main done", flush=True)

    # 2) checkpoints -> branches
    chks = sorted([d for d in os.listdir(CKDIR) if d.startswith("chck_")], key=ck_key)
    print(f"\n=== {len(chks)} checkpoint branches: {chks} ===", flush=True)
    for ck in chks:
        print(f"--- {ck} ---", flush=True)
        create_branch(REPO_ID, branch=ck, exist_ok=True)
        upload_folder(repo_id=REPO_ID, folder_path=f"{CKDIR}/{ck}", revision=ck,
                      commit_message=f"checkpoint {ck}", ignore_patterns=IGNORE)
        print(f"  {ck} done", flush=True)

    print("\nPUSH DONE ->", f"https://huggingface.co/{REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
