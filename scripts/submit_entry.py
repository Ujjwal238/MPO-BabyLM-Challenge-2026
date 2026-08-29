#!/usr/bin/env python
"""Submit/refresh a leaderboard entry via the Gradio API and print our strict-small row.
Usage: submit_entry.py <stem> <model_name> (stem = results dir under strict/results)."""
import json, os, sys, uuid
import requests
from huggingface_hub import get_token

BASE = "https://babylm-community-babylm-leaderboard-2026.hf.space"
H = {"Authorization": f"Bearer {get_token()}"}
STEM, NAME = sys.argv[1], sys.argv[2]
EVAL = os.environ.get("BABYLM_EVAL") or sys.exit("set BABYLM_EVAL to your patched babylm-eval/strict checkout")
JSONF = f"{EVAL}/results/{STEM}/all_full_preds_and_fast_scores_mntp.json"

FORMS = {
    "ujjwal-bored": dict(
        repo="Ujjwal101/Ujjwal-bored", epochs=10, gpu=12.5, flops=3.8e16,
        contribs=[], synth="Not applicable",
        desc=("33M GPT-BERT hybrid (masked+causal) trained from scratch on BabyLM Strict-Small 10M "
              "for 10 epochs. AoA scored with the causal backend; entity tracking causal; mntp elsewhere.")),
    "ujjwal-very-bored": dict(
        repo="Ujjwal101/ujjwal-very-bored", epochs=7.2, gpu=9, flops=2.7e16,
        contribs=["Training objective innovations", "Preference optimization (DPO) post-training"],
        synth=("DPO negatives are length-preserving corruptions (token swaps, function-word subs) of real "
               "corpus sentences; no external model, no new words."),
        desc=("33M GPT-BERT hybrid (masked+causal) pretrained ~7 epochs on BabyLM Strict-Small 10M, then "
              "minimal-pair DPO post-training (~1.3M within-budget word-touches; training ended at ~71.3M "
              "words seen; milestone checkpoints chck_80M/90M/100M equal the final model). AoA causal; "
              "entity causal; mntp elsewhere.")),
}
F = FORMS[NAME]
data = [NAME, "main", F["repo"], "strict-small", None, None, "Hybrid masked+causal",
        F["contribs"], "GPT-BERT", "cosine", F["epochs"], "Byte-Level BPE", "42", 6, 128, 900,
        "BabyLM strict-small", 0, "", 0.0005, "AdamW", 16384, 16384, 12, 33046084, F["flops"], F["gpu"],
        "Not applicable", "", F["synth"], F["desc"], None, "", False, False, False, 0, 0, 0]
assert len(data) == 39

with open(JSONF, "rb") as f:
    up = requests.post(f"{BASE}/gradio_api/upload", headers=H,
                       files={"files": (os.path.basename(JSONF), f, "application/json")})
up.raise_for_status()
data[4] = {"path": up.json()[0], "meta": {"_type": "gradio.FileData"}, "orig_name": os.path.basename(JSONF)}
sess = str(uuid.uuid4())
requests.post(f"{BASE}/gradio_api/queue/join", headers=H,
              json={"data": data, "fn_index": 0, "session_hash": sess}).raise_for_status()
r = requests.get(f"{BASE}/gradio_api/queue/data", headers=H, params={"session_hash": sess},
                 stream=True, timeout=600)
for line in r.iter_lines():
    if not line: continue
    s = line.decode("utf-8", "replace")
    if not s.startswith("data:"): continue
    try: msg = json.loads(s[5:].strip())
    except Exception: continue
    if msg.get("msg") == "process_completed":
        out = msg.get("output", {}).get("data", [])
        status = out[0] if out and isinstance(out[0], str) else "?"
        print("STATUS:", status.split(">")[1].split("<")[0] if ">" in status else status)
        # find strict-small table (any component whose rows have Track == strict-small)
        for comp in out[1:]:
            if not (isinstance(comp, dict) and "data" in comp): continue
            hdrs = comp.get("headers", []); rows = comp.get("data", [])
            if "Track" not in hdrs or not rows: continue
            ti = hdrs.index("Track")
            if not any(r[ti] == "strict-small" for r in rows): continue
            def gv(row, col): return row[hdrs.index(col)] if col in hdrs else None
            key = "NLP Average" if "NLP Average" in hdrs else ("Text Average" if "Text Average" in hdrs else hdrs[3])
            srt = sorted(rows, key=lambda r: -(gv(r, key) or 0))
            print(f"\nSTRICT-SMALL by {key}:")
            for row in srt[:10]:
                nm = str(gv(row, "Model")); nm = nm.split(">")[1].split("<")[0] if "<" in nm else nm
                cols = {c: gv(row, c) for c in ("Overall Average", "NLP Average", "BLiMP", "Entity Tracking", "GlobalPIQA", "(Super)GLUE") if c in hdrs}
                print("  ", nm, "|", cols)
        break
