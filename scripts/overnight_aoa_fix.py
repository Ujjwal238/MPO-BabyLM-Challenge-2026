#!/usr/bin/env python
"""Overnight autonomous pipeline: fix AoA (mntp->causal) and resubmit both entries.

Runs unattended after the Phase-1 causal-AoA job (already running) finishes:
  1. wait for causal AoA (results/Ujjwal-bored/main/zero_shot/causal/AoA_word/surprisal.json, 19 steps)
  2. local proxy: correlate model word-acquisition vs child AoA (cdi_human.csv) for mntp vs causal
  3. rebuild ujjwal-bored JSON with CAUSAL AoA (chck_1M..100M) + resubmit
  4. rebuild ujjwal-very-bored JSON with CAUSAL AoA restricted to chck_1M..70M (DPO's real
     pretraining trajectory) + resubmit
  5. write OVERNIGHT_REPORT.md

Every stage is wrapped so one failure can't abort the rest. All actions logged.
"""
import json, os, shutil, subprocess, sys, time, uuid
from pathlib import Path
import requests
from huggingface_hub import get_token

REPO = Path(os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STRICT = Path(os.environ.get("BABYLM_EVAL") or sys.exit(
    "set BABYLM_EVAL to your patched babylm-eval/strict checkout"))
RES = STRICT / "results"
BASE = "https://babylm-community-babylm-leaderboard-2026.hf.space"
TOKEN = get_token()
H = {"Authorization": f"Bearer {TOKEN}"}
REPORT = REPO / "OVERNIGHT_REPORT.md"
CAUSAL_AOA = RES / "Ujjwal-bored" / "main" / "zero_shot" / "causal" / "AoA_word" / "surprisal.json"
PRETRAIN_70 = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i}M" for i in (10, 20, 30, 40, 50, 60, 70)]

log_lines = []
def rep(msg):
    line = f"{time.strftime('%F %T')} | {msg}"
    print(line, flush=True)
    log_lines.append(line)
    REPORT.write_text("# BabyLM overnight AoA-fix report\n\n```\n" + "\n".join(log_lines) + "\n```\n")

# ---------- local AoA proxy (directional; not the exact server metric) ----------
def human_aoa():
    import csv
    out = {}
    with open(STRICT / "evaluation_data/full_eval/aoa/cdi_human.csv") as f:
        r = csv.reader(f); hdr = next(r)
        months = [int(h) for h in hdr[2:]]
        for row in r:
            w = row[1]; props = [float(x) for x in row[2:]]
            aoa = None
            for m, p in zip(months, props):
                if p >= 0.5: aoa = m; break
            out[w] = aoa if aoa is not None else months[-1] + 1
    return out

def proxy_corr(surprisal_json):
    """Spearman(model acq-step, human AoA). Model acq-step = word_count where a word's
    surprisal first drops below its (max+min)/2 midpoint across the trajectory."""
    try:
        import numpy as np
        from scipy.stats import spearmanr
    except Exception:
        return None
    d = json.load(open(surprisal_json)); res = d["results"]
    by_word = {}
    for r in res:
        by_word.setdefault(r["target_word"], []).append((r["word_count"], r["surprisal"]))
    hum = human_aoa()
    mx, hy = [], []
    for w, seq in by_word.items():
        if w not in hum: continue
        seq = sorted(seq)
        wc = [a for a, _ in seq]; su = [b for _, b in seq]
        if len(su) < 3: continue
        mid = (max(su) + min(su)) / 2.0
        acq = next((wc[i] for i in range(len(su)) if su[i] <= mid), wc[-1])
        mx.append(acq); hy.append(hum[w])
    if len(mx) < 10: return None
    rho, _ = spearmanr(mx, hy)
    return round(float(rho), 4), len(mx)

# ---------- collate + submit helpers ----------
def collate(stem):
    r = subprocess.run([sys.executable, "-m", "evaluation_pipeline.collate_preds",
                        "--model_path_or_name", stem, "--backend", "mntp", "--fast",
                        "--track", "strict-small"], cwd=STRICT, capture_output=True, text=True)
    return (RES / stem / "all_full_preds_and_fast_scores_mntp.json").exists()

def submit(data39, json_path, label):
    with open(json_path, "rb") as f:
        up = requests.post(f"{BASE}/gradio_api/upload", headers=H,
                           files={"files": (os.path.basename(json_path), f, "application/json")})
    up.raise_for_status()
    data39 = list(data39)
    data39[4] = {"path": up.json()[0], "meta": {"_type": "gradio.FileData"},
                 "orig_name": os.path.basename(json_path)}
    sess = str(uuid.uuid4())
    j = requests.post(f"{BASE}/gradio_api/queue/join", headers=H,
                      json={"data": data39, "fn_index": 0, "session_hash": sess}); j.raise_for_status()
    r = requests.get(f"{BASE}/gradio_api/queue/data", headers=H,
                     params={"session_hash": sess}, stream=True, timeout=300)
    status, our = "?", None
    for line in r.iter_lines():
        if not line: continue
        s = line.decode("utf-8", "replace")
        if s.startswith("data:"):
            try: msg = json.loads(s[5:].strip())
            except: continue
            if msg.get("msg") == "process_completed":
                out = msg.get("output", {}).get("data", [])
                if out and isinstance(out[0], str):
                    status = out[0].replace("<p style='color: green; font-size: 20px; text-align: center;'>","").replace("<p style='color: red; font-size: 20px; text-align: center;'>","").replace("</p>","")
                if len(out) > 1 and isinstance(out[1], dict):
                    hdrs = out[1].get("headers", []); rows = out[1].get("data", [])
                    for row in rows:
                        if any(label in str(x).lower() for x in row):
                            our = dict(zip(hdrs, row))
                break
    rep(f"  [{label}] STATUS: {status}")
    if our:
        keep = {k: our.get(k) for k in ("Model","Text Average","BLiMP","AoA","(Super)GLUE") if k in our}
        rep(f"  [{label}] our row (from response): {keep}")
    else:
        rep(f"  [{label}] our row not in returned table (default view is 'strict'; check strict-small filter)")
    return status

# common form values
DESC_DPO = ("33M GPT-BERT hybrid (masked+causal) pretrained ~7 epochs on BabyLM Strict-Small 10M, "
            "then minimal-pair DPO post-training (prefer real sentences over length-preserving "
            "corruptions). AoA scored with the causal backend. mntp elsewhere.")
DESC_P1 = ("33M GPT-BERT hybrid (masked+causal) trained from scratch on BabyLM Strict-Small 10M for "
           "10 epochs. AoA scored with the causal backend; mntp elsewhere.")
SYNTH_DPO = ("DPO negatives are length-preserving corruptions (token swaps, function-word subs) of real "
             "corpus sentences; no external model, no new words.")

def form(model, repo, epochs, gpu_train, flops, desc, synth, contribs):
    # index-aligned to the 39 params; index 4 (results file) filled in submit()
    return [model, "main", repo, "strict-small", None, None,
            "Hybrid masked+causal", contribs, "GPT-BERT", "cosine", epochs,
            "Byte-Level BPE", "42", 6, 128, 900, "BabyLM strict-small", 0, "",
            0.0005, "AdamW", 16384, 16384, 12, 33046084, flops, gpu_train,
            "Not applicable", "", synth, desc, None, "", False, False, False, 0, 0, 0]

# ============================ RUN ============================
rep("=== overnight AoA-fix orchestrator START ===")

# 1) wait for causal AoA
rep("waiting for Phase-1 causal AoA to finish...")
for _ in range(360):  # up to ~6h
    if CAUSAL_AOA.exists():
        try:
            n = len({r["step"] for r in json.load(open(CAUSAL_AOA))["results"]})
            if n >= 19: rep(f"causal AoA ready ({n} steps)"); break
        except Exception: pass
    time.sleep(60)
else:
    rep("TIMEOUT waiting for causal AoA; aborting."); sys.exit(1)

# 2) local proxy validation
try:
    mntp_aoa = RES / "Ujjwal-bored" / "main" / "zero_shot" / "mntp" / "AoA_word" / "surprisal.json"
    pc = proxy_corr(CAUSAL_AOA); pm = proxy_corr(mntp_aoa) if mntp_aoa.exists() else None
    rep(f"local proxy Spearman(acq-step, child-AoA): causal={pc} mntp={pm}  (directional; higher |rho| better)")
except Exception as e:
    rep(f"proxy validation skipped: {e}")

# 3) ujjwal-bored with causal AoA (full chck_1M..100M)
try:
    mntp_aoa_dir = RES / "Ujjwal-bored" / "main" / "zero_shot" / "mntp" / "AoA_word"
    shutil.copy(CAUSAL_AOA, mntp_aoa_dir / "surprisal.json")  # feed causal AoA into the mntp collation
    if collate("Ujjwal-bored"):
        rep("re-collated ujjwal-bored with causal AoA")
        submit(form("ujjwal-bored", "https://huggingface.co/Ujjwal101/Ujjwal-bored",
                    10, 12.5, 3.8e16, DESC_P1, "Not applicable", []),
               RES / "Ujjwal-bored" / "all_full_preds_and_fast_scores_mntp.json", "ujjwal-bored")
    else:
        rep("ERROR: ujjwal-bored collate produced no JSON")
except Exception as e:
    rep(f"ERROR ujjwal-bored stage: {e}")

# 4) ujjwal-very-bored with causal AoA restricted to chck_1M..70M (DPO real trajectory)
try:
    full = json.load(open(CAUSAL_AOA))
    sub = {"metadata": dict(full.get("metadata", {})),
           "results": [r for r in full["results"] if r["step"] in set(PRETRAIN_70)]}
    sub["metadata"]["model_name"] = "Ujjwal101/ujjwal-very-bored"
    sub["metadata"]["total_steps"] = len(PRETRAIN_70)
    sub["metadata"]["completed_steps"] = len(PRETRAIN_70)
    dpo_aoa_dir = RES / "chck_dpo_1250" / "main" / "zero_shot" / "mntp" / "AoA_word"
    dpo_aoa_dir.mkdir(parents=True, exist_ok=True)
    json.dump(sub, open(dpo_aoa_dir / "surprisal.json", "w"))
    rep(f"assembled DPO causal AoA ({len(PRETRAIN_70)} steps, chck_1M..70M)")
    if collate("chck_dpo_1250"):
        rep("re-collated ujjwal-very-bored with causal AoA")
        submit(form("ujjwal-very-bored", "https://huggingface.co/Ujjwal101/ujjwal-very-bored",
                    7.2, 9, 2.7e16, DESC_DPO, SYNTH_DPO,
                    ["Training objective innovations", "Preference optimization (DPO) post-training"]),
               RES / "chck_dpo_1250" / "all_full_preds_and_fast_scores_mntp.json", "ujjwal-very-bored")
    else:
        rep("ERROR: chck_dpo_1250 collate produced no JSON")
except Exception as e:
    rep(f"ERROR ujjwal-very-bored stage: {e}")

rep("=== DONE. Check the leaderboard strict-small filter for updated AoA / Text Average. ===")
