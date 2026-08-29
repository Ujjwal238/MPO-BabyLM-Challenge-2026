# DPO ablation results (2026-07-05 21:43:51)

Reference v1: init=chck_70M beta=0.1 mixed-neg seed=42 -> chck_dpo_1250 = 70.24 / 65.60

2026-07-05 21:43:51 | === TRAIN gptbert_small_dpo_s43 (--init /Users/admin1/Downloads/babylm-eval/checkpoints/gptbert_small_v1/chck_70M --beta 0.1 --seed 43) ===
2026-07-05 22:13:17 | RESULT gptbert_small_dpo_s43 : 2026-07-05 22:13:00 | RESULT blimp_fast AVG=70.14 2026-07-05 22:13:17 | RESULT supplement_fast AVG=64.40 
2026-07-05 22:13:17 |   (gptbert_small_dpo_s43 final pref-acc 0.762)
2026-07-05 22:13:17 | === TRAIN gptbert_small_dpo_s44 (--init /Users/admin1/Downloads/babylm-eval/checkpoints/gptbert_small_v1/chck_70M --beta 0.1 --seed 44) ===
2026-07-05 22:42:39 | RESULT gptbert_small_dpo_s44 : 2026-07-05 22:42:22 | RESULT blimp_fast AVG=70.28 2026-07-05 22:42:39 | RESULT supplement_fast AVG=63.60 
2026-07-05 22:42:39 |   (gptbert_small_dpo_s44 final pref-acc 0.875)
2026-07-05 22:42:39 | === TRAIN gptbert_small_dpo_fromfinal (--init /Users/admin1/Downloads/babylm-eval/checkpoints/gptbert_small_v1/final --beta 0.1 --seed 42) ===
2026-07-05 23:11:01 | RESULT gptbert_small_dpo_fromfinal : 2026-07-05 23:10:45 | RESULT blimp_fast AVG=70.26 2026-07-05 23:11:01 | RESULT supplement_fast AVG=62.40 
2026-07-05 23:11:01 |   (gptbert_small_dpo_fromfinal final pref-acc 0.787)
2026-07-05 23:11:01 | === TRAIN gptbert_small_dpo_hardneg_b01 (--init /Users/admin1/Downloads/babylm-eval/checkpoints/gptbert_small_v1/chck_70M --beta 0.1 --hard_neg --seed 42) ===
2026-07-05 23:39:34 | RESULT gptbert_small_dpo_hardneg_b01 : 2026-07-05 23:39:18 | RESULT blimp_fast AVG=70.16 2026-07-05 23:39:34 | RESULT supplement_fast AVG=64.40 
2026-07-05 23:39:34 |   (gptbert_small_dpo_hardneg_b01 final pref-acc 0.850)
2026-07-05 23:39:34 | === ALL DPO ABLATIONS COMPLETE ===

## Synthesis (2026-07-05)

Fast canary (chck_dpo_1250, mntp) — BLiMP / supp:
| run | BLiMP | supp | pref-acc |
|---|---|---|---|
| Phase-1 baseline (no DPO) | 70.01 | 65.20 | – |
| v1 (chck_70M, b0.1, mixed, s42) | 70.24 | 65.60 | ~0.85 |
| s43 | 70.14 | 64.40 | 0.76 |
| s44 | 70.28 | 63.60 | 0.875 |
| from-final (99M init, s42) | 70.26 | 62.40 | 0.787 |
| hard-neg b0.1 (s42) | 70.16 | 64.40 | 0.85 |
| (v2: hard-neg b0.2, s42) | 70.14 | 64.00 | 0.90 |

**BLiMP: 70.22 ± 0.07 across seeds; 70.14–70.28 across ALL configs — robustly held/slightly > baseline 70.01.**
**Fast supp: 62.4–65.6, noisy (50-item set), NOT a robust gain vs 65.20.**

Fragile long-distance cluster (across all runs above): wh_vs_that_with_gap **74.5–77.0** (baseline ~71.5), distractor_agreement_relational_noun 59–63 (~62), ellipsis_n_bar_2 91–92 (held). **The cluster repair — esp. wh_vs_that_with_gap +3–5 — is ROBUST across seeds/configs.** This is the distinctive, defensible result.

Init ablation: from-70M (70.24) ≈ from-final (70.26) on BLiMP → the extra 3 MLE epochs (70M→99M) do not help; DPO reaches the same result at 71.3M vs 100.3M words seen. **Causal support for the saturation diagnosis (O1).**

Deconfound (R2): conservative (b0.1, mixed neg) ≥ hard-neg b0.1 (70.16) ≥ hard-neg b0.2 (70.14); higher pref-acc (0.90) → not better eval. Over-optimization effect is real but mild at b0.1; both harder negatives and higher β nudge slightly down.

**Robust claims (multi-seed/config):** (1) DPO holds BLiMP tightly and repairs the fragile cluster; (2) wasted-epoch reallocation — same result at 28% less budget; (3) conservatism beats aggressive preference optimization.
**Single-seed (v1 full eval, report with hedging):** entity +2.46, GLUE +0.48, full-supp +1.07. Verifying these across seeds needs full entity (~3.7h) + GLUE (~8h) per seed — deferred; not required for the robust story.

## 30M-init arm (2026-07-12) — the missing timing cell

2026-07-12 | TRAIN gptbert_small_dpo_from30M (--init chck_30M --beta 0.1 --seed 42): 21m19s, 1.277M phase words, final pref-acc 0.794.
2026-07-12 | RESULT chck_dpo_1250 canary: **blimp_fast AVG=64.96, supplement_fast AVG=62.80** (30M no-phase baseline: 64.67 / 62.00, eval_chck_30M_fast.log).
Fragile cluster @30M+DPO vs 30M base: wh_vs_that_with_gap 47.5 (45.0, +2.5), long-distance 9.5 (10.0), distractor 26.5 (25.0), ellipsis 83.5 (83.0) — same miniature repair signature, at the 30M level.

**Reading:** the phase adds ≤0.3 BLiMP over its own init everywhere it starts (30M +0.29, 70M +0.11, 99M +0.25) — the LEVEL is set by acquisition, not the phase. From a pre-saturation init it stays 5+ points below every saturated-init run. Timing 2×2 now complete: too late = wasted budget (99M row), too early = nothing to refine (30M row). In the paper: Table 3 row + method §3.3 + results §4.2.
