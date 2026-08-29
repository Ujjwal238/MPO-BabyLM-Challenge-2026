# Per-operator corruption ablation (2026-07-19 17:56:23)

Reference: mixed uniform (v1) chck_dpo_1250 = 70.24 / 65.60 | baseline 70.01 / 65.20 | chck_70M 70.13 / 65.60

2026-07-19 17:56:23 | === TRAIN gptbert_small_dpo_op_adj (weights 1,0,0,0) ===
2026-07-19 18:25:14 | RESULT gptbert_small_dpo_op_adj : 2026-07-19 18:24:58 | RESULT blimp_fast AVG=69.85 2026-07-19 18:25:14 | RESULT supplement_fast AVG=64.00  (pref-acc 0.863)
2026-07-19 18:25:14 | === TRAIN gptbert_small_dpo_op_dist (weights 0,1,0,0) ===
2026-07-19 18:54:50 | RESULT gptbert_small_dpo_op_dist : 2026-07-19 18:54:32 | RESULT blimp_fast AVG=70.08 2026-07-19 18:54:50 | RESULT supplement_fast AVG=64.80  (pref-acc 0.850)
2026-07-19 18:54:50 | === TRAIN gptbert_small_dpo_op_func (weights 0,0,1,0) ===
2026-07-19 19:24:16 | RESULT gptbert_small_dpo_op_func : 2026-07-19 19:23:59 | RESULT blimp_fast AVG=70.54 2026-07-19 19:24:16 | RESULT supplement_fast AVG=65.20  (pref-acc 0.863)
2026-07-19 19:24:16 | === TRAIN gptbert_small_dpo_op_rand (weights 0,0,0,1) ===
2026-07-19 19:53:27 | RESULT gptbert_small_dpo_op_rand : 2026-07-19 19:53:10 | RESULT blimp_fast AVG=69.83 2026-07-19 19:53:27 | RESULT supplement_fast AVG=64.00  (pref-acc 0.856)
2026-07-19 19:53:27 | === ALL OPERATOR ABLATIONS COMPLETE ===

## Synthesis (2026-07-19)

| operator alone | BLiMP | supp | wh gap | long-d | pref-acc |
|---|---|---|---|---|---|
| adjacent swap | 69.85 | 64.00 | 79.0 | 37.5 | 0.863 |
| distant swap | 70.08 | 64.80 | 79.0 | 37.5 | 0.850 |
| function-word sub | **70.54** | 65.20 | 75.0 | 35.5 | 0.863 |
| random replace | 69.83 | 64.00 | 72.5 | 32.5 | 0.856 |
| mixed (submitted) | 70.24 | **65.60** | 75.5 | 36.5 | 0.85 |
| no phase | 70.01 | 65.20 | 71.5 | 32.5 | – |

Three regularities: (1) aggregate transfer tracks proximity to BLiMP's contrast type (func-sub best alone, rand worst); (2) filler-gap repair tracks order perturbation (swaps 79.0 >> rand 72.5) — operators genuinely complementary; (3) pref-acc identical (0.85-0.86) across all -> differences are transfer, not learnability (proxy/eval dissociation, third instance). No single op dominates all columns; uniform mixture kept (untuned, never worst). In paper: Appendix E (tab:ops) + pointer in method 3.3.
