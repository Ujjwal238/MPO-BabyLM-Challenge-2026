<div align="center">

# Spend the Wasted Epochs

### Preference post-training from the saturation checkpoint for sample-efficient language modeling

**[Ujjwal Mishra](https://scholar.google.co.in/citations?user=Ggw7z6sAAAAJ&hl=en)**

Indian Institute of Information Technology Una

Accepted at the BabyLM Challenge 2026 &nbsp;·&nbsp; [paper](paper/mpo_babylm2026.pdf) &nbsp;·&nbsp; preprint link to follow

</div>

---

## Overview

The BabyLM Strict-Small track allows ten million words of text, traversed at most ten
times. Entries usually respond by adding training signal: a new objective, optimizer,
curriculum, or teacher. This repository contains the code, logs, and checkpoints for a
different response, which is to ask when the existing budget stops paying.

A 33M-parameter GPT-BERT hybrid trained under the track rules **saturates on downstream
evaluation at roughly 70M words seen**, seven of the ten permitted epochs, while training
loss keeps falling. The last three epochs buy a lower loss and no further competence. From
that diagnosis the paper derives three principles, timing, preservation and alignment, and
instantiates them as **MPO** (Minimal-Pair Preference Optimization): a 22-minute
preference phase, begun at the saturation checkpoint, that teaches the model to prefer real
corpus sentences over length-preserving corruptions of them under a frozen-reference penalty.

|                                     | NLP average | words seen |
| ----------------------------------- | ----------- | ---------- |
| GPT-2 (official BabyLM 2026 baseline) | 48.99     | 100M       |
| Interaction (official BabyLM 2026 baseline) | 49.04 | —        |
| this work, 10-epoch baseline        | 51.61       | 99M        |
| **this work, + MPO**                | **51.91**   | **71.3M**  |

MPO improves five of the seven official task columns, concedes under a point on the other
two, and does it at 7.2 rather than 10 epochs. On the last leaderboard pull before the
submission deadline (2026-07-21, 07:55 UTC, 116 strict-small entries) the two entries
placed fourth and eighth.

Everything ran on one laptop: an Apple M1 Pro with 16 GB, on MPS, with no CUDA and no cloud.

### Credit where it is due

Three of the four things this work is built from are not ours, and the paper's argument
depends on that being clear.

The **backbone** is GPT-BERT, by Lucas Georges Gabriel Charpentier and David Samuel, which
won the 2024 Strict-Small track and is the official 2026 baseline architecture. Their
modeling code is vendored under `src/gpt_bert/` with one bug fix marked inline.

The **objective** is DPO, by Rafael Rafailov and colleagues, used unchanged. What MPO
replaces is everything around it: the pairs are a corpus sentence and a synthetic
corruption of it rather than human judgments, the target is grammatical competence rather
than helpfulness, and the phase is placed by measurement at the saturation checkpoint
rather than appended to a finished model.

The **evaluation** is the organizers' 2026 pipeline. No number in the paper's main table
comes from our own harness; both entries are server-scored collations over the complete
suite. Our patch to it is 53 lines and changes no scoring (`patches/`).

The paper also reports that **seven published interventions fail to improve on the baseline
here**. That is a claim about transfer to a 33M-parameter model on a 10M-word budget, and
it is not a claim that those methods do not work. Several of them were validated at other
scales, on other backbones, in other evaluation pipelines, by their authors, and the honest
reading of our Table 2 is that results at this scale are sensitive to interactions among
architecture, recipe, and metric. Re-validation under one's own budget is a prerequisite,
which is exactly the burden we took on ourselves.

## Result

| quantity | value | source |
| --- | --- | --- |
| saturation point `w*` | 70M words seen | `logs/pretraining/eval_chck_*_fast.log` |
| fast BLiMP at `w*`, and at 100M | 70.13, 70.01 | same |
| masked training loss at 40M, and at 98M | 2.71, 2.19 | `logs/pretraining/baseline_train.log` |
| `w*` stable for every tolerance below | ε = 1.20 points | paper Appendix C |
| cross-seed deviation of the same measure | ± 0.07 | `results/mpo_ablations.md` |
| MPO phase cost | 1.277M word-touches, 21 m 57 s | `logs/mpo/mpo_submitted_train.log` |
| total exposure of the submitted model | 71.3M of the 100M cap | paper Appendix D |
| pretraining wall clock, for comparison | 12 h 26 m 45 s | `logs/pretraining/baseline_train.log` |
| NLP average, baseline then + MPO | 51.61 → 51.91 | `results/submissions/*.json.gz` |
| BLiMP supplement, baseline then + MPO | 63.77 → 64.84 | same |
| filler-gap repair (`wh_vs_that_with_gap`) | 71.5 → 75.5 | `logs/mpo/canary_sweep.log` |
| in-training interventions measured, and improvements found | 7, and 0 | `logs/interventions/` |

The two numbers that carry the argument are in the first and last rows. `w*` is stable
across a range of its free parameter seventeen times wider than the measurement noise, so
nothing about the choice of checkpoint is a judgment call. And the empty region of the
design space is what licenses the method: the principles predict that in-training injection
of unanchored signal cannot help here, and as far as this search reaches, it does not.

## Method

Three ingredients.

**The saturation diagnosis.** Writing `A(w)` for downstream accuracy after `w` words seen,
the saturation point is the earliest milestone that no later milestone beats by more than a
tolerance:

```
w* = min { w : max_{w' > w} A(w') − A(w) ≤ ε }
```

This is cheap. It reuses checkpoints a training run is already saving and a fast evaluation
subset, and it is decisive about where post-training signal belongs. On this trajectory
`w* = 70M` for every ε below 1.20 fast-BLiMP points.

**Length-preserving corruptions.** A real sentence `x⁺` becomes `x⁻ = T(x⁺)` under one of
four edits, each operating in place on the tokenized sentence so that `|T(x)| = |x|`: an
adjacent-token swap, a distant-token swap, a function-word substitution, and a random
content-token replacement. On *the cat sat on the mat* these give *the cat on sat the mat*,
*the mat sat on the cat*, *the cat sat of the mat*, and *the cat lamp on the mat*. Because
lengths match, the summed causal log-likelihoods are directly comparable with no
normalization to tune, which mirrors the minimal-pair design of BLiMP itself. The
operators are in `src/contrastive.py` and are shared with the failed contrastive
intervention, which is the point: the same four corruptions cost seven BLiMP points as an
in-training auxiliary loss and gain as an anchored post-training phase.

**An anchor that is part of the objective.** The reference is a frozen copy of the
checkpoint the policy starts from, and `β` is the coefficient on the KL penalty toward it,
so preservation is enforced by the loss rather than left to the learning rate. A retained
hybrid micro-batch is mixed in with probability 0.5 to hold the pretraining signal. This is
what selects DPO among pairwise objectives, and the paper's clearest failure is its nearest
anchor-free relative.

## Reproducing

```bash
conda env create -f environment.yml && conda activate mpo
bash scripts/setup_eval.sh              # clone + pin + patch the official pipeline
export BABYLM_EVAL=../babylm-eval/strict
```

Every script resolves paths from its own location, so no configuration is needed beyond
`BABYLM_EVAL`. Override the repository root with `MPO_ROOT` and the interpreter with
`MPO_PYTHON` if you want to.

**Minutes.** Enough to see the headline result without training anything.

```bash
python src/fetch_checkpoints.py --what headline    # the two models in Table 1
bash scripts/dpo_canary.sh checkpoints/hub/mpo_submitted_dpo1250
```

Expect fast BLiMP 70.24 and supplement 65.60, against the baseline's 70.01 and 65.20. This
is deterministic; it reproduced exactly on a rerun months after the original.

**Half an hour.** The method itself, from the published saturation checkpoint.

```bash
python src/fetch_checkpoints.py --what init
python src/train_dpo.py --init checkpoints/hub/baseline_chck_70M --run_name mpo_repro
bash scripts/dpo_canary.sh checkpoints/mpo_repro/chck_dpo_1250
```

**A few hours.** The ablations behind Tables 3 and 7.

```bash
bash scripts/run_dpo_ablations.sh    # seeds, initialization, hard negatives  (~2 h)
bash scripts/run_op_ablations.sh     # each corruption operator alone         (~2 h)
```

**Half a day each.** The baseline and any of the seven interventions. All of them are
flag-gated in one training script, so the baseline is the same code path with every flag
off.

```bash
python src/train_tokenizer.py && python src/prepare_data.py    # needs the official corpus
python src/train.py --config small --max_epochs 10 --run_name baseline    # 12 h 27 m

python src/train.py --contrastive     --run_name contrastive
python src/train.py --span_masking    --run_name span
python src/train.py --muon            --run_name muon
python src/train.py --adaptive_mask   --run_name adaptive
python src/train.py --slm             --run_name selective_lm   # needs precompute_ref_losses.py
```

The 10M-word corpus and the evaluation data are the organizers' distributions and are not
redistributed here. `scripts/setup_eval.sh` prints the download commands for both.

## Validation

Every number in the paper traces to a log in this repository, and the logs are the reason
the repository is worth having rather than just the code.

- **The trajectory reproduces from primary logs.** Fast BLiMP at 1M, 10M, 30M, 50M, 70M and
  100M words reads 54.21, 59.61, 64.67, 68.74, 70.13, 70.01 in `logs/pretraining/`, which is
  what `paper/figures/fig_saturation.py` plots and what the macros in `paper/results.tex`
  record. There is one source for each number and no number is written twice.
- **Determinism.** A shared canary log was once truncated by a later sweep, which put the
  provenance of one table row in doubt. Rather than trust it, the canary was rerun on the
  submitted checkpoint and reproduced 70.24 and 65.60 exactly, along with the whole fragile
  cluster. The rerun is in `logs/mpo/canary_sweep.log`. Per-run log files, since.
- **Robustness is separated from single measurement.** The claims the paper rests on held
  across three seeds (fast BLiMP 70.22 ± 0.07, and the filler-gap repair in every
  saturation-initialized run). The full-suite gains on entity tracking, GlobalPIQA and GLUE
  come from the single submitted run, and both the paper and `results/mpo_ablations.md` say
  so rather than letting the stronger reading stand.
- **The proxy and the evaluation dissociate, three independent times.** The contrastive loss
  reached near zero as BLiMP fell seven points; Muon fit the masked objective harder than
  AdamW and transferred worse; and within MPO itself, raising preference accuracy from 0.85
  to 0.90 lowered the evaluation. The per-operator ablation is the cleanest form of it, with
  preference accuracy flat at 0.85 to 0.86 across all four operators while transfer varies
  by 0.7 BLiMP points and the filler-gap repair by 6.5.
- **The result is not the training signal leaking into the metric.** The benchmark nearest
  the corruptions moves least, with full BLiMP gaining 0.02, while the gains appear where
  the corruptions never reach: dialogue pragmatics, entity tracking, commonsense, and
  finetuned GLUE. A model exploiting the format would show the opposite.
- **Scoring is not ours.** Both submitted prediction files are included, gzipped, in
  `results/submissions/`. They are the artifacts the organizers' server scored.

## Limits

The study is one architecture, one English corpus, one 10M-word budget. Whether the
diagnosis and MPO carry to other backbones, larger budgets, or other languages is untested.
The full-suite evaluation used one seed, though the two claims the paper rests on held
across three.

Within the result, the repair is partial. MPO improves the two filler-gap columns and is
alone in doing so, but distractor agreement moves within sampling noise and the
long-distance variant remains below chance. The cluster is improved, not solved. World
knowledge is the standing weakness: EWoK and GlobalPIQA are where the leaders pull ahead,
and the paper's own diagnosis says why refinement cannot close it, since world knowledge was
never acquired during pretraining in the first place. Nothing there for refinement to refine.

## Layout

```
src/          training, the MPO phase, the seven interventions, checkpoint fetching
scripts/      evaluation orchestration, ablation drivers, Hub and leaderboard submission
patches/      the 53-line diff against the official evaluation pipeline
logs/         every run referenced above, 69 files
results/      ablation reports and the two server-scored submission files
paper/        LaTeX source, figure scripts, and the compiled paper
artifacts/    the byte-level BPE tokenizer, fit on the training split alone
```

On the path to the result:

| file | role |
| --- | --- |
| `src/train.py` | the hybrid pretraining loop; every intervention is a flag here, all off reproduces the baseline |
| `src/train_dpo.py` | the MPO phase: frozen reference, retained-MLE mix, budget accounting |
| `src/contrastive.py` | the four corruption operators, shared with the failed contrastive arm |
| `src/dataset.py` | packed-stream chunking, and token, span, and adaptive masking |
| `src/model_config.py` | model shapes and the self-contained `trust_remote_code` checkpoint writer |
| `src/prepare_data.py`, `src/train_tokenizer.py` | corpus to token stream, and the 16,384-entry BPE tokenizer |
| `src/prepare_contrastive_data.py` | the 834,593 sentence units that preference pairs are drawn from |
| `src/fetch_checkpoints.py` | pulls any published checkpoint by name from the Hub |
| `scripts/setup_eval.sh` | clones, pins and patches the official evaluation pipeline |
| `scripts/dpo_canary.sh` | the fast BLiMP and supplement A/B used to gate every decision |
| `scripts/run_dpo_ablations.sh`, `scripts/run_op_ablations.sh` | the ablation drivers behind Tables 3 and 7 |

Supporting, not on the path to the result: `src/muon.py`, `src/distill.py`,
`src/precompute_ref_losses.py` and `src/train_morph_tokenizer.py`,
`src/train_uni_tokenizer.py` implement four of the seven interventions;
`src/bench.py` is the throughput benchmark that chose the batch and precision
configuration; `scripts/push_to_hf.py`, `scripts/push_dpo.py`,
`scripts/submit_entry.py` and `scripts/overnight_aoa_fix.py` are the release and
leaderboard machinery, kept because they document how the submitted artifacts were
built.

## Checkpoints

All checkpoints are public and each is a self-contained `trust_remote_code` model
directory. They are not committed here: one set of fp32 weights is 132 MB against
GitHub's 100 MB per-file limit, and they compress by about five percent.

| repository | `main` | branches |
| --- | --- | --- |
| [`Ujjwal101/Ujjwal-bored`](https://huggingface.co/Ujjwal101/Ujjwal-bored) | the 10-epoch baseline, 99M words | `chck_1M` … `chck_100M`, the 19-point saturation trajectory |
| [`Ujjwal101/ujjwal-very-bored`](https://huggingface.co/Ujjwal101/ujjwal-very-bored) | the submitted MPO model, `chck_dpo_1250` | `chck_1M` … `chck_70M` shared pretraining lineage, then `chck_dpo_250` … `chck_dpo_1500` |

In the second repository, `chck_80M`, `chck_90M` and `chck_100M` repeat the final model.
That run stopped maximum-likelihood training at 70M words and never trained past it, so
those milestones carry no additional exposure; they exist because the submission format
asks for them, and reporting them as real training would misstate the budget.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

path = "Ujjwal101/ujjwal-very-bored"
model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(path)
```

## Citation

```bibtex
@inproceedings{mishra2026mpo,
  title     = {Spend the Wasted Epochs: Preference Post-Training from the
               Saturation Checkpoint for Sample-Efficient Language Modeling},
  author    = {Mishra, Ujjwal},
  booktitle = {Proceedings of the BabyLM Challenge},
  year      = {2026}
}
```

## Contact

[ujjwalmishra238@gmail.com](mailto:ujjwalmishra238@gmail.com)
