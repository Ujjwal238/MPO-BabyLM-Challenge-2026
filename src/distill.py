#!/usr/bin/env python
"""Self-distillation from an ensemble of same-budget teachers (BabyLlama-2, Tastet & Timiryasov 2024).

Teachers are our OWN 33M GPT-BERT models trained on the 10M budget (Phase-1 / span / Muon) -> fully
compliant (no external model; the CfP distillation ban applies only to external-list teachers). The
student minimises  alpha * hard-label LM loss  +  (1-alpha) * KL(student || temperature-softened
ensemble mean),  mode-matched (teachers run with the same is_causal flag on the same masked inputs).
Ensemble soft targets carry 'dark knowledge' that lets the student exceed any single teacher.

Teachers load in fp32 + no-grad (33M each -> ~0.5 GB for all 4 models; fp16 load breaks this
custom model's tied-weight assignment and fp16 attention is unreliable on MPS).
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM


def load_teachers(paths, device):
    teachers = []
    for p in paths:
        m = AutoModelForMaskedLM.from_pretrained(p, trust_remote_code=True)
        m = m.to(device).eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        teachers.append(m)
    return teachers


@torch.no_grad()
def ensemble_logits(teachers, input_ids, is_causal):
    """Mean logits of the teacher ensemble, each run in the same mode as the student batch."""
    acc = None
    for m in teachers:
        m.transformer.is_causal = is_causal
        lg = m(input_ids=input_ids, attention_mask=None).logits.float()
        acc = lg if acc is None else acc + lg
    return acc / len(teachers)


def kd_loss(student_logits, ens_logits, target_ids, temp):
    """Temperature-scaled KL at the predicted positions (masked positions for MLM, all for causal)."""
    mask = target_ids != -100
    if not mask.any():
        return student_logits.new_zeros(())
    s = student_logits[mask].float()
    t = ens_logits[mask]
    return F.kl_div(F.log_softmax(s / temp, dim=-1), F.softmax(t / temp, dim=-1),
                    reduction="batchmean") * (temp * temp)
