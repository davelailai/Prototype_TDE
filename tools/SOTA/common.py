"""Small, explicit numerical helpers; no test-driven model selection."""
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np


def asymmetric_loss(logits, targets, gamma_neg=4, gamma_pos=0, clip=.05):
    pos = logits.sigmoid()
    neg = (1 - pos + clip).clamp(max=1)
    log_likelihood = targets * pos.clamp_min(1e-8).log() + (1-targets) * neg.clamp_min(1e-8).log()
    with torch.no_grad():
        weight = (1-pos*targets-neg*(1-targets)).pow(gamma_pos*targets+gamma_neg*(1-targets))
    # Upstream PAT/ProCI use a sum, not a class/batch mean.
    return -(weight * log_likelihood).sum()


def patch_logits(attention, values, batch_size):
    if attention.shape != values.shape or attention.shape[0] % batch_size:
        raise ValueError('Invalid patch-major logits')
    attention = torch.stack(attention.split(batch_size), dim=1)
    values = torch.stack(values.split(batch_size), dim=1)
    return (attention.softmax(dim=1) * values).sum(dim=1)


def ramp(epoch, start, end):
    return min(1., max(0., (epoch-start) / max(1, end-start)))


def metrics(target, score):
    auc = [float(roc_auc_score(target[:, k], score[:, k])) if len(np.unique(target[:, k])) == 2 else None
           for k in range(target.shape[1])]
    ap = [float(average_precision_score(target[:, k], score[:, k])) if target[:, k].sum() else None
          for k in range(target.shape[1])]
    if all(value is None for value in auc):
        raise ValueError('AUC is undefined for every class in this split')
    p = score >= .5
    tp = (p*target).sum(0)
    precision = tp / np.maximum(p.sum(0), 1)
    recall = tp / np.maximum(target.sum(0), 1)
    f1 = 2*precision*recall / np.maximum(precision+recall, 1e-12)
    return dict(mAUC=float(np.mean([v for v in auc if v is not None])),
                mAP=float(np.mean([v for v in ap if v is not None])),
                MAP=float(precision.mean()), MAR=float(recall.mean()), MAF1=float(f1.mean()),
                per_class_AUC=auc, per_class_AP=ap, threshold=.5)


def paired_ce(logits, target):
    return F.binary_cross_entropy_with_logits(logits[:, :-1]-logits[:, -1:], target[:, :-1])


def append_normal(label):
    return torch.cat([label, (label.sum(-1, keepdim=True) == 0).to(label.dtype)], dim=-1)
