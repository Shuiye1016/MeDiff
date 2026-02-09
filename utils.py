import torch
import numpy as np
import config
import math

def calculate_metrics(scores, targets, top_k_list):
    metrics = {}
    if scores.numel() == 0 or targets.numel() == 0 or scores.shape[0] != targets.shape[0]:
        for k in top_k_list:
            for metric in ['Recall', 'NDCG']:
                metrics[f'{metric}@{k}'] = 0.0
        return metrics

    batch_size, num_users = scores.shape
    k_max = min(max(top_k_list), num_users)
    _, top_indices = torch.topk(scores, k=k_max, dim=1)

    for k in top_k_list:
        if k <= 0 or k > num_users: continue
        top_k = top_indices[:, :k]
        relevant_mask = torch.gather(targets.to(scores.device), 1, top_k).float()
        # Recall@K
        num_hits = relevant_mask.sum(dim=1)
        num_relevant = targets.sum(dim=1).clamp(min=1e-8)
        recall = num_hits / num_relevant
        metrics[f'Recall@{k}'] = recall.mean().item()
        # NDCG@K
        ranks = torch.arange(1, k + 1, device=scores.device).float()
        discounts = torch.log2(ranks + 1)
        dcg = (relevant_mask / discounts).sum(dim=1)
        ideal_sorted_targets, _ = torch.sort(targets, dim=1, descending=True)
        ideal_top_k = ideal_sorted_targets[:, :k]
        idcg = (ideal_top_k.float() / discounts).sum(dim=1).clamp(min=1e-8)
        ndcg = dcg / idcg
        ndcg[idcg < 1e-8] = 0.0
        metrics[f'NDCG@{k}'] = ndcg.mean().item()
    return metrics

def compute_early_stopping_score(val_metrics):
    msle = val_metrics.get("MSLE_Size", None)
    recall = val_metrics.get("Recall@500", None)
    acc = val_metrics.get("GroupAcc", None)
    msle_score = -msle
    recall_score = recall
    acc_score = acc
    score = (msle_score + recall_score + acc_score) / 3.0
    return score
