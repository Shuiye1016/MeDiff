import torch
import torch.nn.functional as F
import skfuzzy as fuzz
import numpy as np
import config
import traceback

def fuzzy_cmeans_clustering(embeddings_tensor, n_clusters=config.NUM_CLUSTERS, m=config.FCM_M, error=0.005, maxiter=300, seed=42):
    if embeddings_tensor.shape[0] == 0:
        emb_dim = embeddings_tensor.shape[1] if embeddings_tensor.dim() > 1 else config.INITIAL_EMBEDDING_DIM
        dummy_centers = torch.zeros((n_clusters, emb_dim), dtype=torch.float32, device=config.DEVICE)
        dummy_membership = torch.zeros((0, n_clusters), dtype=torch.float32, device=config.DEVICE)
        return dummy_membership, dummy_centers

    embeddings_np = embeddings_tensor.detach().cpu().numpy().T

    if embeddings_np.shape[1] == 0:
        emb_dim = embeddings_tensor.shape[1]
        dummy_centers = torch.zeros((n_clusters, emb_dim), dtype=torch.float32, device=config.DEVICE)
        dummy_membership = torch.zeros((0, n_clusters), dtype=torch.float32, device=config.DEVICE)
        return dummy_membership, dummy_centers

    num_samples = embeddings_np.shape[1]
    emb_dim = embeddings_np.shape[0]

    if num_samples < n_clusters:
         # print(f"Warning: Number of samples ({num_samples}) is less than n_clusters ({n_clusters}). Using fallback clustering.")
         u = np.ones((n_clusters, num_samples)) / n_clusters
         if num_samples > 0:
             cntr = embeddings_np.mean(axis=1, keepdims=True)
             cntr = np.repeat(cntr, n_clusters, axis=1)
             cntr = cntr.T
         else:
             cntr = np.zeros((n_clusters, emb_dim))
         fpc = 0.0
    else:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            embeddings_np, n_clusters, m, error=error, maxiter=maxiter, init=None, seed=seed
        )
        if np.isnan(u).any() or np.isnan(cntr).any():
                # print("Warning: NaN detected in FCM results. Using fallback.")
                raise ValueError("NaN detected in FCM results")

    membership_matrix = torch.tensor(u.T, dtype=torch.float32).to(embeddings_tensor.device)
    cluster_centers = torch.tensor(cntr, dtype=torch.float32).to(embeddings_tensor.device)

    membership_sum = torch.sum(membership_matrix, dim=1, keepdim=True)
    membership_matrix = membership_matrix / torch.clamp(membership_sum, min=1e-8)

    return membership_matrix, cluster_centers

def get_group_assignments_and_counts(membership_matrix, n_clusters=config.NUM_CLUSTERS):
    if membership_matrix.shape[0] == 0:
         return torch.tensor([], dtype=torch.long, device=membership_matrix.device), \
                torch.zeros(n_clusters, dtype=torch.long, device=membership_matrix.device)

    hard_assignments = torch.argmax(membership_matrix, dim=1)
    group_counts = torch.bincount(hard_assignments, minlength=n_clusters).long()
    return hard_assignments, group_counts

def get_cascade_group_reps_and_counts(user_embeddings, n_clusters=config.NUM_CLUSTERS):
     if user_embeddings.shape[0] == 0:
          emb_dim = user_embeddings.shape[1] if user_embeddings.dim() > 1 else config.INITIAL_EMBEDDING_DIM
          return torch.tensor([], dtype=torch.long, device=user_embeddings.device), \
                 torch.zeros((0, n_clusters), dtype=torch.float32, device=user_embeddings.device), \
                 torch.zeros(n_clusters, dtype=torch.long, device=user_embeddings.device), \
                 torch.zeros((n_clusters, emb_dim), dtype=torch.float32, device=user_embeddings.device)

     memberships, centers = fuzzy_cmeans_clustering(user_embeddings, n_clusters)
     assignments, counts = get_group_assignments_and_counts(memberships, n_clusters)
     group_reps = centers

     return assignments, memberships, counts, group_reps


def get_soft_group_labels(membership_matrix, top_k=1):
    if membership_matrix.shape[0] == 0:
         return torch.tensor([], dtype=torch.long, device=membership_matrix.device), \
                torch.tensor([], dtype=torch.float32, device=membership_matrix.device)

    actual_n_clusters = membership_matrix.shape[1]
    if actual_n_clusters == 0:
         return torch.tensor([], dtype=torch.long, device=membership_matrix.device), \
                torch.tensor([], dtype=torch.float32, device=membership_matrix.device)

    k = min(top_k, config.NUM_CLUSTERS, actual_n_clusters)
    if k <= 0:
         return torch.tensor([], dtype=torch.long, device=membership_matrix.device), \
                torch.tensor([], dtype=torch.float32, device=membership_matrix.device)

    top_k_values, top_k_indices = torch.topk(membership_matrix, k=k, dim=1)
    normalized_values = F.normalize(top_k_values, p=1, dim=1)

    return top_k_indices, normalized_values