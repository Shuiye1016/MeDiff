# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from embeddings import CombinedEmbedder 
from graph_utils import HypergraphConv, TORCH_GEOM_AVAILABLE 
from clustering import get_cascade_group_reps_and_counts 
from attention import GroupAttention 
import pandas as pd
import numpy as np

if not TORCH_GEOM_AVAILABLE:
    raise ImportError("torch_geometric is required for HypergraphConv in GRIDModel.")

try:
    from torchdiffeq import odeint_adjoint as odeint # 使用带伴随方法的odeint以节省内存
except ImportError:
    try:
        from torchdiffeq import odeint
    except ImportError:
        odeint = None
        print("WARNING: torchdiffeq library not found. ODE functionality will not be available.")
        print("Please install it using: pip install torchdiffeq")


class ODENet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        layers = [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(config.ODE_DROPOUT_RATE)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(config.ODE_DROPOUT_RATE)])
        self.net = nn.Sequential(*layers)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh()
        )

    def forward(self, t, z):
        if t.ndim == 0:
            t = t.unsqueeze(0).unsqueeze(1)
        elif t.ndim == 1:
            t = t.unsqueeze(1)

        t_embed = self.time_mlp(t)
        z_with_time = z + t_embed
        return self.net(z_with_time)

def calculate_memberships(data_points_np, centers_np, m=config.FCM_M):
    n_new = data_points_np.shape[0]
    k = centers_np.shape[0]
    if n_new == 0: return np.zeros((0, k))
    if k == 0 or centers_np.ndim != 2 or data_points_np.ndim != 2 or centers_np.shape[1] == 0 or data_points_np.shape[1] == 0:
        return np.ones((n_new, k)) / k if k > 0 else np.zeros((n_new, k))
    if centers_np.shape[1] != data_points_np.shape[1]:
        return np.ones((n_new, k)) / k if k > 0 else np.zeros((n_new, k))
    dist_sq = np.sum((data_points_np[:, np.newaxis, :] - centers_np[np.newaxis, :, :])**2, axis=2)
    dist_sq = np.fmax(dist_sq, np.finfo(np.float64).eps)
    dist = np.sqrt(dist_sq)
    exp = 2. / (m - 1)
    if abs(m - 1.0) < 1e-9: 
        memberships = np.zeros_like(dist)
        if n_new > 0: 
            nearest_center_indices = np.argmin(dist, axis=1)
            memberships[np.arange(n_new), nearest_center_indices] = 1.0
        return memberships
    dist = np.fmax(dist, np.finfo(np.float64).eps) 
    d_raised = dist ** (-exp)
    sum_d_raised = np.sum(d_raised, axis=1, keepdims=True)
    sum_d_raised = np.fmax(sum_d_raised, np.finfo(np.float64).eps) 
    memberships = d_raised / sum_d_raised
    nan_mask = np.isnan(memberships) 
    if np.any(nan_mask):
        memberships[nan_mask] = 1.0 / k if k > 0 else 0.0
    row_sums = memberships.sum(axis=1, keepdims=True) 
    memberships = memberships / np.fmax(row_sums, np.finfo(np.float64).eps)
    return memberships


class GRIDModel(nn.Module):
    def __init__(self, num_users, profile_df, user_to_idx, idx_to_user, all_user_indices):
        super().__init__()
        self.num_users = num_users
        self.n_clusters = config.NUM_CLUSTERS 
        self.user_to_idx = user_to_idx
        self.idx_to_user = idx_to_user
        
        self.initial_feature_dim = config.INITIAL_EMBEDDING_DIM 

        self.global_user_embedding = nn.Embedding(num_users, self.initial_feature_dim)
        nn.init.xavier_uniform_(self.global_user_embedding.weight)

        hgnn_input_dim = self.initial_feature_dim
        self.hgnn_output_dim = config.GNN_EMBEDDING_DIM 
        
        self.hgc_layers = nn.ModuleList()
        if config.GNN_LAYERS > 0 : 
            for _ in range(config.GNN_LAYERS):
                self.hgc_layers.append(HypergraphConv(hgnn_input_dim, self.hgnn_output_dim, use_attention=False))
                hgnn_input_dim = self.hgnn_output_dim 
        
        self.bn_hgnn = nn.BatchNorm1d(self.hgnn_output_dim) if config.GNN_LAYERS > 0 else nn.Identity()
        self.dropout_hgnn = nn.Dropout(p=config.DROPOUT_RATE) if config.GNN_LAYERS > 0 else nn.Identity()

        self.ode_func = None
        self.use_ode = odeint is not None
        if self.use_ode:
            ode_input_output_dim = self.hgnn_output_dim if config.GNN_LAYERS > 0 else self.initial_feature_dim
            if ode_input_output_dim > 0 :
                 self.ode_func = ODENet(ode_input_output_dim)
            else:
                 self.use_ode = False
                 print("GRIDModel Warning: ODE input/output dimension is 0. Disabling ODE.")
        
        self.cascade_rep_aggregation_type = config.CASCADE_REP_AGGREGATION
        aggregation_input_dim = self.hgnn_output_dim if config.GNN_LAYERS > 0 else self.initial_feature_dim

        self.group_attention_aggregator = None 
        if self.cascade_rep_aggregation_type == "group_attention":
            if config.GROUP_ATTENTION_DIM > 0 and config.ATTENTION_HEADS > 0 and \
               config.GROUP_ATTENTION_DIM % config.ATTENTION_HEADS == 0:
                self.group_attention_aggregator = GroupAttention(
                    embed_dim=aggregation_input_dim, 
                    num_heads=config.ATTENTION_HEADS,
                    dropout=config.DROPOUT_RATE
                )
                self.current_cascade_rep_dim = aggregation_input_dim
            else:
                print(f"GRIDModel Warning: Invalid GroupAttention config. Falling back to mean_pool.")
                self.cascade_rep_aggregation_type = "mean_pool" 
                self.current_cascade_rep_dim = aggregation_input_dim
        elif self.cascade_rep_aggregation_type == "mean_pool":
            self.current_cascade_rep_dim = aggregation_input_dim
        else:
            raise ValueError(f"Unsupported CASCADE_REP_AGGREGATION: {self.cascade_rep_aggregation_type}")

        self.final_cascade_rep_projection = nn.Identity()
        if self.current_cascade_rep_dim != config.CASCADE_MODEL_DIM and self.current_cascade_rep_dim > 0 and config.CASCADE_MODEL_DIM > 0:
            self.final_cascade_rep_projection = nn.Linear(self.current_cascade_rep_dim, config.CASCADE_MODEL_DIM)
        elif self.current_cascade_rep_dim == 0 and config.CASCADE_MODEL_DIM > 0: 
             print(f"GRIDModel Warning: current_cascade_rep_dim is 0, but CASCADE_MODEL_DIM is {config.CASCADE_MODEL_DIM}.")

        self.cascade_representation_final_dim = config.CASCADE_MODEL_DIM

        self.fallback_proj_z_to_cascade_model_dim = nn.Linear(self.initial_feature_dim, self.cascade_representation_final_dim) \
                                                    if self.initial_feature_dim > 0 and self.cascade_representation_final_dim > 0 \
                                                    else nn.Identity()

        self.group_predictor = nn.Linear(self.cascade_representation_final_dim, self.n_clusters) \
                               if self.cascade_representation_final_dim > 0 and self.n_clusters > 0 \
                               else nn.Identity() 
        self.size_predictor = nn.Sequential(
            nn.Linear(self.cascade_representation_final_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        ) if self.cascade_representation_final_dim > 0 and self.cascade_representation_final_dim !=0 else nn.Identity()
        
        self.user_score_proj = nn.Identity()
        if self.cascade_representation_final_dim != self.global_user_embedding.embedding_dim and \
           self.cascade_representation_final_dim > 0 and self.global_user_embedding.embedding_dim > 0:
             self.user_score_proj = nn.Linear(self.cascade_representation_final_dim, self.global_user_embedding.embedding_dim)


    def forward(self, batch, combined_embedder_instance): 
        global_user_indices_list = batch['global_user_indices']
        input_uids_global_list = batch['input_uids_global']
        timestamps_list = batch['timestamps']
        content_list = batch['content']
        first_timestamp_list = batch['first_timestamp']
        target_indices_global_list = batch['target_indices_global'] 

        batch_size = len(global_user_indices_list)
        batch_user_scores_list, batch_group_logits_list, batch_size_predictions_list = [], [], []
        batch_target_group_dist_list, valid_sample_original_indices = [], []

        for i in range(batch_size):
            num_local_nodes = batch['num_local_nodes'][i] 
            if num_local_nodes == 0: continue

            current_content = content_list[i]
            current_input_uids = input_uids_global_list[i] 
            current_timestamps = timestamps_list[i]
            current_first_timestamp = first_timestamp_list[i]
            
            local_node_features_z = combined_embedder_instance(
                current_content, current_input_uids, current_timestamps, current_first_timestamp
            )
            
            if local_node_features_z.shape[0] == 0 or local_node_features_z.shape[0] != len(current_input_uids) :
                continue

            cascade_rep = None
            cluster_centers_for_target_dist = None 
            use_fallback_path = (local_node_features_z.shape[0] < self.n_clusters)

            if use_fallback_path:
                pooled_z = torch.mean(local_node_features_z, dim=0) 
                cascade_rep = self.fallback_proj_z_to_cascade_model_dim(pooled_z) \
                              if not isinstance(self.fallback_proj_z_to_cascade_model_dim, nn.Identity) \
                              else pooled_z
            else:
                assignments_tensor, _, group_counts_input_nodes, initial_group_centers_z = get_cascade_group_reps_and_counts(
                    local_node_features_z, self.n_clusters
                ) 
                cluster_centers_for_target_dist = initial_group_centers_z.detach().cpu().numpy()
                
                nodes_for_hgnn = local_node_features_z 

                if config.GNN_LAYERS == 0:
                    group_representations_list = []
                    for k_idx in range(self.n_clusters):
                        member_nodes_mask = (assignments_tensor == k_idx)
                        member_node_Z_features = nodes_for_hgnn[member_nodes_mask]
                        if member_node_Z_features.shape[0] > 0:
                            group_representations_list.append(torch.mean(member_node_Z_features, dim=0))
                        else:
                            group_representations_list.append(torch.zeros(self.initial_feature_dim, device=config.DEVICE))
                    group_representations = torch.stack(group_representations_list)
                else:
                    local_node_indices_h, hyperedge_indices_h = [], []
                    for node_idx_local, group_idx_assigned in enumerate(assignments_tensor): 
                        local_node_indices_h.append(node_idx_local)
                        hyperedge_indices_h.append(group_idx_assigned.item())
                    
                    if not local_node_indices_h: 
                        pooled_z = torch.mean(nodes_for_hgnn, dim=0)
                        cascade_rep = self.fallback_proj_z_to_cascade_model_dim(pooled_z) \
                                      if not isinstance(self.fallback_proj_z_to_cascade_model_dim, nn.Identity) \
                                      else pooled_z
                        cluster_centers_for_target_dist = None
                        if cascade_rep is None: cascade_rep = torch.zeros(self.cascade_representation_final_dim, device=config.DEVICE)
                    else:
                        hyperedge_index = torch.tensor([local_node_indices_h, hyperedge_indices_h], dtype=torch.long).to(config.DEVICE)
                        refined_node_embeddings = nodes_for_hgnn 
                        for hgc_layer in self.hgc_layers:
                            refined_node_embeddings = hgc_layer(refined_node_embeddings, hyperedge_index)
                        if refined_node_embeddings.shape[0] > 1: refined_node_embeddings = self.bn_hgnn(refined_node_embeddings)
                        refined_node_embeddings = F.relu(refined_node_embeddings)
                        refined_node_embeddings = self.dropout_hgnn(refined_node_embeddings)

                        group_reps_list, actual_group_counts_for_agg = [], []
                        for k_idx in range(self.n_clusters):
                            member_nodes_mask = (assignments_tensor == k_idx)
                            member_node_embeddings = refined_node_embeddings[member_nodes_mask]
                            if member_node_embeddings.shape[0] > 0:
                                group_reps_list.append(torch.mean(member_node_embeddings, dim=0))
                                actual_group_counts_for_agg.append(member_node_embeddings.shape[0])
                            else:
                                group_reps_list.append(torch.zeros(self.hgnn_output_dim, device=config.DEVICE))
                                actual_group_counts_for_agg.append(0)
                        group_representations = torch.stack(group_reps_list) 

                evolved_group_representations = group_representations
                if self.use_ode and self.ode_func is not None and group_representations.numel() > 0:
                    c = group_representations.shape[0]
                    timestamps_float = torch.tensor([
                        (t_i - current_first_timestamp).total_seconds() / 60.0
                        for t_i in current_timestamps
                    ], dtype=torch.float32, device=config.DEVICE)

                    group_times = []
                    for k_idx in range(self.n_clusters):
                        member_nodes_mask = (assignments_tensor == k_idx)
                        if member_nodes_mask.sum() > 0:
                            avg_time = timestamps_float[member_nodes_mask].mean()
                        else:
                            avg_time = torch.tensor(0.0, device=config.DEVICE)
                        group_times.append(avg_time)

                    group_times = torch.stack(group_times)
                    group_times_norm = (group_times - group_times.min()) / (
                                group_times.max() - group_times.min() + 1e-6)
                    try:
                        sorted_times, indices = torch.sort(group_times_norm)
                        perturb = torch.arange(len(sorted_times), device=config.DEVICE) * 1e-5
                        sorted_times = sorted_times + perturb
                        sorted_group_reps = group_representations[indices]
                        ode_solution = odeint(self.ode_func, sorted_group_reps, sorted_times, method='rk4')
                        evolved_sorted = ode_solution.mean(dim=0)

                        recover_indices = torch.argsort(indices)
                        evolved_group_representations = evolved_sorted[recover_indices]
                    except Exception as e_ode:
                        print(f"GRIDModel: Error during ODE integration for sample {i}: {e_ode}. Using non-evolved group reps.")
                        evolved_group_representations = group_representations

                group_counts_tensor = torch.tensor(actual_group_counts_for_agg if 'actual_group_counts_for_agg' in locals() else group_counts_input_nodes.cpu().numpy(), dtype=torch.float32, device=config.DEVICE)
                group_representations_batched = evolved_group_representations.unsqueeze(0) 
                group_counts_batched = group_counts_tensor.unsqueeze(0)       

                cascade_rep_intermediate = torch.zeros(self.current_cascade_rep_dim, device=config.DEVICE) 
                if self.cascade_rep_aggregation_type == "group_attention" and self.group_attention_aggregator is not None:
                    if group_representations_batched.shape[1] > 0 : 
                        attended_group_reps = self.group_attention_aggregator(
                            query_reps=group_representations_batched, key_reps=group_representations_batched,   
                            value_reps=group_representations_batched, group_counts_for_keys=group_counts_batched )
                        if attended_group_reps.shape[1] > 0:
                            valid_groups_mask = (group_counts_batched.squeeze(0) > 1e-9)
                            if torch.any(valid_groups_mask):
                                 cascade_rep_intermediate = torch.mean(attended_group_reps.squeeze(0)[valid_groups_mask], dim=0)
                            elif attended_group_reps.numel() > 0 : 
                                 cascade_rep_intermediate = torch.mean(attended_group_reps.squeeze(0), dim=0)
                elif self.cascade_rep_aggregation_type == "mean_pool":
                    if group_representations_batched.shape[1] > 0: 
                        weights = group_counts_batched.squeeze(0).float() 
                        sum_weights = torch.sum(weights)
                        if sum_weights > 1e-9: 
                            weights = weights / sum_weights
                            cascade_rep_intermediate = torch.sum(group_representations_batched.squeeze(0) * weights.unsqueeze(1), dim=0)
                        elif group_representations_batched.numel() > 0: 
                            cascade_rep_intermediate = torch.mean(group_representations_batched.squeeze(0), dim=0)
                
                cascade_rep = self.final_cascade_rep_projection(cascade_rep_intermediate) \
                              if not isinstance(self.final_cascade_rep_projection, nn.Identity) \
                              else cascade_rep_intermediate
            
            current_target_dist = torch.ones(self.n_clusters, device=config.DEVICE) / max(1, self.n_clusters) 
            target_indices_global = target_indices_global_list[i].to(config.DEVICE)
            if target_indices_global.numel() > 0 and cluster_centers_for_target_dist is not None and \
               cluster_centers_for_target_dist.shape[0] == self.n_clusters and \
               cluster_centers_for_target_dist.shape[1] == self.initial_feature_dim: 
                target_user_initial_features = self.global_user_embedding(target_indices_global) 
                target_user_features_np = target_user_initial_features.detach().cpu().numpy()
                if target_user_features_np.shape[1] == cluster_centers_for_target_dist.shape[1]:
                    target_memberships_np = calculate_memberships(
                        target_user_features_np, cluster_centers_for_target_dist, m=config.FCM_M)
                    if target_memberships_np.shape[0] > 0: 
                        avg_target_membership = np.mean(target_memberships_np, axis=0)
                        target_dist_sum = np.sum(avg_target_membership)
                        if target_dist_sum > 1e-9: 
                            current_target_dist = torch.tensor(avg_target_membership / target_dist_sum, dtype=torch.float32, device=config.DEVICE)
                        elif self.n_clusters > 0: 
                             current_target_dist = torch.ones(self.n_clusters, device=config.DEVICE) / self.n_clusters
            
            if cascade_rep is None:
                 cascade_rep = torch.zeros(self.cascade_representation_final_dim, device=config.DEVICE)
            
            group_logits_output = self.group_predictor(cascade_rep) if not isinstance(self.group_predictor, nn.Identity) else torch.zeros(self.n_clusters, device=config.DEVICE)
            raw_size_out = self.size_predictor(cascade_rep).squeeze() if not isinstance(self.size_predictor, nn.Identity) else torch.tensor(0.0, device=config.DEVICE)
            raw_size_out = raw_size_out.float()
            size_prediction_output = F.softplus(raw_size_out)
            size_prediction_output = torch.nan_to_num(size_prediction_output, nan=0.0, posinf=1e6, neginf=0.0)
            batch_group_logits_list.append(group_logits_output)
            batch_size_predictions_list.append(size_prediction_output)
            batch_target_group_dist_list.append(current_target_dist)
            
            projected_cascade_rep = cascade_rep
            if not isinstance(self.user_score_proj, nn.Identity) and cascade_rep.numel() > 0:
                projected_cascade_rep = self.user_score_proj(cascade_rep)

            if projected_cascade_rep.ndim == 1 and self.global_user_embedding.weight.ndim == 2 and \
               projected_cascade_rep.shape[0] == self.global_user_embedding.weight.shape[1]:
                user_scores_for_sample = torch.matmul(
                    projected_cascade_rep.unsqueeze(0),
                    self.global_user_embedding.weight.t()
                ).squeeze(0)
                user_scores_for_sample = user_scores_for_sample / config.RETRIEVAL_TEMPERATURE
            else:
                user_scores_for_sample = torch.zeros(self.num_users, device=config.DEVICE) 
            
            batch_user_scores_list.append(user_scores_for_sample)
            valid_sample_original_indices.append(i) 

        if not valid_sample_original_indices: return None 

        final_user_scores = torch.stack(batch_user_scores_list) if batch_user_scores_list else torch.empty(0, self.num_users, device=config.DEVICE)
        final_group_logits = torch.stack(batch_group_logits_list) if batch_group_logits_list else torch.empty(0, self.n_clusters, device=config.DEVICE)
        final_size_predictions = torch.stack(batch_size_predictions_list) if batch_size_predictions_list else torch.empty(0, device=config.DEVICE)
        final_target_group_dists = torch.stack(batch_target_group_dist_list) if batch_target_group_dist_list else torch.empty(0, self.n_clusters, device=config.DEVICE)
        
        return {
            'user_scores': final_user_scores, 'group_logits': final_group_logits,
            'size_predictions': final_size_predictions, 'target_group_dist': final_target_group_dists,
            'valid_indices': valid_sample_original_indices }
