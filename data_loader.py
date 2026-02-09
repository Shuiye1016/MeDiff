# data_loader.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import config
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Any, Tuple

from clustering import get_cascade_group_reps_and_counts
from embeddings import CombinedEmbedder

def load_cascades(cascade_dir: str) -> Dict[str, pd.DataFrame]:
    all_cascades = {}
    if not os.path.isdir(cascade_dir):
        # print(f"Error: Cascade directory not found at {cascade_dir}")
        return {}
    file_limit = float('inf') 
    loaded_count = 0
    for filename in os.listdir(cascade_dir):
        if loaded_count >= file_limit: break
        if filename.endswith(".csv"):
            cascade_id = filename.split('.')[0]
            file_path = os.path.join(cascade_dir, filename)
            try:
                df = pd.read_csv(file_path, dtype={'origin_uid': str, 'uid': str, 'origin_id':str, 'id':str}, keep_default_na=False, na_values=['NULL', '', 'None'])
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                df.dropna(subset=['created_at', 'uid', 'id'], inplace=True) 
                if df.empty: continue
                df = df.sort_values(by='created_at').reset_index(drop=True)
                df['origin_id'] = df['origin_id'].fillna('ROOT')
                df['origin_uid'] = df['origin_uid'].fillna('ROOT')
                df['content'] = df['content'].fillna('')
                df['uid'] = df['uid'].astype(str) 
                all_cascades[cascade_id] = df
                loaded_count += 1
            except Exception as e:
                print(f"Error loading cascade file {filename}: {e}")
                continue
    return all_cascades

def load_user_profiles(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    scalers = {}
    if not os.path.isfile(filepath):
        print(f"Error: User profile file not found at {filepath}")
        return pd.DataFrame(index=pd.Index([], name='用户ID')), scalers

    profiles_df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
    if '用户ID' not in profiles_df.columns:
        print("Error: '用户ID' column not found in profile file.")
        return pd.DataFrame(index=pd.Index([], name='用户ID')), scalers
    
    profiles_df['用户ID'] = profiles_df['用户ID'].astype(str) 
    profiles_df.set_index('用户ID', inplace=True)

    numerical_features_to_standardize = config.SELECTED_PROFILE_FEATURES.get('numerical', [])
    for feature in numerical_features_to_standardize:
        actual_col_name = '微博数' if feature == '发微博数量' else feature 
        if actual_col_name in profiles_df.columns:
            temp_series = profiles_df[actual_col_name].copy()
            if temp_series.dtype == 'object': 
                temp_series = temp_series.astype(str).str.replace(',', '', regex=False)
                is_wan = temp_series.str.contains('万', na=False)
                temp_series.loc[is_wan] = pd.to_numeric(temp_series.loc[is_wan].str.replace('万', '', regex=False), errors='coerce') * 10000
            
            profiles_df[actual_col_name] = pd.to_numeric(temp_series, errors='coerce')
            mean = profiles_df[actual_col_name].mean()
            std = profiles_df[actual_col_name].std()
            scalers[actual_col_name] = {'mean': mean, 'std': std} 
            profiles_df[actual_col_name] = profiles_df[actual_col_name].fillna(mean)
            if pd.notna(std) and std > 1e-8:
                profiles_df[actual_col_name] = (profiles_df[actual_col_name] - mean) / std
            else: 
                profiles_df[actual_col_name] = 0.0
    return profiles_df, scalers

def load_and_preprocess_data() -> Dict[str, Any]:
    all_cascades = load_cascades(config.CASCADE_DIR)
    profiles_df, scalers = load_user_profiles(config.USER_PROFILE_FILE)
    all_users_from_cascades = set()
    for df_val in all_cascades.values(): 
        all_users_from_cascades.update(df_val['uid'].astype(str).unique())
        all_users_from_cascades.update(df_val['origin_uid'].astype(str).unique())
    all_users_from_cascades.discard('ROOT') 
    all_users_from_profiles = set(profiles_df.index.astype(str)) if not profiles_df.empty else set()
    all_users = sorted(list(all_users_from_cascades.union(all_users_from_profiles)))
    user_to_idx, idx_to_user, num_users, all_user_indices = {}, {}, 0, torch.empty(0, dtype=torch.long)
    if all_users:
        user_to_idx = {uid: i for i, uid in enumerate(all_users)}
        idx_to_user = {i: uid for uid, i in user_to_idx.items()}
        num_users = len(all_users)
        all_user_indices = torch.arange(num_users)
    processed_cascades_for_dataset = []
    for cid, df_cascade in all_cascades.items():
        if len(df_cascade) < config.MIN_CASCADE_LENGTH: continue
        if not any(str(uid) in user_to_idx for uid in df_cascade['uid'].unique()): continue
        processed_cascades_for_dataset.append({'cid': cid, 'dataframe': df_cascade})
    train_data, val_data, test_data = [], [], []
    if processed_cascades_for_dataset:
        train_data, test_val_data = train_test_split(processed_cascades_for_dataset, test_size=0.3, random_state=42, shuffle=True)
        if test_val_data: 
            val_data, test_data = ([], []) if not test_val_data else \
                              (test_val_data, []) if len(test_val_data) == 1 else \
                              train_test_split(test_val_data, test_size=1/3, random_state=42, shuffle=True)
    return {
        "train": train_data, "val": val_data, "test": test_data,
        "num_users": num_users, "profiles_df": profiles_df, "scalers": scalers,
        "user_to_idx": user_to_idx, "idx_to_user": idx_to_user, "all_user_indices": all_user_indices,
    }

class CascadeDataset(Dataset):
    def __init__(self, processed_data: List[Dict[str, Any]], 
                 user_to_idx: Dict[str, int], 
                 combined_embedder_instance: CombinedEmbedder 
                ):
        self.data = processed_data
        self.global_user_to_idx = user_to_idx
        self.num_users = len(user_to_idx)
        self.combined_embedder = combined_embedder_instance

    def get_preliminary_features_for_split(self, 
                                           uids_for_Z: List[str], 
                                           contents_for_Z: List[str], 
                                           timestamps_for_Z: pd.Series, 
                                           first_timestamp_for_Z: pd.Timestamp
                                          ) -> torch.Tensor:
        if not uids_for_Z or self.combined_embedder is None:
            return torch.zeros((len(uids_for_Z), config.INITIAL_EMBEDDING_DIM), device=config.DEVICE)
        with torch.no_grad(): 
            if not isinstance(contents_for_Z, list): contents_for_Z = list(contents_for_Z)
            if not isinstance(timestamps_for_Z, pd.Series): timestamps_for_Z = pd.Series(timestamps_for_Z)
            features_tensor_Z = self.combined_embedder(contents_for_Z, uids_for_Z, timestamps_for_Z, first_timestamp_for_Z)
            if features_tensor_Z is None or features_tensor_Z.shape[0] != len(uids_for_Z) or \
               features_tensor_Z.shape[1] != config.INITIAL_EMBEDDING_DIM:
                return torch.zeros((len(uids_for_Z), config.INITIAL_EMBEDDING_DIM), device=config.DEVICE)
            return features_tensor_Z

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item = self.data[idx]
        df_cascade_full = item['dataframe'] 
        
        uids_for_Z_generation = [str(uid) for uid in df_cascade_full['uid'].tolist() if str(uid) in self.global_user_to_idx]
        if not uids_for_Z_generation: return None 

        df_for_Z_features = df_cascade_full[df_cascade_full['uid'].astype(str).isin(uids_for_Z_generation)]
        if df_for_Z_features.empty: return None

        contents_for_Z_generation = df_for_Z_features['content'].tolist()
        timestamps_for_Z_generation = df_for_Z_features['created_at'] 
        first_timestamp_overall = df_cascade_full['created_at'].iloc[0] if not df_cascade_full.empty else pd.Timestamp.now(tz=None)

        preliminary_features_Z = self.get_preliminary_features_for_split(
            uids_for_Z_generation, contents_for_Z_generation, 
            timestamps_for_Z_generation, first_timestamp_overall
        )
        
        assignments = None
        if preliminary_features_Z.numel() > 0 and preliminary_features_Z.shape[0] > 0 and preliminary_features_Z.shape[1] > 0:
            try:
                assignments_tensor, _, _, _ = get_cascade_group_reps_and_counts(
                    preliminary_features_Z.cpu(), config.NUM_CLUSTERS 
                )
                if assignments_tensor.numel() > 0: assignments = assignments_tensor.numpy()
            except Exception: assignments = None 
        
        split_index = -1
        if assignments is not None and len(assignments) == len(uids_for_Z_generation):
            uid_to_assignment = {uid: assign for uid, assign in zip(uids_for_Z_generation, assignments)}
            last_user_uid_in_Z_list = uids_for_Z_generation[-1] 
            if last_user_uid_in_Z_list in uid_to_assignment:
                last_group_assignment = uid_to_assignment[last_user_uid_in_Z_list]
                first_occurrence_original_idx = -1
                for original_df_idx, row_data in df_cascade_full.iterrows():
                    current_uid_str = str(row_data['uid'])
                    if current_uid_str in uid_to_assignment and uid_to_assignment[current_uid_str] == last_group_assignment:
                        first_occurrence_original_idx = original_df_idx; break
                if first_occurrence_original_idx != -1: split_index = first_occurrence_original_idx 
        
        input_df, target_df = None, None
        if len(df_cascade_full) > 0 : 
            if split_index != -1 and split_index > 0 and split_index < len(df_cascade_full) : 
                input_df = df_cascade_full.iloc[:split_index] 
                target_df = df_cascade_full.iloc[split_index:]
                if input_df.empty: split_index = -1 
            if split_index == -1: 
                horizon_size = max(1, int(len(df_cascade_full) * config.PREDICTION_HORIZON_RATIO))
                input_len = max(1, len(df_cascade_full) - horizon_size)
                input_df = df_cascade_full.iloc[:input_len]
                target_df = df_cascade_full.iloc[input_len:]
        
        if input_df is None or input_df.empty: return None
        if target_df is None: target_df = pd.DataFrame(columns=df_cascade_full.columns) 

        local_user_list, global_idx_map, local_user_to_idx = [], {}, {}
        local_node_idx_counter = 0
        potential_graph_uids = pd.concat([input_df['uid'], input_df['origin_uid']]).astype(str).unique()
        for uid_str in potential_graph_uids:
            if uid_str != 'ROOT' and uid_str in self.global_user_to_idx:
                if uid_str not in local_user_to_idx: 
                    local_idx = local_node_idx_counter
                    global_idx_map[local_idx] = self.global_user_to_idx[uid_str]
                    local_user_to_idx[uid_str] = local_idx
                    local_user_list.append(uid_str); local_node_idx_counter += 1
        
        num_local_nodes = len(local_user_list)
        if num_local_nodes == 0: return None 

        source_indices, target_indices = [], []
        for _, row in input_df.iterrows():
            src_uid, tgt_uid = str(row['origin_uid']), str(row['uid'])
            if src_uid != 'ROOT' and src_uid in local_user_to_idx and tgt_uid in local_user_to_idx:
                src_idx, tgt_idx = local_user_to_idx[src_uid], local_user_to_idx[tgt_uid]
                if src_idx != tgt_idx: source_indices.append(src_idx); target_indices.append(tgt_idx)
        if num_local_nodes > 1 and not source_indices: 
            for i in range(num_local_nodes): source_indices.append(i); target_indices.append(i)
        local_edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
        
        input_uids_for_Z_final = [str(uid) for uid in input_df['uid'].tolist() if str(uid) in self.global_user_to_idx]
        if not input_uids_for_Z_final: return None
        
        aligned_input_df_for_Z = input_df[input_df['uid'].astype(str).isin(input_uids_for_Z_final)]
        if aligned_input_df_for_Z.empty: return None
            
        target_uids_global = [str(uid) for uid in target_df['uid'].tolist() if str(uid) in self.global_user_to_idx]
        target_indices_global = [self.global_user_to_idx[uid] for uid in target_uids_global]
        target_user_vec = torch.zeros(self.num_users, dtype=torch.float32)
        if target_indices_global: target_user_vec[target_indices_global] = 1.0
        
        first_ts_for_Z_final = aligned_input_df_for_Z['created_at'].iloc[0] if not aligned_input_df_for_Z.empty else first_timestamp_overall

        return {
            'input_uids_global': input_uids_for_Z_final, 
            'timestamps': aligned_input_df_for_Z['created_at'], 
            'content': aligned_input_df_for_Z['content'].tolist(), 
            'first_timestamp': first_ts_for_Z_final, 
            'local_edge_index': local_edge_index, 
            'num_local_nodes': num_local_nodes, 
            'global_user_indices': torch.tensor([global_idx_map[i] for i in range(num_local_nodes)], dtype=torch.long), 
            'target_user_vector': target_user_vec,
            'target_size': torch.tensor(len(target_indices_global), dtype=torch.float32),
            'target_indices_global': torch.tensor(target_indices_global, dtype=torch.long) 
        }
    
    def __len__(self) -> int:
        return len(self.data)

def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    batch = [item for item in batch if item is not None] 
    if not batch: return None
    first_item_keys = batch[0].keys()
    collated_batch = {key: [d.get(key) for d in batch] for key in first_item_keys}

    tensor_keys_to_stack = ['target_user_vector', 'target_size']
    for key in tensor_keys_to_stack:
        if key in collated_batch:
            items_to_stack = [item for item in collated_batch[key] if isinstance(item, torch.Tensor)]
            if items_to_stack:
                 try:
                    collated_batch[key] = torch.stack(items_to_stack)
                 except RuntimeError as e:
                    print(f"Collate_fn: Error stacking key '{key}': {e}. Item shapes: {[t.shape for t in items_to_stack]}. Keeping as list.")
    return collated_batch
