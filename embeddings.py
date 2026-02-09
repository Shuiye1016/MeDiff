# embeddings.py
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import config
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

if config.EMBEDDING_COMBINATION == 'hetero':
    try:
        from torch_geometric.nn import HeteroConv, Linear as PyGLinear, SAGEConv
        from torch_geometric.data import HeteroData
        PYG_AVAILABLE = True
    except ImportError:
        # print("WARNING: PyTorch Geometric (torch_geometric) not found. 'hetero' mode for CombinedEmbedder will not be available.")
        PYG_AVAILABLE = False
else:
    PYG_AVAILABLE = False

class TextEmbedder(nn.Module):
    def __init__(self, model_name=config.PRETRAINED_MODEL_NAME, output_dim=config.TEXT_EMBEDDING_DIM):
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Identity()
        self.model = None
        self.tokenizer = None
        if self.output_dim > 0:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                raw_embedding_dim = self.model.config.hidden_size
                if raw_embedding_dim != self.output_dim:
                    self.projection = nn.Linear(raw_embedding_dim, self.output_dim)
            except Exception as e:
                # print(f"Error loading {model_name} for TextEmbedder: {e}. TextEmbedder will output zeros.")
                self.model = None; self.tokenizer = None

    def forward(self, texts: List[str]) -> torch.Tensor:
        batch_size = len(texts) if isinstance(texts, list) else 0
        if not self.model or not self.tokenizer or self.output_dim == 0:
            return torch.zeros((batch_size, self.output_dim), device=config.DEVICE)
        current_texts = [str(t) if pd.notna(t) and t is not None else "" for t in texts]
        if not current_texts and batch_size > 0 :
            return torch.zeros((batch_size, self.output_dim), device=config.DEVICE)
        if not current_texts and batch_size == 0:
            return torch.zeros((0, self.output_dim), device=config.DEVICE)
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(current_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = self.model(**inputs)
        raw_embeddings = outputs.last_hidden_state.mean(dim=1)
        return self.projection(raw_embeddings).to(config.DEVICE)

class DateTimeFeatureEncoder(nn.Module):
    def __init__(self, output_dim=10, ref_year=2020):
        super().__init__()
        self.output_dim = output_dim
        self.ref_year = ref_year
        self.month_embed = nn.Embedding(13, 3, padding_idx=0)
        self.weekday_embed = nn.Embedding(8, 3, padding_idx=0)
        internal_dim = 1 + 1 + self.month_embed.embedding_dim + self.weekday_embed.embedding_dim
        self.projection = nn.Linear(internal_dim, output_dim) if internal_dim > 0 and output_dim > 0 else nn.Identity()

    def forward(self, datetime_str_series: pd.Series) -> torch.Tensor:
        batch_size = len(datetime_str_series)
        if batch_size == 0 or self.output_dim == 0:
            return torch.zeros((batch_size, self.output_dim), device=config.DEVICE)
        years_diff, months, days_of_month, weekdays = [], [], [], []
        for dt_str in datetime_str_series:
            dt_obj = pd.to_datetime(dt_str, errors='coerce')
            if pd.isna(dt_obj): years_diff.append(0.0); months.append(0); days_of_month.append(0.0); weekdays.append(0)
            else:
                years_diff.append(float(dt_obj.year - self.ref_year)); months.append(dt_obj.month)
                days_of_month.append(float(dt_obj.day)); weekdays.append(dt_obj.weekday() + 1)
        years_t = torch.tensor(years_diff, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        months_t = torch.tensor(months, dtype=torch.long).to(config.DEVICE)
        days_t = torch.tensor(days_of_month, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        weekdays_t = torch.tensor(weekdays, dtype=torch.long).to(config.DEVICE)
        month_e = self.month_embed(months_t); weekday_e = self.weekday_embed(weekdays_t)
        combined = torch.cat([years_t / 10.0, days_t / 31.0, month_e, weekday_e], dim=1)
        return self.projection(combined)

class TimeEmbedder(nn.Module):
    def __init__(self, output_dim=config.TIME_EMBEDDING_DIM):
        super().__init__()
        self.output_dim = output_dim
        self.inv_freq = None
        if output_dim > 0:
            internal_dim = output_dim + (output_dim % 2)
            self.inv_freq = 1.0 / (10000 ** (torch.arange(0, internal_dim, 2).float() / max(1e-6, internal_dim)))

    def forward(self, timestamps_series: pd.Series, first_timestamp_obj: pd.Timestamp) -> torch.Tensor:
        batch_size = len(timestamps_series)
        if self.output_dim <= 0 or self.inv_freq is None or batch_size == 0:
            return torch.zeros((batch_size, self.output_dim), device=config.DEVICE)
        ts_dt = pd.to_datetime(timestamps_series, errors='coerce')
        ref_dt = pd.to_datetime(first_timestamp_obj, errors='coerce')
        if pd.isna(ref_dt): ref_dt = ts_dt.dropna().iloc[0] if not ts_dt.dropna().empty else pd.Timestamp.now(tz=None)
        ts_dt = ts_dt.fillna(ref_dt)
        diff_sec = (ts_dt - ref_dt).dt.total_seconds().fillna(0)
        diff_t = torch.tensor(diff_sec.values, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
        inv_freq_dev = self.inv_freq.to(config.DEVICE)
        pos_a = torch.sin(diff_t * inv_freq_dev); pos_b = torch.cos(diff_t * inv_freq_dev)
        pos_enc = torch.cat([pos_a, pos_b], dim=-1)
        return pos_enc[:, :self.output_dim]

class ProfileEmbedder(nn.Module):
    def __init__(self, profile_df: pd.DataFrame, selected_features_config: dict, output_dim=config.PROFILE_EMBEDDING_DIM):
        super().__init__()
        self.profile_df = profile_df.copy() if profile_df is not None else pd.DataFrame()
        self.output_dim = output_dim
        self.feature_encoders = nn.ModuleDict()
        self.cat_feature_vocabs = {}
        self.internal_text_embedder = TextEmbedder(output_dim=config.TEXT_EMBEDDING_DIM) if config.TEXT_EMBEDDING_DIM > 0 and selected_features_config.get('textual', []) else None

        current_combined_dim_for_final_proj = 0
        self.cat_agg_proj, self.num_agg_proj, self.textual_profile_agg_proj, self.datetime_agg_proj = None, None, None, None

        self.categorical_feature_names = selected_features_config.get('categorical', [])
        cat_internal_total_dim = 0
        if self.categorical_feature_names and config.PROFILE_CAT_AGG_DIM > 0:
            cat_emb_dim_per_feat = config.PROFILE_CAT_AGG_DIM // len(self.categorical_feature_names) if len(self.categorical_feature_names) > 0 else config.PROFILE_CAT_AGG_DIM
            cat_emb_dim_per_feat = max(1, cat_emb_dim_per_feat)
            for fname in self.categorical_feature_names:
                if fname in self.profile_df.columns:
                    unique_vals = self.profile_df[fname].fillna("MISSING_CAT").astype(str).unique()
                    self.cat_feature_vocabs[fname] = {v: i + 1 for i, v in enumerate(unique_vals)}
                    emb = nn.Embedding(len(unique_vals) + 1, cat_emb_dim_per_feat, padding_idx=0)
                    self.feature_encoders[f'{fname}_cat_embed'] = emb
                    cat_internal_total_dim += emb.embedding_dim
        if cat_internal_total_dim > 0:
            self.cat_agg_proj = nn.Linear(cat_internal_total_dim, config.PROFILE_CAT_AGG_DIM)
            current_combined_dim_for_final_proj += config.PROFILE_CAT_AGG_DIM

        self.actual_numerical_feature_names = []
        for cfg_name in selected_features_config.get('numerical', []):
            actual_col = '微博数' if cfg_name == '发微博数量' else cfg_name
            if actual_col in self.profile_df.columns: self.actual_numerical_feature_names.append(actual_col)
        if self.actual_numerical_feature_names and config.PROFILE_NUM_AGG_DIM > 0:
            self.num_agg_proj = nn.Linear(len(self.actual_numerical_feature_names), config.PROFILE_NUM_AGG_DIM)
            current_combined_dim_for_final_proj += config.PROFILE_NUM_AGG_DIM

        self.textual_profile_feature_names = selected_features_config.get('textual', [])
        textual_profile_internal_total_dim = 0
        if self.internal_text_embedder and self.textual_profile_feature_names and config.PROFILE_TEXTUAL_AGG_DIM > 0:
            text_proj_dim_per_field = config.PROFILE_TEXTUAL_AGG_DIM // len(self.textual_profile_feature_names) if len(self.textual_profile_feature_names) > 0 else config.PROFILE_TEXTUAL_AGG_DIM
            text_proj_dim_per_field = max(1, text_proj_dim_per_field)
            for fname in self.textual_profile_feature_names:
                if fname in self.profile_df.columns:
                    self.feature_encoders[f'{fname}_text_proj'] = nn.Linear(config.TEXT_EMBEDDING_DIM, text_proj_dim_per_field)
                    textual_profile_internal_total_dim += text_proj_dim_per_field
        if textual_profile_internal_total_dim > 0:
            self.textual_profile_agg_proj = nn.Linear(textual_profile_internal_total_dim, config.PROFILE_TEXTUAL_AGG_DIM)
            current_combined_dim_for_final_proj += config.PROFILE_TEXTUAL_AGG_DIM

        self.datetime_feature_names = selected_features_config.get('datetime', [])
        datetime_internal_total_dim = 0
        if self.datetime_feature_names and config.PROFILE_DATETIME_AGG_DIM > 0:
            dt_enc_out_dim_per_field = config.PROFILE_DATETIME_AGG_DIM // len(self.datetime_feature_names) if len(self.datetime_feature_names) > 0 else config.PROFILE_DATETIME_AGG_DIM
            dt_enc_out_dim_per_field = max(1, dt_enc_out_dim_per_field)
            for fname in self.datetime_feature_names:
                if fname in self.profile_df.columns:
                    self.feature_encoders[f'{fname}_datetime_encode'] = DateTimeFeatureEncoder(output_dim=dt_enc_out_dim_per_field)
                    datetime_internal_total_dim += dt_enc_out_dim_per_field
        if datetime_internal_total_dim > 0:
            self.datetime_agg_proj = nn.Linear(datetime_internal_total_dim, config.PROFILE_DATETIME_AGG_DIM)
            current_combined_dim_for_final_proj += config.PROFILE_DATETIME_AGG_DIM

        self.final_proj = nn.Linear(current_combined_dim_for_final_proj, self.output_dim) if current_combined_dim_for_final_proj > 0 and self.output_dim > 0 else nn.Identity()
        self.activation = nn.ReLU()

    def _get_user_data_batch(self, uids_str_list: List[str]) -> pd.DataFrame:
        all_expected_cols = list(set(self.categorical_feature_names + self.actual_numerical_feature_names + self.textual_profile_feature_names + self.datetime_feature_names))
        default_row_data = {**{col: "MISSING_CAT" for col in self.categorical_feature_names}, **{col: 0.0 for col in self.actual_numerical_feature_names}, **{col: "" for col in self.textual_profile_feature_names}, **{col: None for col in self.datetime_feature_names}}
        batch_data_list = []
        for uid_str in uids_str_list:
            if uid_str in self.profile_df.index: batch_data_list.append(self.profile_df.loc[uid_str])
            else: series_data = {col: default_row_data.get(col) for col in all_expected_cols if col in default_row_data}; batch_data_list.append(pd.Series(series_data, name=uid_str))
        if not batch_data_list: return pd.DataFrame(columns=all_expected_cols)
        return pd.DataFrame(batch_data_list)

    def _clean_numerical_value(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s_val = value.replace(',', '')
            if '万' in s_val:
                try:
                    return float(s_val.replace('万', '')) * 10000
                except ValueError:
                    return 0.0
            try:
                return float(s_val)
            except ValueError:
                return 0.0
        return 0.0

    def forward(self, uids_str_list: List[str]) -> Dict[str, torch.Tensor]:
        batch_size = len(uids_str_list)
        def_cat_dim = max(1, config.PROFILE_CAT_AGG_DIM)
        def_num_dim = max(1, config.PROFILE_NUM_AGG_DIM)
        def_text_dim = max(1, config.PROFILE_TEXTUAL_AGG_DIM)
        def_date_dim = max(1, config.PROFILE_DATETIME_AGG_DIM)
        def_out_dim = max(1, self.output_dim)

        return_dict = {
            "final_profile_embedding": torch.zeros((batch_size, def_out_dim), device=config.DEVICE),
            "categorical_summary": torch.zeros((batch_size, def_cat_dim), device=config.DEVICE),
            "numerical_summary": torch.zeros((batch_size, def_num_dim), device=config.DEVICE),
            "textual_profile_summary": torch.zeros((batch_size, def_text_dim), device=config.DEVICE),
            "datetime_summary": torch.zeros((batch_size, def_date_dim), device=config.DEVICE),
        }
        if batch_size == 0: return return_dict

        batch_df = self._get_user_data_batch(uids_str_list)
        all_cat_embeds, all_textual_profile_embeds, all_datetime_embeds = [], [], []

        if self.categorical_feature_names and config.PROFILE_CAT_AGG_DIM > 0:
            for fname in self.categorical_feature_names:
                key = f'{fname}_cat_embed';
                if key in self.feature_encoders:
                    vocab = self.cat_feature_vocabs[fname]; series = batch_df[fname].fillna("MISSING_CAT").astype(str)
                    indices = [vocab.get(v, vocab.get("MISSING_CAT", 0)) for v in series]
                    all_cat_embeds.append(self.feature_encoders[key](torch.tensor(indices, dtype=torch.long).to(config.DEVICE)))
            if all_cat_embeds and self.cat_agg_proj: return_dict["categorical_summary"] = self.cat_agg_proj(torch.cat(all_cat_embeds, dim=1))

        if self.actual_numerical_feature_names and self.num_agg_proj and config.PROFILE_NUM_AGG_DIM > 0:
            num_data = [pd.to_numeric(batch_df[col].apply(self._clean_numerical_value), errors='coerce').fillna(0.0).values for col in self.actual_numerical_feature_names]
            if num_data: return_dict["numerical_summary"] = self.num_agg_proj(torch.tensor(np.array(num_data).T, dtype=torch.float32).to(config.DEVICE))

        if self.internal_text_embedder and self.textual_profile_feature_names and config.PROFILE_TEXTUAL_AGG_DIM > 0:
            for fname in self.textual_profile_feature_names:
                key = f'{fname}_text_proj'
                if key in self.feature_encoders and fname in batch_df.columns:
                    texts = batch_df[fname].fillna("").tolist()
                    if texts: raw_embeds = self.internal_text_embedder(texts); all_textual_profile_embeds.append(self.feature_encoders[key](raw_embeds))
            if all_textual_profile_embeds and self.textual_profile_agg_proj: return_dict["textual_profile_summary"] = self.textual_profile_agg_proj(torch.cat(all_textual_profile_embeds, dim=1))

        if self.datetime_feature_names and config.PROFILE_DATETIME_AGG_DIM > 0:
            for fname in self.datetime_feature_names:
                key = f'{fname}_datetime_encode'
                if key in self.feature_encoders and fname in batch_df.columns: all_datetime_embeds.append(self.feature_encoders[key](batch_df[fname]))
            if all_datetime_embeds and self.datetime_agg_proj: return_dict["datetime_summary"] = self.datetime_agg_proj(torch.cat(all_datetime_embeds, dim=1))

        features_for_final_proj = []
        if return_dict["categorical_summary"].numel() > 0 and return_dict["categorical_summary"].shape[1] > 0 : features_for_final_proj.append(return_dict["categorical_summary"])
        if return_dict["numerical_summary"].numel() > 0 and return_dict["numerical_summary"].shape[1] > 0: features_for_final_proj.append(return_dict["numerical_summary"])
        if return_dict["textual_profile_summary"].numel() > 0 and return_dict["textual_profile_summary"].shape[1] > 0: features_for_final_proj.append(return_dict["textual_profile_summary"])
        if return_dict["datetime_summary"].numel() > 0 and return_dict["datetime_summary"].shape[1] > 0: features_for_final_proj.append(return_dict["datetime_summary"])

        if features_for_final_proj and self.output_dim > 0:
            combined = torch.cat(features_for_final_proj, dim=1)
            if not isinstance(self.final_proj, nn.Identity) and combined.shape[1] > 0 and hasattr(self.final_proj, 'in_features') and combined.shape[1] == self.final_proj.in_features :
                return_dict["final_profile_embedding"] = self.activation(self.final_proj(combined))
            elif isinstance(self.final_proj, nn.Identity) and combined.shape[1] == self.output_dim:
                return_dict["final_profile_embedding"] = self.activation(combined)
        return return_dict

class SemanticAttention(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.Tanh()
        )
        self.q = nn.Parameter(torch.Tensor(1, hidden_size))
        nn.init.xavier_uniform_(self.q)

    def forward(self, z: torch.Tensor):
        w = torch.matmul(self.project(z), self.q.t())
        return w

class HeteroGNN(nn.Module):
    def __init__(self, node_type_dims: Dict[str, int], hidden_dim: int, out_dim: int,
                edge_types: List[Tuple[str, str, str]], num_layers=2, dropout=0.3, num_heads_gat=4):
        super().__init__()
        if not PYG_AVAILABLE: raise ImportError("HeteroGNN requires PyTorch Geometric.")
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.initial_lins = nn.ModuleDict()
        for node_type, in_dim in node_type_dims.items():
            if in_dim > 0:
                self.initial_lins[node_type] = PyGLinear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        self.convs = nn.ModuleList()
        for i_layer in range(num_layers):
            conv_dict = {}
            for src, rel, dst in edge_types:
                if node_type_dims.get(src, 0) > 0 and node_type_dims.get(dst, 0) > 0:
                    conv_dict[(src, rel, dst)] = SAGEConv(self.hidden_dim, self.hidden_dim, aggr="sum", bias=False)
            if conv_dict:
                self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            else:
                print(f"HeteroGNN: No valid convolutions defined for layer {i_layer}.")

        self.dropout = nn.Dropout(dropout)
        self.final_user_proj = PyGLinear(hidden_dim, out_dim) if hidden_dim != out_dim and hidden_dim > 0 and out_dim > 0 else nn.Identity()

    def forward(self, x_dict_initial: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str,str,str], torch.Tensor]) -> torch.Tensor:
        x_dict = {}
        for node_type, x_feat in x_dict_initial.items():
            if node_type in self.initial_lins and x_feat.numel() > 0 and x_feat.shape[0] > 0:
                x_dict[node_type] = self.initial_lins[node_type](x_feat)
            elif x_feat.numel() > 0 and x_feat.shape[0] > 0:
                x_dict[node_type] = x_feat

        if not self.convs:
            user_final_emb = x_dict.get('user')
            if user_final_emb is not None and user_final_emb.numel() > 0:
                return self.final_user_proj(user_final_emb)
            else:
                batch_s = 1
                for val_tensor in x_dict_initial.values():
                    if val_tensor.ndim > 1 and val_tensor.shape[0] > 0: batch_s = val_tensor.shape[0]; break
                return torch.zeros((batch_s, self.out_dim), device=config.DEVICE)


        for conv_layer in self.convs:
            current_valid_x_dict = {k:v for k,v in x_dict.items() if v.numel() > 0 and v.shape[0] > 0}
            current_valid_edge_index_dict = {}
            for edge_type, edge_idx_val in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                if src_type in current_valid_x_dict and dst_type in current_valid_x_dict:
                    current_valid_edge_index_dict[edge_type] = edge_idx_val

            if not current_valid_x_dict or (not current_valid_edge_index_dict and len(current_valid_x_dict)>1):
                break

            x_dict_out = conv_layer(current_valid_x_dict, current_valid_edge_index_dict)

            temp_x_dict = {}
            for key, value in x_dict_out.items():
                if value.numel() > 0 and value.shape[0] > 0:
                    temp_x_dict[key] = F.relu(value)
                    temp_x_dict[key] = self.dropout(temp_x_dict[key])
            x_dict = temp_x_dict

        user_final_emb = x_dict.get('user')
        batch_s = 1
        if user_final_emb is None or user_final_emb.numel() == 0 or user_final_emb.shape[0] == 0:
            if 'user' in x_dict_initial and x_dict_initial['user'].numel() > 0 :
                initial_user_feat = x_dict_initial['user']
                if 'user' in self.initial_lins:
                    initial_user_feat_proj = self.initial_lins['user'](initial_user_feat)
                else:
                    initial_user_feat_proj = initial_user_feat if initial_user_feat.shape[-1] == self.hidden_dim else torch.zeros((initial_user_feat.shape[0], self.hidden_dim), device=config.DEVICE)

                if isinstance(self.final_user_proj, nn.Identity) and initial_user_feat_proj.shape[-1] == self.out_dim:
                    return initial_user_feat_proj
                elif not isinstance(self.final_user_proj, nn.Identity):
                    return self.final_user_proj(initial_user_feat_proj)
            for val_tensor in x_dict_initial.values():
                if val_tensor.ndim > 1 and val_tensor.shape[0] > 0: batch_s = val_tensor.shape[0]; break
            return torch.zeros((batch_s, self.out_dim), device=config.DEVICE)

        return self.final_user_proj(user_final_emb)

class CombinedEmbedder(nn.Module):
    def __init__(self, text_embedder: Optional[TextEmbedder],
                profile_embedder: Optional[ProfileEmbedder],
                time_embedder: Optional[TimeEmbedder],
                num_users: int,
                user_to_idx: Dict[str, int]):
        super().__init__()
        self.text_embedder = text_embedder
        self.profile_embedder = profile_embedder
        self.time_embedder = time_embedder
        self.combination_type = config.EMBEDDING_COMBINATION
        self.user_to_idx = user_to_idx

        self.hetero_gnn = None
        self.user_type_embedding = None

        if self.combination_type == 'hetero':
            if not PYG_AVAILABLE: raise ImportError("PyTorch Geometric is required for 'hetero' mode.")
            self.node_type_input_dims = {
                'user': config.USER_NODE_INIT_DIM,
                'text_modality': config.TEXT_EMBEDDING_DIM if self.text_embedder and config.TEXT_EMBEDDING_DIM > 0 else 0,
                'time_modality': config.TIME_EMBEDDING_DIM if self.time_embedder and config.TIME_EMBEDDING_DIM > 0 else 0,
                'profile_base': config.PROFILE_EMBEDDING_DIM if self.profile_embedder and config.PROFILE_EMBEDDING_DIM > 0 else 0,
                'cat_summary': config.PROFILE_CAT_AGG_DIM if self.profile_embedder and config.PROFILE_CAT_AGG_DIM > 0 else 0,
                'num_summary': config.PROFILE_NUM_AGG_DIM if self.profile_embedder and config.PROFILE_NUM_AGG_DIM > 0 else 0,
                'textual_profile_summary': config.PROFILE_TEXTUAL_AGG_DIM if self.profile_embedder and config.PROFILE_TEXTUAL_AGG_DIM > 0 else 0,
                'datetime_summary': config.PROFILE_DATETIME_AGG_DIM if self.profile_embedder and config.PROFILE_DATETIME_AGG_DIM > 0 else 0,
            }
            self.node_type_input_dims = {k:v for k,v in self.node_type_input_dims.items() if v > 0}

            self.edge_types_list = [
                ('user', 'to_text', 'text_modality'), ('text_modality', 'rev_to_text', 'user'),
                ('user', 'to_time', 'time_modality'), ('time_modality', 'rev_to_time', 'user'),
                ('user', 'to_profile_base', 'profile_base'), ('profile_base', 'rev_to_profile_base', 'user'),
                ('profile_base', 'has_cat', 'cat_summary'), ('cat_summary', 'rev_has_cat', 'profile_base'),
                ('profile_base', 'has_num', 'num_summary'), ('num_summary', 'rev_has_num', 'profile_base'),
                ('profile_base', 'has_textual_profile', 'textual_profile_summary'), ('textual_profile_summary', 'rev_has_textual_profile', 'profile_base'),
                ('profile_base', 'has_datetime', 'datetime_summary'), ('datetime_summary', 'rev_has_datetime', 'profile_base'),
            ]
            self.valid_edge_types_for_gnn = []
            for src, rel, dst in self.edge_types_list:
                if src in self.node_type_input_dims and dst in self.node_type_input_dims:
                    self.valid_edge_types_for_gnn.append((src,rel,dst))

            if config.USER_NODE_INIT_DIM > 0 and num_users > 0:
                self.user_type_embedding = nn.Embedding(num_users, config.USER_NODE_INIT_DIM)

            if self.valid_edge_types_for_gnn and config.HETERO_GNN_HIDDEN_DIM > 0 and config.INITIAL_EMBEDDING_DIM > 0 and config.HETERO_GNN_LAYERS > 0:
                self.hetero_gnn = HeteroGNN(
                    node_type_dims=self.node_type_input_dims, hidden_dim=config.HETERO_GNN_HIDDEN_DIM,
                    out_dim=config.INITIAL_EMBEDDING_DIM, edge_types=self.valid_edge_types_for_gnn,
                    num_layers=config.HETERO_GNN_LAYERS, dropout=config.HETERO_DROPOUT_RATE)
            else:
                print("CombinedEmbedder ('hetero'): HeteroGNN not initialized due to no valid edge types or GNN config (hidden_dim/out_dim/layers <= 0).")
                self.hetero_gnn = None

            self.final_combined_dim = config.INITIAL_EMBEDDING_DIM

        elif self.combination_type == 'concat':
            dim_sum = 0
            if self.text_embedder and config.TEXT_EMBEDDING_DIM > 0: dim_sum += config.TEXT_EMBEDDING_DIM
            if self.profile_embedder and config.PROFILE_EMBEDDING_DIM > 0: dim_sum += config.PROFILE_EMBEDDING_DIM
            if self.time_embedder and config.TIME_EMBEDDING_DIM > 0: dim_sum += config.TIME_EMBEDDING_DIM
            self.final_combined_dim = dim_sum if dim_sum > 0 else 1

        elif self.combination_type == 'weighted_sum':
            self.final_combined_dim = config.INITIAL_EMBEDDING_DIM
            self.projections_for_sum = nn.ModuleDict()
            num_active_for_sum = 0; active_sum_names = []
            main_modalities = {'text': config.TEXT_EMBEDDING_DIM, 'profile_base': config.PROFILE_EMBEDDING_DIM, 'time': config.TIME_EMBEDDING_DIM}
            for name, dim_cfg in main_modalities.items():
                if dim_cfg > 0:
                    self.projections_for_sum[name] = nn.Linear(dim_cfg, self.final_combined_dim) if dim_cfg != self.final_combined_dim else nn.Identity()
                    num_active_for_sum +=1; active_sum_names.append(name)
            self.sum_weights = nn.Parameter(torch.ones(num_active_for_sum)) if num_active_for_sum > 0 else None
            self.active_names_for_weighted_sum = active_sum_names
        else: raise ValueError(f"Unknown EMBEDDING_COMBINATION: {self.combination_type}")

        if self.final_combined_dim != config.INITIAL_EMBEDDING_DIM:
            print(f"CombinedEmbedder ({self.combination_type}) WARNING: Final Z dim ({self.final_combined_dim}) differs from config.INITIAL_EMBEDDING_DIM ({config.INITIAL_EMBEDDING_DIM}). Using: {self.final_combined_dim}.")

    def forward(self, contents: List[str], uids_str_list: List[str],
                timestamps_series: pd.Series, first_timestamp_obj: pd.Timestamp) -> torch.Tensor:
        batch_size = len(uids_str_list)
        if batch_size == 0: return torch.zeros((0, self.final_combined_dim), device=config.DEVICE)

        text_e = self.text_embedder(contents) if self.text_embedder and config.TEXT_EMBEDDING_DIM > 0 else None
        time_e = self.time_embedder(timestamps_series, first_timestamp_obj) if self.time_embedder and config.TIME_EMBEDDING_DIM > 0 else None
        profile_outputs = self.profile_embedder(uids_str_list) if self.profile_embedder else {}

        if self.combination_type == 'hetero':
            if not PYG_AVAILABLE or self.hetero_gnn is None or self.user_type_embedding is None:
                return torch.zeros((batch_size, self.final_combined_dim), device=config.DEVICE)
            output_z_list = []
            for i in range(batch_size):
                current_x_dict = {}; current_edge_index_dict = {}

                if config.USER_NODE_INIT_DIM > 0:
                    uid_str = uids_str_list[i]
                    if uid_str in self.user_to_idx:
                        user_idx = self.user_to_idx[uid_str]
                        user_initial_embedding = self.user_type_embedding(torch.tensor([user_idx], device=config.DEVICE))
                        current_x_dict['user'] = user_initial_embedding

                if text_e is not None and text_e.shape[0] == batch_size: current_x_dict['text_modality'] = text_e[i].unsqueeze(0)
                if time_e is not None and time_e.shape[0] == batch_size: current_x_dict['time_modality'] = time_e[i].unsqueeze(0)

                prof_base_e = profile_outputs.get("final_profile_embedding")
                if prof_base_e is not None and prof_base_e.shape[0] == batch_size: current_x_dict['profile_base'] = prof_base_e[i].unsqueeze(0)

                sub_profile_features = {
                    "cat_summary": profile_outputs.get("categorical_summary"),
                    "num_summary": profile_outputs.get("numerical_summary"),
                    "textual_profile_summary": profile_outputs.get("textual_profile_summary"),
                    "datetime_summary": profile_outputs.get("datetime_summary")
                }
                for key, tensor_val in sub_profile_features.items():
                    if tensor_val is not None and tensor_val.shape[0] == batch_size and self.node_type_input_dims.get(key,0) > 0:
                        current_x_dict[key] = tensor_val[i].unsqueeze(0)

                for src_type, rel_type, dst_type in self.valid_edge_types_for_gnn:
                    if src_type in current_x_dict and dst_type in current_x_dict:
                        current_edge_index_dict[(src_type, rel_type, dst_type)] = torch.tensor([[0], [0]], dtype=torch.long, device=config.DEVICE)

                if 'user' not in current_x_dict:
                    output_z_list.append(torch.zeros((1, self.final_combined_dim), device=config.DEVICE)); continue

                if not current_edge_index_dict and len(current_x_dict) > 1:
                    user_initial_feat = current_x_dict['user']
                    if 'user' in self.hetero_gnn.initial_lins: user_initial_feat = self.hetero_gnn.initial_lins['user'](user_initial_feat)
                    user_z_for_sample = self.hetero_gnn.final_user_proj(user_initial_feat)
                elif not current_edge_index_dict and len(current_x_dict) == 1 and 'user' in current_x_dict:
                    user_initial_feat = current_x_dict['user']
                    if 'user' in self.hetero_gnn.initial_lins: user_initial_feat = self.hetero_gnn.initial_lins['user'](user_initial_feat)
                    user_z_for_sample = self.hetero_gnn.final_user_proj(user_initial_feat)
                elif not current_edge_index_dict :
                    user_z_for_sample = torch.zeros((1, self.final_combined_dim), device=config.DEVICE)
                else:
                    user_z_for_sample = self.hetero_gnn(current_x_dict, current_edge_index_dict)

                output_z_list.append(user_z_for_sample)

            if not output_z_list: return torch.zeros((batch_size, self.final_combined_dim), device=config.DEVICE)
            return torch.cat(output_z_list, dim=0)

        else:
            profile_final_e = profile_outputs.get("final_profile_embedding")
            embed_list, names_for_sum_actual = [], []
            if text_e is not None and text_e.numel() > 0 and text_e.shape[0] == batch_size: embed_list.append(text_e); names_for_sum_actual.append('text')
            if profile_final_e is not None and profile_final_e.numel() > 0 and profile_final_e.shape[0] == batch_size: embed_list.append(profile_final_e); names_for_sum_actual.append('profile_base')
            if time_e is not None and time_e.numel() > 0 and time_e.shape[0] == batch_size: embed_list.append(time_e); names_for_sum_actual.append('time')
            if not embed_list: return torch.zeros((batch_size, self.final_combined_dim), device=config.DEVICE)
            if self.combination_type == 'concat':
                current_concat_dim = sum(e.shape[1] for e in embed_list)
                if current_concat_dim == self.final_combined_dim and current_concat_dim > 0 : return torch.cat(embed_list, dim=1)
                return torch.zeros((batch_size, self.final_combined_dim), device=config.DEVICE)
            elif self.combination_type == 'weighted_sum':
                if self.sum_weights is not None and self.active_names_for_weighted_sum and \
                len(names_for_sum_actual) == len(self.active_names_for_weighted_sum) and \
                len(embed_list) == len(self.active_names_for_weighted_sum) and \
                self.sum_weights.shape[0] == len(self.active_names_for_weighted_sum):

                    norm_weights = F.softmax(self.sum_weights, dim=0)
                    projected_weighted = []

                    for idx, name_key in enumerate(self.active_names_for_weighted_sum):
                        try:
                            actual_emb_idx = names_for_sum_actual.index(name_key)
                            emb_to_project = embed_list[actual_emb_idx]
                            if name_key in self.projections_for_sum:
                                projected = self.projections_for_sum[name_key](emb_to_project)
                                projected_weighted.append(projected * norm_weights[idx])
                        except ValueError: pass
                    if projected_weighted: return torch.stack(projected_weighted).sum(dim=0)
            return torch.zeros((batch_size, self.final_combined_dim), device=config.DEVICE)