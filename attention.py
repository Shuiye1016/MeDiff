# attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import math

def group_softmax(O, group_counts, dim=-1):
    if dim < 0:
        dim = O.dim() + dim

    k = O.shape[dim] 
    if k == 0: return O 

    view_shape = list(O.shape)
    for i in range(O.dim()):
        if i != dim and i != 0 and i !=1 :
            view_shape[i] = 1
    view_shape[dim] = k
    
    if group_counts.dim() == O.dim() -1:
        group_counts_exp = group_counts.unsqueeze(-2)
        if dim != -1 and dim != O.dim() -1 :
            pass
    elif group_counts.dim() == O.dim():
         group_counts_exp = group_counts
    else:
        b_o, h_o = O.shape[0], O.shape[1]
        k_keys_o = O.shape[-1]
        
        if group_counts.shape[0] == b_o and group_counts.shape[1] == h_o and group_counts.shape[2] == k_keys_o and dim == -1:
            target_view_shape_for_counts = [b_o, h_o, 1, k_keys_o]
            group_counts_exp = group_counts.float().to(O.device).view(target_view_shape_for_counts)
        else:
            shape = list(O.shape)
            for i in range(len(shape)):
                if i not in [0, 1, dim if dim >=0 else O.dim()+dim]:
                    shape[i] = 1

            num_dims_to_unsqueeze = O.dim() - group_counts.dim() -1
            
            temp_group_counts = group_counts.float().to(O.device)
            for _ in range(num_dims_to_unsqueeze):
                temp_group_counts = temp_group_counts.unsqueeze(-2)

            if O.dim() == 4 and group_counts.dim() == 3 and dim == -1:
                 group_counts_exp = group_counts.float().to(O.device).unsqueeze(2)
            else:
                 # print(f"Warning: group_softmax shape mismatch or unhandled case. O:{O.shape}, counts:{group_counts.shape}, dim:{dim}")
                 group_counts_exp = group_counts.float().to(O.device)
    exp_O = torch.exp(O)
    weighted_exp_O = exp_O * group_counts_exp 

    denominator = torch.sum(weighted_exp_O, dim=dim, keepdim=True)
    denominator = torch.clamp(denominator, min=1e-9) 

    return weighted_exp_O / denominator


class GroupAttention(nn.Module):
    def __init__(self, embed_dim=config.GROUP_ATTENTION_DIM, num_heads=config.ATTENTION_HEADS, dropout=config.DROPOUT_RATE):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
             raise ValueError("Embedding dimension and number of heads must be positive.")
        if embed_dim % num_heads != 0:
             raise ValueError(f"Embedding dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query_reps, key_reps, value_reps, group_counts_for_keys, mask=None):
        batch_size, kq, _ = query_reps.shape
        kk = key_reps.shape[1]

        Q = self.Wq(query_reps)
        K = self.Wk(key_reps)
        V = self.Wv(value_reps)

        Q = Q.view(batch_size, kq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, kk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, kk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3: mask = mask.unsqueeze(1)
            energy = energy.masked_fill(mask == 0, -1e10)

        counts_expanded_for_heads = group_counts_for_keys.unsqueeze(1).repeat(1, self.num_heads, 1)
        
        attention = group_softmax(energy, counts_expanded_for_heads, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, kq, self.embed_dim)
        out = self.fc_out(out)
        return out