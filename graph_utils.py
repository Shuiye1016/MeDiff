# graph_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import config
import numpy as np
import pandas as pd

try:
    from torch_geometric.nn import MessagePassing, HypergraphConv
    from torch_geometric.utils import degree, scatter
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    TORCH_GEOM_AVAILABLE = True
except ImportError:
    TORCH_GEOM_AVAILABLE = False

    class MessagePassing(nn.Module):
        def __init__(self, aggr='add', **kwargs): super().__init__()
        def propagate(self, edge_index, size=None, **kwargs): raise ImportError("torch_geometric required")
    class HypergraphConv(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); raise ImportError("torch_geometric required")
        def forward(self, *args, **kwargs): raise ImportError("torch_geometric required")
    def degree(index, num_nodes=None, dtype=None): raise ImportError("torch_geometric required")
    def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"): raise ImportError("torch_geometric required")
    def maybe_num_nodes(edge_index, num_nodes=None): return num_nodes
print("graph_utils.py: Ready for HypergraphConv usage in model.")