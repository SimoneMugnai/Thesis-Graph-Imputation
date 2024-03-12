from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor


def connectivity_to_row_col(edge_index: Adj) -> Tuple[Tensor, Tensor]:
    if isinstance(edge_index, Tensor):
        return edge_index[0], edge_index[1]
    elif isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.t().coo()
        return row, col
    else:
        raise NotImplementedError()


def connectivity_to_edge_index(
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    if isinstance(edge_index, Tensor):
        return edge_index, edge_attr
    elif isinstance(edge_index, SparseTensor):
        row, col, edge_attr = edge_index.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        return edge_index, edge_attr
    else:
        raise NotImplementedError()


def connectivity_to_adj_t(edge_index: Adj,
                          edge_attr: Optional[Tensor] = None,
                          num_nodes: Optional[int] = None) -> SparseTensor:
    if isinstance(edge_index, SparseTensor):
        return edge_index
    elif isinstance(edge_index, Tensor):
        adj_t = SparseTensor.from_edge_index(edge_index, edge_attr,
                                             (num_nodes, num_nodes)).t()
        return adj_t
    else:
        raise NotImplementedError()
