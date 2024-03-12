from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from tsl.nn.models import SPINHierarchicalModel


class SPINHierarchicalPredictionModel(SPINHierarchicalModel):
    r"""The Hierarchical Spatiotemporal Point Inference Network (SPIN-H) from
    the paper `"Learning to Reconstruct Missing Data from Spatiotemporal Graphs
    with Sparse Observations" <https://arxiv.org/abs/2205.13479>`_
    (Marisca et al., NeurIPS 2022).
    """
    return_type = tuple

    def __init__(self,
                 input_size: int,
                 h_size: int,
                 z_size: int,
                 n_nodes: int,
                 z_heads: int = 1,
                 horizon: Optional[int] = None,
                 exog_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 n_layers: int = 5,
                 eta: int = 3,
                 message_layers: int = 1,
                 reweigh: Optional[str] = 'softmax',
                 update_z_cross: bool = True,
                 norm: bool = True,
                 spatial_aggr: str = 'add'):
        super(SPINHierarchicalPredictionModel, self).__init__(
            input_size=input_size,
            h_size=h_size,
            z_size=z_size,
            n_nodes=n_nodes,
            z_heads=z_heads,
            exog_size=exog_size,
            output_size=output_size,
            n_layers=n_layers,
            eta=eta,
            message_layers=message_layers,
            reweigh=reweigh,
            update_z_cross=update_z_cross,
            norm=norm,
            spatial_aggr=spatial_aggr,
        )
        self.horizon = horizon

    def forward(self,
                x: Tensor,
                input_mask: Tensor,
                edge_index: Tensor,
                u: OptTensor = None,
                u_horizon: OptTensor = None,
                node_index: OptTensor = None,
                target_nodes: OptTensor = None) -> Tuple:
        """"""
        # x: [batch window nodes features]
        # u / u_horizon : [batch window (nodes) features]
        b, w, n, f, = x.size()
        # Concatenate (and possibly build) exogenous
        if u is None and u_horizon is None:
            assert self.horizon is not None
            s = w + self.horizon
            u = torch.zeros((b, s, f), dtype=x.dtype, device=x.device)
            u += (torch.linspace(.1, .9, s, dtype=x.dtype, device=x.device).
                  view(1, s, 1))
        else:
            assert u is not None and u_horizon is not None
            u = torch.cat([u, u_horizon], dim=1)

        h = u.size(1) - w
        x_h = torch.zeros((b, h, n, f), dtype=x.dtype, device=x.device)
        x = torch.cat([x, x_h], dim=1)

        mask_h = torch.zeros((b, h, n, f), dtype=torch.bool, device=x.device)
        mask = torch.cat([input_mask, mask_h], dim=1)

        x_hat, imputations = super().forward(x=x,
                                             u=u,
                                             mask=mask,
                                             edge_index=edge_index,
                                             node_index=node_index,
                                             target_nodes=target_nodes)
        imputations.append(x_hat)
        imputations = [imputation[:, :w] for imputation in imputations]

        y_hat = x_hat[:, w:]

        return y_hat, imputations
