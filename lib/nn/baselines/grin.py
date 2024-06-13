from typing import Optional, List

from torch import Tensor
from torch_geometric.typing import OptTensor
from tsl.nn.models import GRINModel as GRINModel_


class GRINModel(GRINModel_):

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
                mask: OptTensor = None,
                u: Optional[Tensor] = None) -> List:
        """"""
        if u is not None and u.ndim == 3:
            u = u.unsqueeze(-2).expand(-1, -1, x.size(-2), -1)
        return super().forward(x=x,
                               mask=mask,
                               u=u,
                               edge_index=edge_index,
                               edge_weight=edge_weight)
