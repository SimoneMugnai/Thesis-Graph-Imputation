from einops import rearrange
from torch import Tensor
from tsl.nn.layers import GRUCell, NodeEmbedding


class AmortizedGRUCell(GRUCell):
    r"""GRU cell with amortized memory: the hidden state is initialized with a
    different (trainable) embedding for each node.

    Args:
        input_size (int): The number of features in the input sample.
        hidden_size (int): The number of features in the hidden state.
        bias (bool): If :obj:`True`, then the layer will learn an additive
            bias for each gate.
            (default: :obj:`True`)
        device (optional): The device of the parameters.
            (default: :obj:`None`)
        dtype (optional): The data type of the parameters.
            (default: :obj:`None`)
    """

    def __init__(self, input_size: int, hidden_size: int, n_nodes: int,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(input_size, hidden_size, bias, device, dtype)
        self.n_nodes = n_nodes
        self.history_encoding = NodeEmbedding(n_nodes, hidden_size)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'history_encoding'):
            self.history_encoding.reset_parameters()

    def initialize_state(self, x) -> Tensor:
        b = x.size(0) / self.n_nodes
        assert int(b) == b
        h0 = self.history_encoding(expand=[int(b), -1, -1])  # repeat batch
        return rearrange(h0, 'b n f -> (b n) f')
