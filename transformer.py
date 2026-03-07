from torch import Tensor, nn
from fnn import FNN


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_qkv_layers: int,
        num_heads: int,
        embed_size: int,
        num_output_layers: int,
    ) -> None:
        super().__init__()
        self._q_net, self._k_net, self._v_net = [
            FNN(
                input_size = input_size,
                hidden_size = embed_size,
                num_hidden_layers = num_qkv_layers,
                output_size = embed_size,
            )
            for _ in range(3)
        ]
        self._attn = nn.MultiheadAttention(
            embed_dim = embed_size,
            num_heads = num_heads,
        )
        self._output_net = FNN(
            input_size = embed_size,
            hidden_size = embed_size,
            num_hidden_layers = num_output_layers,
            output_size = embed_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        q = self._q_net(x)
        k = self._k_net(x)
        v = self._v_net(x)
        attn_out, _ = self._attn(q, k, v)
        meaned = attn_out.mean(dim = 0)
        return self._output_net(meaned)
