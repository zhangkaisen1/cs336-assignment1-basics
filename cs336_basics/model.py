import torch
import torch.nn as nn
import math 
from einops import rearrange,einsum

class Linear(nn.Module):
    def __init__(
        self, 
        in_features : int, 
        out_features : int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
    # initialize nn.Parameter
        super().__init__()
        # initialize w as para
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, 0.0, sigma, -3.0 * sigma, 3.0 * sigma)


    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )
        nn.init.trunc_normal_(self.weight, 0.0, 1.0, -3.0, 3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(
                d_model,
                device=device,
                dtype=dtype
            )
        )
        nn.init.ones_(self.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMSa = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True)
            + self.eps
        )
        return (x / RMSa * self.weight).to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)

    @staticmethod
    def silu(x:torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        inv_freq = 1 / (
            theta ** (
                torch.arange(
                    0,
                    d_k,
                    2,
                    device=device,
                    dtype=torch.float32,
                ) / d_k
            )
        )
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        angles = positions[:, None] * inv_freq[None, :] # [max_seq_len, d_k]
        
        self.register_buffer(
            "cos",
            torch.cos(angles),
        )

        self.register_buffer(
            "sin",
            torch.sin(angles),
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        return torch.stack(
            [rotated_even, rotated_odd],
            dim=-1,
        ).flatten(-2)
        

def softmax(v : torch.Tensor, dim : int) -> torch.Tensor:
    # Tips : prevent overflow
    max_value = torch.max(v, dim = dim, keepdim = True).values
    v = v - max_value

    exp_v = torch.exp(v)
    sum_v = torch.sum(
        exp_v,
        dim=dim,
        keepdim=True,
    )

    return  exp_v / sum_v


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.tensor:
    d_k = Q.shape[-1] 
    QK_t = Q @ K.transpose(-2, -1)
    softmax_v = QK_t / (d_k ** 0.5)

    # mask
    if mask is not None:
        softmax_v = softmax_v.masked_fill(
            mask == 0,
            float("-inf"),
        )
    attention_weights = softmax(
        softmax_v, 
        dim = -1,
    )

    return attention_weights @ V

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        if_RoPE:bool = False,
        max_seq_len: int = 4096, 
        theta: float = 10000.0,
        **kwargs
    ):
        assert d_model % num_heads == 0

        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)

        self.output_proj = Linear(d_model, d_model)
        self.if_RoPE = if_RoPE

        if self.if_RoPE == True:
            self.rope = RotaryPositionalEmbedding(theta = theta, d_k = self.d_k, max_seq_len = max_seq_len)
    
    def forward(
        self,
        x:torch.Tensor,
        token_positions=None
    ) -> torch.Tensor:
        seq_len = x.size(1)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # multi head
        q = rearrange(q, '... seq_len (h d) -> ... h seq_len d', h = self.h)
        k = rearrange(k, '... seq_len (h d) -> ... h seq_len d', h = self.h)
        v = rearrange(v, '... seq_len (h d) -> ... h seq_len d', h = self.h)
        ## rope
        if self.if_RoPE == True:
            if token_positions is None:
                token_positions = torch.arange(
                    x.shape[-2],
                    device=x.device
                )
                
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 变为下三角
        mask = torch.ones((seq_len, seq_len), device=x.device).tril()
        mask = rearrange(mask, 'T1 T2 -> 1 1 T1 T2')

        xo = scaled_dot_product_attention(q, k, v, mask)
        xo = rearrange(xo, '... h seq_len d -> ... seq_len (h d)')
        return self.output_proj(xo)

# wait to alter
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        d_ff:int,
        max_seq_len: int = 4096, 
        theta: float = 10000.0,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = MultiheadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            if_RoPE=True, 
            max_seq_len=max_seq_len, 
            theta=theta
        )
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(self.d_model) 
        self.ln2 = RMSNorm(self.d_model) 
    
    def forward(
        self,
        x:torch.Tensor,
        token_positions=None,
    ) -> torch.Tensor:

        y = x + self.attn(self.ln1(x), token_positions)
        o = y + self.ffn(self.ln2(y))

        return o

# wait to alter
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self, 
        token_ids:torch.Tensor, 
        token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        y = self.lm_head(x)
        return y


