from einops import einsum, rearrange
import torch
import torch.nn as nn
import math
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        sigma = math.sqrt(2 / (in_features + out_features))
        w = torch.empty(out_features, in_features, device = device, dtype= dtype)
        nn.init.trunc_normal_(w, mean = 0, std = sigma, a = -3*sigma, b = 3*sigma)
        self.W = nn.Parameter(w)
    def forward(self, x):
        return einsum(x, self.W, "... d_in, out d_in -> ... out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim, device = device, dtype= dtype)
        nn.init.trunc_normal_(weight, mean = 0, std = 1, a = -3, b = 3)
        self.weight = nn.Parameter(weight)
    def forward(self, token_ids):
        return self.weight[token_ids]   

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / RMS * self.gain
        return result.to(in_dtype)

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device = None, dtype = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model,d_ff, device=device, dtype=dtype)
    def forward(self, x):
        return self.w2( silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        j = torch.arange(d_k // 2, device=device)
        freqs = theta ** (-2 * j / d_k)              # (d_k/2,)  每对的频率
        positions = torch.arange(max_seq_len, device=device)   # (max_seq_len,)
        angles = positions[:, None] * freqs[None, :]  # (max_seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    def forward(self, x, token_postions):
        cos = self.cos[token_postions]
        sin = self.sin[token_postions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        result = torch.stack([out1, out2], dim = -1).flatten(-2)
        return result

def softmax(x, dim):
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)                         # QK^T / √d_k → (..., queries, keys)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))   # False位置设 -inf
    attn = softmax(scores, dim=-1)        # 沿 keys 维归一化
    return attn @ V                           # attn × V → (..., queries, d_v)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope     # 可选,带RoPE版本时传进来
    def forward(self, x, token_position = None):
        *batch, S, d_model = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... s (h dk) -> ... h s dk", h = self.num_heads)
        K = rearrange(K, "... s (h dk) -> ... h s dk", h = self.num_heads)
        V = rearrange(V, "... s (h dk) -> ... h s dk", h = self.num_heads)
        if self.rope is not None:
            if token_position is None:
                token_position = torch.arange(S, device=x.device)
            Q = self.rope(Q, token_position)
            K = self.rope(K, token_position)
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, "... h s dk -> ... s (h dk)")
        return self.output_proj(out)            

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    def forward(self, x, token_positions=None):
        y = x + self.attn(self.ln1(x), token_positions)   # 子层1: 残差 + 注意力(norm)
        z = y + self.ffn(self.ln2(y))                       # 子层2: 残差 + 前馈(norm)
        return z

class TransformerLM(nn.Module):
    def __init__(self,vocab_size, context_length, d_model, num_layers,
                 num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        d_k = d_model // num_heads
        rope = RotaryPositionalEmbedding(rope_theta, d_k, max_seq_len=context_length, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope, device=device,dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
        