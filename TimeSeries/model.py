import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None): 
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1) #sfm
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        assert input_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.input_dim = input_dim
        self.num_heads = num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * input_dim)
        self.o_proj = nn.Linear(input_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        # qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.input_dim)
        # qkv = qkv.permute(0, 2, 1)  # [Batch, Head, SeqLen]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads=1, dropout=0.05):
        super().__init__()
        # Attn
        self.self_attn = MultiheadSelfAttention(input_dim=input_dim, num_heads=num_heads)
        # Dense out
        self.dense = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask=mask)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.norm(x)

        return x
    
class Attn4TS(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1, dropout=0.05):
        super().__init__()
        self.attn = SelfAttentionLayer(input_dim, num_heads, dropout)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, mask=None):
        x = self.attn(x, mask)
        return self.fc(x)