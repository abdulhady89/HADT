import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.models_tools import init, get_active_func, get_init_method

"""Transformer modules."""

class CrossAgentAttention(nn.Module):
    """A new inter-agent cross-attention mechanism."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.scale = self.dk ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x: (B, N, d_model)
        mask: (B, N) binary mask (1=valid, 0=padding)
        attention_bias: (B, N, N) optional attention bias
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.dk).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.dk).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.dk).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_ = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_logits = attn_logits.masked_fill(mask_ == 0, -1e9)

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out_proj(out)
        out = self.norm(x + out)

        return out, attn


class TransformerBasedCritic(nn.Module):
    def __init__(self, args, obs_dim, num_agent_types=2, n_heads=4):
        super().__init__()

        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]
        init_method = get_init_method(self.initialization_method)
        gain = nn.init.calculate_gain(self.activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        d_model=self.hidden_sizes[0]

        self.obs_dim = int(obs_dim)
        self.obs_proj = nn.Linear(self.obs_dim, d_model)
        
        # Add agent types factor as heterogeneity
        self.type_embed = nn.Embedding(num_agent_types, d_model)
        self.agent_types = torch.tensor([args["agent_types"]]).to("cuda:0") 
        # Cross-agent attention
        self.cross_attn = CrossAgentAttention(d_model=d_model, n_heads=n_heads)

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
        )

        # Critic head (aggregated value)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, obs, mask=None):
        """
        obs: (B, N, obs_dim)
        agent_types: (B, N)
        mask: (B, N)
        attention_bias: (B, N, N)
        """

        # --- auto-split if obs is flattened ---
        if obs.ndim == 2:  # (B, N*obs_dim)
            obs, N = self._split_flat(obs)
            self.agent_types = self.agent_types[:, :N]
            if mask is not None:
                mask = mask[:, :N]

        x = self.obs_proj(obs)
        x = x + self.type_embed(self.agent_types)

        # Inter-agent attention
        x, attn = self.cross_attn(x, mask)
        x = self.ff(x)

        # Aggregate across agents
        if mask is not None:
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        else:
            pooled = x.mean(dim=1)

        value = self.value_head(pooled)
        return value, attn
    
    def _split_flat(self, obs):
        B, total_dim = obs.shape
        num_agents = total_dim // self.obs_dim
        obs = obs.view(B, num_agents, self.obs_dim)
        return obs, num_agents

