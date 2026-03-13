import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method

class CrossAgentAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x, attn_mask=None):
        """
        x: [B, N, D]  (batch, agents, embedding)
        """
        out, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask
        )
        return out, attn_weights
    
class CrossAgentEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=1):
        super().__init__()

        self.attn = CrossAgentAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            # nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        # Cross-agent attention
        attn_out, attn_weights = self.attn(x, attn_mask)
        x = self.norm1(x + attn_out)

        # dropout
        x = self.dropout(x)

        # Feed-forward
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x, attn_weights
    

class CrossAgentEncoderCritic(nn.Module):
    def __init__(self, obs_dim, embed_dim, num_heads, num_layers):
        super().__init__()

        self.embedding = nn.Linear(obs_dim, embed_dim)

        self.layers = nn.ModuleList([
            CrossAgentEncoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])


        
        self.out = (nn.Linear(embed_dim,1))
        nn.init.constant_(self.out.weight.data,0)
        nn.init.constant_(self.out.bias.data,0)

    def forward(self, obs, attn_mask=None):
        """
        obs: [B, N, obs_dim]
        """
        x = self.embedding(obs)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_w = layer(x, attn_mask)
            attn_weights_all.append(attn_w)
        x = self.out(x)

        return x, attn_weights_all
    
    
    
if __name__=="__main__":
    B, N, obs_dim = 32, 3, 31
    encoder = CrossAgentEncoderCritic(
        obs_dim=obs_dim,
        embed_dim=64,
        num_heads=1,
        num_layers=1
    )
    
    print(encoder)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(total_params)

    obs = torch.randn(B, N, obs_dim)
    agent_critic, attn_maps = encoder(obs)
    print(agent_critic.shape)
    print(attn_maps)