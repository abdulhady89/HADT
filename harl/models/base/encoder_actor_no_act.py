import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method
# from harl.models.base.act import ACTLayer
# import argparse

class CrossActionAttention(nn.Module):
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
    
class CrossActionEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=1):
        super().__init__()

        self.attn = CrossActionAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.0)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            # nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, h=None,attn_mask=None):
        # Cross-agent attention
        if h is not None:
            x = torch.cat((x, h), 1)

        attn_out, attn_weights = self.attn(x, attn_mask)
        x = self.norm1(x + attn_out)

        # dropout
        x = self.dropout(x)

        # Feed-forward
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x, attn_weights
    

class CrossActionEncoderPolicy(nn.Module):
    def __init__(self, args, obs_dim, embed_dim, num_heads, num_layers, initialization_method="orthogonal_"):
        super().__init__()

        self.args = args
        self.embedding = nn.Linear(obs_dim, embed_dim)

        self.layers = nn.ModuleList([
            CrossActionEncoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.embed_dim = embed_dim
        self.initialization_method = initialization_method
        
        for name, param in self.embedding.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                init_method = get_init_method(initialization_method)
                init_method(param)

        for name, param in self.layers.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and "norm" not in name:
                init_method = get_init_method(initialization_method)
                init_method(param)
    
    def init_hidden_state(self,b):
        return torch.zeros(b, self.embed_dim).cuda()

    def forward(self, obs, hidden_state=None, attn_mask=None):
        """
        obs: [B, N, obs_dim]
        """
        x = self.embedding(obs)

        if hidden_state is not None:
            x = torch.cat((x,hidden_state.view(hidden_state.shape[0],1,hidden_state.shape[-1])),1)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_w = layer(x, attn_mask)
            attn_weights_all.append(attn_w)

        x_out = x[:,0,:]

        if hidden_state is not None:
            h = x[:,-1,:]
            return x_out, h
        else:
            return x_out, attn_weights_all
    

    
if __name__=="__main__":

    # parser = argparse.ArgumentParser(description='Unit Testing')
    # parser.add_argument('--n_tgt_obs', default='5', type=int)

    # args = parser.parse_args()
    
    B, N, obs_dim = 32, 9, 4 # Batch, N-actions, act_obs)
    
    encoder = CrossActionEncoderPolicy(args=None,
        obs_dim=obs_dim,
        embed_dim=64,
        num_heads=1,
        num_layers=1
    )
    encoder.to(device='cuda:0')
    # hid = encoder.init_hidden_state(B)
    
    print(encoder)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(total_params)

    obs = torch.randn(B, N, obs_dim).cuda()
    agent_actions, h = encoder(obs)
    # agent_actions, h = encoder(obs, hid)
    print(agent_actions.shape)
    print(h.shape)