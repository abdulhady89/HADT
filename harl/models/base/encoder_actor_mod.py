import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method
# from harl.models.base.act import ACTLayer
# import argparse

class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class CrossActionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(
            embed_dim=embed_dim//2,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim//2,
            num_heads=num_heads,
            batch_first=True
        )
        # self.mix = nn.Linear(embed_dim//2, embed_dim)
        self.mixer = nn.Parameter(torch.zeros(embed_dim//2,embed_dim).normal_(mean=0,std=0.1))
        self.diff_lambda = nn.Parameter(torch.zeros(1).normal_(mean=0,std=0.1))

    def forward(self, x, attn_mask=None):
        """
        x: [B, N, D]  (batch, agents, embedding)
        """
        b,n,d = x.shape
        x1 = x[:,:,:d//2]
        x2 = x[:,:,d//2:]
        out1, attn_weights1 = self.attn1(
            query=x1,
            key=x1,
            value=x1,
            attn_mask=attn_mask
        )
        out2, attn_weights2 = self.attn2(
            query=x2,
            key=x2,
            value=x2,
            attn_mask=attn_mask
        )
        # diff_out = out1 - (self.diff_lambda + 0.8) * out2
        diff_out = out1 - self.diff_lambda * out2
        out = diff_out @ self.mixer
        attn_weights= torch.cat((attn_weights1,attn_weights2),dim=0)
        
        return out, attn_weights
    
class CrossActionEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
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
    

class CrossActionEncoderPolicy(nn.Module):
    def __init__(self, args, obs_dim, embed_dim, num_heads, num_layers,initialization_method="orthogonal_"):
        super().__init__()

        self.args = args
        self.embedding = nn.Linear(obs_dim, embed_dim)

        self.layers = nn.ModuleList([
            CrossActionEncoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.act_out = nn.Linear(embed_dim,1)
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

        for name, param in self.act_out.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                init_method = get_init_method(initialization_method)
                init_method(param)
        

    def forward(self, obs, hidden_state=None,available_actions=None, deterministic=False, attn_mask=None):
        """
        obs: [B, N, obs_dim]
        """
        
        x = self.embedding(obs)
        if hidden_state is not None:
            x = torch.cat((x,hidden_state.view(hidden_state.shape[0],1,hidden_state.shape[-1])),1)
            
        # else:
        #     x = self.embedding(obs)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_w = layer(x, attn_mask)
            attn_weights_all.append(attn_w)

        if hidden_state is not None:
            h = x[:,-1,:]

        x_out = self.act_out(x[:,:-1,:])
        x_out = x_out.view(x_out.shape[0],x_out.shape[1])
        if available_actions is not None:
            x_out[available_actions == 0] = -1e10
 
        action_distribution = FixedCategorical(logits=x_out)
        actions = (
                action_distribution.mode()
                if deterministic
                else action_distribution.sample()
            )
        action_log_probs = action_distribution.log_probs(actions)

        return actions, action_log_probs, h
    
    def evaluate_actions(self, obs, action, hidden_state=None, available_actions=None, active_masks=None, attn_mask=None):

        
        x = self.embedding(obs)
        if hidden_state is not None:
            x = torch.cat((x,hidden_state.view(hidden_state.shape[0],1,hidden_state.shape[-1])),1)
            
        # else:
        #     x = self.embedding(obs)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_w = layer(x, attn_mask)
            attn_weights_all.append(attn_w)

        if hidden_state is not None:
            h = x[:,-1,:]

        x_out = self.act_out(x[:,:-1,:])
        x_out = x_out.view(x_out.shape[0],x_out.shape[1])
        if available_actions is not None:
            x_out[available_actions == 0] = -1e10
 
        action_distribution = FixedCategorical(logits=x_out)
        
        action_log_probs = action_distribution.log_probs(action)
        if active_masks is not None:
            dist_entropy = (
                action_distribution.entropy() * active_masks.squeeze(-1)
            ).sum() / active_masks.sum()
        else:
            dist_entropy = action_distribution.entropy().mean()

        return action_log_probs, dist_entropy, action_distribution, h
    

    
if __name__=="__main__":

    # parser = argparse.ArgumentParser(description='Unit Testing')
    # parser.add_argument('--n_tgt_obs', default='5', type=int)

    # args = parser.parse_args()
    
    B, N, obs_dim = 32, 9, 4 # Batch, N-actions, act_obs)
    encoder = CrossActionEncoderPolicy(args=None,
        obs_dim=obs_dim,
        embed_dim=64,
        num_heads=16,
        num_layers=1
    )
    encoder.to(device='cuda:0')
    hid = torch.zeros(B, 64).cuda()
    print(encoder)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(total_params)

    obs = torch.randn(B, N, obs_dim).cuda()
    # agent_actions, attn_maps = encoder(obs)
    agent_actions, log_probs, hid = encoder(obs, hid)
    print(agent_actions.shape)
    # print(attn_maps)