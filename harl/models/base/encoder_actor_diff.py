import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method
from harl.utils.diff_transf_tools import apply_rotary_emb, RMSNorm
# from harl.models.base.act import ACTLayer
# import argparse
import torch.nn.functional as F
import math

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        x,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn, attn_weights

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
    def __init__(self, embed_dim, num_heads, num_kv_heads=None):
        super().__init__()
        # self.attn = nn.MultiheadAttention(
        #     embed_dim=embed_dim,
        #     num_heads=num_heads,
        #     batch_first=True
        # )
        self.attn = MultiheadDiffAttn(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            depth=1,
            # batch_first=True
        )

    def forward(self, x, attn_mask=None):
        """
        x: [B, N, D]  (batch, agents, embedding)
        """
        # out, attn_weights = self.attn(
        #     query=x,
        #     key=x,
        #     value=x,
        #     attn_mask=attn_mask
        # )
        out, attn_weights = self.attn(
            x,
            attn_mask=attn_mask
        )
        return out, attn_weights
    
class CrossActionEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, mlp_ratio=4):
        super().__init__()

        self.attn = CrossActionAttention(embed_dim, num_heads, num_kv_heads)
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
    def __init__(self, args, obs_dim, embed_dim, num_heads, num_layers, num_kv_heads=None, initialization_method="orthogonal_"):
        super().__init__()

        self.args = args
        self.embedding = nn.Linear(obs_dim, embed_dim)

        self.layers = nn.ModuleList([
            CrossActionEncoderBlock(embed_dim, num_heads, num_kv_heads)
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
            # elif "weight" in name and "norm" not in name and "attn" not in name:
            elif "weight" in name and "norm" not in name and "subln" not in name:
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
        num_heads=4,
        num_kv_heads=2,
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