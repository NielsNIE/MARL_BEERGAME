from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from mappo_beergame.utils.nn import build_mlp, ortho_init, get_act

# ===== Single source of actor/critic（统一版本） =====

class BetaActor(nn.Module):
    """
    Beta(α,β) policy on (0,1) scaled to [0, A_max].
    - 可配置主干（hidden_sizes/act/layernorm/dropout）与 head_hidden
    """
    def __init__(self, obs_dim: int, act_dim: int, A_max: float,
                 hidden_sizes=(256,256,256), act="silu",
                 layernorm=False, dropout=0.0, head_hidden=64):
        super().__init__()
        self.A_max = float(A_max)
        self.backbone, feat_dim = build_mlp(
            obs_dim, list(hidden_sizes), act=act, layernorm=layernorm, dropout=dropout
        )
        if head_hidden and head_hidden > 0:
            self.alpha_head = nn.Sequential(
                nn.Linear(feat_dim, head_hidden), get_act(act),
                nn.Linear(head_hidden, act_dim)
            )
            self.beta_head  = nn.Sequential(
                nn.Linear(feat_dim, head_hidden), get_act(act),
                nn.Linear(head_hidden, act_dim)
            )
        else:
            self.alpha_head = nn.Linear(feat_dim, act_dim)
            self.beta_head  = nn.Linear(feat_dim, act_dim)

        # Init
        self.apply(lambda m: ortho_init(m, gain=math.sqrt(2)))
        ortho_init(self.alpha_head[-1] if isinstance(self.alpha_head, nn.Sequential) else self.alpha_head, 0.01)
        ortho_init(self.beta_head[-1]  if isinstance(self.beta_head,  nn.Sequential) else self.beta_head,  0.01)

    def _dist(self, obs: torch.Tensor) -> Beta:
        h = self.backbone(obs)
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta  = F.softplus(self.beta_head(h))  + 1.0
        return Beta(alpha, beta)

    @torch.no_grad()
    def sample_action(self, obs: torch.Tensor):
        dist = self._dist(obs)
        u = dist.sample()
        a = u * self.A_max
        logp = dist.log_prob(torch.clamp(u, 1e-6, 1-1e-6)).sum(dim=-1) - math.log(self.A_max)
        return a, logp

    @torch.no_grad()
    def mean_action(self, obs: torch.Tensor):
        dist = self._dist(obs)
        u_mean = dist.concentration1 / (dist.concentration1 + dist.concentration0)
        return u_mean * self.A_max

    def log_prob_of(self, obs: torch.Tensor, a: torch.Tensor):
        dist = self._dist(obs)
        u = torch.clamp(a / self.A_max, 1e-6, 1-1e-6)
        logp = dist.log_prob(u).sum(dim=-1) - math.log(self.A_max)
        ent = dist.entropy().sum(dim=-1)
        return logp, ent

class Critic(nn.Module):
    """Centralized value function V(joint_obs).（集中式价值函数）"""
    def __init__(self, joint_obs_dim: int,
                 hidden_sizes=(256,256,256), act="silu",
                 layernorm=False, dropout=0.0):
        super().__init__()
        self.backbone, feat_dim = build_mlp(
            joint_obs_dim, list(hidden_sizes), act=act, layernorm=layernorm, dropout=dropout
        )
        self.v = nn.Linear(feat_dim, 1)
        self.apply(lambda m: ortho_init(m, gain=math.sqrt(2)))
        ortho_init(self.v, gain=1.0)

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone(joint_obs)
        return self.v(h).squeeze(-1)