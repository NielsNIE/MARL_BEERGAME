# mappo_beergame/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class EnvCfg:
    """Environment configuration.（环境配置）"""
    n_agents: int = 3
    L: int = 1                 # lead time（提前期）
    A_max: float = 30.0
    h: float = 3.0             # holding cost（持有成本）
    p: float = 5.0             # backlog penalty（欠货罚金）
    lam: float = 0.5           # smoothing penalty（平滑项）
    demand_lambda: float = 8.0
    horizon: int = 128

@dataclass
class TrainCfg:
    """Training loop configuration.（训练流程配置）"""
    episodes: int = 400
    snapshot_every: int = 50
    gamma: float = 0.95
    lam: float = 0.95
    epochs: int = 5
    batch_size: int = 128

@dataclass
class OptimCfg:
    """Optimizer configuration.（优化器参数配置）"""
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    grad_clip: float = 0.5

@dataclass
class PPOCfg:
    """PPO losses & tricks.（PPO 超参与技巧）"""
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    value_clip_range: float = 0.2
    target_kl: float | None = 0.02
    normalize_adv: bool = False

@dataclass
class NetCfg:
    """Network config for actor/critic.（网络结构配置）"""
    act: str = "silu"                         # tanh/relu/silu/gelu
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256, 256])
    layernorm: bool = False
    dropout: float = 0.0
    head_hidden: int = 64                    # extra hidden for actor heads; 0 disables（仅Actor使用）

@dataclass
class BaselineCfg:
    """Baseline evaluation config.（基线评估配置）"""
    service_level: float = 0.95
    demand_sample_steps: int = 3000
    eval_episodes: int = 5

@dataclass
class GlobalCfg:
    """Top-level configuration holder.（全局配置容器）"""
    seed: int = 42
    device: str = "cpu"
    env: EnvCfg = field(default_factory=EnvCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    net: NetCfg = field(default_factory=NetCfg)
    baseline: BaselineCfg = field(default_factory=BaselineCfg)