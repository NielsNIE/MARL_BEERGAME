import math
import torch
import torch.nn as nn

# ===== Tiny, unified NN utilities（统一的网络工具，去重） =====

def ortho_init(layer: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal init for Linear only.（线性层正交初始化）"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.)

def get_act(name: str) -> nn.Module:
    """Activation factory.（激活函数工厂）"""
    n = (name or "tanh").lower()
    if n == "relu": return nn.ReLU()
    if n == "silu": return nn.SiLU()
    if n == "gelu": return nn.GELU()
    return nn.Tanh()

def build_mlp(in_dim: int, hidden_sizes, act="tanh", layernorm=False, dropout=0.0):
    """
    Build a configurable MLP trunk.（可配置 MLP 主干）
    Returns (sequential, out_dim).
    """
    layers = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(get_act(act))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    return nn.Sequential(*layers), last