from __future__ import annotations
from typing import List, Tuple
import numpy as np

class RolloutBuffer:
    """Lightweight rollout buffer for on-policy PPO.（轻量GAE缓存）"""
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs_local, self.obs_joint = [], []
        self.actions, self.logprobs = [], []
        self.rewards, self.dones, self.values = [], [], []

    def add(self, obs_local, obs_joint, actions, logprobs, reward, done, value):
        self.obs_local.append(np.stack(obs_local, axis=0))  # [n_agents, obs_dim]
        self.obs_joint.append(obs_joint.copy())
        self.actions.append(np.array(actions, dtype=np.float32).reshape(-1, 1))
        self.logprobs.append(np.array(logprobs, dtype=np.float32))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def compute_gae(self, last_value: float, gamma=0.95, lam=0.95):
        T = len(self.rewards)
        adv = np.zeros(T, dtype=np.float32)
        ret = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            done = self.dones[t]
            next_v = 0.0 if done else (last_value if t == T-1 else self.values[t+1])
            delta = self.rewards[t] + gamma * next_v - self.values[t]
            gae = delta + (0.0 if done else gamma * lam) * gae
            adv[t] = gae
            ret[t] = adv[t] + self.values[t]
        # We return raw adv/ret; normalization is handled by trainer if enabled.
        return adv, ret