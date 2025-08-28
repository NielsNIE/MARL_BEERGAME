from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mappo_beergame.core.networks import BetaActor, Critic

class MAPPO:
    def __init__(self,
                 obs_dim: int,
                 joint_obs_dim: int,
                 act_dim: int,
                 A_max: float,
                 device: str,
                 # Optim
                 actor_lr: float,
                 critic_lr: float,
                 grad_clip: float,
                 # PPO
                 clip_range: float,
                 value_coef: float,
                 entropy_coef: float,
                 value_clip_range: float,
                 target_kl: float | None,
                 normalize_adv: bool,
                 # Nets
                 net_cfg: Dict[str, Any],):
        
        self.device = torch.device(device)

        self.actor = BetaActor(
            obs_dim=obs_dim, act_dim=act_dim, A_max=A_max, **net_cfg
        ).to(self.device)
        self.critic = Critic(
            joint_obs_dim=joint_obs_dim, **{k: v for k, v in net_cfg.items() if k != "head_hidden"}
        ).to(self.device)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.grad_clip = grad_clip
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.value_clip_range = value_clip_range
        self.target_kl = target_kl
        self.normalize_adv = normalize_adv

    # ---------- Interaction ----------
    @torch.no_grad()
    def act(self, obs_local_list, joint_obs, deterministic=False):
        actions, logps = [], []
        j = torch.tensor(joint_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        value = self.critic(j).item()
        for obs in obs_local_list:
            o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic:
                a = self.actor.mean_action(o)
                lp, _ = self.actor.log_prob_of(o, a)
            else:
                a, lp = self.actor.sample_action(o)
            actions.append(a.squeeze(0).cpu().numpy().tolist()[0])
            logps.append(lp.item())
        return actions, logps, value

    # ---------- Learning ----------
    def update(self, buffer, epochs=10, batch_size=256, gamma=0.95, lam=0.95) -> Dict[str, float]:
        obs_local = np.array(buffer.obs_local, dtype=np.float32)   # [T, n_agents, obs_dim]
        actions   = np.array(buffer.actions,   dtype=np.float32)   # [T, n_agents, 1]
        obs_joint = np.array(buffer.obs_joint, dtype=np.float32)   # [T, joint_obs_dim]
        old_logps = np.array(buffer.logprobs,  dtype=np.float32)   # [T, n_agents]

        T, n_agents, obs_dim = obs_local.shape
        obs_local = obs_local.reshape(T * n_agents, obs_dim)
        actions   = actions.reshape(T * n_agents, 1)
        old_logps = old_logps.reshape(T * n_agents)

        # GAE on centralized value
        with torch.no_grad():
            last_joint = torch.tensor(obs_joint[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.critic(last_joint).item()
        adv, ret = buffer.compute_gae(last_value, gamma=gamma, lam=lam)

        adv_agents = np.repeat(adv, n_agents)
        ret_agents = np.repeat(ret, n_agents)  # kept for clarity if needed later

        obs_local_t = torch.tensor(obs_local, dtype=torch.float32, device=self.device)
        actions_t   = torch.tensor(actions,   dtype=torch.float32, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        adv_t       = torch.tensor(adv_agents, dtype=torch.float32, device=self.device)
        ret_t_full  = torch.tensor(ret, dtype=torch.float32, device=self.device)
        obs_joint_t = torch.tensor(obs_joint, dtype=torch.float32, device=self.device)

        if self.normalize_adv:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        with torch.no_grad():
            V_old = self.critic(obs_joint_t).detach()

        N = obs_local_t.size(0)
        idx = np.arange(N)

        info = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0,
                    approx_kl=0.0, clipfrac=0.0)
        total_batches = 0
        early_stop = False

        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, N, batch_size):
                b = idx[start:start + batch_size]
                o_b, a_b, adv_b, oldlp_b = obs_local_t[b], actions_t[b], adv_t[b], old_logps_t[b]

                # Actor
                logp, ent = self.actor.log_prob_of(o_b, a_b)
                ratio = torch.exp(logp - oldlp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                approx_kl = (oldlp_b - logp).mean().clamp(min=0.0)
                clipfrac  = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()

                self.optim_actor.zero_grad()
                (policy_loss - self.entropy_coef * ent.mean()).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.optim_actor.step()

                # Critic (fit whole trajectory for stability)
                V_pred = self.critic(obs_joint_t)
                if self.value_clip_range and self.value_clip_range > 0:
                    v_clip = V_old + torch.clamp(V_pred - V_old, -self.value_clip_range, self.value_clip_range)
                    loss_unclipped = (V_pred - ret_t_full).pow(2)
                    loss_clipped   = (v_clip - ret_t_full).pow(2)
                    value_loss = torch.max(loss_unclipped, loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(V_pred, ret_t_full)

                self.optim_critic.zero_grad()
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.optim_critic.step()

                # Logging
                info["policy_loss"] += float(policy_loss.item())
                info["value_loss"]  += float(value_loss.item())
                info["entropy"]     += float(ent.mean().item())
                info["approx_kl"]   += float(approx_kl.item())
                info["clipfrac"]    += float(clipfrac.item())
                total_batches += 1

                # ========== KL early stop ==========
                if (self.target_kl is not None) and (approx_kl.item() > 1.2 * self.target_kl):
                    early_stop = True
                    break
                        
            if early_stop:
                print(f"[INFO] Early stopping")
                break

        if total_batches > 0:
            for k in info:
                info[k] /= total_batches
        return info