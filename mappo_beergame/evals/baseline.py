from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch 

from mappo_beergame.envs.beer_game import BeerGame3Env

def compute_bullwhip_chain(order_seqs, demand_seq):
    """Bullwhip per level vs downstream.（逐级牛鞭效应）"""
    n_agents = len(order_seqs)
    bw = [0.0] * n_agents
    den0 = float(np.var(demand_seq) + 1e-8)
    bw[0] = float(np.var(order_seqs[0]) / den0)
    for i in range(1, n_agents):
        bw[i] = float(np.var(order_seqs[i]) / (np.var(order_seqs[i - 1]) + 1e-8))
    return bw

def compute_bull_overall(bw_chain):
    """Overall bullwhip: product of chain.（整体牛鞭效应）"""
    return float(np.prod(bw_chain)) if len(bw_chain) > 0 else 0.0

def order_up_to_action(S, I, P_sum, B):
    """Base-stock policy.（订至策略）"""
    return max(0.0, float(S) - (float(I) + float(P_sum) - float(B)))

def rollout_baseline_n_agents(env: BeerGame3Env, S_list, horizon=200):
    """Rollout the base-stock baseline.（基线策略试跑）"""
    assert len(S_list) == env.n_agents
    obs = env.reset()
    total_raw_reward = 0.0
    orders_seq = [[] for _ in range(env.n_agents)]
    demand_seq = []
    ship_seq = []

    # --- service metrics accumulators ---
    steps = 0
    dem_sum = 0.0
    shipped_sum = 0.0
    backlog_sum = 0.0
    inventory_sum = 0.0

    for _ in range(horizon):
        actions = []
        for i, st in enumerate(env.states):
            a = order_up_to_action(S_list[i], st.I, float(np.sum(st.P_in)), st.B)
            actions.append(a)
        obs, _, done, info = env.step(actions)
        total_raw_reward += info["raw_reward"]
        for i in range(env.n_agents):
            orders_seq[i].append(actions[i])
        demand_seq.append(info["ext_demand"])
        ship_seq.append(info["ship_downstream"][0])   # <<< 新增

        # --- accumulate service stats ---
        steps += 1
        dem_sum += float(info["ext_demand"])
        shipped_sum += float(info["ship_downstream"][0])
        backlog_sum += float(sum(info["B_list"]))
        inventory_sum += float(sum(info["I_list"]))

        if done:
            obs = env.reset()

    bw = compute_bullwhip_chain(orders_seq, demand_seq)
    bw_overall = compute_bull_overall(bw)
    avg_cost_per_step = - total_raw_reward / horizon
    fr_sla_k = compute_fill_rate_sla_k(demand_seq, ship_seq, k=env.Ls)   # <<< 新增
    svc = {
        "fill_rate": shipped_sum / (dem_sum + 1e-8),
        "fill_rate_sla_k": fr_sla_k,   # <<< 新增
        "avg_backlog": backlog_sum / (steps * env.n_agents),
        "avg_inventory": inventory_sum / (steps * env.n_agents),
    }
    return bw, bw_overall, avg_cost_per_step, svc

def _z_from_service_level(service: float) -> float:
    """Service level -> z score (approx).（服务水平到z值）"""
    table = {
        0.80: 0.8416, 0.85: 1.0364, 0.90: 1.2816, 0.95: 1.6449,
        0.975: 1.9600, 0.99: 2.3263, 0.995: 2.5758,
    }
    key = min(table.keys(), key=lambda k: abs(k - service))
    return table[key]

def estimate_external_demand_samples(A_max, demand_lambda, horizon, seed, steps=3000, L=1):
    """Sample external demand to estimate μ/σ.（采样外部需求用于估计均值方差）"""
    tmp_env = BeerGame3Env(n_agents=3, Ls=L, A_max=A_max,
                           demand_lambda=demand_lambda, horizon=horizon, seed=seed)
    obs = tmp_env.reset()
    samples = []
    for _ in range(steps):
        obs, _, done, info = tmp_env.step([0.0, 0.0, 0.0])
        samples.append(info["ext_demand"])
        if done:
            obs = tmp_env.reset()
    return np.array(samples, dtype=np.float32)

def compute_order_up_to_S_from_samples(demand_samples, L=1, service_level=0.95):
    """Base-stock formula: S = μ*(L+1) + z*σ*sqrt(L+1).（订至水平公式）"""
    mu = float(np.mean(demand_samples))
    sigma = float(np.std(demand_samples))
    z = _z_from_service_level(service_level)
    Lp1 = L + 1
    S = mu * Lp1 + z * sigma * (Lp1 ** 0.5)
    return S, {"mu": mu, "sigma": sigma, "z": z, "L": L, "service_level": service_level}

@torch.no_grad()
def evaluate_policy(env: BeerGame3Env, agent, episodes=5):
    """Deterministic eval for learned policy.（确定性评估）"""
    total_cost = 0.0
    order_seqs = [[] for _ in range(env.n_agents)]
    demand_seq = []
    ship_seq = []

    # --- service metrics accumulators ---
    steps = 0
    dem_sum = 0.0
    shipped_sum = 0.0      # shipments to customer = info["ship_downstream"][0]
    backlog_sum = 0.0      # sum over all agents
    inventory_sum = 0.0    # sum over all agents

    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            (obs_list, joint_obs) = obs
            acts, _, _ = agent.act(obs_list, joint_obs, deterministic=True)
            obs, _, done, info = env.step(acts)

            total_cost += -info["raw_reward"]
            for i in range(env.n_agents):
                order_seqs[i].append(acts[i])
            demand_seq.append(info["ext_demand"])

            # --- accumulate service stats ---
            steps += 1
            dem_sum += float(info["ext_demand"])
            shipped_sum += float(info["ship_downstream"][0])
            backlog_sum += float(sum(info["B_list"]))
            inventory_sum += float(sum(info["I_list"]))
            ship_seq.append(info["ship_downstream"][0])  # <<< 新增

    bw = compute_bullwhip_chain(order_seqs, demand_seq)
    bw_overall = compute_bull_overall(bw)
    avg_cost_per_step = total_cost / (episodes * env.horizon)

    # --- finalize service metrics ---
    fr_sla_k = compute_fill_rate_sla_k(demand_seq, ship_seq, k=env.Ls)   # <<< 新增
    svc = {
        "fill_rate": shipped_sum / (dem_sum + 1e-8),
        "fill_rate_sla_k": fr_sla_k,   # <<< 新增
        "avg_backlog": backlog_sum / (steps * env.n_agents),
        "avg_inventory": inventory_sum / (steps * env.n_agents),
    }
    return bw, bw_overall, avg_cost_per_step, svc

def rollout_mean_baseline(env: BeerGame3Env, mean_q: float, horizon: int = 128):
    """
    Constant-mean ordering baseline.（恒定均值下单基线）
    """
    obs = env.reset()
    total_raw_reward = 0.0
    orders_seq = [[] for _ in range(env.n_agents)]
    demand_seq = []
    ship_seq = []   # <<< 新增

    # --- service metrics accumulators ---
    steps = 0
    dem_sum = 0.0
    shipped_sum = 0.0
    backlog_sum = 0.0
    inventory_sum = 0.0

    for _ in range(horizon):
        actions = [float(mean_q)] * env.n_agents
        obs, _, done, info = env.step(actions)
        total_raw_reward += info["raw_reward"]
        for i in range(env.n_agents):
            orders_seq[i].append(actions[i])
        demand_seq.append(info["ext_demand"])
        ship_seq.append(info["ship_downstream"][0])   # <<< 新增

        # --- accumulate service stats ---
        steps += 1
        dem_sum += float(info["ext_demand"])
        shipped_sum += float(info["ship_downstream"][0])
        backlog_sum += float(sum(info["B_list"]))
        inventory_sum += float(sum(info["I_list"]))

        if done:
            obs = env.reset()

    bw = compute_bullwhip_chain(orders_seq, demand_seq)
    bw_overall = compute_bull_overall(bw)
    avg_cost_per_step = - total_raw_reward / horizon
    fr_sla_k = compute_fill_rate_sla_k(demand_seq, ship_seq, k=env.Ls)   # <<< 新增
    svc = {
        "fill_rate": shipped_sum / (dem_sum + 1e-8),
        "fill_rate_sla_k": fr_sla_k,   # <<< 新增
        "avg_backlog": backlog_sum / (steps * env.n_agents),
        "avg_inventory": inventory_sum / (steps * env.n_agents),
    }
    return bw, bw_overall, avg_cost_per_step, svc

from collections import deque

def compute_fill_rate_sla_k(demand_seq, ship_seq, k: int) -> float:
    """
    SLA-k 按期满足率：需求在生成后的 k 步内被发货的比例（含当期，Δt < k）。
    demand_seq[t]: t期外部需求
    ship_seq[t]:   t期实际对顾客发货（零售商向下游）
    """
    if k is None or k <= 0:
        k = 1
    T = min(len(demand_seq), len(ship_seq))
    total_dem = float(np.sum(demand_seq[:T]))
    if total_dem <= 0:
        return 1.0

    q = deque()  # 元素: (remaining_qty, t_birth)
    served_in_k = 0.0

    for t in range(T):
        d = float(demand_seq[t])
        if d > 0:
            q.append([d, t])
        s = float(ship_seq[t])

        while s > 0 and q:
            qty, tb = q[0]
            take = min(qty, s)
            # 在 k 步内(含当期)：age = t - tb < k
            if (t - tb) < k:
                served_in_k += take
            qty -= take
            s   -= take
            if qty <= 1e-12:
                q.popleft()
            else:
                q[0][0] = qty

    return served_in_k / (total_dem + 1e-8)