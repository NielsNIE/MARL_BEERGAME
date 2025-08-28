from __future__ import annotations
import os, time, json, argparse
import yaml
import numpy as np
import torch
import argparse

from mappo_beergame.config import GlobalCfg, EnvCfg, TrainCfg, OptimCfg, PPOCfg, NetCfg, BaselineCfg
from mappo_beergame.envs.beer_game import BeerGame3Env
from mappo_beergame.core.buffer import RolloutBuffer
from mappo_beergame.core.mappo import MAPPO
from mappo_beergame.evals.baseline import (
    evaluate_policy, estimate_external_demand_samples,
    compute_order_up_to_S_from_samples, rollout_baseline_n_agents,
    compute_bullwhip_chain, compute_bull_overall, rollout_mean_baseline
)
from mappo_beergame.utils.io import CSVLogger, save_snapshot, save_summary
from mappo_beergame.utils.plotting import plot_learning_curves, plot_bullwhip_from_snapshots, plot_baselines_panel

def load_cfg(path: str) -> GlobalCfg:

    def as_float(x, default):
        try:
            return float(x)
        except Exception:
            return default

    def as_int(x, default):
        try:
            return int(x)
        except Exception:
            return default

    def as_bool(x, default):
        # Accept bools, "true"/"false"/"1"/"0"
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("true", "1", "yes", "y", "on"):  return True
            if s in ("false", "0", "no", "n", "off"): return False
        return default

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    env_raw = raw.get("env", {}) or {}
    train_raw = raw.get("train", {}) or {}
    optim_raw = raw.get("optim", {}) or {}
    ppo_raw = raw.get("ppo", {}) or {}
    net_raw = raw.get("net", {}) or {}
    base_raw = raw.get("baseline", {}) or {}

    # —— Env
    env = EnvCfg(
        n_agents=as_int(env_raw.get("n_agents", 3), 3),
        L=as_int(env_raw.get("L", 1), 1),
        A_max=as_float(env_raw.get("A_max", 30.0), 30.0),
        h=as_float(env_raw.get("h", 3.0), 3.0),
        p=as_float(env_raw.get("p", 5.0), 5.0),
        lam=as_float(env_raw.get("lam", 0.5), 0.5),
        demand_lambda=as_float(env_raw.get("demand_lambda", 8.0), 8.0),
        horizon=as_int(env_raw.get("horizon", 128), 128),
    )

    # —— Train
    train = TrainCfg(
        episodes=as_int(train_raw.get("episodes", 400), 400),
        snapshot_every=as_int(train_raw.get("snapshot_every", 50), 50),
        gamma=as_float(train_raw.get("gamma", 0.95), 0.95),
        lam=as_float(train_raw.get("lam", 0.95), 0.95),
        epochs=as_int(train_raw.get("epochs", 5), 5),
        batch_size=as_int(train_raw.get("batch_size", 128), 128),
    )

    # —— Optim
    optim = OptimCfg(
        actor_lr=as_float(optim_raw.get("actor_lr", 5e-4), 5e-4),
        critic_lr=as_float(optim_raw.get("critic_lr", 5e-4), 5e-4),
        grad_clip=as_float(optim_raw.get("grad_clip", 0.5), 0.5),
    )

    # —— PPO
    ppo = PPOCfg(
        clip_range=as_float(ppo_raw.get("clip_range", 0.2), 0.2),
        value_coef=as_float(ppo_raw.get("value_coef", 0.5), 0.5),
        entropy_coef=as_float(ppo_raw.get("entropy_coef", 0.01), 0.01),
        value_clip_range=as_float(ppo_raw.get("value_clip_range", 0.2), 0.2),
        target_kl=as_float(ppo_raw.get("target_kl", 0.02), 0.02),
        normalize_adv=as_bool(ppo_raw.get("normalize_adv", False), False),
    )

    # —— Net
    hs = net_raw.get("hidden_sizes", [256, 256, 256]) or [256, 256, 256]
    # 强制把 hidden_sizes 转成 int 列表
    try:
        hs = [int(x) for x in hs]
    except Exception:
        hs = [256, 256, 256]

    net = NetCfg(
        act=str(net_raw.get("act", "silu")),
        hidden_sizes=hs,
        layernorm=as_bool(net_raw.get("layernorm", False), False),
        dropout=as_float(net_raw.get("dropout", 0.0), 0.0),
        head_hidden=as_int(net_raw.get("head_hidden", 64), 64),
    )

    # —— Baseline
    baseline = BaselineCfg(
        service_level=as_float(base_raw.get("service_level", 0.95), 0.95),
        demand_sample_steps=as_int(base_raw.get("demand_sample_steps", 3000), 3000),
        eval_episodes=as_int(base_raw.get("eval_episodes", 5), 5),
    )

    return GlobalCfg(
        seed=as_int(raw.get("seed", 42), 42),
        device=str(raw.get("device", "cpu")),
        env=env, train=train, optim=optim, ppo=ppo, net=net, baseline=baseline
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    
    # 在读取 cfg 后、构建 agent 前做一次校验
    assert 0.1 <= cfg.ppo.clip_range <= 0.2, \
        f"ppo.clip_range={cfg.ppo.clip_range} is out of recommended [0.1, 0.2]"
    assert 0.0 < cfg.ppo.target_kl <= 0.1, \
        f"ppo.target_kl={cfg.ppo.target_kl} looks too large; try ~0.01–0.02"

    # ===== Env & Agent =====
    env = BeerGame3Env(
        n_agents=cfg.env.n_agents, Ls=cfg.env.L, A_max=cfg.env.A_max,
        h=cfg.env.h, p=cfg.env.p, lam=cfg.env.lam,
        demand_lambda=cfg.env.demand_lambda, horizon=cfg.env.horizon, seed=cfg.seed
    )
    obs_dim = env.obs_dim
    joint_dim = env.obs_dim * env.n_agents

    agent = MAPPO(
        obs_dim=obs_dim, joint_obs_dim=joint_dim, act_dim=1, A_max=cfg.env.A_max,
        device=cfg.device,
        actor_lr=cfg.optim.actor_lr, critic_lr=cfg.optim.critic_lr, grad_clip=cfg.optim.grad_clip,
        clip_range=cfg.ppo.clip_range, value_coef=cfg.ppo.value_coef, entropy_coef=cfg.ppo.entropy_coef,
        value_clip_range=cfg.ppo.value_clip_range, target_kl=cfg.ppo.target_kl,
        normalize_adv=cfg.ppo.normalize_adv,
        net_cfg=dict(
            hidden_sizes=cfg.net.hidden_sizes, act=cfg.net.act,
            layernorm=cfg.net.layernorm, dropout=cfg.net.dropout, head_hidden=cfg.net.head_hidden
        ),
    )

    # ===== Output dirs =====
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"beer3_mappo_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger = CSVLogger(
        run_dir,
        extra_fields=["avg_hold_cost", "avg_backlog_cost", "avg_smooth_cost", "avg_total_cost"]
    )

    print(f"[INFO] Start training: episodes={cfg.train.episodes}, horizon={cfg.env.horizon}, "
          f"L={cfg.env.L}, A_max={cfg.env.A_max}, demand_lambda={cfg.env.demand_lambda}")

    # ===== Training loop =====
    for ep in range(1, cfg.train.episodes + 1):
        buf = RolloutBuffer()
        obs = env.reset()
        ep_return_raw = 0.0
        # episode 级别累计器
        ep_hold = 0.0
        ep_backlog = 0.0
        ep_smooth = 0.0

        seq_orders = [[] for _ in range(env.n_agents)]
        seq_demand = []

        done = False
        while not done:
            (obs_list, joint_obs) = obs
            actions, logps, value = agent.act(obs_list, joint_obs, deterministic=False)
            next_obs, rewards, done, info = env.step(actions)

            r = rewards[0]  # shared, scaled
            buf.add(obs_list, joint_obs, actions, logps, r, done, value)
            ep_return_raw += info["raw_reward"]
            
            ep_hold   += info.get("cost_hold", 0.0)
            ep_backlog+= info.get("cost_backlog", 0.0)
            ep_smooth += info.get("cost_smooth", 0.0)

            for i in range(env.n_agents):
                seq_orders[i].append(actions[i])
            seq_demand.append(info["ext_demand"])

            obs = next_obs

        upd = agent.update(
            buffer=buf, epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
            gamma=cfg.train.gamma, lam=cfg.train.lam
        )

        avg_return_raw = ep_return_raw / cfg.env.horizon
        avg_hold   = ep_hold    / cfg.env.horizon
        avg_back   = ep_backlog / cfg.env.horizon
        avg_smooth = ep_smooth  / cfg.env.horizon
        avg_total  = avg_hold + avg_back + avg_smooth  # 与 -avg_return_raw 接近（尺度差异来自 reward_scaled 的 0.01）

        logger.log(
            ep, avg_return_raw,
            extras={
                "avg_hold_cost":   avg_hold,
                "avg_backlog_cost":avg_back,
                "avg_smooth_cost": avg_smooth,
                "avg_total_cost":  avg_total,
            }
        )

        print(f"EP {ep:03d} | avg_return_raw={avg_return_raw:10.2f} | "
              f"avg_cost/step={-avg_return_raw:8.2f} | "
              f"kl={upd.get('approx_kl',0):.4f} | clipfrac={upd.get('clipfrac',0):.3f} | ")

        if ep % cfg.train.snapshot_every == 0:
            bw_chain = compute_bullwhip_chain(seq_orders, seq_demand)
            bw_overall = compute_bull_overall(bw_chain)
            snapshot = {
                "episode": int(ep),
                "avg_cost_per_step": float(-avg_return_raw),
                "bullwhip_chain": [float(x) for x in bw_chain],
                "bull_overall": float(bw_overall),
            }
            print(f"[SNAPSHOT @EP{ep:03d}] {json.dumps(snapshot, ensure_ascii=False)}")
            save_snapshot(run_dir, ep, snapshot)

    # ===== Final evaluation & baseline =====
    print("\n[INFO] Final evaluation (deterministic policy)")

    eval_env = BeerGame3Env(
        n_agents=cfg.env.n_agents, Ls=cfg.env.L, A_max=cfg.env.A_max,
        demand_lambda=cfg.env.demand_lambda, horizon=cfg.env.horizon, seed=321,
        h=cfg.env.h, p=cfg.env.p, lam=cfg.env.lam  # or smooth_penalty=cfg.env.smooth_penalty
    )
    bw_eval, bw_eval_overall, cost_eval, svc_rl = evaluate_policy(eval_env, agent, episodes=cfg.baseline.eval_episodes)
    bw_eval = [float(x) for x in bw_eval]; cost_eval = float(cost_eval)

    demand_samples = estimate_external_demand_samples(
        A_max=cfg.env.A_max, demand_lambda=cfg.env.demand_lambda,
        horizon=cfg.env.horizon, seed=999, steps=cfg.baseline.demand_sample_steps, L=cfg.env.L
    )
    S_auto, stats = compute_order_up_to_S_from_samples(
        demand_samples, L=cfg.env.L, service_level=cfg.baseline.service_level
    )
    S_list = [S_auto] * cfg.env.n_agents

    base_env = BeerGame3Env(
        n_agents=cfg.env.n_agents, Ls=cfg.env.L, A_max=cfg.env.A_max,
        demand_lambda=cfg.env.demand_lambda, horizon=cfg.env.horizon, seed=123,
        h=cfg.env.h, p=cfg.env.p, lam=cfg.env.lam
    )
    bw_base, bw_base_overall, cost_base, svc_base = rollout_baseline_n_agents(base_env, S_list, horizon=cfg.env.horizon)
    bw_base = [float(x) for x in bw_base]; cost_base = float(cost_base)
    
    # ===== Mean-order baseline (constant μ) =====
    # 使用前面 demand_samples 估计得到的 mu（与 BASELINE S 一致，更公平）
    mean_q = stats["mu"]

    mean_env = BeerGame3Env(
        n_agents=cfg.env.n_agents, Ls=cfg.env.L, A_max=cfg.env.A_max,
        demand_lambda=cfg.env.demand_lambda, horizon=cfg.env.horizon, seed=456,
        h=cfg.env.h, p=cfg.env.p, lam=cfg.env.lam
    )

    bw_mean, bw_mean_overall, cost_mean, svc_mean = rollout_mean_baseline(mean_env, mean_q, horizon=cfg.env.horizon)
    bw_mean = [float(x) for x in bw_mean]; cost_mean = float(cost_mean)
    
    print(f"[BASELINE S] service={stats['service_level']:.3f} | mu={stats['mu']:.3f} | "
        f"sigma={stats['sigma']:.3f} | z={stats['z']:.3f} | L={stats['L']} | S={S_auto:.3f}")
    print(f"[BASELINE] Bullwhip={bw_base} | Bull_overall={bw_base_overall:.3f} | AvgCost/step={cost_base:.2f} | "
        f"FillRate={svc_base['fill_rate']:.3f} | AvgB={svc_base['avg_backlog']:.3f} | AvgI={svc_base['avg_inventory']:.3f}")
    print(f"[LEARNED ] Bullwhip={bw_eval} | Bull_overall={bw_eval_overall:.3f} | AvgCost/step={cost_eval:.2f} | "
        f"FillRate={svc_rl['fill_rate']:.3f} | AvgB={svc_rl['avg_backlog']:.3f} | AvgI={svc_rl['avg_inventory']:.3f}")
    print(f"[MEAN   ] q={mean_q:.3f} | Bullwhip={bw_mean} | Bull_overall={bw_mean_overall:.3f} | AvgCost/step={cost_mean:.2f} | "
        f"FillRate={svc_mean['fill_rate']:.3f} | AvgB={svc_mean['avg_backlog']:.3f} | AvgI={svc_mean['avg_inventory']:.3f}")
    
    # ===== Scenario 对比图（同一需求轨迹）=====
    from mappo_beergame.utils.plotting import plot_scenario_comparison

    scenario_seed = 13579  # 固定种子

    def rollout_series(policy_kind: str):
        env_scn = BeerGame3Env(
            n_agents=cfg.env.n_agents, Ls=cfg.env.L, A_max=cfg.env.A_max,
            demand_lambda=cfg.env.demand_lambda, horizon=cfg.env.horizon, seed=scenario_seed,
            h=cfg.env.h, p=cfg.env.p, lam=cfg.env.lam
        )
        obs = env_scn.reset()
        I, B, A, D = [], [], [], []   # ← 新增 D
        done = False

        def base_stock_action(st, S_level):
            ip = float(st.I) + float(np.sum(st.P_in)) - float(st.B)
            a = max(0.0, S_level - ip)
            return float(np.clip(a, 0.0, cfg.env.A_max))

        mean_q_local = stats["mu"]

        while not done:
            obs_list, joint_obs = obs
            st0 = env_scn.states[0]
            I.append(float(st0.I))
            B.append(float(st0.B))

            if policy_kind == "RL":
                acts, _, _ = agent.act(obs_list, joint_obs, deterministic=True)
            elif policy_kind == "Baseline":
                acts = []
                for i, st in enumerate(env_scn.states):
                    acts.append(base_stock_action(st, S_list[i]))
            elif policy_kind == "Mean":
                acts = [float(np.clip(mean_q_local, 0.0, cfg.env.A_max)) for _ in range(env_scn.n_agents)]
            else:
                raise ValueError(policy_kind)

            A.append(float(acts[0]))
            obs, _, done, info = env_scn.step(acts)

            # 记录外部需求
            D.append(float(info.get("ext_demand", np.nan)))   # ← 新增这一行

        return {"I": I, "B": B, "A": A, "D": D}   # ← 返回包含 D

    series = {
        "Baseline": rollout_series("Baseline"),
        "Mean":     rollout_series("Mean"),
        "RL":       rollout_series("RL"),
    }
    plot_scenario_comparison(run_dir, series, agent_idx=0)
    
    # ===== SLA-k 曲线：同一需求种子下，RL / Baseline / Mean 的 FillRate@k 随 k 变化 =====
    from mappo_beergame.utils.plotting import plot_slak_curves
    from mappo_beergame.evals.baseline import compute_fill_rate_sla_k  

    scenario_seed = 24680  # 固定种子，确保三策略面对相同需求
    K_max = min(20, cfg.env.horizon)  # 取一个不大的上限，用于画曲线

    def rollout_dem_ship(policy_kind: str):
        env_scn = BeerGame3Env(
            n_agents=cfg.env.n_agents, Ls=cfg.env.L, A_max=cfg.env.A_max,
            demand_lambda=cfg.env.demand_lambda, horizon=cfg.env.horizon, seed=scenario_seed,
            h=cfg.env.h, p=cfg.env.p, lam=cfg.env.lam
        )
        obs = env_scn.reset()
        dem, ship = [], []
        done = False

        def base_stock_action(st, S_level):
            ip = float(st.I) + float(np.sum(st.P_in)) - float(st.B)
            a = max(0.0, S_level - ip)
            return float(np.clip(a, 0.0, cfg.env.A_max))

        mean_q_local = stats["mu"]

        while not done:
            obs_list, joint_obs = obs
            if policy_kind == "RL":
                acts, _, _ = agent.act(obs_list, joint_obs, deterministic=True)
            elif policy_kind == "Baseline":
                acts = [base_stock_action(st, S_list[i]) for i, st in enumerate(env_scn.states)]
            elif policy_kind == "Mean":
                acts = [float(np.clip(mean_q_local, 0.0, cfg.env.A_max)) for _ in range(env_scn.n_agents)]
            else:
                raise ValueError(policy_kind)

            obs, _, done, info = env_scn.step(acts)
            dem.append(float(info["ext_demand"]))
            ship.append(float(info["ship_downstream"][0]))  # 零售商对顾客发货
        return dem, ship

    dem_rl,  ship_rl  = rollout_dem_ship("RL")
    dem_b,   ship_b   = rollout_dem_ship("Baseline")
    dem_mean,ship_mean= rollout_dem_ship("Mean")

    k_vals = list(range(1, K_max + 1))
    curves = {
        "Baseline": {"k": k_vals, "sla": [compute_fill_rate_sla_k(dem_b,    ship_b,    k) for k in k_vals]},
        "Mean":     {"k": k_vals, "sla": [compute_fill_rate_sla_k(dem_mean, ship_mean, k) for k in k_vals]},
        "RL":       {"k": k_vals, "sla": [compute_fill_rate_sla_k(dem_rl,   ship_rl,   k) for k in k_vals]},
    }
    plot_slak_curves(run_dir, curves)
        
    # ===== Plots =====
    from mappo_beergame.utils.plotting import plot_learning_curves, plot_bullwhip_from_snapshots
    plot_learning_curves(run_dir)
    plot_bullwhip_from_snapshots(run_dir, n_agents=cfg.env.n_agents)
    
    from mappo_beergame.utils.plotting import (
        plot_cost_breakdown, plot_baselines_panel, plot_policy_portrait
    )
    from mappo_beergame.evals.portrait import dump_policy_portrait

    # 成本分解曲线
    plot_cost_breakdown(run_dir)

    # 动作-状态画像（以零售商为例）
    portrait_npz = dump_policy_portrait(
        BeerGame3Env(n_agents=3, Ls=cfg.env.L, A_max=cfg.env.A_max,
                    demand_lambda=cfg.env.demand_lambda, horizon=cfg.env.horizon, seed=888),
        agent, episodes=2, agent_idx=0, max_points=4000,
        out_path=os.path.join(run_dir, "portrait_eval.npz")
    )
    plot_policy_portrait(portrait_npz, out_path=os.path.join(run_dir, "plots", "portrait_B.png"), x_feature="B")
    # 也可以再画 I / Pin_sum
    # plot_policy_portrait(portrait_npz, out_path=os.path.join(run_dir, "plots", "portrait_I.png"), x_feature="I")
    # plot_policy_portrait(portrait_npz, out_path=os.path.join(run_dir, "plots", "portrait_Pin_sum.png"), x_feature="Pin_sum")

    # ===== Save weights & summary =====
    weights_dir = os.path.join(run_dir, "weights"); os.makedirs(weights_dir, exist_ok=True)
    torch.save(agent.actor.state_dict(), os.path.join(weights_dir, "actor.pt"))
    torch.save(agent.critic.state_dict(), os.path.join(weights_dir, "critic.pt"))
    print(f"[INFO] Saved weights to: {weights_dir}")

    summary = {
        "config": {
            "episodes": int(cfg.train.episodes),
            "horizon": int(cfg.env.horizon),
            "lead_time": int(cfg.env.L),
            "A_max": float(cfg.env.A_max),
            "demand_lambda": float(cfg.env.demand_lambda),
            "device": str(cfg.device),
            "snapshot_every": int(cfg.train.snapshot_every),
            "baseline_service_level": float(cfg.baseline.service_level),
        },
        "baseline": {
            "S_per_level": [float(x) for x in S_list],
            "S_stats": {k: (float(v) if isinstance(v, (int,float)) else v) for k, v in stats.items()}
        },
        "final_eval": {
            "RL": {
                "bullwhip_chain": bw_eval,
                "bull_overall": float(bw_eval_overall),
                "avg_cost_per_step": float(cost_eval),
                "fill_rate": float(svc_rl["fill_rate"]),
                "fill_rate_sla_k": float(svc_rl.get("fill_rate_sla_k", np.nan)),
                "avg_backlog": float(svc_rl["avg_backlog"]),
                "avg_inventory": float(svc_rl["avg_inventory"]),
            },
            "Baseline": {
                "bullwhip_chain": bw_base,
                "bull_overall": float(bw_base_overall),
                "avg_cost_per_step": float(cost_base),
                "fill_rate": float(svc_base["fill_rate"]),
                "fill_rate_sla_k": float(svc_base.get("fill_rate_sla_k", np.nan)), 
                "avg_backlog": float(svc_base["avg_backlog"]),
                "avg_inventory": float(svc_base["avg_inventory"]),
            },
            "Mean": {
                "q_mean": float(mean_q),
                "bullwhip_chain": bw_mean,
                "bull_overall": float(bw_mean_overall),
                "avg_cost_per_step": float(cost_mean),
                "fill_rate": float(svc_mean["fill_rate"]),
                "fill_rate_sla_k": float(svc_mean.get("fill_rate_sla_k", np.nan)), 
                "avg_backlog": float(svc_mean["avg_backlog"]),
                "avg_inventory": float(svc_mean["avg_inventory"]),
            }
        },
        "artifacts": {
            "run_dir": run_dir,
            "csv_log": os.path.join(run_dir, "train_log.csv"),
            "plots": {
                "avg_return": os.path.join(run_dir, "plot_avg_return.png"),
                "avg_cost": os.path.join(run_dir, "plot_avg_cost.png"),
                "bullwhip": os.path.join(run_dir, "plot_bullwhip.png"),
            },
            "weights": {
                "actor": os.path.join(weights_dir, "actor.pt"),
                "critic": os.path.join(weights_dir, "critic.pt"),
            }
        }
    }
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))
    save_summary(run_dir, summary)
    plot_baselines_panel(run_dir)
    logger.close()

if __name__ == "__main__":
    main()