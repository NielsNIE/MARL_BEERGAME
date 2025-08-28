import os, numpy as np
from typing import List, Tuple
import torch

@torch.no_grad()
def dump_policy_portrait(env, agent, episodes=2, agent_idx=0,
                         max_points=5000, out_path=None):
    """
    评估时采样 (obs, action) → 保存 npz 供画图
    - 默认仅采零售商（agent_idx=0）；也可循环采三层分别另存
    - max_points: 采样上限，避免文件过大
    """
    samples = []
    count = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            obs_list, joint_obs = obs
            acts, _, _ = agent.act(obs_list, joint_obs, deterministic=True)
            st = env.states[agent_idx]
            feat = [
                float(st.I),
                float(st.B),
                float(np.sum(st.P_in)),
                float(st.P_out[0]),
                float(st.last_a),
                float(st.last_down_order),
                float(acts[agent_idx]),
            ]
            samples.append(feat)
            count += 1
            if count >= max_points:
                done = True
                break
            obs, _, done, _ = env.step(acts)

    arr = np.array(samples, dtype=np.float32)
    cols = np.array(["I","B","Pin_sum","Pout0","last_a","last_down","action"])
    if out_path is None:
        out_path = os.path.join("runs", "portrait_eval.npz")
    np.savez(out_path, data=arr, cols=cols)
    return out_path