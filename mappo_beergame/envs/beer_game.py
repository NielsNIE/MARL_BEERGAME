from __future__ import annotations
from typing import Tuple, List
import random
import numpy as np

class AgentState:
    __slots__ = ("I", "B", "P_in", "P_out", "last_a", "last_down_order")
    def __init__(self, I=10.0, B=0.0, Ls=1):
        self.I = float(I)
        self.B = float(B)
        self.P_in = np.zeros(Ls, dtype=np.float32)
        self.P_out = np.zeros(Ls, dtype=np.float32)
        self.last_a = 0.0
        self.last_down_order = 0.0

class BeerGame3Env:
    """
    3 agents in a line: Retailer(0) <- Wholesaler(1) <- Factory(2).
    Shared reward = negative cost.（共享奖励=成本的负值）
    """
    def __init__(self,
                 n_agents: int = 3,
                 Ls: int = 1,
                 A_max: float = 30.0,
                 h: float = 3.0,
                 p: float = 5.0,
                 lam: float = 0.5,
                 demand_lambda: float = 8.0,
                 horizon: int = 52,
                 seed: int = 42):
        assert n_agents == 3, "This demo targets 3 agents."
        self.n_agents = n_agents
        self.Ls = int(Ls)
        self.A_max = float(A_max)
        self.h, self.p, self.lam = float(h), float(p), float(lam)
        self.demand_lambda = float(demand_lambda)
        self.horizon = int(horizon)

        self.rng = np.random.RandomState(seed)
        random.seed(seed); np.random.seed(seed)

        self.obs_dim = 6   # [I,B,P_in0,P_out0,last_a,last_down_order]
        self.act_dim = 1
        self.reset()

    def reset(self):
        self.t = 0
        self.states = [AgentState(I=10.0, B=0.0, Ls=self.Ls) for _ in range(self.n_agents)]
        for st in self.states:
            st.P_in[:] = 0.0; st.P_out[:] = 0.0
            st.last_a = 0.0; st.last_down_order = 0.0
        return self._build_obs()

    def step(self, actions: List[float]):
        # 保险裁剪
        actions = [float(np.clip(a, 0.0, self.A_max)) for a in actions]
        n = self.n_agents

        # 1) Arrivals from pipeline
        arrivals_in = [self.states[i].P_in[0] for i in range(n)]

        # 2) Downstream orders
        downstream_orders = [0.0] * n
        downstream_orders[0] = self._external_demand()
        for i in range(1, n):
            downstream_orders[i] = actions[i - 1]

        # 3) Fulfill demand -> update I/B
        shipments_down = [0.0] * n
        for i in range(n):
            st = self.states[i]
            supply = st.I + arrivals_in[i]
            need = downstream_orders[i] + st.B
            give = min(supply, need)
            st.I = max(0.0, supply - need)
            st.B = max(0.0, need - supply)
            shipments_down[i] = give

        # 4) Pipeline shift
        for i in range(n):
            self.states[i].P_in = np.roll(self.states[i].P_in, -1);  self.states[i].P_in[-1]  = 0.0
            self.states[i].P_out = np.roll(self.states[i].P_out, -1); self.states[i].P_out[-1] = 0.0

        # Shipments move
        for i in range(1, n):
            self.states[i].P_out[-1] += shipments_down[i]
            self.states[i - 1].P_in[-1] += shipments_down[i]
        # Retailer to customer (for observation)
        self.states[0].P_out[-1] += shipments_down[0]
        # Factory orders from infinite upstream
        self.states[n - 1].P_in[-1] += actions[n - 1]

        # 5) Cost / reward（分解：holding / backlog / smoothing）
        hold_cost_sum   = 0.0
        backlog_cost_sum = 0.0
        smooth_cost_sum  = 0.0
        total_cost       = 0.0

        for i in range(n):
            st = self.states[i]
            a = actions[i]
            hold_i   = self.h * st.I
            back_i   = self.p * st.B
            smooth_i = self.lam * (a - st.last_a) ** 2

            cost_i = hold_i + back_i + smooth_i
            total_cost += cost_i
            hold_cost_sum += hold_i
            backlog_cost_sum += back_i
            smooth_cost_sum += smooth_i

            # 维护 last_*（放在成本之后，使用的是上一步动作）
            st.last_a = a
            st.last_down_order = downstream_orders[i]

        reward = -total_cost
        reward_scaled = reward * 0.01  # 与现有训练的一致缩放

        self.t += 1
        done = (self.t >= self.horizon)

        obs = self._build_obs()
        info = {
            "raw_reward": reward,
            "ext_demand": downstream_orders[0],
            "actions": actions,
            "ship_downstream": shipments_down,
            "arrivals_in": arrivals_in,
            "I_list": [self.states[i].I for i in range(n)],
            "B_list": [self.states[i].B for i in range(n)],
            # === 新增：三类成本（每步总和）===
            "cost_hold": float(hold_cost_sum),
            "cost_backlog": float(backlog_cost_sum),
            "cost_smooth": float(smooth_cost_sum),
        }
        return obs, [reward_scaled] * n, done, info

    def _build_obs(self):
        obs_list = []
        for st in self.states:
            obs = np.array([st.I, st.B, st.P_in[0], st.P_out[0], st.last_a, st.last_down_order], dtype=np.float32)
            obs_list.append(obs)
        joint_obs = np.concatenate(obs_list, axis=0).astype(np.float32)
        return obs_list, joint_obs

    def _external_demand(self) -> float:
        val = self.rng.poisson(self.demand_lambda)
        return float(np.clip(val, 0, int(3 * self.demand_lambda)))