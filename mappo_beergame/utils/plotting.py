import os, glob, json
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(run_dir: str):
    csv_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(csv_path):
        print("[WARN] No train_log.csv found for plotting.")
        return

    ep, ret, cost = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        for i, line in enumerate(lines):
            if i == 0:  # header
                continue
            cols = line.split(",")
            ep.append(int(cols[0])); ret.append(float(cols[1])); cost.append(float(cols[2]))

    plt.figure()
    plt.plot(ep, ret)
    plt.xlabel("Episode"); plt.ylabel("Avg Raw Return (per step)")
    plt.title("Training Curve: Avg Return"); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "plot_avg_return.png"), dpi=160); plt.close()

    plt.figure()
    plt.plot(ep, cost)
    plt.xlabel("Episode"); plt.ylabel("Avg Cost (per step)")
    plt.title("Training Curve: Avg Cost"); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "plot_avg_cost.png"), dpi=160); plt.close()

def plot_bullwhip_from_snapshots(run_dir: str, n_agents: int = 3):
    snap_files = sorted(glob.glob(os.path.join(run_dir, "snapshot_ep*.json")))
    if not snap_files:
        print("[WARN] No snapshot files for bullwhip plotting.")
        return

    xs = []
    bws = [[] for _ in range(n_agents)]
    for p in snap_files:
        ep = int(os.path.basename(p).split("ep")[1].split(".")[0])
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        bw_chain = data.get("bullwhip_chain", [None]*n_agents)
        xs.append(ep)
        for i in range(n_agents):
            bws[i].append(bw_chain[i])

    plt.figure()
    labels = ["R vs Demand", "W vs R", "F vs W"]
    for i in range(n_agents):
        plt.plot(xs, bws[i], label=labels[i] if i < len(labels) else f"Level {i}")
    plt.xlabel("Episode (snapshot)"); plt.ylabel("Bullwhip Index")
    plt.title("Bullwhip over Training (snapshots)"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "plot_bullwhip.png"), dpi=160); plt.close()
    
def plot_cost_breakdown(run_dir: str):
    """
    Plot per-episode cost breakdown:
      stacked area: hold / backlog / smooth
      line: total
    保存: plots/cost_breakdown.png
    """
    import os, csv
    import numpy as np
    import matplotlib.pyplot as plt

    csv_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(csv_path):
        print("[WARN] No train_log.csv for cost breakdown.")
        return

    # 读取
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # 列索引
    def col(name, default=None):
        return header.index(name) if name in header else default

    idx_ep   = col("episode", 0)
    idx_hold = col("avg_hold_cost")
    idx_back = col("avg_backlog_cost")
    idx_smth = col("avg_smooth_cost")
    idx_tot  = col("avg_total_cost")

    if None in (idx_hold, idx_back, idx_smth, idx_tot):
        print("[WARN] cost columns not found in CSV; skip cost_breakdown.")
        return

    ep    = np.array([int(r[idx_ep]) for r in rows])
    hold  = np.array([float(r[idx_hold]) for r in rows])
    back  = np.array([float(r[idx_back]) for r in rows])
    smth  = np.array([float(r[idx_smth]) for r in rows])
    total = np.array([float(r[idx_tot])  for r in rows])

    # 指定颜色
    colors = ["#4593CF", "#f6c062", "#579332"]  # 蓝/橙/绿
    total_color = "black"

    # 画图
    plt.figure(figsize=(20,8))
    plt.stackplot(ep, hold, back, smth,
                  labels=["Holding", "Backlog", "Smoothing"],
                  colors=colors, alpha=0.4)
    plt.plot(ep, total, linewidth=1.0, label="Total", color=total_color)
    plt.xlabel("Episode")
    plt.ylabel("Avg Cost per Step")
    plt.title("Cost Breakdown per Episode (stacked) + Total")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    out = os.path.join(run_dir, "plots", "cost_breakdown.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved {out}")

    
def plot_baselines_panel(run_dir: str):
    """
    Read summary.json and draw:
      - Bars: AvgCost/step, FillRate, AvgBacklog, AvgInventory for (S/Mean/RL)
      - Radar: Bullwhip chain (3 dims) for (S/Mean/RL)
    保存: plots/panel_baselines.png
    """

    s_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(s_path):
        print("[WARN] No summary.json for baselines panel.")
        return

    with open(s_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    fe = s.get("final_eval", {})
    RL   = fe.get("RL", {})
    Base = fe.get("Baseline", {})
    Mean = fe.get("Mean", {})

    # 取值
    names = ["Baseline", "Mean", "RL"]
    cost  = [Base.get("avg_cost_per_step", np.nan),
             Mean.get("avg_cost_per_step", np.nan),
             RL.get("avg_cost_per_step",   np.nan)]
    fill  = [Base.get("fill_rate", np.nan),
             Mean.get("fill_rate", np.nan),
             RL.get("fill_rate",   np.nan)]
    back  = [Base.get("avg_backlog", np.nan),
             Mean.get("avg_backlog", np.nan),
             RL.get("avg_backlog",   np.nan)]
    inv   = [Base.get("avg_inventory", np.nan),
             Mean.get("avg_inventory", np.nan),
             RL.get("avg_inventory",   np.nan)]

    bw_labels = ["R vs Demand","W vs R","F vs W"]
    bw_base = Base.get("bullwhip_chain", [np.nan]*3)
    bw_mean = Mean.get("bullwhip_chain", [np.nan]*3)
    bw_rl   = RL.get("bullwhip_chain",   [np.nan]*3)

    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1,1.2])

    # 柱图函数
    def bar(ax, vals, title):
        ax.bar(names, vals)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    ax1 = fig.add_subplot(gs[0,0]); bar(ax1, cost, "Avg Cost / step")
    ax2 = fig.add_subplot(gs[0,1]); bar(ax2, fill, "Fill Rate")
    ax3 = fig.add_subplot(gs[0,2]); bar(ax3, back, "Avg Backlog")
    ax4 = fig.add_subplot(gs[1,0]); bar(ax4, inv,  "Avg Inventory")

    # 雷达图
    def radar_factory(num_vars, frame='polygon'):
        # from mpl docs: create radar axes
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        # rotate theta such that the first axis is at the top
        theta += np.pi / 2
        return theta

    theta = radar_factory(len(bw_labels))
    ax5 = fig.add_subplot(gs[1,1:], polar=True)
    def close(vals):
        return np.concatenate([vals, [vals[0]]])

    ax5.plot(close(theta), close(np.array(bw_base, dtype=float)), label="Baseline")
    ax5.plot(close(theta), close(np.array(bw_mean, dtype=float)), label="Mean")
    ax5.plot(close(theta), close(np.array(bw_rl,   dtype=float)), label="RL")
    ax5.set_xticks(theta)
    ax5.set_xticklabels(bw_labels)
    ax5.set_title("Bullwhip Radar (lower is better; 1=reference)")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="upper right", bbox_to_anchor=(1.15, 1.10))

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    out = os.path.join(run_dir, "plots", "panel_baselines.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved {out}")
    
def plot_policy_portrait(npz_path: str, out_path: str = None, x_feature: str = "B"):
    """
    读取 dump_policy_portrait 保存的 npz，画 hexbin:
      x = x_feature（如 I / B / Pin_sum），y = action
      附带分箱均值曲线
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    cols = [c for c in d["cols"]]
    data = d["data"]

    if x_feature not in cols:
        print(f"[WARN] feature {x_feature} not in {cols}")
        return

    xi = cols.index(x_feature)
    yi = cols.index("action")
    x = data[:, xi]
    y = data[:, yi]

    plt.figure(figsize=(6,5))
    hb = plt.hexbin(x, y, gridsize=40, bins="log")
    cb = plt.colorbar(hb); cb.set_label("log(count)")
    plt.xlabel(x_feature)
    plt.ylabel("action")

    # 分箱均值
    bins = np.linspace(x.min(), x.max(), 20)
    idx = np.digitize(x, bins)
    x_centers = 0.5*(bins[1:]+bins[:-1])
    y_means = np.array([y[idx==i].mean() if np.any(idx==i) else np.nan for i in range(1,len(bins))])
    plt.plot(x_centers, y_means, lw=2, label="binned mean")

    plt.title(f"Policy Portrait: action vs {x_feature}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(npz_path), "plots", f"portrait_{x_feature}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[PLOT] saved {out_path}")

def plot_baselines_panel(run_dir: str, summary: dict | None = None):
    # -------- 读数据 --------
    if summary is None:
        s_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(s_path):
            print("[WARN] No summary.json for baselines panel.")
            return
        with open(s_path, "r", encoding="utf-8") as f:
            s = json.load(f)
    else:
        s = summary

    fe   = s.get("final_eval", {})
    RL   = fe.get("RL", {})
    Base = fe.get("Baseline", {})
    Mean = fe.get("Mean", {})
    k_for_sla = int(s.get("config", {}).get("lead_time", 1))

    strategies = ["S", "Mean", "RL"]

    # 核心标量指标（非牛鞭）
    metrics = [
        ("avg_cost_per_step", "Avg Cost / step",  "lower_better"),
        ("fill_rate_sla_k",   f"Fill Rate@{k_for_sla}", "higher_better"),  # ← 替换
        ("avg_backlog",       "Avg Backlog",      "lower_better"),
        ("avg_inventory",     "Avg Inventory",    "lower_better"),
    ]

    # 牛鞭三维（均为“越小越好”）
    bw_labels = ["R vs Demand", "W vs R", "F vs W"]
    bw_S  = np.array(Base.get("bullwhip_chain", [np.nan]*3), dtype=float)
    bw_M  = np.array(Mean.get("bullwhip_chain", [np.nan]*3), dtype=float)
    bw_RL = np.array(RL.get("bullwhip_chain",   [np.nan]*3), dtype=float)

    # -------- 准备柱状图数据（绝对值）--------
    def get_val(block, key, default=np.nan):
        return block.get(key, default)

    raw_values = {
        "S":    {m[0]: get_val(Base, m[0]) for m in metrics},
        "Mean": {m[0]: get_val(Mean, m[0]) for m in metrics},
        "RL":   {m[0]: get_val(RL,   m[0]) for m in metrics},
    }

    # 追加牛鞭三维
    for i, lab in enumerate(bw_labels):
        key = f"bw_{i}_{lab.replace(' ', '')}"
        for name, arr in zip(["S","Mean","RL"], [bw_S, bw_M, bw_RL]):
            val = float(arr[i]) if (arr is not None and len(arr) > i) else np.nan
            raw_values[name][key] = val
        metrics.append((key, f"Bullwhip: {lab}", "lower_better"))

    # ✅ 新增：整体牛鞭 bull_overall（若字段缺失，用 chain 和回退，可按需改均值/平方和）
    def overall_from(block, chain_arr):
        v = block.get("bull_overall", None)
        if v is None:
            return float(np.nansum(chain_arr)) if chain_arr.size else np.nan
        return float(v)

    overall_S   = overall_from(Base, bw_S)
    overall_M   = overall_from(Mean, bw_M)
    overall_RL  = overall_from(RL,   bw_RL)

    for name, v in zip(["S","Mean","RL"], [overall_S, overall_M, overall_RL]):
        raw_values[name]["bull_overall"] = v
    metrics.append(("bull_overall", "Bullwhip: Overall", "lower_better"))

    # 确保输出目录
    outdir = os.path.join(run_dir, "plots")
    os.makedirs(outdir, exist_ok=True)

    # -------- 1) 柱状图（绝对值）--------
    n_metrics = len(metrics)
    cols = 4
    rows = int(np.ceil(n_metrics / cols))

    fig1, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.4*rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, (key, title, _) in enumerate(metrics):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        vals = [raw_values[st][key] for st in strategies]
        ax.bar(strategies, vals, color=["#799BD1", "#8EC69B", "#D88D8F"])
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    for j in range(n_metrics, rows*cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig1.suptitle("Baselines Panel — All Metrics (Absolute Bars)", fontsize=14, y=1.02)
    plt.tight_layout()
    out1 = os.path.join(outdir, "panel_baselines_bars.png")
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig1)
    print(f"[PLOT] saved {out1}")

    # -------- 2) 雷达图（统一归一化，“越大越好”）--------
    eps = 1e-12
    pad_lo, pad_hi = 0.6, 0.95
    norm_scores = {st: [] for st in strategies}
    metric_names_for_radar = []

    def _to_score(vals, direction):
        v = np.array(vals, dtype=float)
        s = v.copy() if direction == "higher_better" else -v
        mask = np.isfinite(s)
        if mask.sum() < 2:
            z = np.full_like(s, 0.5, dtype=float)
            z[~mask] = 0.5
            return z
        s_f = s[mask]
        vmin, vmax = np.min(s_f), np.max(s_f)
        if abs(vmax - vmin) < eps:
            z = np.full_like(s, 0.5, dtype=float)
            z[~mask] = 0.5
            return z
        z = np.zeros_like(s, dtype=float)
        z[mask] = (s[mask] - vmin) / (vmax - vmin + eps)
        z = pad_lo + (pad_hi - pad_lo) * z
        z[~mask] = 0.5
        return z

    # 只给雷达图用的指标：去掉单独 bullwhip 分量（bw_*），保留 bull_overall 与其他指标
    radar_metrics = [(key, title, direction) for (key, title, direction) in metrics
                    if not key.startswith("bw_")]

    for key, title, direction in radar_metrics:
        vals = [raw_values[st][key] for st in strategies]  # [S, Mean, RL]
        z = _to_score(vals, direction)
        if key == "fill_rate_sla_k":
            arr = np.array(vals, dtype=float)
            if np.all(np.isfinite(arr)):
                r = np.round(arr, 2)
                if (np.max(r) - np.min(r)) == 0:
                    z = np.full_like(arr, pad_hi, dtype=float)
        # 先占位，后面一起改
        for st, zi in zip(strategies, z):
            norm_scores[st].append(float(zi))
        metric_names_for_radar.append(title)

    # 兜底：长度一致 + NaN→0.5
    n_dims = len(metric_names_for_radar)
    for st in strategies:
        y = np.array(norm_scores[st], dtype=float)
        assert len(y) == n_dims
        y = np.where(np.isfinite(y), y, 0.5)
        norm_scores[st] = y.tolist()

    # --- 角度：均匀分布 0..2π（不在 theta 上加偏移）---
    theta = np.linspace(0, 2*np.pi, n_dims, endpoint=False)

    # 正确闭合：角度尾点用 theta[0] + 2π（避免短弧扇形）
    def close(vals, thetas):
        return (
            np.concatenate([vals, vals[:1]]),
            np.concatenate([thetas, thetas[:1] + 2*np.pi]),
        )

    fig2 = plt.figure(figsize=(8, 7))
    ax = fig2.add_subplot(111, polar=True)

    COL_S, COL_M, COL_RL = "#799BD1", "#8EC69B", "#D88D8F"

    def plot_one(st, col):
        y = np.array(norm_scores[st], dtype=float)
        y, t = close(y, theta)
        ax.plot(t, y, color=col, lw=2.0, label=st)
        ax.fill(t, y, color=col, alpha=0.15)
        for ang, val in zip(theta, y[:-1]):
            ax.plot([ang], [val], marker="o", ms=4, color=col)

    plot_one("S", COL_S)
    plot_one("Mean", COL_M)
    plot_one("RL", COL_RL)

    ax.set_xticks(theta)
    ax.set_xticklabels([lbl.replace(": ", ":\n") for lbl in metric_names_for_radar], fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.00"], fontsize=12, color="#666666")
    ax.grid(True, alpha=0.35)
    ax.set_title("Baselines Panel — Unified Radar (Normalized, higher is better)", fontsize=15, pad=15)

    # 起点旋到正上方（替代直接改 theta）
    ax.set_theta_offset(np.pi/2)
    # 强制整圈显示，避免被判成短弧
    ax.set_thetalim(0, 2*np.pi)

    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), frameon=True)
    fig2.subplots_adjust(left=0.06, right=0.82, top=0.90, bottom=0.08)

    out2 = os.path.join(outdir, "panel_baselines_radar_all.png")
    plt.savefig(out2, dpi=160)  # 别用 bbox_inches="tight"
    plt.close(fig2)
    print(f"[PLOT] saved {out2}")
    
def plot_scenario_comparison(run_dir: str, series: dict, agent_idx: int = 0):
    """
    series: {
        "Baseline": {"I": [...], "B": [...], "A": [...], "D": [...]},
        "Mean":     {"I": [...], "B": [...], "A": [...], "D": [...]},
        "RL":       {"I": [...], "B": [...], "A": [...], "D": [...]},
    }
    仅对比零售商(默认 agent_idx=0)的 I/B/A 以及共同的外部需求 D 时间序列
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    cols = {"Baseline": "#799BD1", "Mean": "#8EC69B", "RL": "#D88D8F"}

    # 时间轴长度：以 RL 的长度为基准（各策略同一场景应一致）
    T = max(len(v["I"]) for v in series.values())
    t = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)  # 4 行：D / A / I / B
    (axD, axA, axI, axB) = axes

    # --- Demand ---
    # 需求对三种策略应相同，取任意一个（优先 RL）
    D_ref = None
    for name in ["RL", "Baseline", "Mean"]:
        if name in series and "D" in series[name] and len(series[name]["D"]) == T:
            D_ref = series[name]["D"]
            break
    if D_ref is None:
        # 回退：若未提供 D，则画空图提示
        D_ref = [np.nan] * T
    axD.plot(t, D_ref, color="#216F87", lw=1.5)
    axD.set_ylabel("Demand (D)")
    axD.set_title("Scenario Comparison (same demand seed)")
    axD.grid(True, alpha=0.3)

    # --- Orders (A) ---
    for name, col in cols.items():
        axA.plot(t, series[name]["A"], label=name, color=col, lw=1.5)
    axA.set_ylabel("Order (A)")
    axA.grid(True, alpha=0.3)

    # --- Inventory (I) ---
    for name, col in cols.items():
        axI.plot(t, series[name]["I"], label=name, color=col, lw=1.5)
    axI.set_ylabel("Inventory (I)")
    axI.grid(True, alpha=0.3)

    # --- Backlog (B) ---
    for name, col in cols.items():
        axB.plot(t, series[name]["B"], label=name, color=col, lw=1.5)
    axB.set_ylabel("Backlog (B)")
    axB.set_xlabel("Time step")
    axB.grid(True, alpha=0.3)

    # 图例放在 A 子图的右上角
    axA.legend(loc="upper right", bbox_to_anchor=(1.12, 1.0), frameon=True)

    fig.tight_layout()
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    out = os.path.join(run_dir, "plots", "scenario_comparison.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved {out}")
    
def plot_slak_curves(run_dir: str, curves: dict):
    """
    curves: {
      "Baseline": {"k": [1,2,...], "sla": [..]},
      "Mean":     {"k": [...],     "sla": [..]},
      "RL":       {"k": [...],     "sla": [..]},
    }
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    cols = {"Baseline": "#799BD1", "Mean": "#8EC69B", "RL": "#D88D8F"}

    plt.figure(figsize=(7.5, 5.5))
    for name, data in curves.items():
        kx = np.array(data["k"], dtype=float)
        vy = np.array(data["sla"], dtype=float)
        plt.plot(kx, vy, marker="o", lw=1.8, label=name, color=cols.get(name, None))
    plt.xlabel("k (steps)")
    plt.ylabel("Fill Rate@k")
    plt.title("SLA-k Curves (Retailer)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    out = os.path.join(run_dir, "plots", "slak_curves.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print(f"[PLOT] saved {out}")