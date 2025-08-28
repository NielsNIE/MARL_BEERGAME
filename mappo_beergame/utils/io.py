from __future__ import annotations
import os, csv, json
from typing import Dict, Any, List, Optional

class CSVLogger:
    def __init__(self, out_dir: str, filename: str = "train_log.csv",
                 extra_fields: Optional[List[str]] = None):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, filename)

        # 基础列
        self.base_fields = ["episode", "avg_return_raw", "avg_cost_per_step"]
        # 额外列（可为空）
        self.extra_fields = list(extra_fields or [])

        write_header = not os.path.exists(self.path)
        self.fp = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.fp)

        if write_header:
            header = self.base_fields + self.extra_fields
            self.writer.writerow(header)
            self.fp.flush()

    def log(self, episode: int, avg_return_raw: float, extras: Optional[Dict[str, Any]] = None):
        avg_cost = -avg_return_raw

        # 额外列值（按列名顺序）
        extra_vals = []
        for k in self.extra_fields:
            v = (extras or {}).get(k, "")
            extra_vals.append(v)

        row = [episode, avg_return_raw, avg_cost] + extra_vals
        self.writer.writerow(row)
        self.fp.flush()

    def close(self):
        self.fp.close()

def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_snapshot(out_dir: str, episode: int, snapshot: Dict[str, Any]):
    save_json(os.path.join(out_dir, f"snapshot_ep{episode:03d}.json"), snapshot)

def save_summary(out_dir: str, summary: Dict[str, Any]):
    save_json(os.path.join(out_dir, "summary.json"), summary)