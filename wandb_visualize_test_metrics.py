#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import sys
from collections import defaultdict
from datetime import datetime

import wandb


def collect_all_runs(root_dir):
    """
    从 root_dir 递归查找所有 test_metrics.json，
    返回:
        all_runs: key = (model, dataset) -> list[run_dict]
    run_dict 包含:
        MAE, RMSE, MAPE, path, setting, time, timestamp
    """
    all_runs = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "test_metrics.json" not in filenames:
            continue

        metrics_path = os.path.join(dirpath, "test_metrics.json")

        # 读取 overall 指标
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
            overall = data.get("overall", {})
            mae = overall.get("MAE", None)
            mape = overall.get("MAPE", None)
            rmse = overall.get("RMSE", None)
        except Exception as e:
            print(f"[WARN] 读取 {metrics_path} 失败: {e}", file=sys.stderr)
            continue

        # checkpoints 目录结构: model/setting/hash/...
        rel_dir = os.path.relpath(dirpath, root_dir)
        parts = rel_dir.split(os.sep)
        if len(parts) < 2:
            continue

        model = parts[0]
        setting = parts[1]  # 如 PEMS04_300_12_12
        dataset = setting.split("_")[0]

        # 用文件的 mtime 作为这次实验的时间
        mtime = os.path.getmtime(metrics_path)
        time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        key = (model, dataset)
        all_runs[key].append({
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "path": metrics_path,
            "setting": setting,
            "time": time_str,
            "timestamp": mtime,
        })

    return all_runs


def choose_best(all_runs):
    """从 all_runs 中为每个 (model, dataset) 选一个 MAE 最小的实验。"""
    best = {}

    for key, runs in all_runs.items():
        def mae_or_inf(r):
            v = r.get("MAE")
            return v if isinstance(v, (int, float)) else float("inf")

        best_run = min(runs, key=mae_or_inf)
        best[key] = best_run

    return best


def collect_horizon_rows(root_dir):
    """
    额外收集每个 test_metrics.json 里的 horizon_k 指标，
    返回一个 list[dict]，每个 dict 一行：
        model, dataset, setting, horizon, MAE, MAPE, RMSE, time, path, timestamp
    """
    rows = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "test_metrics.json" not in filenames:
            continue

        metrics_path = os.path.join(dirpath, "test_metrics.json")

        # checkpoints 目录结构: model/setting/hash/...
        rel_dir = os.path.relpath(dirpath, root_dir)
        parts = rel_dir.split(os.sep)
        if len(parts) < 2:
            continue

        model = parts[0]
        setting = parts[1]
        dataset = setting.split("_")[0]

        mtime = os.path.getmtime(metrics_path)
        time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取 {metrics_path} 失败: {e}", file=sys.stderr)
            continue

        for key, val in data.items():
            if key == "overall":
                horizon = 0  # 约定 0 代表 overall
            else:
                if not key.startswith("horizon_"):
                    continue
                try:
                    horizon = int(key.split("_")[1])
                except Exception:
                    continue

            mae = val.get("MAE")
            mape = val.get("MAPE")
            rmse = val.get("RMSE")

            rows.append({
                "model": model,
                "dataset": dataset,
                "setting": setting,
                "horizon": horizon,
                "MAE": mae,
                "MAPE": mape,
                "RMSE": rmse,
                "time": time_str,
                "path": metrics_path,
                "timestamp": mtime,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="汇总 BasicTS checkpoints/test_metrics.json 并同步到 wandb 可视化"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="checkpoints",
        help="checkpoints 根目录（默认：当前目录下的 checkpoints）",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="basicts-test-metrics",
        help="wandb project 名",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="wandb run 名，不填则自动生成",
    )
    args = parser.parse_args()

    root_dir = args.root

    # 1. 复用你原来的逻辑：聚合 & 选最优
    all_runs = collect_all_runs(root_dir)
    best_metrics = choose_best(all_runs)
    horizon_rows = collect_horizon_rows(root_dir)

    # 2. 开一个 wandb run（相当于一个“汇总实验”）
    run = wandb.init(
        project=args.project,
        name=args.run_name or f"metrics-summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={"root_dir": root_dir},
    )

    # 2.1 best_summary 表：每个 (model, dataset) 一行，保留最优一次
    best_table = wandb.Table(
        columns=["model", "dataset", "MAE", "RMSE", "MAPE", "setting", "time", "path"]
    )
    for (model, dataset), info in sorted(best_metrics.items()):
        best_table.add_data(
            model,
            dataset,
            float(info["MAE"]) if info["MAE"] is not None else None,
            float(info["RMSE"]) if info["RMSE"] is not None else None,
            float(info["MAPE"]) if info["MAPE"] is not None else None,
            info["setting"],
            info["time"],
            info["path"],
        )
    wandb.log({"summary/best_runs": best_table})

    # 2.2 all_runs 表：每一次实验一行
    all_table = wandb.Table(
        columns=["model", "dataset", "MAE", "RMSE", "MAPE", "setting", "time", "path"]
    )
    for (model, dataset), runs in sorted(all_runs.items()):
        for r in runs:
            all_table.add_data(
                model,
                dataset,
                float(r["MAE"]) if r["MAE"] is not None else None,
                float(r["RMSE"]) if r["RMSE"] is not None else None,
                float(r["MAPE"]) if r["MAPE"] is not None else None,
                r["setting"],
                r["time"],
                r["path"],
            )
    wandb.log({"detail/all_runs": all_table})

    # 2.3 horizon_metrics 表：方便画 MAE/MAPE/RMSE vs horizon 的曲线
    horizon_table = wandb.Table(
        columns=[
            "model", "dataset", "setting",
            "horizon", "MAE", "MAPE", "RMSE",
            "time", "path"
        ]
    )
    for row in horizon_rows:
        horizon_table.add_data(
            row["model"],
            row["dataset"],
            row["setting"],
            int(row["horizon"]),
            float(row["MAE"]) if row["MAE"] is not None else None,
            float(row["MAPE"]) if row["MAPE"] is not None else None,
            float(row["RMSE"]) if row["RMSE"] is not None else None,
            row["time"],
            row["path"],
        )
    wandb.log({"horizon/metrics": horizon_table})

    run.finish()

    print("同步到 wandb 完成。你可以在网页上用 Tables 构建各种图表。")


if __name__ == "__main__":
    main()
