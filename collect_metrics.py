import os
import json
import argparse
import sys
from collections import defaultdict
from datetime import datetime


def collect_all_runs(root_dir):
    """
    从 root_dir 递归查找所有 test_metrics.json，
    返回一个 dict:
        key: (model, dataset)
        value: list[run_dict]

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
        setting = parts[1]
        # 约定: setting 形如 PEMS04_300_12_12
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


def format_mape(mape):
    """MAPE 通常是 0.x，转换成百分比字符串。"""
    if mape is None:
        return ""
    try:
        v = float(mape)
    except Exception:
        return str(mape)
    if v < 1:
        v = v * 100.0
    return f"{v:.2f}%"


def print_summary_table(best_metrics):
    """
    打印一个“论文风”的汇总大表（每个模型-数据集只保留最优一次）：
    并且按“跨数据集的平均 MAE”从小到大对模型做排序。
    """
    if not best_metrics:
        print("没有在指定目录下找到任何 test_metrics.json")
        return

    models = sorted({m for (m, d) in best_metrics.keys()})
    datasets = sorted({d for (m, d) in best_metrics.keys()})
    d2i = {d: i for i, d in enumerate(datasets)}

    # === 先按“平均 MAE”对模型做一个排序 ===
    model_avg_mae = {}
    for m in models:
        maes = []
        for d in datasets:
            info = best_metrics.get((m, d))
            if info is None:
                continue
            mae = info.get("MAE")
            if isinstance(mae, (int, float)):
                maes.append(mae)
        if maes:
            model_avg_mae[m] = sum(maes) / len(maes)
        else:
            # 没有任何合法 MAE 的模型，放到最后
            model_avg_mae[m] = float("inf")

    # 从小到大：平均 MAE 越小，排得越靠前
    models = sorted(models, key=lambda mm: model_avg_mae.get(mm, float("inf")))
    # 如果你想从大到小，就加 reverse=True：
    # models = sorted(models, key=lambda mm: model_avg_mae.get(mm, float("inf")), reverse=True)

    # 先把所有数字转成字符串，后面好统一算宽度
    values = {}
    for m in models:
        for d in datasets:
            info = best_metrics.get((m, d))
            if info is None:
                mae_s = rmse_s = mape_s = ""
            else:
                mae = info.get("MAE")
                rmse = info.get("RMSE")
                mape = info.get("MAPE")
                mae_s = f"{mae:.2f}" if isinstance(mae, (int, float)) else ""
                rmse_s = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else ""
                mape_s = format_mape(mape)
            values[(m, d)] = (mae_s, rmse_s, mape_s)

    # 后面计算宽度、打印表头和行的逻辑保持不变
    num_cols = 1 + 3 * len(datasets)
    col_widths = [0] * num_cols

    col_widths[0] = max(
        len("Model"),
        len("Dataset"),
        len("Metric"),
        max(len(m) for m in models),
    )

    for d in datasets:
        di = d2i[d]
        base = 1 + 3 * di
        col_widths[base] = max(col_widths[base], len("MAE"), len(d))
        col_widths[base + 1] = max(col_widths[base + 1], len("RMSE"))
        col_widths[base + 2] = max(col_widths[base + 2], len("MAPE"))

    for (m, d), (mae_s, rmse_s, mape_s) in values.items():
        di = d2i[d]
        base = 1 + 3 * di
        col_widths[base] = max(col_widths[base], len(mae_s))
        col_widths[base + 1] = max(col_widths[base + 1], len(rmse_s))
        col_widths[base + 2] = max(col_widths[base + 2], len(mape_s))

    def border_line():
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def format_row(cells, aligns):
        row = "|"
        for i, cell in enumerate(cells):
            text = "" if cell is None else str(cell)
            w = col_widths[i]
            align = aligns[i] if isinstance(aligns, list) else aligns
            if align == "left":
                padded = text.ljust(w)
            elif align == "right":
                padded = text.rjust(w)
            else:
                padded = text.center(w)
            row += " " + padded + " |"
        return row

    # ===== 打印表头 =====
    print(border_line())
    header1_cells = ["Dataset"]
    for d in datasets:
        header1_cells.extend([d, "", ""])
    print(format_row(header1_cells, aligns="center"))

    header2_cells = ["Metric"]
    for _ in datasets:
        header2_cells.extend(["MAE", "RMSE", "MAPE"])
    print(format_row(header2_cells, aligns="center"))
    print(border_line())

    # ===== 打印数据行（已经按平均 MAE 排好序的 models）=====
    for m in models:
        cells = [m]
        for d in datasets:
            mae_s, rmse_s, mape_s = values[(m, d)]
            cells.extend([mae_s, rmse_s, mape_s])
        aligns = ["left"] + ["right"] * (num_cols - 1)
        print(format_row(cells, aligns))
    print(border_line())


def print_detail_tables(all_runs):
    """
    按“数据集”分别打印一个详细表，
    把同一模型在该数据集上的所有实验都列出来。

    只展示：Model, Time, MAE, RMSE, MAPE
    Time 来自 test_metrics.json 的修改时间。
    """
    if not all_runs:
        return

    # 收集所有数据集
    datasets = sorted({d for (_m, d) in all_runs.keys()})

    for d in datasets:
        # 汇总该数据集下的所有 run
        rows = []
        for (m, dd), runs in all_runs.items():
            if dd != d:
                continue
            for r in runs:
                rows.append({
                    "model": m,
                    "time": r.get("time", ""),
                    "timestamp": r.get("timestamp", 0.0),
                    "MAE": r.get("MAE"),
                    "RMSE": r.get("RMSE"),
                    "MAPE": r.get("MAPE"),
                })

        if not rows:
            continue

        # 按 MAE 排个序，方便看最优；你要想改成按时间排序，把这里改成 key=lambda x: x["timestamp"] 即可
        def mae_or_inf(row):
            v = row.get("MAE")
            return v if isinstance(v, (int, float)) else float("inf")

        rows.sort(key=mae_or_inf)

        # 先把数字转成字符串
        for row in rows:
            mae = row["MAE"]
            rmse = row["RMSE"]
            mape = row["MAPE"]
            row["MAE_s"] = f"{mae:.4f}" if isinstance(mae, (int, float)) else ""
            row["RMSE_s"] = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else ""
            row["MAPE_s"] = format_mape(mape)

        # 计算列宽：Model, Time, MAE, RMSE, MAPE
        headers = ["Model", "Time", "MAE", "RMSE", "MAPE"]
        col_widths = [len(h) for h in headers]

        for row in rows:
            col_widths[0] = max(col_widths[0], len(row["model"]))
            col_widths[1] = max(col_widths[1], len(row["time"]))
            col_widths[2] = max(col_widths[2], len(row["MAE_s"]))
            col_widths[3] = max(col_widths[3], len(row["RMSE_s"]))
            col_widths[4] = max(col_widths[4], len(row["MAPE_s"]))

        def border_line():
            return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        def format_row(cells, aligns):
            row = "|"
            for i, cell in enumerate(cells):
                text = "" if cell is None else str(cell)
                w = col_widths[i]
                align = aligns[i] if isinstance(aligns, list) else aligns
                if align == "left":
                    padded = text.ljust(w)
                elif align == "right":
                    padded = text.rjust(w)
                else:
                    padded = text.center(w)
                row += " " + padded + " |"
            return row

        # 打印
        print()  # 空一行
        print(f"===== {d} =====")
        print(border_line())
        print(format_row(headers, aligns="center"))
        print(border_line())
        for row in rows:
            cells = [
                row["model"],
                row["time"],
                row["MAE_s"],
                row["RMSE_s"],
                row["MAPE_s"],
            ]
            aligns = ["left", "left", "right", "right", "right"]
            print(format_row(cells, aligns))
        print(border_line())


def main():
    parser = argparse.ArgumentParser(
        description="汇总 BasicTS checkpoints 下各模型在不同数据集上的 MAE/RMSE/MAPE，并打印成大表格"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="checkpoints",
        help="checkpoints 根目录（默认：当前目录下的 checkpoints）",
    )
    args = parser.parse_args()

    all_runs = collect_all_runs(args.root)
    best_metrics = choose_best(all_runs)

    # 1) 汇总大表（每个模型-数据集只保留最优一次）
    print_summary_table(best_metrics)

    # 2) 按数据集展开的详细表，展示每次实验的时间 + 指标
    print_detail_tables(all_runs)


if __name__ == "__main__":
    main()
