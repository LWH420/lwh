import argparse
import json
import os
import subprocess
import sys


def run_cmd(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_metrics(out_dir):
    metrics_file = os.path.join(out_dir, "metrics.json")
    if not os.path.isfile(metrics_file):
        return {}
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_result(tag, out_dir):
    m = load_metrics(out_dir)
    q_stats = m.get("by_variable_norm", {}).get("Q", {})
    return {
        "tag": tag,
        "out_dir": out_dir,
        "q_r2_norm": q_stats.get("r2"),
        "q_rmse_norm": q_stats.get("rmse"),
        "q_mae_norm": q_stats.get("mae"),
        "q_rmse_upper_norm": m.get("q_rmse_upper_norm"),
        "q_rmse_lower_norm": m.get("q_rmse_lower_norm"),
        "q_rmse_all_norm": m.get("q_rmse_all_norm"),
        "overall_rmse_norm": m.get("overall_rmse_norm"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_script", type=str, default="train_ResCNN.py")
    parser.add_argument("--global_input", type=str, required=True)
    parser.add_argument("--global_output", type=str, required=True)
    parser.add_argument("--level_input", type=str, required=True)
    parser.add_argument("--level_output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ntimes_input", type=int, default=5)
    parser.add_argument("--data_path", type=str,
                        default="/data2/share/llj/lwh/ML_code/lwh_NN_model/normalized_output/")
    parser.add_argument("--result_json", type=str, default="dual_norm_results.json")
    args = parser.parse_args()

    # 全变量统一归一化数据
    run_cmd(
        [
            sys.executable,
            args.train_script,
            "--input_file", args.global_input,
            "--output_file", args.global_output,
            "--norm_mode", "global",
            "--epochs", str(args.epochs),
            "--ntimes_input", str(args.ntimes_input),
        ]
    )

    # 逐层归一化数据
    run_cmd(
        [
            sys.executable,
            args.train_script,
            "--input_file", args.level_input,
            "--output_file", args.level_output,
            "--norm_mode", "level",
            "--epochs", str(args.epochs),
            "--ntimes_input", str(args.ntimes_input),
        ]
    )

    # 通过 train_ResCNN.py 的命名规则推测目录
    def infer_out_dir(model_type, ntimes_input, epochs, lr, bs):
        lr_str = f"{lr:.0e}" if lr < 1e-3 else str(lr)
        run_name = f"model_{model_type}_seq{ntimes_input}_lr{lr_str}_bs{bs}_ep{epochs}_vertnorm_newresRNN"
        return run_name

    global_run = os.path.join(
        args.data_path,
        infer_out_dir("rescnn_qbranch", args.ntimes_input, args.epochs, 1e-4, 256)
    )
    level_run = os.path.join(
        args.data_path,
        infer_out_dir("rescnn", args.ntimes_input, args.epochs, 8e-5, 256)
    )

    # 仅保存 run_name，具体 out_dir 由 train_ResCNN.py 的 data_path 决定
    summary = [
        summarize_result("global_norm", global_run),
        summarize_result("level_norm", level_run),
    ]

    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {args.result_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
