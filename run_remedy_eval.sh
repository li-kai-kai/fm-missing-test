#!/usr/bin/env bash
# 评估修复实验的三个臂，每个臂三个训练种子。
#
# 必须带 --no-save-pred：写预测 NIfTI 需要原始 BraTS 的仿射矩阵，而
# /SimCLR/data 已不存在（见 run_eval.py 的 get_affine）。指标计算不受影响。
# --run-tag 同时决定 runs/ 下的运行目录名和 evaluation/test/<tag>/ 输出目录，
# 因此修复臂的结果与主结果（evaluation/test/main/）完全隔离，不会互相污染。
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)"

run_arm() {
  local arm=$1
  for seed in 0 1 2; do
    echo "[$(date +%H:%M:%S)] 评估 ${arm} seed=${seed}"
    python3 run_eval.py --out experiment_multiclass --split test \
        --methods B --seeds "$seed" --run-tag "_${arm}" --no-save-pred \
        > "logs/remedy_eval_${arm}_seed${seed}.log" 2>&1 \
      || echo "  !! ${arm} seed=${seed} 失败，见日志"
  done
  echo "[$(date +%H:%M:%S)] ${arm} 完成"
}

run_arm tg1 &
run_arm fgw9 &
wait
run_arm fgw9tg1

echo "[$(date +%H:%M:%S)] 全部评估完成"
