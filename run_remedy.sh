#!/usr/bin/env bash
# 修复实验：沿诊断报告定位的两条机制轴，对 B 的速度损失做体素重加权。
#
#   fgw9    : fm_fg_weight=9   非背景体素 10 倍权重（针对「L2 被背景体素主导」）
#   tg1     : fm_time_gamma=1  w(t) ∝ 1/(1-t+0.1)（针对「高 t 段误差被 1/(1-t) 放大」）
#   fgw9tg1 : 两者叠加
#
# 除损失权重外，数据、噪声、划分、步数、验证安排与既有 B_seed* 完全一致，
# 因此可直接与 experiment_multiclass/runs/B_seed* 对照。
# 队列按「先补齐 seed 0」排序，中途中断也留有可用的完整消融行。
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)"
export PYTORCH_ALLOC_CONF=expandable_segments:True

CONC=2   # 4090 24G，单次 B 峰值 8.8G，两路并发留有余量

queue=(
  "tg1      0"
  "fgw9     0"
  "fgw9tg1  0"
  "tg1      1"
  "fgw9     1"
  "fgw9tg1  1"
  "tg1      2"
  "fgw9     2"
  "fgw9tg1  2"
)

declare -A ARGS=(
  [tg1]="--fm-time-gamma 1"
  [fgw9]="--fm-fg-weight 9"
  [fgw9tg1]="--fm-fg-weight 9 --fm-time-gamma 1"
)

mkdir -p logs
running=0
for item in "${queue[@]}"; do
  read -r tag seed <<< "$item"
  # shellcheck disable=SC2086
  python3 run_train.py --method B --seed "$seed" --out experiment_multiclass \
      --tag "_${tag}" ${ARGS[$tag]} > "logs/remedy_B_${tag}_seed${seed}.log" 2>&1 &
  running=$((running + 1))
  # 错开启动，避免多个进程同时写 out/config_B.yaml
  sleep 20
  if [ "$running" -ge "$CONC" ]; then
    wait -n
    running=$((running - 1))
  fi
  echo "[$(date +%H:%M:%S)] launched ${tag} seed=${seed}"
done
wait
echo "[$(date +%H:%M:%S)] 全部完成"
