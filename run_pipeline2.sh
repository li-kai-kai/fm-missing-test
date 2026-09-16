#!/usr/bin/env bash
# 阶段三/四 + 方案 §6 规定的排查 + 复验 + 出图出报告。
# 顺序：种子0测试 -> 16/32 步排查 -> 学习率同等机会调参 -> 种子1/2 复验 -> 图与报告
set -u
cd "$(dirname "$0")"
OUT=experiment
LOG=logs

wait_for() { while pgrep -f "run_train.py.*--seed $1" > /dev/null; do sleep 20; done; }

echo "[p2] 等待种子 0 的 B 训练结束 ..."
wait_for 0
echo "[p2] 种子 0 训练结束"

# ---- 阶段三：种子 0 锁定配置后测试 ----
python3 -u run_eval.py --out $OUT --split test --methods A,B --seeds 0 > $LOG/eval_seed0.log 2>&1
echo "[p2] 种子 0 测试评价完成"

# ---- 方案 §6 阶段二排查：同 checkpoint、同初始噪声下 16 步 vs 32 步 ----
python3 -u run_fm_steps.py --out $OUT --seed 0 --steps 16,32 --split val > $LOG/fm_steps.log 2>&1
echo "[p2] 16/32 步排查完成"

# ---- 方案 §6：学习率敏感性，两个模型同等机会，各试 3e-5（两者并行）----
python3 -u run_train.py --method A --seed 0 --lr 3e-5 --tag _lr3e-5 > $LOG/A_lr3e-5.log 2>&1 &
sleep 2
python3 -u run_train.py --method B --seed 0 --lr 3e-5 --tag _lr3e-5 > $LOG/B_lr3e-5.log 2>&1 &
while pgrep -f "run_train.py.*_lr3e-5" > /dev/null; do sleep 20; done
python3 -u run_eval.py --out $OUT --split val --methods A,B --seeds 0 --run-tag _lr3e-5 \
  --no-save-pred > $LOG/eval_lr3e-5.log 2>&1
echo "[p2] 学习率 3e-5 调参完成"

# ---- 阶段四：种子 1、2 复验（保持划分与配置不变）----
for S in 1 2; do
  python3 -u run_train.py --method A --seed $S > $LOG/A_seed$S.log 2>&1 &
  sleep 2
  python3 -u run_train.py --method B --seed $S > $LOG/B_seed$S.log 2>&1 &
  wait_for $S
  python3 -u run_eval.py --out $OUT --split test --methods A,B --seeds $S > $LOG/eval_seed$S.log 2>&1
  echo "[p2] 种子 $S 复验完成"
done

# ---- 阶段一复跑（方案要求 1000~2000 次更新）----
python3 -u run_sanity.py --out $OUT --n-cases 4 --steps 1500 > $LOG/sanity.log 2>&1

# ---- 图与报告 ----
python3 -u run_figures.py --out $OUT --seeds 0,1,2 > $LOG/figures.log 2>&1
python3 -u run_case_figs.py --out $OUT --seed 0 > $LOG/case_figs.log 2>&1
python3 -u run_report.py --out $OUT --split test > $LOG/report.log 2>&1
echo "[p2] 全部完成"
