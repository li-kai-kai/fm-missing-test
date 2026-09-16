#!/usr/bin/env bash
# 端到端流水线：等种子 0 训练结束 -> 测试 -> 种子 1/2 复验 -> 图与报告
set -u
cd "$(dirname "$0")"
OUT=experiment
LOG=logs

wait_for() {  # 等待指定 seed 的训练进程结束
  while pgrep -f "run_train.py.*--seed $1" > /dev/null; do sleep 20; done
}

echo "[pipeline] 等待种子 0 训练结束 ..."
wait_for 0
echo "[pipeline] 种子 0 训练结束，开始测试评价"
python3 -u run_eval.py --out $OUT --split test --methods A,B --seeds 0 > $LOG/eval_seed0.log 2>&1
echo "[pipeline] 种子 0 评价完成"

for S in 1 2; do
  echo "[pipeline] 启动种子 $S 训练"
  python3 -u run_train.py --method A --seed $S > $LOG/A_seed$S.log 2>&1 &
  sleep 2
  python3 -u run_train.py --method B --seed $S > $LOG/B_seed$S.log 2>&1 &
  wait_for $S
  echo "[pipeline] 种子 $S 训练结束，开始测试评价"
  python3 -u run_eval.py --out $OUT --split test --methods A,B --seeds $S > $LOG/eval_seed$S.log 2>&1
done

echo "[pipeline] 阶段一复跑（方案要求 1000~2000 次更新）"
python3 -u run_sanity.py --out $OUT --n-cases 4 --steps 1500 > $LOG/sanity.log 2>&1

echo "[pipeline] 生成图与报告（汇总由 run_eval 内部的 rebuild_combined 完成，不重跑推理）"
python3 -u run_figures.py --out $OUT --seeds 0,1,2 > $LOG/figures.log 2>&1
python3 -u run_case_figs.py --out $OUT --seed 0 > $LOG/case_figs.log 2>&1
python3 -u run_report.py --out $OUT --split test > $LOG/report.log 2>&1
echo "[pipeline] 全部完成"
