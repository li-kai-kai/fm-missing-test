# 论文工程与投稿清单

**题目**：Why Flow Matching Fails at Missing-Modality Brain Tumour Segmentation:
A Diagnostic Study

**目标会议**：IEEE ISPA 2026（第 24 届 IEEE International Symposium on Parallel
and Distributed Processing with Applications）

| 节点 | 日期 |
| :--- | :--- |
| 正文截稿 | **2026-09-30 (AoE)** |
| 作者通知 | 2026-10-30 |
| Camera-ready | 2026-11-30 |
| 会期 | 2026-12-27 ~ 30，马来西亚吉隆坡 |

**格式**：IEEE Computer Society Proceedings Format，正文 **上限 8 页**，
EDAS 提交（<https://edas.info/N35627>）。

> 页数与日期的口径来自 ISPA 2026 官网与其 CFP。二手 CFP 站点上出现过 8/20、
> 9/15 等不同的截稿日期，**投稿前请在 EDAS 上再确认一次**。

---

## 一、文件清单

### 手写的内容（改这些）

| 文件 | 作用 |
| :--- | :--- |
| `main.tex` | 正文。方法、实验设置、五项预注册判读规则、结果、诊断、讨论 |
| `remedy.tex` | 补救实验一节。用 `\REMEDY*` 宏集中管理，正文各处引用 |
| `refs.bib` | 参考文献，27 条，逐条核实到原始出处 |
| `.gitignore` | 忽略 LaTeX 中间产物（保留 `main.pdf`） |

### 生成的内容（不要手改，改生成器）

| 文件 | 由谁生成 |
| :--- | :--- |
| `tables/tab_main.tex` | `make_tables.py` |
| `tables/tab_binary.tex` | `make_tables.py` |
| `tables/tab_remedy.tex` | `make_tables.py` |
| `tables/tab_region.tex` | `make_tables.py`（**当前未被正文引用**，备用） |
| `figures/fig_mechanism.pdf` | `make_figures.py` |
| `figures/fig_sweep.pdf` | `make_figures.py` |
| `figures/fig_qualitative.pdf` | `make_figures.py` |

生成器直接从实验产物读数，**表格和正文里的数字都不经过手抄**：

| 脚本 | 读什么 |
| :--- | :--- |
| `make_tables.py` | `experiment_multiclass/evaluation/test/{main,_tg1,_fgw9,_fgw9tg1}/summary.csv`、`experiment/summary.csv` |
| `make_figures.py` | `experiment_multiclass/diagnostics/*.json`、`experiment_multiclass/predictions/test/`、`experiment_multiclass/cache/` |
| `remedy_stats.py` | 修复臂与基线的 `metrics_per_case.csv`，算逐病例配对 bootstrap（正文里那些区间） |

## 二、从零重建

```bash
cd paper
python3 make_tables.py      # 重新生成 tables/
python3 make_figures.py     # 重新生成 figures/
python3 remedy_stats.py     # 打印修复臂的配对 bootstrap（不写文件）

pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

跑两遍 `pdflatex` 是必须的：交叉引用与页码要第二轮才稳定。`bibtex` 只需在
`refs.bib` 变动后重跑。

当前状态：**8 页整、0 Overfull、0 未定义引用**。

## 三、投稿前检查清单

- [ ] **确认 EDAS 上的截稿时间与页数上限**（二手来源口径不一致）
- [ ] **填作者**：`main.tex` 里 `\author{}` 目前是占位符，需替换为姓名、
      单位、邮箱，并去掉 `\thanks{}` 里的 "Author list pending"
- [ ] **注册 / 登录 EDAS 账号**，截稿前别卡最后一天
- [ ] **选 track**：本文是生成式模型在 3D 体数据上的代价与可靠性分析，
      对应 Track 2（Technologies and Tools，含 generative AI）或
      Track 3（Applications and Services，含高性能科学计算）
- [ ] 通读一遍摘要与结论：作者位、致谢、基金号
- [ ] 确认 8 页没被挤破（重跑编译看 `Output written`）
- [ ] 若需要，把 `main.pdf` 与 `figures/` 一并打包

### 关于 scope 的风险

ISPA 是并行分布式系统会议，医学影像分割本身不在它的典型 scope 内。正文已经
刻意往「生成式分割模型的**算力代价与可靠性**」上靠（14× 推理开销、2.5× 训练
开销、run-to-run 混沌、以及"更准的积分反而更差"），投稿时选对 track 很重要。

## 四、正文数字的可追溯性

正文里每个数值都能追到实验产物，核验方式是重跑对应脚本比对：

| 正文位置 | 来源 |
| :--- | :--- |
| 表 1（四分类主结果） | `evaluation/test/main/metrics_per_case.csv` |
| 表 2（二分类） | `experiment/summary.csv` |
| 表 3（补救臂） | `evaluation/test/_{tg1,fgw9,fgw9tg1}/metrics_per_case.csv` |
| 五项诊断读数 | `experiment_multiclass/diagnostics/*.json` |
| 「36 个配对全部为负」 | `evaluation/test/main/bootstrap.json` |
| 补救臂的显著性区间 | `remedy_stats.py` |

主结果、诊断与补救三类产物由 `run_eval.py`、`run_diagnostics.py`、
`run_remedy.sh` + `run_remedy_eval.sh` 分别产出，互不覆盖。

## 五、已知的坑

1. **`experiment_multiclass/config_B.yaml` 会被每次 B 训练覆写**，内容是最后
   启动的那次运行的配置。它入过库两次、每次都归位成基线 0.0。看配置请以
   `runs/<run>/config.yaml` 为准，那才是逐次运行独立保存的。

2. **修复臂的 `evaluation/test/_<臂>/bootstrap.json` 是空 dict**。`run_eval` 的
   配对比较要求同一次评估里同时存在 A 与 B，而修复臂只有 B 行。这不代表
   「没有差异」——真正的配对分析在 `remedy_stats.py`。

3. **原始 BraTS 数据 `/SimCLR/data/BraTS2020` 已不存在**，只剩预处理缓存
   （`experiment*/cache/`）。因此 `run_preprocess.py` 无法重跑；评估必须带
   `--no-save-pred`，因为写预测 NIfTI 需要原始数据的仿射矩阵。

4. **`make_tables.py` 改了以后要重跑**才会更新 `tables/*.tex`。提交历史里出现过
   一次生成器已改、产物还是旧的（主表标题描述了一个并不存在的列）。

5. 参考文献的 `\cite` key 与 `refs.bib` 必须对得上。核对命令：

   ```bash
   python3 - <<'PY'
   import re
   tex = open('main.tex').read() + open('remedy.tex').read()
   bib = open('refs.bib').read()
   cited = {k.strip() for m in re.findall(r'\\cite\{([^}]*)\}', tex) for k in m.split(',')}
   defined = set(re.findall(r'@\w+\{([^,]+),', bib))
   print('引了但没有:', sorted(cited - defined) or '无')
   print('有但没引:', sorted(defined - cited) or '无')
   PY
   ```

6. **提交 3 的信息里写「24 条文献」，实际是 27 条**。产物本身没问题，只是提交
   信息数错了，之后整理时顺手 amend 即可。
