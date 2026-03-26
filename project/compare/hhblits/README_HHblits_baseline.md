# HHblits 基线接入说明

这个方案把 **HHblits** 作为你当前项目的一个**外部检索基线**接入，而不是替换你现有的深度模型训练流程。

你的现有项目评估逻辑是：
1. 为每个 query 找到候选蛋白；
2. 用 `TM-align` 对候选进行打分；
3. 汇报 `Average Top-1 Remote homologous score`、`Average Top-k Remote homologous score`、`Average Top-1 TM-score`、`Average Top-1 SeqID`。

`hhblits_baseline.py` 沿用了这套评价口径，只是把“候选生成器”从双塔模型/FAISS，换成了 **HHblits**。

---

## 一、我给你做了什么

新增文件：

- `hhblits_baseline.py`

它提供三个子命令：

- `export`：把你的库蛋白和测试 query 导出为 `FASTA/A3M`
- `build-db`：把导出的 A3M 构造成 HH-suite 数据库
- `eval`：对测试 query 跑 HHblits，再调用你现有的 `TM-align` 评价函数输出最终指标

---

## 二、推荐流程

### 1）导出序列

```bash
python hhblits_baseline.py export \
  --data_path ./data/sorted_1300_p0_h1.pt \
  --pair_file ./data/tmalign.out \
  --out_dir ./hhblits_workspace
```

导出后会得到：

- `./hhblits_workspace/library_fasta/`
- `./hhblits_workspace/library_a3m/`
- `./hhblits_workspace/test_queries_fasta/`
- `./hhblits_workspace/test_queries_a3m/`
- `./hhblits_workspace/test_split.tsv`

---

### 2）构建 HH-suite 数据库

#### 方案 A：优先推荐，调用官方 `hhsuitedb.py`

```bash
python hhblits_baseline.py build-db \
  --a3m_dir ./hhblits_workspace/library_a3m \
  --db_prefix ./hhblits_workspace/hhdb/library \
  --cpu 8 \
  --use_hhsuitedb \
  --hhsuitedb_path /path/to/hhsuitedb.py
```

#### 方案 B：不用 `hhsuitedb.py`，走手工 fallback

你需要显式提供 HH-suite 的数据库资源文件：

```bash
python hhblits_baseline.py build-db \
  --a3m_dir ./hhblits_workspace/library_a3m \
  --db_prefix ./hhblits_workspace/hhdb/library \
  --cpu 8 \
  --cs219_lib /path/to/cs219.lib \
  --context_lib /path/to/context_data.lib
```

如需自定义二进制路径，还可以额外传：

- `--ffindex_build_path`
- `--ffindex_apply_path`
- `--hhmake_path`
- `--cstranslate_path`
- `--ffindex_order_path`

构建成功后，应该能看到：

- `library_a3m.ffdata / .ffindex`
- `library_hhm.ffdata / .ffindex`
- `library_cs219.ffdata / .ffindex`

这里的前缀来自 `--db_prefix`。

---

### 3）评估 HHblits 基线

```bash
python hhblits_baseline.py eval \
  --data_path ./data/sorted_1300_p0_h1.pt \
  --pair_file ./data/tmalign.out \
  --db_prefix ./hhblits_workspace/hhdb/library \
  --out_dir ./hhblits_eval \
  --pdb_root ../../data/pdb \
  --tmalign_path ./TMalign \
  --cpu 8 \
  --n_iter 2 \
  --search_k 200 \
  --topk 12
```

输出文件包括：

- `./hhblits_eval/metrics.json`
- `./hhblits_eval/hhblits_hits.tsv`
- `./hhblits_eval/hhblits_top1.tsv`
- `./hhblits_eval/test_split.tsv`

其中 `metrics.json` 会给出：

- `avg_top1_score`
- `avg_topk_score`
- `avg_top1_tm`
- `avg_top1_seqid`
- `self_hit_rate_before_filtering`

这些指标与你当前训练代码的评估口径保持一致。

---

## 三、论文里怎么写更合适

你可以把 HHblits 描述为：

> 作为经典远程同源检索基线方法，本文引入 HHblits 对同一测试集合进行检索，并沿用统一的 TM-align 后验评价流程，对比其在 Top-1/Top-k remote homology score、Top-1 TM-score 与 Top-1 SeqID 指标上的表现。

这样写的好处是：

- **比较公平**：大家都用同一套 TM-align 后验指标
- **方法边界清晰**：HHblits 是检索基线，不是训练模型
- **实验可复现**：流程非常明确

---

## 四、两个你要特别注意的点

### 1）token 到氨基酸字母的映射

脚本默认你的序列编码是：

- `0` = padding
- `1..20` 对应 `ACDEFGHIKLMNPQRSTVWY`

如果你数据里 1..20 的氨基酸顺序不是这个，需要加：

```bash
--alphabet YOUR_ORDER
```

否则导出的 FASTA 会错。

### 2）这个 HHblits 基线本质上是“外部候选生成器”

所以它不会改你现在的 `train.py`，也不会参与梯度训练。它的作用是：

- 生成候选
- 用你的原评价逻辑打分
- 给论文提供一个经典方法对比

这正是最适合论文实验的接法。

---

## 五、我建议你的实验表这样设计

| 方法 | 候选生成 | 重排/后验评分 | Avg Top-1 Score | Avg Top-k Score | Avg Top-1 TM | Avg Top-1 SeqID |
|---|---|---|---:|---:|---:|---:|
| HHblits | HHblits | TM-align |  |  |  |  |
| 你的方法 | Dual Encoder + FAISS | TM-align |  |  |  |  |

如果你愿意，还可以补一个：

| 方法 | 候选生成 | 重排/后验评分 |
|---|---|---|
| FAISS-only embedding retrieval | 深度嵌入检索 | 无 |

这样可以把“候选能力”和“后验结构一致性”分开讨论。
