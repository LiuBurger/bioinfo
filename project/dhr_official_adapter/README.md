# Official DHR checkpoint direct-inference adapter

这一版的目标不是重新训练 DHR，而是**直接加载官方 checkpoint 做推理**，再把结果对接到你现在的项目评测流程里。

## 这版做了什么

- 直接加载官方 `dhr_qencoder.pt` / `dhr_cencoder.pt`
- 支持三种输入来源：
  - TSV（`id<TAB>sequence`）
  - FASTA
  - 你当前项目的 `.pt` 数据对象
- 可以直接构建候选库 embedding + FAISS index
- 可以直接对 query 做检索
- 可以直接在你的 `pair_file + PDB + TM-align` 上跑评测

## 文件说明

- `dhr_official_model.py`：官方 checkpoint 兼容加载器
- `dhr_official_adapter.py`：主入口
- `default_token_map_acdefghiklmnpqrstvwy.json`：默认整数 token 到氨基酸字母的映射示例

## 环境要求

至少需要这些依赖：

- `torch`
- `fair-esm` 或兼容的 `esm`
- `faiss`
- `numpy`
- `pandas`
- `tqdm`

如果你要跑 `project-eval`，还需要：

- `TMalign`
- 你的 PDB 根目录

## 官方 checkpoint

先从官方 DHR README 提供的 checkpoint 链接下载，并解压到某个目录，例如：

```bash
/path/to/dhr_ckpt/
  dhr_qencoder.pt
  dhr_cencoder.pt
```

下面命令里都把这个目录记作 `CKPT_DIR`。

## 一、如果你已经有 TSV / FASTA

### 1）构建候选库

```bash
python dhr_official_adapter.py build-db \
  --input_tsv /abs/path/to/library.tsv \
  --ckpt_dir /abs/path/to/CKPT_DIR \
  --output_dir ./dhr_db \
  --metric l2 \
  --batch_size 8
```

### 2）检索 query

```bash
python dhr_official_adapter.py retrieve \
  --query_tsv /abs/path/to/query.tsv \
  --database_dir ./dhr_db \
  --ckpt_dir /abs/path/to/CKPT_DIR \
  --output_tsv ./retrieval.tsv \
  --topk 200 \
  --batch_size 8
```

## 二、如果你要直接接你当前项目的 `.pt`

### 方式 A：先导出 TSV

```bash
python dhr_official_adapter.py export-project-tsv \
  --project_pt ./data/sorted_1300_p0_h1.pt \
  --output_tsv ./project_library.tsv
```

如果你的整数 token 顺序**不是**默认的
`1:A, 2:C, 3:D, ..., 20:Y`，请显式传入你自己的映射：

```bash
python dhr_official_adapter.py export-project-tsv \
  --project_pt ./data/sorted_1300_p0_h1.pt \
  --token_map_json ./my_token_map.json \
  --output_tsv ./project_library.tsv
```

### 方式 B：一步跑完整项目评测

这个命令最适合你现在的需求：

```bash
python dhr_official_adapter.py project-eval \
  --project_pt ./data/sorted_1300_p0_h1.pt \
  --pair_file ./data/tmalign.out \
  --ckpt_dir /abs/path/to/CKPT_DIR \
  --work_dir ./dhr_official_eval \
  --pdb_root ../../data/pdb \
  --tmalign_path ./TMalign \
  --eval_subset test_split \
  --test_size 1024 \
  --random_state 42 \
  --topk 12 \
  --search_k 200 \
  --metric l2 \
  --batch_size 8
```

执行后会在 `work_dir` 下生成：

- `database/index-ebd.index`
- `database/df-ebd.pkl`
- `summary.json`
- `retrieval_rows.tsv`

其中 `summary.json` 会给出：

- `avg_top1_score`
- `avg_topk_score`
- `avg_top1_tm`
- `avg_top1_seqid`
- `self_hit_rate`

## 关于 token 映射

这是最需要你注意的地方：

你的项目 `.pt` 里序列是**整数 token**，而官方 DHR checkpoint 吃的是**氨基酸字符串**，然后交给 ESM 的 batch converter。

所以这版适配器做了两层兼容：

1. 如果你的 `.pt` 里本来就是字符串，直接用。
2. 如果是整数 token，就按 `token_map_json` 或默认映射转成字符串。

默认映射只是一个**保守假设**，不保证一定与你当前项目的数据编码一致。要做论文正式实验时，最好使用你自己的真实 token→氨基酸映射表。

## 推荐你现在直接跑的命令

如果你就是要“官方 checkpoint 直推理 + 接你项目评测”，先跑这个：

```bash
python dhr_official_adapter.py project-eval \
  --project_pt ./data/sorted_1300_p0_h1.pt \
  --pair_file ./data/tmalign.out \
  --ckpt_dir /abs/path/to/CKPT_DIR \
  --work_dir ./dhr_official_eval \
  --pdb_root ../../data/pdb \
  --tmalign_path ./TMalign \
  --eval_subset test_split \
  --test_size 1024 \
  --random_state 42 \
  --topk 12 \
  --search_k 200 \
  --batch_size 8
```

如果报错最常见的原因只有三个：

1. checkpoint 路径下没有 `dhr_qencoder.pt / dhr_cencoder.pt`
2. 本机没装 `esm`
3. 你的项目 token 映射和默认映射不一致
