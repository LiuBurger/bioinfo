#!/usr/bin/env bash

python hhblits_baseline.py export \
  --data_path ./data/sorted_1300_p0_h1.pt \
  --pair_file ./data/tmalign.out \
  --out_dir ./hhblits_workspace

python hhblits_baseline.py build-db \
  --a3m_dir ./hhblits_workspace/library_a3m \
  --db_prefix ./hhblits_workspace/hhdb/library \
  --cpu 32 \
  --cs219_lib /home/burger/miniconda3/envs/cu121/data/cs219.lib \
  --context_lib /home/burger/miniconda3/envs/cu121/data/context_data.lib

python hhblits_baseline.py eval \
  --data_path ./data/sorted_1300_p0_h1.pt \
  --pair_file ./data/tmalign.out \
  --db_prefix ./hhblits_workspace/hhdb/library \
  --out_dir ./hhblits_eval \
  --pdb_root ../../data/pdb \
  --tmalign_path ./TMalign \
  --cpu 32 \
  --n_iter 2 \
  --search_k 200 \
  --topk 12