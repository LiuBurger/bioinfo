#!/usr/bin/env bash
set -euo pipefail

python tmvec_baseline_eval_protobuffix.py \
  --data_path ./data/sorted_1300_p0_h1.pt \
  --pair_file ./data/tmalign.out \
  --pdb_root ../../data/pdb \
  --tmalign_path ./TMalign \
  --tmvec_model ./pretrained/tm_vec_swiss_model_large.ckpt \
  --tmvec_config ./pretrained/49181524_tm_vec_swiss_model_large_params.json \
  --protrans_model ./Rostlab \
  --device gpu \
  --topk 12 \
  --eval_search_k 200 \
  --test_size 1024 \
  --random_state 42 \
  --output_dir ./tmvec_eval_swiss_large
