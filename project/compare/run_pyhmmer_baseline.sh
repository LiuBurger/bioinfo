python pyhmmer_baseline.py \
  --data_path ./data/sorted_1300_p0_h1.pt \
  --pair_file ./data/tmalign.out \
  --pdb_root ../../data/pdb \
  --tmalign_path ./TMalign \
  --output_dir ./pyhmmer_runs \
  --cpus 32 \
  --eval_search_k 200 \
  --topk 12
