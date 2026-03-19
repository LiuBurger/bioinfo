import os
import torch as pt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.data import ProteinDataset, QueryHomologyDataset, collate_fun_train, collate_fun_emb
from utils.model import DualEncoderRetriever
from utils.tools import gen_embeddings, build_idx, calculate_remote_homology_score, save_model


@dataclass
class TrainConfig:
    epochs: int = 10
    gpu: int = 8
    batch_size: int = 128
    num_workers: int = 4
    model_name: str = "DualEncoder"
    lr: float = 2e-4
    pdb_root: str = "../../data/pdb"
    tmalign_path: str = "./TMalign"
    topk: int = 12
    eval_search_k: int = 128
    train_top_m: int = 16
    tmalign_reference: int = 1
    tmalign_workers: int = None


class Trainer:
    def __init__(self, model, config: TrainConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.log_file = open(f"{config.model_name}.txt", "w", encoding="utf-8")
        self.max_top1, self.max_topk = float("-inf"), float("-inf")
        self.optimizer = pt.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = pt.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr / 100
        )

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

    def _filter_self_hits(self, I: np.ndarray, test_map: np.ndarray, lib_map: np.ndarray):
        filtered = []
        self_hits = 0
        for row, q_global_idx in zip(I, test_map):
            row_filtered = []
            hit_self = False
            for j_local in row:
                j_local = int(j_local)
                cand_global_idx = int(lib_map[j_local])  # 关键：局部索引 -> 全局索引
                if cand_global_idx == int(q_global_idx):
                    hit_self = True
                    continue
                row_filtered.append(j_local)
                if len(row_filtered) >= self.config.topk:
                    break

            filtered.append(row_filtered)
            self_hits += int(hit_self)
        self_hit_rate = self_hits / max(len(test_map), 1)
        return filtered, self_hit_rate

    def train(self, train_loader: DataLoader, test_loader: DataLoader, lib_loader: DataLoader, test_set, datalib, test_map):
        try:
            for epoch in range(self.config.epochs):
                print(f"======================= Train epoch {epoch + 1} =======================")
                self.log_file.write(f"\nEpoch {epoch + 1} Training\n")
                self.model.train()
                for i, batch in enumerate(tqdm(train_loader, unit="batch")):
                    batch_gpu = {}
                    for k, v in batch.items():
                        if hasattr(v, 'to'):
                            batch_gpu[k] = v.to(self.config.gpu, non_blocking=True)
                        else:
                            batch_gpu[k] = v

                    output = self.model(batch_gpu)
                    loss = output["loss"]
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = pt.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    if (i + 1) % 100 == 0:
                        msg = (
                            f"Epoch [{epoch + 1}/{self.config.epochs}], Step [{i+1}]\n"
                            f"Train Loss: {loss:.4f}\n"
                            f"Grad_norm: {float(grad_norm):.4f}"
                        )
                        print(msg)
                        self.log_file.write(msg + "\n")
                self.scheduler.step()

                print(f"======================= Test epoch {epoch + 1} =======================")
                self.log_file.write(f"\nEpoch {epoch + 1} Testing\n")
                pt.cuda.empty_cache()
                embs_lib = gen_embeddings(self.model, lib_loader, self.config.gpu, mode='cand')
                embs_test = gen_embeddings(self.model, test_loader, self.config.gpu, mode='query')
                search_k = max(self.config.topk + 1, self.config.eval_search_k)
                I, _ = build_idx(embs_lib, embs_test, topk=search_k)
                I_filtered, self_hit_rate = self._filter_self_hits(I, test_map, datalib.map)
                sorted_results, avg_top1_score, avg_topk_score, avg_top1_tm, avg_top1_seqid = calculate_remote_homology_score(
                    query=test_set,
                    database=datalib,
                    idx=I_filtered,
                    k=self.config.topk,
                    pdb_root=self.config.pdb_root,
                    tmalign_path=self.config.tmalign_path,
                    reference=self.config.tmalign_reference,
                    num_workers=self.config.tmalign_workers,
                )
                msg = (
                    f"Self-hit rate before filtering: {self_hit_rate:.4f}\n"
                    f"Average Top-1 Remote homologous score: {avg_top1_score:.6f}\n"
                    f"Average Top-{self.config.topk} Remote homologous score: {avg_topk_score:.6f}\n"
                    f"Average Top-1 TM-score: {avg_top1_tm:.6f}\n"
                    f"Average Top-1 SeqID: {avg_top1_seqid:.6f}"
                )
                print(msg)
                self.log_file.write(msg + "\n")
                # if avg_top1_score > self.max_top1 or avg_topk_score > self.max_topk:
                #     self.max_top1 = avg_top1_score
                #     self.max_topk = avg_topk_score
                #     save_model(self.model, self.config.model_name, epoch)
                print("======================================================================")
        finally:
            self.close()


if __name__ == "__main__":
    start_time = datetime.now()
    config = TrainConfig()
    pt.cuda.set_device(config.gpu)

    print('loading data')
    data = pt.load(f'./data/sorted_1300_p0_h0.pt', weights_only=False)
    lib_data = ProteinDataset(data, mode='cand')
    lib_map = np.arange(len(lib_data), dtype=np.int64)
    print('number of library proteins:', len(lib_map))

    pdb2idx = {lib_data[i]['lab']: i for i in range(len(lib_map))}
    train_set = QueryHomologyDataset(lib_data, './data/tmalign.out', pdb2idx)
    train_map = np.arange(len(train_set), dtype=np.int64)
   
    train_map, test_map = train_test_split(train_map, test_size=1024, random_state=42)
    train_map, val_map = train_test_split(train_map, test_size=2048, random_state=42)
    train_map = np.sort(train_map)
    val_map = np.sort(val_map)
    test_map = np.sort(test_map)
    print('Train query num:', len(train_map))
    print('Val query num:', len(val_map))
    print('Test query num:', len(test_map))
    
    val_set = ProteinDataset(lib_data, mapping=val_map, mode='query')
    test_set = ProteinDataset(lib_data, mapping=test_map, mode='query')      

    
    print('Query_i homology_ij pair num:', len(train_set))  

    batch_size = config.batch_size
    libloader = DataLoader(
        lib_data,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fun_emb(mode='cand'),
        pin_memory=True,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fun_train(lib_data),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fun_emb(mode='query'),
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fun_emb(mode='query'),
        pin_memory=True,
        drop_last=False,
    )

    model = DualEncoderRetriever(
        vocab_size=21,
        node_feat_dim=whole_data[0]['graph'].x.shape[1],
        edge_feat_dim=whole_data[0]['graph'].edge_attr.shape[1]
    ).cuda(config.gpu)

    trainer = Trainer(model, config)
    trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        lib_loader=libloader,
        test_set=test_set,
        datalib=lib_data,
        test_map=test_map,
    )
    pt.cuda.empty_cache()
    end_time = datetime.now()
    print(f"whole time consuming: {end_time - start_time}")
