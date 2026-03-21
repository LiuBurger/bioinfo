import argparse
from datetime import datetime

import numpy as np
import torch as pt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data4 import ProteinDataset, QueryHomologyDataset, collate_fun_train, collate_fun_emb
from utils.model4 import DualEncoderRetriever, MultiTaskOptimizer
from utils.tools4 import build_idx, calculate_remote_homology_score



def parse_args():
    parser = argparse.ArgumentParser(description='Seq-augmented Graph-Graph DualEncoder training config')

    # training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    # optimization
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--grad_clip', type=float, default=2.0)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--tm_threshold', type=float, default=0.6)
    parser.add_argument('--positive_top_ratio', type=float, default=0.3)

    # model
    parser.add_argument('--vocab_size', type=int, default=21)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--gnn_num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'add', 'max'])
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--no_normalize', action='store_false', dest='normalize')

    # retrieval / evaluation
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--eval_search_k', type=int, default=128)
    parser.add_argument('--tmalign_reference', type=int, default=1)
    parser.add_argument('--tmalign_workers', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--eval_interval', type=int, default=10)

    # paths
    parser.add_argument('--data_path', type=str, default='./data/sorted_1300_p0_h0.pt')
    parser.add_argument('--pdb_root', type=str, default='../../data/pdb')
    parser.add_argument('--tmalign_path', type=str, default='./TMalign')
    parser.add_argument('--pair_file', type=str, default='./data/tmalign.out')
    parser.add_argument('--model_name', type=str, default='GraphSeqDualTower')

    return parser.parse_args()


class Trainer:
    def __init__(self, model, config: argparse.Namespace):
        self.model = model
        self.config = config
        self.device = pt.device(f'cuda:{config.gpu}' if pt.cuda.is_available() else 'cpu')
        self.log_file = open(f'{config.model_name}.txt', 'w', encoding='utf-8')
        self.optimizer = MultiTaskOptimizer(
            ranker=self.model,
            lr=config.lr,
            weight_decay=config.weight_decay,
            alpha=config.alpha,
            tm_threshold=config.tm_threshold,
        )
        self.max_top1_score = float('-inf')
        self.max_topk_score = float('-inf')

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

    def _log(self, msg: str):
        print(msg)
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def _to_device(self, batch):
        batch_gpu = {}
        for k, v in batch.items():
            batch_gpu[k] = v.to(self.device, non_blocking=True) if hasattr(v, 'to') else v
        return batch_gpu

    def _filter_self_hits(self, indices: np.ndarray, query_map: np.ndarray, lib_map: np.ndarray):
        filtered = []
        self_hits = 0
        for row, q_global_idx in zip(indices, query_map):
            row_filtered = []
            hit_self = False
            for j_local in row:
                j_local = int(j_local)
                cand_global_idx = int(lib_map[j_local])
                if cand_global_idx == int(q_global_idx):
                    hit_self = True
                    continue
                row_filtered.append(j_local)
                if len(row_filtered) >= self.config.topk:
                    break
            filtered.append(row_filtered)
            self_hits += int(hit_self)
        self_hit_rate = self_hits / max(len(query_map), 1)
        return filtered, self_hit_rate

    @pt.no_grad()
    def _gen_eval_embeddings(self, loader, mode: str):
        embs = []
        self.model.eval()
        for data in loader:
            data_gpu = self._to_device(data)
            emb = self.model.encode(data_gpu, mode=mode)
            embs.append(emb.detach().cpu().numpy().astype(np.float32))
        if len(embs) == 0:
            raise ValueError(f'No embeddings generated for mode={mode}.')
        return np.concatenate(embs, axis=0)

    def train(self, train_loader, test_loader, lib_loader, test_map, test_set, datalib):
        try:
            for epoch in range(self.config.epochs):
                self._log(f'======================= Train epoch {epoch + 1} =======================')
                self.model.train()

                running = {
                    'loss': 0.0,
                    'info_nce': 0.0,
                    'tm_bce': 0.0,
                    'tm_acc': 0.0,
                    'grad_norm': 0.0,
                }
                steps = 0

                for batch in tqdm(train_loader, unit='batch'):
                    batch_gpu = self._to_device(batch)
                    output = self.model(batch_gpu)
                    step_stats = self.optimizer._step(
                        output['info_nce_loss'],
                        output['tm_logit'],
                        batch_gpu['tmscores'],
                        grad_clip=self.config.grad_clip,
                    )

                    tm_label = step_stats['tm_label']
                    tm_pred_label = (output['tm_prob'].detach() >= 0.5).float()
                    tm_acc = (tm_pred_label == tm_label).float().mean()

                    steps += 1
                    running['loss'] += float(step_stats['loss'])
                    running['info_nce'] += float(output['info_nce_loss'].detach())
                    running['tm_bce'] += float(step_stats['aux_loss'])
                    running['tm_acc'] += float(tm_acc.detach())
                    running['grad_norm'] += float(step_stats['grad_norm'])

                self._log(
                    f"Train loss: {running['loss'] / max(steps, 1):.6f} | "
                    f"InfoNCE: {running['info_nce'] / max(steps, 1):.6f} | "
                    f"TM BCE: {running['tm_bce'] / max(steps, 1):.6f} | "
                    f"TM Acc: {running['tm_acc'] / max(steps, 1):.6f} | "
                    f"Grad norm: {running['grad_norm'] / max(steps, 1):.6f} | "
                    f"Alpha: {self.optimizer.alpha:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.8f}"
                )

                self._log('======================================================================')
                if (epoch + 1) % self.config.eval_interval == 0:
                    self._log('======================= Test =======================')
                    embs_lib = self._gen_eval_embeddings(lib_loader, mode='cand')
                    embs_test = self._gen_eval_embeddings(test_loader, mode='query')

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
                    _ = sorted_results
                    msg = (
                        f'Self-hit rate before filtering: {self_hit_rate:.4f}\n'
                        f'FAISS candidates per query: {search_k}\n'
                        f'Average Top-1 Remote homologous score: {avg_top1_score:.6f}\n'
                        f'Average Top-{self.config.topk} Remote homologous score: {avg_topk_score:.6f}\n'
                        f'Average Top-1 TM-score: {avg_top1_tm:.6f}\n'
                        f'Average Top-1 SeqID: {avg_top1_seqid:.6f}'
                    )
                    self._log(msg)
                    if avg_top1_score > self.max_top1_score or avg_topk_score > self.max_topk_score:
                        self.max_top1_score = max(self.max_top1_score, avg_top1_score)
                        self.max_topk_score = max(self.max_topk_score, avg_topk_score)
        finally:
            self.close()


if __name__ == '__main__':
    start_time = datetime.now()
    config = parse_args()

    if pt.cuda.is_available():
        pt.cuda.set_device(config.gpu)

    print('loading data')
    data = pt.load(config.data_path, weights_only=False)

    lib_data = ProteinDataset(data, mode='graph')
    lib_map = np.arange(len(lib_data), dtype=np.int64)
    pdb2idx = {str(lib_data[i]['lab']): i for i in range(len(lib_map))}
    print('number of library proteins:', len(lib_data))

    queryhomo = QueryHomologyDataset(None, config.pair_file, pdb2idx)
    train_map = np.arange(len(queryhomo), dtype=np.int64)
    train_map, test_map = train_test_split(
        train_map,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    test_map = np.sort(test_map)

    train_set = QueryHomologyDataset(queryhomo, mapping=train_map)
    test_set = QueryHomologyDataset(queryhomo, mapping=test_map)

    print('Train query-group num:', len(train_set))
    print('Test query num:', len(test_set))

    test_global_idx = np.array([int(test_set[i]['idx1']) for i in range(len(test_set))], dtype=np.int64)

    lib_loader = DataLoader(
        lib_data,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fun_emb(mode='cand'),
        pin_memory=True,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fun_train(lib_data, positive_top_ratio=config.positive_top_ratio),
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fun_emb(mode='query', protein_dataset=lib_data),
        pin_memory=True,
        drop_last=False,
    )

    sample_graph = lib_data[0]['graph']
    device = pt.device(f'cuda:{config.gpu}' if pt.cuda.is_available() else 'cpu')

    model = DualEncoderRetriever(
        vocab_size=config.vocab_size,
        node_feat_dim=sample_graph.x.shape[1],
        edge_feat_dim=sample_graph.edge_attr.shape[1] if getattr(sample_graph, 'edge_attr', None) is not None else 1,
        hidden_dim=config.hidden_dim,
        num_layers=config.gnn_num_layers,
        dropout=config.dropout,
        normalize=config.normalize,
        temperature=config.temperature,
        pooling=config.pooling,
    ).to(device)

    trainer = Trainer(model, config)
    trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        lib_loader=lib_loader,
        test_map=test_global_idx,
        test_set=test_set,
        datalib=lib_data,
    )

    pt.cuda.empty_cache()
    print(f'whole time consuming: {datetime.now() - start_time}')
