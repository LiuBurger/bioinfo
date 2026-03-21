import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as pt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils.data4_seq_transformer_seqid_remote import ProteinDataset, QueryHomologyDataset, collate_fun_train, collate_fun_emb
from utils.model4_seq_transformer_seqid_remote_v2 import DualEncoderRetriever, MultiTaskOptimizer
from utils.tools4 import build_idx, calculate_remote_homology_score


def parse_args():
    parser = argparse.ArgumentParser(description='Shared-trunk SeqTransformer + GINE dual-encoder training config')

    # training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    # optimization
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--alpha_remote', type=float, default=0.2)
    parser.add_argument('--alpha_seqid', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=2.0)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--positive_top_ratio', type=float, default=0.3)
    parser.add_argument('--aux_loss_type', type=str, default='smooth_l1', choices=['smooth_l1', 'mse', 'l1'])
    parser.add_argument('--aux_beta', type=float, default=0.05)

    # model
    parser.add_argument('--vocab_size', type=int, default=21)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--gnn_num_layers', type=int, default=3)
    parser.add_argument('--transformer_num_layers', type=int, default=1)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--no_normalize', action='store_false', dest='normalize')

    # retrieval / evaluation
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--eval_search_k', type=int, default=200)
    parser.add_argument('--tmalign_reference', type=int, default=1)
    parser.add_argument('--tmalign_workers', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--eval_interval', type=int, default=10)

    # checkpoint / early stop / scheduler
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--monitor', type=str, default='top1', choices=['top1', 'topk'])
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='number of evaluation rounds without improvement before stopping')
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=int, default=2,
                        help='ReduceLROnPlateau patience in evaluation rounds')
    parser.add_argument('--scheduler_threshold', type=float, default=1e-4)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6)
    parser.add_argument('--save_last', action='store_true', default=True)
    parser.add_argument('--no_save_last', action='store_false', dest='save_last')

    # paths
    parser.add_argument('--data_path', type=str, default='./data/sorted_1300_p0_h0.pt')
    parser.add_argument('--pdb_root', type=str, default='../../data/pdb')
    parser.add_argument('--tmalign_path', type=str, default='./TMalign')
    parser.add_argument('--pair_file', type=str, default='./data/tmalign.out')
    parser.add_argument('--model_name', type=str, default='SharedTransformerGINE_AttnPool')

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
            alpha_remote=config.alpha_remote,
            alpha_seqid=config.alpha_seqid,
            aux_loss_type=config.aux_loss_type,
            aux_beta=config.aux_beta,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            threshold=config.scheduler_threshold,
            threshold_mode='abs',
            min_lr=config.scheduler_min_lr,
        )

        self.save_dir = Path(config.save_dir) / config.model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = float('-inf')
        self.best_top1_score = float('-inf')
        self.best_topk_score = float('-inf')
        self.best_epoch = -1
        self.no_improve_count = 0

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

    def _get_monitor_value(self, metrics: dict) -> float:
        if self.config.monitor == 'top1':
            return float(metrics['avg_top1_score'])
        if self.config.monitor == 'topk':
            return float(metrics['avg_topk_score'])
        raise ValueError(f'Unsupported monitor: {self.config.monitor}')

    def _is_better(self, metric_value: float) -> bool:
        return metric_value > (self.best_metric + self.config.min_delta)

    def _save_checkpoint(self, epoch: int, metrics: dict, filename: str, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'best_top1_score': self.best_top1_score,
            'best_topk_score': self.best_topk_score,
            'best_epoch': self.best_epoch,
            'config': vars(self.config),
            'metrics': metrics,
        }
        save_path = self.save_dir / filename
        pt.save(ckpt, save_path)
        tag = 'BEST' if is_best else 'LAST'
        self._log(f'[{tag} CHECKPOINT] saved to: {save_path}')

    @pt.no_grad()
    def _evaluate(self, test_loader, lib_loader, test_map, test_set, datalib):
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

        metrics = {
            'self_hit_rate': float(self_hit_rate),
            'search_k': int(search_k),
            'avg_top1_score': float(avg_top1_score),
            'avg_topk_score': float(avg_topk_score),
            'avg_top1_tm': float(avg_top1_tm),
            'avg_top1_seqid': float(avg_top1_seqid),
        }
        monitor_value = self._get_monitor_value(metrics)

        msg = (
            f'Self-hit rate before filtering: {metrics["self_hit_rate"]:.4f}\n'
            f'FAISS candidates per query: {metrics["search_k"]}\n'
            f'Average Top-1 Remote homologous score: {metrics["avg_top1_score"]:.6f}\n'
            f'Average Top-{self.config.topk} Remote homologous score: {metrics["avg_topk_score"]:.6f}\n'
            f'Average Top-1 TM-score: {metrics["avg_top1_tm"]:.6f}\n'
            f'Average Top-1 SeqID: {metrics["avg_top1_seqid"]:.6f}\n'
            f'Monitor ({self.config.monitor}): {monitor_value:.6f}'
        )
        self._log(msg)
        return sorted_results, metrics, monitor_value

    def train(self, train_loader, test_loader, lib_loader, test_map, test_set, datalib):
        try:
            for epoch in range(self.config.epochs):
                self._log(f'======================= Train epoch {epoch + 1} =======================')
                self.model.train()

                running = {
                    'loss': 0.0,
                    'info_nce': 0.0,
                    'remote_aux': 0.0,
                    'seqid_aux': 0.0,
                    'remote_mae': 0.0,
                    'seqid_mae': 0.0,
                    'grad_norm': 0.0,
                }
                steps = 0

                for batch in tqdm(train_loader, unit='batch'):
                    batch_gpu = self._to_device(batch)
                    output = self.model(batch_gpu)
                    step_stats = self.optimizer._step(
                        output['info_nce_loss'],
                        output['remote_score_pred'],
                        batch_gpu['remote_scores'],
                        output['seqid_pred'],
                        batch_gpu['seqids'],
                        grad_clip=self.config.grad_clip,
                    )

                    remote_mae = (output['remote_score_pred'].detach() - batch_gpu['remote_scores']).abs().mean()
                    seqid_mae = (output['seqid_pred'].detach() - batch_gpu['seqids']).abs().mean()

                    steps += 1
                    running['loss'] += float(step_stats['loss'])
                    running['info_nce'] += float(output['info_nce_loss'].detach())
                    running['remote_aux'] += float(step_stats['remote_aux'])
                    running['seqid_aux'] += float(step_stats['seqid_aux'])
                    running['remote_mae'] += float(remote_mae.detach())
                    running['seqid_mae'] += float(seqid_mae.detach())
                    running['grad_norm'] += float(step_stats['grad_norm'])

                current_lr = self.optimizer.param_groups[0]['lr']
                self._log(
                    f"Train loss: {running['loss'] / max(steps, 1):.6f} | "
                    f"InfoNCE: {running['info_nce'] / max(steps, 1):.6f} | "
                    f"Remote aux: {running['remote_aux'] / max(steps, 1):.6f} | "
                    f"SEQID aux: {running['seqid_aux'] / max(steps, 1):.6f} | "
                    f"Remote MAE: {running['remote_mae'] / max(steps, 1):.6f} | "
                    f"SEQID MAE: {running['seqid_mae'] / max(steps, 1):.6f} | "
                    f"Grad norm: {running['grad_norm'] / max(steps, 1):.6f} | "
                    f"AlphaRemote: {self.optimizer.alpha_remote:.4f} | "
                    f"AlphaSEQID: {self.optimizer.alpha_seqid:.4f} | "
                    f"LR: {current_lr:.8f}"
                )

                self._log('======================================================================')

                if (epoch + 1) % self.config.eval_interval != 0:
                    continue

                _, eval_metrics, monitor_value = self._evaluate(
                    test_loader=test_loader,
                    lib_loader=lib_loader,
                    test_map=test_map,
                    test_set=test_set,
                    datalib=datalib,
                )

                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(monitor_value)
                new_lr = self.optimizer.param_groups[0]['lr']
                if abs(new_lr - old_lr) > 1e-12:
                    self._log(
                        f'[LR SCHEDULER] monitor={self.config.monitor} | '
                        f'{old_lr:.8f} -> {new_lr:.8f}'
                    )

                improved = self._is_better(monitor_value)
                if improved:
                    self.best_metric = monitor_value
                    self.best_top1_score = eval_metrics['avg_top1_score']
                    self.best_topk_score = eval_metrics['avg_topk_score']
                    self.best_epoch = epoch + 1
                    self.no_improve_count = 0
                    self._save_checkpoint(
                        epoch=epoch + 1,
                        metrics=eval_metrics,
                        filename='best.pt',
                        is_best=True,
                    )
                    self._log(
                        f'[BEST UPDATED] epoch={self.best_epoch} | '
                        f'best_{self.config.monitor}={self.best_metric:.6f} | '
                        f'best_top1={self.best_top1_score:.6f} | '
                        f'best_top{self.config.topk}={self.best_topk_score:.6f}'
                    )
                else:
                    self.no_improve_count += 1
                    self._log(
                        f'[NO IMPROVEMENT] eval_rounds_without_improvement='
                        f'{self.no_improve_count}/{self.config.early_stop_patience} | '
                        f'best_epoch={self.best_epoch} | '
                        f'best_{self.config.monitor}={self.best_metric:.6f}'
                    )

                if self.config.save_last:
                    self._save_checkpoint(
                        epoch=epoch + 1,
                        metrics=eval_metrics,
                        filename='last.pt',
                        is_best=False,
                    )

                if self.no_improve_count >= self.config.early_stop_patience:
                    self._log(
                        f'[EARLY STOP] Stop training at epoch {epoch + 1}. '
                        f'Best epoch: {self.best_epoch}, '
                        f'best_{self.config.monitor}: {self.best_metric:.6f}, '
                        f'best_top1: {self.best_top1_score:.6f}, '
                        f'best_top{self.config.topk}: {self.best_topk_score:.6f}'
                    )
                    break
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
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate_fun_emb(mode='cand'),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate_fun_emb(mode='query', protein_dataset=lib_data),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate_fun_train(protein_dataset=lib_data, positive_top_ratio=config.positive_top_ratio),
    )

    device = pt.device(f'cuda:{config.gpu}' if pt.cuda.is_available() else 'cpu')
    sample_graph = lib_data[0]['graph']
    model = DualEncoderRetriever(
        vocab_size=config.vocab_size,
        node_feat_dim=sample_graph.x.shape[1],
        edge_feat_dim=sample_graph.edge_attr.shape[1] if getattr(sample_graph, 'edge_attr', None) is not None else 1,
        hidden_dim=config.hidden_dim,
        proj_dim=config.proj_dim,
        gnn_num_layers=config.gnn_num_layers,
        transformer_num_layers=config.transformer_num_layers,
        transformer_heads=config.transformer_heads,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        normalize=config.normalize,
        temperature=config.temperature,
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

    end_time = datetime.now()
    print(f'Total runtime: {end_time - start_time}')
