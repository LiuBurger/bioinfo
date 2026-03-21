import os
import re
import faiss
import numpy as np
import torch as pt
import torch.nn as nn
import subprocess
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed


@pt.no_grad()
def gen_embeddings(model: nn.Module, loader: DataLoader, gpu: int, mode: str = 'query'):
    embs = []
    model.eval()
    device = pt.device(f'cuda:{gpu}' if pt.cuda.is_available() else 'cpu')
    for data in loader:
        data_gpu = {}
        for k, v in data.items():
            data_gpu[k] = v.to(device, non_blocking=True) if hasattr(v, 'to') else v
        emb = model.encode(data_gpu, mode).detach().cpu().numpy().astype(np.float32)
        embs.append(emb)
    if len(embs) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    embs = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, a_min=1e-12, a_max=None)
    pt.cuda.empty_cache()
    return embs


def build_idx(embs_lib: np.ndarray, embs_query: np.ndarray, topk: int = 200, keep_index: bool = False):
    assert embs_lib.shape[1] == embs_query.shape[1], 'Dimension not match'
    index_flat = faiss.IndexFlatIP(embs_lib.shape[1])
    index_flat.add(embs_lib.astype('float32'))
    time_start = datetime.now()
    scores, indices = index_flat.search(embs_query.astype('float32'), topk)
    print('faiss cpu searching time:', datetime.now() - time_start)
    if not keep_index:
        index_flat.reset()
    pt.cuda.empty_cache()
    return indices, scores


def save_model(model: nn.Module, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    pt.save(model.state_dict(), save_path)


def make_pdb_path(pdb_root, pdb_name: str) -> Path:
    return Path(pdb_root) / pdb_name[1:3] / f'{pdb_name}.pdb'


def generate_tasks(
    query, database, idx: list,
    k: int = 12,
    pdb_root: str = '../../data/pdb',
    tmalign_path: str = './TMalign',
    reference: int = 1,
):
    pdb_root = Path(pdb_root)
    tmalign_path = Path(tmalign_path).resolve()
    if not tmalign_path.is_file():
        raise FileNotFoundError(f'TMalign binary not found: {tmalign_path}')
    for i in range(len(query)):
        q_name = str(database[query[i]['idx1']]['lab'])
        q_file = make_pdb_path(pdb_root, q_name)
        if not q_file.is_file():
            raise FileNotFoundError(f'Query PDB file not found: {q_file}')
        for j in idx[i][:k]:
            c_name = str(database[int(j)]['lab'])
            c_file = make_pdb_path(pdb_root, c_name)
            if not c_file.is_file():
                raise FileNotFoundError(f'Candidate PDB file not found: {c_file}')
            yield (i, int(j), q_file, c_file, tmalign_path, reference)


def run_tmalign(task):
    q_idx, c_idx, pdb1, pdb2, tmalign_path, reference = task
    cmd = [str(tmalign_path), str(pdb1), str(pdb2)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cmd_str = ' '.join(cmd)
        raise RuntimeError(
            f'TMalign failed.\n'
            f'cmd: {cmd_str}\n'
            f'returncode: {result.returncode}\n'
            f'stderr: {result.stderr}'
        )
    output = result.stdout
    seqid_match = re.search(r'Seq_ID=.*?=\s*([0-9.]+)', output)
    tm_scores = re.findall(r'TM-score=\s*([0-9.]+)', output)
    if seqid_match is None:
        raise ValueError(f'Failed to parse Seq_ID from TMalign output.\nstdout:\n{output}')
    if not (1 <= reference <= len(tm_scores)):
        raise ValueError(f'Invalid reference={reference}. Parsed {len(tm_scores)} TM-scores.\nstdout:\n{output}')

    seqid = float(seqid_match.group(1))
    tm_score = float(tm_scores[reference - 1])
    score = tm_score - 0.6 + min(0.4 - seqid, 0.0)

    return q_idx, c_idx, {
        'score': score,
        'tm_score': tm_score,
        'seqid': seqid,
    }


def calculate_remote_homology_score(
    query, database, idx: list,
    k: int = 12,
    pdb_root: str = '../../data/pdb',
    tmalign_path: str = './TMalign',
    reference: int = 1,
    num_workers: int = None,
):
    n = len(query)
    assert n == len(idx), 'The length of query and idx should be the same.'
    tasks = list(
        generate_tasks(
            query=query,
            database=database,
            idx=idx,
            k=k,
            pdb_root=pdb_root,
            tmalign_path=tmalign_path,
            reference=reference,
        )
    )
    if len(tasks) == 0:
        raise ValueError('No TM-align tasks were generated.')
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, min(len(tasks), cpu_count // 2 if cpu_count > 1 else 1))

    results_dict = {i: [] for i in range(n)}
    avg_topk_score = 0.0
    success_cnt = 0
    error_messages = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_tmalign, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc='TM-align', leave=False):
            try:
                q_idx, c_idx, result = future.result()
                results_dict[q_idx].append({
                    'c_idx': c_idx,
                    'score': result['score'],
                    'tm_score': result['tm_score'],
                    'seqid': result['seqid'],
                })
                avg_topk_score += result['score']
                success_cnt += 1
            except Exception as e:
                error_messages.append(str(e))
    if success_cnt == 0:
        raise RuntimeError('All TM-align tasks failed.\n' + ('\n\nFirst error:\n' + error_messages[0] if error_messages else ''))
    if error_messages:
        print(f'[Warning] {len(error_messages)} TM-align tasks failed. First error:\n{error_messages[0]}')
    avg_topk_score /= success_cnt

    sorted_results = {}
    avg_top1_score = 0.0
    avg_top1_tm = 0.0
    avg_top1_seqid = 0.0
    valid_queries = 0
    for q_idx in range(n):
        c_list = results_dict[q_idx]
        c_list.sort(key=lambda x: x['score'], reverse=True)
        sorted_results[q_idx] = c_list
        if len(c_list) > 0:
            avg_top1_score += c_list[0]['score']
            avg_top1_tm += c_list[0]['tm_score']
            avg_top1_seqid += c_list[0]['seqid']
            valid_queries += 1
    if valid_queries > 0:
        avg_top1_score /= valid_queries
        avg_top1_tm /= valid_queries
        avg_top1_seqid /= valid_queries

    return sorted_results, avg_top1_score, avg_topk_score, avg_top1_tm, avg_top1_seqid
