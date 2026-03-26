import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch as pt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.data import ProteinDataset, QueryHomologyDataset
from utils.tools import calculate_remote_homology_score


DEFAULT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "pyhmmer-based remote homology retrieval baseline. "
            "This script replaces the learned embedding + FAISS retrieval stage "
            "with pyhmmer.phmmer sequence search, while keeping the original "
            "TM-align-based evaluation."
        )
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the torch-saved protein library file, same as your original train.py.')
    parser.add_argument('--pair_file', type=str, required=True,
                        help='Path to the pair file used by QueryHomologyDataset, same as your original train.py.')
    parser.add_argument('--pdb_root', type=str, required=True,
                        help='Root directory of PDB files, used by TM-align evaluation.')
    parser.add_argument('--tmalign_path', type=str, required=True,
                        help='Path to the TMalign executable.')
    parser.add_argument('--output_dir', type=str, default='./pyhmmer_runs',
                        help='Directory for evaluation results.')

    parser.add_argument('--cpus', type=int, default=1,
                        help='Number of worker threads used inside pyhmmer.hmmer.phmmer.')
    parser.add_argument('--evalue', type=float, default=1000.0,
                        help='Loose reporting threshold passed as E, so enough candidates are kept before top-k truncation.')
    parser.add_argument('--domE', type=float, default=None,
                        help='Optional domain reporting threshold passed as domE.')
    parser.add_argument('--incE', type=float, default=None,
                        help='Optional inclusion threshold passed as incE.')
    parser.add_argument('--eval_search_k', type=int, default=200,
                        help='How many pyhmmer candidates to keep per query before TM-align reranking/evaluation.')
    parser.add_argument('--topk', type=int, default=12,
                        help='Final top-k candidates evaluated by TM-align, matching your original project.')
    parser.add_argument('--tmalign_reference', type=int, default=1,
                        help='Reference TM-score index used in your original project.')
    parser.add_argument('--tmalign_workers', type=int, default=None,
                        help='Number of workers used inside TM-align evaluation.')

    parser.add_argument('--test_size', type=int, default=1024,
                        help='Test split size, aligned with your original train.py default.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for query split, aligned with your original train.py default.')
    parser.add_argument('--use_all_queries', action='store_true',
                        help='Evaluate all query groups instead of the train/test split from the original project.')

    parser.add_argument('--alphabet', type=str, default=DEFAULT_ALPHABET,
                        help=(
                            'Amino-acid alphabet for sequence decoding, corresponding to token ids 1..20. '
                            'Padding is assumed to be 0. Default: ACDEFGHIKLMNPQRSTVWY. '
                            'Change this if your integer encoding uses a different order.'
                        ))
    return parser.parse_args()


def decode_sequence(tokens, alphabet: str) -> str:
    if len(alphabet) != 20:
        raise ValueError(f'alphabet must contain exactly 20 amino-acid characters, got {len(alphabet)}: {alphabet}')

    if isinstance(tokens, pt.Tensor):
        arr = tokens.detach().cpu().numpy()
    else:
        arr = np.asarray(tokens)

    aa = []
    for x in arr.tolist():
        idx = int(x)
        if idx == 0:
            continue
        if 1 <= idx <= 20:
            aa.append(alphabet[idx - 1])
        else:
            aa.append('X')
    return ''.join(aa)


def normalize_label(label) -> str:
    if isinstance(label, bytes):
        label = label.decode('utf-8')
    return str(label)


def decode_bytes(x) -> str:
    if x is None:
        return ''
    if isinstance(x, bytes):
        return x.decode('utf-8', errors='replace')
    return str(x)


def build_digital_sequence(name: str, seq: str, alphabet_obj):
    import pyhmmer

    if not seq:
        raise ValueError(f'Empty sequence for {name}')

    text_seq = pyhmmer.easel.TextSequence(
        name=name.encode('utf-8'),
        sequence=seq.encode('utf-8'),
    )
    return text_seq.digitize(alphabet_obj)


def build_library_sequences(dataset: ProteinDataset, alphabet: str):
    import pyhmmer

    alphabet_obj = pyhmmer.easel.Alphabet.amino()
    sequences = []
    label_to_global_idx: Dict[str, int] = {}

    for local_idx in tqdm(range(len(dataset)), desc='Preparing library sequences'):
        item = dataset[local_idx]
        label = normalize_label(item['lab'])
        seq = decode_sequence(item['seq'], alphabet)
        if not seq:
            continue
        sequences.append(build_digital_sequence(label, seq, alphabet_obj))
        label_to_global_idx[label] = int(dataset.map[local_idx])

    return alphabet_obj, sequences, label_to_global_idx


def build_query_sequences(query_set: QueryHomologyDataset, lib_data: ProteinDataset, alphabet_obj, alphabet: str):
    query_sequences = []
    query_names = []
    query_global_indices = []

    for q_pos in tqdm(range(len(query_set)), desc='Preparing query sequences'):
        q_group = query_set[q_pos]
        q_global_idx = int(q_group['idx1'])
        q_item = lib_data[q_global_idx]
        q_name = normalize_label(q_item['lab'])
        q_seq = decode_sequence(q_item['seq'], alphabet)
        query_sequences.append(build_digital_sequence(q_name, q_seq, alphabet_obj))
        query_names.append(q_name)
        query_global_indices.append(q_global_idx)

    return query_sequences, query_names, query_global_indices


def get_best_domain_stats(hit) -> tuple[float, float]:
    domains = getattr(hit, 'domains', None)
    if domains is None or len(domains) == 0:
        return math.nan, math.nan

    best_domain_score = -math.inf
    best_domain_evalue = math.inf
    for dom in domains:
        score = float(getattr(dom, 'score', math.nan))
        i_evalue = float(getattr(dom, 'i_evalue', math.inf))
        if score > best_domain_score:
            best_domain_score = score
        if i_evalue < best_domain_evalue:
            best_domain_evalue = i_evalue

    if best_domain_score == -math.inf:
        best_domain_score = math.nan
    if best_domain_evalue == math.inf:
        best_domain_evalue = math.nan
    return best_domain_evalue, best_domain_score


def search_with_pyhmmer(
    query_sequences: Sequence,
    query_names: Sequence[str],
    query_global_indices: Sequence[int],
    target_sequences: Sequence,
    label_to_global_idx: Dict[str, int],
    cpus: int,
    evalue: float,
    domE: float | None,
    incE: float | None,
    eval_search_k: int,
):
    import pyhmmer

    kwargs = {
        'cpus': cpus,
        'E': evalue,
    }
    if domE is not None:
        kwargs['domE'] = domE
    if incE is not None:
        kwargs['incE'] = incE

    hits_iter = pyhmmer.hmmer.phmmer(query_sequences, target_sequences, **kwargs)

    all_candidates: List[List[int]] = [[] for _ in range(len(query_sequences))]
    all_hit_records: List[dict] = []
    parsed_hit_count = np.zeros(len(query_sequences), dtype=np.int64)

    for q_pos, hits in enumerate(tqdm(hits_iter, total=len(query_sequences), desc='pyhmmer queries')):
        q_name = query_names[q_pos]
        q_global_idx = query_global_indices[q_pos]

        candidate_indices = []
        seen_global = set()
        total_hits = len(hits)
        parsed_hit_count[q_pos] = total_hits

        for rank, hit in enumerate(hits, start=1):
            target_name = decode_bytes(hit.name)
            if target_name == q_name:
                continue
            if target_name not in label_to_global_idx:
                continue

            cand_global_idx = int(label_to_global_idx[target_name])
            if cand_global_idx == q_global_idx or cand_global_idx in seen_global:
                continue

            seen_global.add(cand_global_idx)
            best_domain_evalue, best_domain_score = get_best_domain_stats(hit)
            candidate_indices.append(cand_global_idx)
            all_hit_records.append({
                'query_pos': q_pos,
                'query_name': q_name,
                'query_global_idx': q_global_idx,
                'rank_from_pyhmmer': rank,
                'target_name': target_name,
                'target_global_idx': cand_global_idx,
                'full_evalue': float(getattr(hit, 'evalue', math.nan)),
                'full_score': float(getattr(hit, 'score', math.nan)),
                'best_domain_evalue': best_domain_evalue,
                'best_domain_score': best_domain_score,
            })
            if len(candidate_indices) >= eval_search_k:
                break

        all_candidates[q_pos] = candidate_indices

    return all_candidates, all_hit_records, parsed_hit_count


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyhmmer  # noqa: F401
    except Exception as e:
        raise ImportError(
            'pyhmmer is required for this script. Install it with '\
            '`pip install pyhmmer` or `conda install -c bioconda pyhmmer`. '
            f'Original import error: {e}'
        )

    print('[INFO] Loading data...')
    data = pt.load(args.data_path, weights_only=False)
    lib_data = ProteinDataset(data, mode='graph')
    pdb2idx = {normalize_label(lib_data[i]['lab']): i for i in range(len(lib_data))}
    queryhomo = QueryHomologyDataset(None, args.pair_file, pdb2idx)

    if args.use_all_queries:
        query_set = queryhomo
    else:
        full_map = np.arange(len(queryhomo), dtype=np.int64)
        _, test_map = train_test_split(
            full_map,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        test_map = np.sort(test_map)
        query_set = QueryHomologyDataset(queryhomo, mapping=test_map)

    print(f'[INFO] Library size: {len(lib_data)}')
    print(f'[INFO] Query size: {len(query_set)}')

    alphabet_obj, target_sequences, label_to_global_idx = build_library_sequences(lib_data, args.alphabet)
    query_sequences, query_names, query_global_indices = build_query_sequences(
        query_set=query_set,
        lib_data=lib_data,
        alphabet_obj=alphabet_obj,
        alphabet=args.alphabet,
    )

    print('[INFO] Running pyhmmer retrieval...')
    all_candidates, all_hit_records, parsed_hit_count = search_with_pyhmmer(
        query_sequences=query_sequences,
        query_names=query_names,
        query_global_indices=query_global_indices,
        target_sequences=target_sequences,
        label_to_global_idx=label_to_global_idx,
        cpus=args.cpus,
        evalue=args.evalue,
        domE=args.domE,
        incE=args.incE,
        eval_search_k=max(args.eval_search_k, args.topk + 1),
    )

    insufficient = sum(len(x) < args.topk for x in all_candidates)
    print(f'[INFO] Queries with fewer than topk={args.topk} candidates after pyhmmer/self-hit filtering: {insufficient}/{len(all_candidates)}')
    print(f'[INFO] Mean parsed pyhmmer hits per query (before filtering): {parsed_hit_count.mean():.2f}')

    sorted_results, avg_top1_score, avg_topk_score, avg_top1_tm, avg_top1_seqid = calculate_remote_homology_score(
        query=query_set,
        database=lib_data,
        idx=all_candidates,
        k=args.topk,
        pdb_root=args.pdb_root,
        tmalign_path=args.tmalign_path,
        reference=args.tmalign_reference,
        num_workers=args.tmalign_workers,
    )

    metrics = {
        'tool': 'pyhmmer.phmmer',
        'library_size': int(len(lib_data)),
        'query_size': int(len(query_set)),
        'topk': int(args.topk),
        'eval_search_k': int(max(args.eval_search_k, args.topk + 1)),
        'cpus': int(args.cpus),
        'evalue': float(args.evalue),
        'domE': None if args.domE is None else float(args.domE),
        'incE': None if args.incE is None else float(args.incE),
        'avg_parsed_hits_before_filter': float(parsed_hit_count.mean()) if len(parsed_hit_count) > 0 else 0.0,
        'queries_with_insufficient_candidates': int(insufficient),
        'avg_top1_score': float(avg_top1_score),
        'avg_topk_score': float(avg_topk_score),
        'avg_top1_tm': float(avg_top1_tm),
        'avg_top1_seqid': float(avg_top1_seqid),
    }

    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')

    if all_hit_records:
        pd.DataFrame(all_hit_records).sort_values(['query_pos', 'rank_from_pyhmmer']).to_csv(
            output_dir / 'pyhmmer_hits.tsv',
            sep='\t',
            index=False,
        )

    summary_rows = []
    for q_pos, cand_list in sorted_results.items():
        q_name = normalize_label(lib_data[int(query_set[q_pos]['idx1'])]['lab'])
        if len(cand_list) == 0:
            summary_rows.append({
                'query_pos': q_pos,
                'query_name': q_name,
                'best_target_name': None,
                'best_score': math.nan,
                'best_tm_score': math.nan,
                'best_seqid': math.nan,
                'num_tmalign_valid_hits': 0,
            })
            continue
        best = cand_list[0]
        summary_rows.append({
            'query_pos': q_pos,
            'query_name': q_name,
            'best_target_name': normalize_label(lib_data[int(best['c_idx'])]['lab']),
            'best_score': best['score'],
            'best_tm_score': best['tm_score'],
            'best_seqid': best['seqid'],
            'num_tmalign_valid_hits': len(cand_list),
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / 'top1_summary.tsv', sep='\t', index=False)

    print('==================== pyhmmer evaluation summary ====================')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f'[INFO] metrics.json saved to: {output_dir / "metrics.json"}')
    print(f'[INFO] pyhmmer_hits.tsv saved to: {output_dir / "pyhmmer_hits.tsv"}')
    print(f'[INFO] top1_summary.tsv saved to: {output_dir / "top1_summary.tsv"}')


if __name__ == '__main__':
    main()
