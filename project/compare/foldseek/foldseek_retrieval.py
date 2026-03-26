#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch as pt
from sklearn.model_selection import train_test_split

from utils.data import ProteinDataset, QueryHomologyDataset
from utils.tools import calculate_remote_homology_score


STRUCTURE_EXTENSIONS = [
    '.pdb', '.cif', '.mmcif', '.ent',
    '.pdb.gz', '.cif.gz', '.mmcif.gz', '.ent.gz',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Use Foldseek to perform structure retrieval for the current protein search task.'
    )
    parser.add_argument('--data_path', type=str, required=True, help='Same PT file used by train.py')
    parser.add_argument('--pdb_root', type=str, required=True, help='Root directory of structure files')
    parser.add_argument('--pair_file', type=str, default=None,
                        help='Optional TM-align pair file used to define the query set and evaluation set')

    parser.add_argument('--query_split', type=str, default='all', choices=['all', 'train', 'test'],
                        help='If pair_file is given, choose which query subset to use')
    parser.add_argument('--test_size', type=int, default=1024,
                        help='Same meaning as in train.py when query_split is train/test')
    parser.add_argument('--random_state', type=int, default=42)

    parser.add_argument('--foldseek', type=str, default='foldseek', help='Foldseek executable path')
    parser.add_argument('--threads', type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument('--gpu', action='store_true', help='Enable Foldseek GPU search if your build supports it')
    parser.add_argument('--alignment_type', type=int, default=2, choices=[0, 1, 2, 3],
                        help='0=3Di SW, 1=TMalign, 2=3Di+AA(default), 3=LoLalign')
    parser.add_argument('-s', '--sensitivity', type=float, default=None,
                        help='Optional Foldseek sensitivity parameter')
    parser.add_argument('--topk', type=int, default=12, help='Top-k hits kept per query after self-hit filtering')
    parser.add_argument('--remove_self_hit', action='store_true', default=True)
    parser.add_argument('--keep_self_hit', action='store_false', dest='remove_self_hit')
    parser.add_argument('--create_index', action='store_true',
                        help='Call foldseek createindex on the target DB for repeated searches')

    parser.add_argument('--workspace', type=str, default='./foldseek_workspace')
    parser.add_argument('--target_link_dir', type=str, default=None,
                        help='Optional directory to store flattened target structure symlinks')
    parser.add_argument('--query_link_dir', type=str, default=None,
                        help='Optional directory to store flattened query structure symlinks')
    parser.add_argument('--target_db', type=str, default=None,
                        help='Optional target DB path. If exists, createdb will be skipped')
    parser.add_argument('--tmp_dir', type=str, default=None,
                        help='Temporary directory for Foldseek runtime files')
    parser.add_argument('--result_tsv', type=str, default='./foldseek_results_raw.tsv')
    parser.add_argument('--rank_tsv', type=str, default='./foldseek_results_topk.tsv')
    parser.add_argument('--json_out', type=str, default='./foldseek_results_topk.json')

    parser.add_argument('--evaluate_with_tmalign', action='store_true',
                        help='Evaluate Foldseek top-k results using your existing TM-align-based metric')
    parser.add_argument('--tmalign_path', type=str, default='./TMalign')
    parser.add_argument('--tmalign_reference', type=int, default=1)
    parser.add_argument('--tmalign_workers', type=int, default=None)
    parser.add_argument('--metrics_out', type=str, default='./foldseek_metrics.json')

    return parser.parse_args()


def resolve_executable(path_or_name: str) -> str:
    resolved = shutil.which(path_or_name)
    if resolved:
        return resolved
    p = Path(path_or_name)
    if p.is_file():
        return str(p.resolve())
    raise FileNotFoundError(
        f'Cannot find Foldseek executable: {path_or_name}. '
        f'Please install Foldseek or pass --foldseek /path/to/foldseek.'
    )


def run_cmd(cmd: Sequence[str]) -> None:
    printable = ' '.join(map(str, cmd))
    print(f'[CMD] {printable}')
    result = subprocess.run(list(map(str, cmd)), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'Command failed: {printable}\n\n'
            f'stdout:\n{result.stdout}\n\n'
            f'stderr:\n{result.stderr}'
        )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())


def normalize_structure_name(name: str) -> str:
    base = Path(str(name)).name
    lowered = base.lower()
    changed = True
    while changed:
        changed = False
        for ext in STRUCTURE_EXTENSIONS:
            if lowered.endswith(ext):
                base = base[:-len(ext)]
                lowered = base.lower()
                changed = True
                break
    return base


def resolve_structure_path(pdb_root: str, pdb_name: str) -> Path:
    name = str(pdb_name)
    direct = Path(name)
    if direct.is_file():
        return direct.resolve()

    stem = normalize_structure_name(name)
    root = Path(pdb_root)
    subdir = stem[1:3] if len(stem) >= 3 else ''

    candidates: List[Path] = []
    if subdir:
        for ext in STRUCTURE_EXTENSIONS:
            candidates.append(root / subdir / f'{stem}{ext}')
    for ext in STRUCTURE_EXTENSIONS:
        candidates.append(root / f'{stem}{ext}')

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        f'Cannot resolve structure file for {pdb_name!r} under pdb_root={pdb_root}. '
        f'Tried {len(candidates)} candidates.'
    )


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_structure_dir(names: Iterable[str], pdb_root: str, out_dir: str) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, Path] = {}

    for raw_name in names:
        norm_name = normalize_structure_name(raw_name)
        if norm_name in mapping:
            continue
        src = resolve_structure_path(pdb_root, norm_name)
        suffix = ''.join(src.suffixes)
        if not suffix:
            suffix = '.pdb'
        dst = out / f'{norm_name}{suffix}'
        symlink_or_copy(src, dst)
        mapping[norm_name] = dst
    return mapping


def load_library(data_path: str) -> ProteinDataset:
    data = pt.load(data_path, weights_only=False)
    return ProteinDataset(data, mode='graph')


def split_query_dataset(
    lib_data: ProteinDataset,
    pair_file: Optional[str],
    query_split: str,
    test_size: int,
    random_state: int,
) -> Tuple[Optional[QueryHomologyDataset], List[str]]:
    if pair_file is None:
        query_names = [normalize_structure_name(lib_data[i]['lab']) for i in range(len(lib_data))]
        return None, query_names

    pdb2idx = {normalize_structure_name(str(lib_data[i]['lab'])): i for i in range(len(lib_data))}
    queryhomo = QueryHomologyDataset(None, pair_file, pdb2idx)
    all_map = np.arange(len(queryhomo), dtype=np.int64)

    if query_split == 'all':
        selected_map = all_map
    else:
        train_map, test_map = train_test_split(
            all_map,
            test_size=test_size,
            random_state=random_state,
        )
        selected_map = train_map if query_split == 'train' else np.sort(test_map)

    dataset = QueryHomologyDataset(queryhomo, mapping=np.asarray(selected_map, dtype=np.int64))
    query_names = [normalize_structure_name(str(lib_data[int(dataset[i]['idx1'])]['lab'])) for i in range(len(dataset))]
    return dataset, query_names


def maybe_create_target_db(
    foldseek_bin: str,
    target_dir: str,
    target_db: str,
    tmp_dir: str,
    create_index: bool,
    threads: int,
) -> None:
    db_path = Path(target_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db_exists = db_path.exists() or (db_path.parent / (db_path.name + '.dbtype')).exists()
    if not db_exists:
        run_cmd([foldseek_bin, 'createdb', target_dir, target_db])
    else:
        print(f'[INFO] Reuse existing Foldseek DB: {target_db}')

    if create_index:
        run_cmd([foldseek_bin, 'createindex', target_db, tmp_dir, '--threads', str(threads)])


def run_foldseek_search(
    foldseek_bin: str,
    query_dir: str,
    target_db: str,
    result_tsv: str,
    tmp_dir: str,
    alignment_type: int,
    threads: int,
    gpu: bool,
    sensitivity: Optional[float],
) -> None:
    ensure_parent(result_tsv)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        foldseek_bin,
        'easy-search',
        query_dir,
        target_db,
        result_tsv,
        tmp_dir,
        '--alignment-type', str(alignment_type),
        '--threads', str(threads),
        '--format-output', 'query,target,evalue,bits,alntmscore,qtmscore,ttmscore,fident,alnlen',
    ]
    if gpu:
        cmd += ['--gpu', '1']
    if sensitivity is not None:
        cmd += ['-s', str(sensitivity)]

    run_cmd(cmd)


FIELD_NAMES = ['query', 'target', 'evalue', 'bits', 'alntmscore', 'qtmscore', 'ttmscore', 'fident', 'alnlen']


def parse_foldseek_results(
    result_tsv: str,
    pdb2idx: Dict[str, int],
    query_names: Sequence[str],
    topk: int,
    remove_self_hit: bool,
) -> Tuple[List[List[int]], List[List[dict]]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)

    with open(result_tsv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row:
                continue
            rec = {FIELD_NAMES[i]: row[i] if i < len(row) else '' for i in range(len(FIELD_NAMES))}
            q = normalize_structure_name(rec['query'])
            t = normalize_structure_name(rec['target'])
            if t not in pdb2idx:
                continue
            grouped[q].append({
                'query': q,
                'target': t,
                'target_idx': int(pdb2idx[t]),
                'evalue': _to_float(rec['evalue']),
                'bits': _to_float(rec['bits']),
                'alntmscore': _to_float(rec['alntmscore']),
                'qtmscore': _to_float(rec['qtmscore']),
                'ttmscore': _to_float(rec['ttmscore']),
                'fident': _to_float(rec['fident']),
                'alnlen': _to_int(rec['alnlen']),
            })

    ranked_idx: List[List[int]] = []
    ranked_records: List[List[dict]] = []
    for q in query_names:
        hits: List[dict] = []
        seen = set()
        for rec in grouped.get(q, []):
            t_idx = rec['target_idx']
            if remove_self_hit and rec['target'] == q:
                continue
            if t_idx in seen:
                continue
            seen.add(t_idx)
            hits.append(rec)
            if len(hits) >= topk:
                break
        ranked_idx.append([h['target_idx'] for h in hits])
        ranked_records.append(hits)
    return ranked_idx, ranked_records


def _to_float(x: str) -> Optional[float]:
    x = str(x).strip()
    if x == '':
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _to_int(x: str) -> Optional[int]:
    x = str(x).strip()
    if x == '':
        return None
    try:
        return int(float(x))
    except ValueError:
        return None


def save_topk_outputs(
    rank_tsv: str,
    json_out: str,
    ranked_records: Sequence[Sequence[dict]],
) -> None:
    ensure_parent(rank_tsv)
    ensure_parent(json_out)

    with open(rank_tsv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['query', 'rank', 'target', 'target_idx', 'evalue', 'bits', 'alntmscore', 'qtmscore', 'ttmscore', 'fident', 'alnlen'])
        for hit_list in ranked_records:
            for rank, rec in enumerate(hit_list, start=1):
                writer.writerow([
                    rec['query'], rank, rec['target'], rec['target_idx'], rec['evalue'], rec['bits'],
                    rec['alntmscore'], rec['qtmscore'], rec['ttmscore'], rec['fident'], rec['alnlen']
                ])

    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(ranked_records, f, ensure_ascii=False, indent=2)


def evaluate_with_tmalign(
    query_set: Optional[QueryHomologyDataset],
    lib_data: ProteinDataset,
    ranked_idx: Sequence[Sequence[int]],
    topk: int,
    pdb_root: str,
    tmalign_path: str,
    reference: int,
    num_workers: Optional[int],
) -> dict:
    if query_set is None:
        raise ValueError('--evaluate_with_tmalign requires --pair_file so that the query set is defined.')

    sorted_results, avg_top1_score, avg_topk_score, avg_top1_tm, avg_top1_seqid = calculate_remote_homology_score(
        query=query_set,
        database=lib_data,
        idx=list(ranked_idx),
        k=topk,
        pdb_root=pdb_root,
        tmalign_path=tmalign_path,
        reference=reference,
        num_workers=num_workers,
    )
    return {
        'avg_top1_score': float(avg_top1_score),
        'avg_topk_score': float(avg_topk_score),
        'avg_top1_tm': float(avg_top1_tm),
        'avg_top1_seqid': float(avg_top1_seqid),
        'num_queries': int(len(query_set)),
        'sorted_results': sorted_results,
    }


def main() -> None:
    args = parse_args()
    foldseek_bin = resolve_executable(args.foldseek)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    target_link_dir = args.target_link_dir or str(workspace / 'target_structures')
    query_link_dir = args.query_link_dir or str(workspace / 'query_structures')
    target_db = args.target_db or str(workspace / 'targetDB')
    tmp_dir = args.tmp_dir or str(workspace / 'tmp')

    print('[INFO] Loading library data...')
    lib_data = load_library(args.data_path)
    pdb2idx = {normalize_structure_name(str(lib_data[i]['lab'])): i for i in range(len(lib_data))}
    print(f'[INFO] Library size: {len(lib_data)}')

    query_set, query_names = split_query_dataset(
        lib_data=lib_data,
        pair_file=args.pair_file,
        query_split=args.query_split,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f'[INFO] Query size: {len(query_names)}')

    print('[INFO] Preparing flattened target structure directory...')
    target_names = [normalize_structure_name(str(lib_data[i]['lab'])) for i in range(len(lib_data))]
    prepare_structure_dir(target_names, args.pdb_root, target_link_dir)

    print('[INFO] Preparing flattened query structure directory...')
    prepare_structure_dir(query_names, args.pdb_root, query_link_dir)

    print('[INFO] Building/reusing Foldseek target database...')
    maybe_create_target_db(
        foldseek_bin=foldseek_bin,
        target_dir=target_link_dir,
        target_db=target_db,
        tmp_dir=tmp_dir,
        create_index=args.create_index,
        threads=args.threads,
    )

    print('[INFO] Running Foldseek search...')
    run_foldseek_search(
        foldseek_bin=foldseek_bin,
        query_dir=query_link_dir,
        target_db=target_db,
        result_tsv=args.result_tsv,
        tmp_dir=tmp_dir,
        alignment_type=args.alignment_type,
        threads=args.threads,
        gpu=args.gpu,
        sensitivity=args.sensitivity,
    )

    print('[INFO] Parsing Foldseek results...')
    ranked_idx, ranked_records = parse_foldseek_results(
        result_tsv=args.result_tsv,
        pdb2idx=pdb2idx,
        query_names=query_names,
        topk=args.topk,
        remove_self_hit=args.remove_self_hit,
    )

    missing = sum(1 for x in ranked_records if len(x) == 0)
    print(f'[INFO] Queries with zero retained hits: {missing}/{len(ranked_records)}')
    save_topk_outputs(args.rank_tsv, args.json_out, ranked_records)
    print(f'[INFO] Saved Top-k TSV to: {args.rank_tsv}')
    print(f'[INFO] Saved Top-k JSON to: {args.json_out}')

    if args.evaluate_with_tmalign:
        print('[INFO] Evaluating with TM-align-based remote homology metric...')
        metrics = evaluate_with_tmalign(
            query_set=query_set,
            lib_data=lib_data,
            ranked_idx=ranked_idx,
            topk=args.topk,
            pdb_root=args.pdb_root,
            tmalign_path=args.tmalign_path,
            reference=args.tmalign_reference,
            num_workers=args.tmalign_workers,
        )
        sorted_results = metrics.pop('sorted_results')
        ensure_parent(args.metrics_out)
        with open(args.metrics_out, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(Path(args.metrics_out).with_suffix('.sorted_results.json'), 'w', encoding='utf-8') as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)
        print('[METRICS]')
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f'[INFO] Saved metrics to: {args.metrics_out}')


if __name__ == '__main__':
    main()
