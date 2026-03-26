#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch as pt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

STANDARD_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


class SimpleProteinDataset:
    def __init__(self, dataset):
        if isinstance(dataset, tuple):
            self.seq = dataset[0]
            self.lab = dataset[1]
        else:
            self.seq = dataset.seq
            self.lab = dataset.lab
        self.map = np.arange(len(self.lab), dtype=np.int64)

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        idx = int(self.map[idx])
        return {
            'seq': self.seq[idx],
            'lab': self.lab[idx],
        }


class QueryHomologyDatasetLite:
    def __init__(self, pair_file: str, pdb2idx: Dict[str, int], mapping: Optional[np.ndarray] = None):
        df = pd.read_csv(pair_file, names=['name1', 'name2', 'tmscore', 'seqid'], delimiter='\t')
        df['name1'] = df['name1'].astype(str).str.split('/').str[-1].str.removesuffix('.pdb')
        df['name2'] = df['name2'].astype(str).str.split('/').str[-1].str.removesuffix('.pdb')
        df['idx1'] = df['name1'].map(pdb2idx)
        df['idx2'] = df['name2'].map(pdb2idx)
        df = df.dropna(subset=['idx1', 'idx2']).copy()
        df['idx1'] = df['idx1'].astype(np.int64)
        df['idx2'] = df['idx2'].astype(np.int64)

        groups = []
        for idx1, g in df.groupby('idx1', sort=False):
            groups.append({
                'idx1': int(idx1),
                'idx2_list': g['idx2'].to_numpy(dtype=np.int64).tolist(),
                'tmscore_list': pt.tensor(g['tmscore'].to_numpy(dtype=np.float32)),
                'seqid_list': pt.tensor(g['seqid'].to_numpy(dtype=np.float32)),
            })
        self.groups = groups
        self.map = np.arange(len(self.groups), dtype=np.int64) if mapping is None else mapping.astype(np.int64)

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        return self.groups[int(self.map[idx])]


def make_pdb_path(pdb_root: str, pdb_name: str) -> Path:
    return Path(pdb_root) / pdb_name[1:3] / f'{pdb_name}.pdb'


def run_tmalign_once(pdb1: Path, pdb2: Path, tmalign_path: str, reference: int = 1):
    cmd = [str(tmalign_path), str(pdb1), str(pdb2)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f'TMalign failed for {pdb1.name} vs {pdb2.name}.\n'
            f'stdout:\n{proc.stdout}\n\n'
            f'stderr:\n{proc.stderr}'
        )
    import re
    output = proc.stdout
    seqid_match = re.search(r'Seq_ID=.*?=\s*([0-9.]+)', output)
    tm_scores = re.findall(r'TM-score=\s*([0-9.]+)', output)
    if seqid_match is None or not (1 <= reference <= len(tm_scores)):
        raise ValueError(f'Failed to parse TM-align output for {pdb1.name} vs {pdb2.name}.')
    seqid = float(seqid_match.group(1))
    tm_score = float(tm_scores[reference - 1])
    remote_score = tm_score - 0.6 + min(0.4 - seqid, 0.0)
    return remote_score, tm_score, seqid


def calculate_remote_homology_score(
    query, database, idx: List[List[int]], k: int, pdb_root: str, tmalign_path: str, reference: int = 1
):
    n = len(query)
    if n != len(idx):
        raise ValueError('The length of query and idx should be the same.')

    sorted_results: Dict[int, List[Dict[str, float]]] = {}
    avg_topk_score = 0.0
    total_pairs = 0
    avg_top1_score = 0.0
    avg_top1_tm = 0.0
    avg_top1_seqid = 0.0
    valid_queries = 0

    for q_rank in tqdm(range(n), desc='TM-align rerank', leave=False):
        q_idx = int(query[q_rank]['idx1'])
        q_name = normalize_label(database[q_idx]['lab'])
        q_file = make_pdb_path(pdb_root, q_name)
        if not q_file.is_file():
            raise FileNotFoundError(f'Query PDB file not found: {q_file}')

        cur = []
        for cand_idx in idx[q_rank][:k]:
            c_name = normalize_label(database[int(cand_idx)]['lab'])
            c_file = make_pdb_path(pdb_root, c_name)
            if not c_file.is_file():
                raise FileNotFoundError(f'Candidate PDB file not found: {c_file}')
            remote_score, tm_score, seqid = run_tmalign_once(q_file, c_file, tmalign_path, reference)
            cur.append({
                'c_idx': int(cand_idx),
                'score': float(remote_score),
                'tm_score': float(tm_score),
                'seqid': float(seqid),
            })
            avg_topk_score += float(remote_score)
            total_pairs += 1

        cur.sort(key=lambda x: x['score'], reverse=True)
        sorted_results[q_rank] = cur
        if cur:
            avg_top1_score += cur[0]['score']
            avg_top1_tm += cur[0]['tm_score']
            avg_top1_seqid += cur[0]['seqid']
            valid_queries += 1

    if total_pairs == 0:
        raise RuntimeError('No TM-align tasks were generated.')

    avg_topk_score /= total_pairs
    if valid_queries > 0:
        avg_top1_score /= valid_queries
        avg_top1_tm /= valid_queries
        avg_top1_seqid /= valid_queries

    return sorted_results, avg_top1_score, avg_topk_score, avg_top1_tm, avg_top1_seqid



def normalize_label(label) -> str:
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    label = str(label).strip()
    label = Path(label).name
    if label.endswith(".pdb"):
        label = label[:-4]
    return label


def decode_sequence(tokens, alphabet: str = STANDARD_ALPHABET, unknown: str = "X") -> str:
    if isinstance(tokens, pt.Tensor):
        arr = tokens.detach().cpu().numpy()
    else:
        arr = np.asarray(tokens)
    letters: List[str] = []
    for x in arr:
        idx = int(x)
        if idx == 0:
            continue
        if 1 <= idx <= len(alphabet):
            letters.append(alphabet[idx - 1])
        else:
            letters.append(unknown)
    return "".join(letters)


def write_fasta(path: Path, name: str, seq: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i + 80] + "\n")


def write_a3m(path: Path, name: str, seq: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        f.write(seq + "\n")


def prepare_datasets(data_path: str, pair_file: str, test_size: int, random_state: int):
    data = pt.load(data_path, weights_only=False)
    lib_data = SimpleProteinDataset(data)
    pdb2idx = {normalize_label(lib_data[i]['lab']): i for i in range(len(lib_data))}

    if len(pdb2idx) != len(lib_data):
        raise ValueError('Duplicate protein labels detected after normalization. Please inspect your labels.')

    queryhomo = QueryHomologyDatasetLite(pair_file, pdb2idx)
    all_query_map = np.arange(len(queryhomo), dtype=np.int64)
    _, test_map = train_test_split(
        all_query_map,
        test_size=test_size,
        random_state=random_state,
    )
    test_map = np.sort(test_map)
    test_set = QueryHomologyDatasetLite(pair_file, pdb2idx, mapping=test_map)
    test_global_idx = np.array([int(test_set[i]['idx1']) for i in range(len(test_set))], dtype=np.int64)
    return lib_data, pdb2idx, test_set, test_global_idx


def run_cmd(cmd: Sequence[str], env: Optional[dict] = None, cwd: Optional[Path] = None):
    cmd = [str(x) for x in cmd]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}):\n"
            f"{' '.join(shlex.quote(x) for x in cmd)}\n\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def which_or_raise(name: str, user_path: Optional[str] = None) -> str:
    if user_path:
        path = shutil.which(user_path) or user_path
    else:
        path = shutil.which(name)
    if not path:
        raise FileNotFoundError(f"Required executable not found: {name}")
    return path


def detect_cstranslate_supports_b_flag(cstranslate_path: str) -> bool:
    for flag in ("-h", "--help"):
        try:
            proc = subprocess.run([cstranslate_path, flag], capture_output=True, text=True)
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if "-b" in text:
                return True
            if "cstranslate has no -b flag anymore" in text:
                return False
        except Exception:
            continue
    return False


def export_library(args):
    out_dir = Path(args.out_dir)
    lib_fasta_dir = out_dir / 'library_fasta'
    lib_a3m_dir = out_dir / 'library_a3m'
    query_fasta_dir = out_dir / 'test_queries_fasta'
    query_a3m_dir = out_dir / 'test_queries_a3m'
    out_dir.mkdir(parents=True, exist_ok=True)

    lib_data, _, test_set, _ = prepare_datasets(
        data_path=args.data_path,
        pair_file=args.pair_file,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    rows = []
    seen = set()
    for i in tqdm(range(len(lib_data)), desc='Export library'):
        item = lib_data[i]
        name = normalize_label(item['lab'])
        if name in seen:
            raise ValueError(f'Duplicate label after normalization: {name}')
        seen.add(name)
        seq = decode_sequence(item['seq'], alphabet=args.alphabet, unknown=args.unknown_aa)
        write_fasta(lib_fasta_dir / f'{name}.fasta', name, seq)
        write_a3m(lib_a3m_dir / f'{name}.a3m', name, seq)
        rows.append({'idx': i, 'label': name, 'length': len(seq)})

    pd.DataFrame(rows).to_csv(out_dir / 'library_metadata.tsv', sep='\t', index=False)

    split_rows = []
    for i in tqdm(range(len(test_set)), desc='Export test queries'):
        idx1 = int(test_set[i]['idx1'])
        item = lib_data[idx1]
        name = normalize_label(item['lab'])
        seq = decode_sequence(item['seq'], alphabet=args.alphabet, unknown=args.unknown_aa)
        write_fasta(query_fasta_dir / f'{name}.fasta', name, seq)
        write_a3m(query_a3m_dir / f'{name}.a3m', name, seq)
        split_rows.append({'query_rank': i, 'global_idx': idx1, 'label': name, 'length': len(seq)})

    pd.DataFrame(split_rows).to_csv(out_dir / 'test_split.tsv', sep='\t', index=False)
    print(f'Library FASTA exported to: {lib_fasta_dir}')
    print(f'Library A3M exported to:   {lib_a3m_dir}')
    print(f'Test FASTA exported to:    {query_fasta_dir}')
    print(f'Test split saved to:       {out_dir / "test_split.tsv"}')


def build_db_with_hhsuitedb(a3m_dir: Path, db_prefix: Path, cpu: int, hhsuitedb_path: str):
    glob_expr = str(a3m_dir / '*.a3m')
    run_cmd([
        sys.executable,
        hhsuitedb_path,
        '--ia3m', glob_expr,
        '-o', str(db_prefix),
        '--cpu', str(cpu),
    ])


def build_db_manually(
    a3m_dir: Path,
    db_prefix: Path,
    cpu: int,
    ffindex_build_path: str,
    ffindex_apply_path: str,
    hhmake_path: str,
    cstranslate_path: str,
    ffindex_order_path: str,
    cs219_lib: str,
    context_lib: str,
):
    a3m_files = sorted(a3m_dir.glob('*.a3m'))
    if not a3m_files:
        raise FileNotFoundError(f'No .a3m files found in {a3m_dir}')

    db_prefix.parent.mkdir(parents=True, exist_ok=True)
    filelist = db_prefix.parent / f'{db_prefix.name}_a3m_files.txt'
    with open(filelist, 'w', encoding='utf-8') as f:
        for path in a3m_files:
            f.write(str(path.resolve()) + '\n')

    a3m_base = str(db_prefix) + '_a3m'
    hhm_base = str(db_prefix) + '_hhm'
    cs_base = str(db_prefix) + '_cs219'

    run_cmd([
        ffindex_build_path,
        '-s',
        a3m_base + '.ffdata',
        a3m_base + '.ffindex',
        '-f',
        str(filelist),
    ])

    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = str(max(1, int(cpu)))

    run_cmd([
        ffindex_apply_path,
        a3m_base + '.ffdata',
        a3m_base + '.ffindex',
        '-d',
        hhm_base + '.ffdata',
        '-i',
        hhm_base + '.ffindex',
        '--',
        hhmake_path,
        '-i', 'stdin',
        '-o', 'stdout',
        '-v', '0',
    ], env=env)

    cstranslate_cmd = [
        cstranslate_path,
        '-A', cs219_lib,
        '-D', context_lib,
        '-x', '0.3',
        '-c', '4',
        '--ffindex',
        '-i', a3m_base,
        '-o', cs_base,
        '-I', 'a3m',
    ]
    if detect_cstranslate_supports_b_flag(cstranslate_path):
        cstranslate_cmd.append('-b')
    run_cmd(cstranslate_cmd, env=env)

    sorting_file = db_prefix.parent / f'{db_prefix.name}_sorting.dat'
    with open(cs_base + '.ffindex', 'r', encoding='utf-8') as f:
        entries = [line.rstrip('\n').split() for line in f if line.strip()]
    entries.sort(key=lambda x: int(x[2]))
    with open(sorting_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(entry[0] + '\n')

    run_cmd([
        ffindex_order_path,
        str(sorting_file),
        hhm_base + '.ffdata',
        hhm_base + '.ffindex',
        hhm_base + '_ordered.ffdata',
        hhm_base + '_ordered.ffindex',
    ])
    run_cmd([
        ffindex_order_path,
        str(sorting_file),
        a3m_base + '.ffdata',
        a3m_base + '.ffindex',
        a3m_base + '_ordered.ffdata',
        a3m_base + '_ordered.ffindex',
    ])

    shutil.move(hhm_base + '_ordered.ffdata', hhm_base + '.ffdata')
    shutil.move(hhm_base + '_ordered.ffindex', hhm_base + '.ffindex')
    shutil.move(a3m_base + '_ordered.ffdata', a3m_base + '.ffdata')
    shutil.move(a3m_base + '_ordered.ffindex', a3m_base + '.ffindex')


def build_db(args):
    a3m_dir = Path(args.a3m_dir)
    db_prefix = Path(args.db_prefix)
    db_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.use_hhsuitedb:
        hhsuitedb_path = which_or_raise('hhsuitedb.py', args.hhsuitedb_path)
        build_db_with_hhsuitedb(
            a3m_dir=a3m_dir,
            db_prefix=db_prefix,
            cpu=args.cpu,
            hhsuitedb_path=hhsuitedb_path,
        )
    else:
        ffindex_build_path = which_or_raise('ffindex_build', args.ffindex_build_path)
        ffindex_apply_path = which_or_raise('ffindex_apply', args.ffindex_apply_path)
        hhmake_path = which_or_raise('hhmake', args.hhmake_path)
        cstranslate_path = which_or_raise('cstranslate', args.cstranslate_path)
        ffindex_order_path = which_or_raise('ffindex_order', args.ffindex_order_path)
        if not args.cs219_lib or not args.context_lib:
            raise ValueError('Manual build mode requires --cs219_lib and --context_lib.')
        build_db_manually(
            a3m_dir=a3m_dir,
            db_prefix=db_prefix,
            cpu=args.cpu,
            ffindex_build_path=ffindex_build_path,
            ffindex_apply_path=ffindex_apply_path,
            hhmake_path=hhmake_path,
            cstranslate_path=cstranslate_path,
            ffindex_order_path=ffindex_order_path,
            cs219_lib=args.cs219_lib,
            context_lib=args.context_lib,
        )

    print('HH-suite database build finished. Expected files:')
    print(f'  {db_prefix}_a3m.ffdata / .ffindex')
    print(f'  {db_prefix}_hhm.ffdata / .ffindex')
    print(f'  {db_prefix}_cs219.ffdata / .ffindex')


def parse_blasttab(blasttab_path: Path) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    if not blasttab_path.is_file():
        return hits

    seen_targets = set()
    with open(blasttab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split('\t')
            if len(cols) < 11:
                cols = line.split()
            if len(cols) < 11:
                continue
            query_name = normalize_label(cols[0].split()[0])
            target_name = normalize_label(cols[1].split()[0])
            if target_name in seen_targets:
                continue
            seen_targets.add(target_name)
            try:
                evalue = float(cols[-2])
            except ValueError:
                evalue = float('inf')
            try:
                score = float(cols[-1])
            except ValueError:
                score = float('-inf')
            hits.append({
                'query': query_name,
                'target': target_name,
                'evalue': evalue,
                'score': score,
            })
    hits.sort(key=lambda x: (x['evalue'], -x['score']))
    return hits


def build_candidate_lists(
    raw_hits_per_query: List[List[Dict[str, object]]],
    query_global_idx: np.ndarray,
    pdb2idx: Dict[str, int],
    search_k: int,
) -> Tuple[List[List[int]], float, int]:
    filtered_rows: List[List[int]] = []
    self_hits = 0
    mapped_hits = 0

    for q_hits, q_idx in zip(raw_hits_per_query, query_global_idx):
        row: List[int] = []
        hit_self = False
        for hit in q_hits:
            target = str(hit['target'])
            if target not in pdb2idx:
                continue
            cand_idx = int(pdb2idx[target])
            mapped_hits += 1
            if cand_idx == int(q_idx):
                hit_self = True
                continue
            row.append(cand_idx)
            if len(row) >= search_k:
                break
        filtered_rows.append(row)
        self_hits += int(hit_self)

    self_hit_rate = self_hits / max(len(query_global_idx), 1)
    return filtered_rows, float(self_hit_rate), mapped_hits


def save_raw_hits(raw_hits_per_query: List[List[Dict[str, object]]], path: Path):
    rows = []
    for q_rank, hits in enumerate(raw_hits_per_query):
        for rank, hit in enumerate(hits, start=1):
            rows.append({
                'query_rank': q_rank,
                'hit_rank': rank,
                'query': hit['query'],
                'target': hit['target'],
                'evalue': hit['evalue'],
                'score': hit['score'],
            })
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False)


@pt.no_grad()
def eval_hhblits(args):
    out_dir = Path(args.out_dir)
    query_dir = out_dir / 'queries_fasta'
    hh_out_dir = out_dir / 'hhblits_raw'
    out_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    hh_out_dir.mkdir(parents=True, exist_ok=True)

    hhblits_path = which_or_raise('hhblits', args.hhblits_path)

    lib_data, pdb2idx, test_set, test_global_idx = prepare_datasets(
        data_path=args.data_path,
        pair_file=args.pair_file,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    raw_hits_per_query: List[List[Dict[str, object]]] = []
    split_rows = []
    for q_rank in tqdm(range(len(test_set)), desc='HHblits retrieval'):
        idx1 = int(test_set[q_rank]['idx1'])
        item = lib_data[idx1]
        query_name = normalize_label(item['lab'])
        query_seq = decode_sequence(item['seq'], alphabet=args.alphabet, unknown=args.unknown_aa)
        query_fasta = query_dir / f'{query_name}.fasta'
        hhr_path = hh_out_dir / f'{query_name}.hhr'
        blasttab_path = hh_out_dir / f'{query_name}.blasttab'

        write_fasta(query_fasta, query_name, query_seq)
        cmd = [
            hhblits_path,
            '-i', str(query_fasta),
            '-d', args.db_prefix,
            '-o', str(hhr_path),
            '-blasttab', str(blasttab_path),
            '-cpu', str(args.cpu),
            '-n', str(args.n_iter),
            '-Z', str(args.search_k),
            '-B', str(args.search_k),
            '-v', str(args.verbose),
        ]
        run_cmd(cmd)
        hits = parse_blasttab(blasttab_path)
        raw_hits_per_query.append(hits)
        split_rows.append({
            'query_rank': q_rank,
            'global_idx': idx1,
            'label': query_name,
            'length': len(query_seq),
            'num_raw_hits': len(hits),
        })

    pd.DataFrame(split_rows).to_csv(out_dir / 'test_split.tsv', sep='\t', index=False)
    save_raw_hits(raw_hits_per_query, out_dir / 'hhblits_hits.tsv')

    candidate_rows, self_hit_rate, mapped_hits = build_candidate_lists(
        raw_hits_per_query=raw_hits_per_query,
        query_global_idx=test_global_idx,
        pdb2idx=pdb2idx,
        search_k=args.search_k,
    )

    sorted_results, avg_top1_score, avg_topk_score, avg_top1_tm, avg_top1_seqid = calculate_remote_homology_score(
        query=test_set,
        database=lib_data,
        idx=candidate_rows,
        k=args.topk,
        pdb_root=args.pdb_root,
        tmalign_path=args.tmalign_path,
        reference=args.tmalign_reference,
    )

    top1_rows = []
    for q_rank, c_list in sorted_results.items():
        if not c_list:
            continue
        best = c_list[0]
        q_idx = int(test_set[q_rank]['idx1'])
        q_name = normalize_label(lib_data[q_idx]['lab'])
        c_name = normalize_label(lib_data[int(best['c_idx'])]['lab'])
        top1_rows.append({
            'query_rank': int(q_rank),
            'query': q_name,
            'candidate': c_name,
            'candidate_idx': int(best['c_idx']),
            'remote_score': float(best['score']),
            'tm_score': float(best['tm_score']),
            'seqid': float(best['seqid']),
        })
    pd.DataFrame(top1_rows).to_csv(out_dir / 'hhblits_top1.tsv', sep='\t', index=False)

    metrics = {
        'num_test_queries': int(len(test_set)),
        'search_k': int(args.search_k),
        'topk_for_tmalign': int(args.topk),
        'hhblits_iterations': int(args.n_iter),
        'self_hit_rate_before_filtering': float(self_hit_rate),
        'mapped_hits': int(mapped_hits),
        'avg_top1_score': float(avg_top1_score),
        'avg_topk_score': float(avg_topk_score),
        'avg_top1_tm': float(avg_top1_tm),
        'avg_top1_seqid': float(avg_top1_seqid),
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description='HHblits baseline for the current protein retrieval project.')
    sub = parser.add_subparsers(dest='command', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data_path', type=str, required=True)
    common.add_argument('--pair_file', type=str, required=True)
    common.add_argument('--test_size', type=int, default=1024)
    common.add_argument('--random_state', type=int, default=42)
    common.add_argument('--alphabet', type=str, default=STANDARD_ALPHABET,
                        help='Index-to-amino-acid mapping for token ids 1..N.')
    common.add_argument('--unknown_aa', type=str, default='X')

    p_export = sub.add_parser('export', parents=[common], help='Export library/test sequences to FASTA/A3M.')
    p_export.add_argument('--out_dir', type=str, required=True)
    p_export.set_defaults(func=export_library)

    p_build = sub.add_parser('build-db', help='Build a custom HH-suite database from exported A3M files.')
    p_build.add_argument('--a3m_dir', type=str, required=True)
    p_build.add_argument('--db_prefix', type=str, required=True,
                         help='Database prefix passed to hhblits -d, without _a3m/_hhm/_cs219 suffix.')
    p_build.add_argument('--cpu', type=int, default=8)
    p_build.add_argument('--use_hhsuitedb', action='store_true',
                         help='Use official hhsuitedb.py script instead of the manual fallback pipeline.')
    p_build.add_argument('--hhsuitedb_path', type=str, default=None)
    p_build.add_argument('--ffindex_build_path', type=str, default=None)
    p_build.add_argument('--ffindex_apply_path', type=str, default=None)
    p_build.add_argument('--hhmake_path', type=str, default=None)
    p_build.add_argument('--cstranslate_path', type=str, default=None)
    p_build.add_argument('--ffindex_order_path', type=str, default=None)
    p_build.add_argument('--cs219_lib', type=str, default=None)
    p_build.add_argument('--context_lib', type=str, default=None)
    p_build.set_defaults(func=build_db)

    p_eval = sub.add_parser('eval', parents=[common], help='Run HHblits retrieval and evaluate with TM-align metrics.')
    p_eval.add_argument('--db_prefix', type=str, required=True,
                        help='Same prefix used in hhblits -d (without _a3m/_hhm/_cs219 suffix).')
    p_eval.add_argument('--out_dir', type=str, required=True)
    p_eval.add_argument('--hhblits_path', type=str, default=None)
    p_eval.add_argument('--cpu', type=int, default=8)
    p_eval.add_argument('--n_iter', type=int, default=2)
    p_eval.add_argument('--search_k', type=int, default=200)
    p_eval.add_argument('--topk', type=int, default=12)
    p_eval.add_argument('--verbose', type=int, default=1)
    p_eval.add_argument('--pdb_root', type=str, required=True)
    p_eval.add_argument('--tmalign_path', type=str, required=True)
    p_eval.add_argument('--tmalign_reference', type=int, default=1)
    p_eval.set_defaults(func=eval_hhblits)

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
