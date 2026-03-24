from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch as pt
from sklearn.model_selection import train_test_split

try:
    from utils.data import ProteinDataset, QueryHomologyDataset
    from utils.tools import calculate_remote_homology_score
except ImportError:  # uploaded flat files / standalone mode
    from data import ProteinDataset, QueryHomologyDataset
    from tools import calculate_remote_homology_score


# 你的序列 token -> 氨基酸字母表。
# 假设 0 是 padding，1..20 依次对应下面 20 个天然氨基酸。
# 若你的编码顺序不同，请改这个字符串，例如 "ARNDCQEGHILKMFPSTWYV"。
DEFAULT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# 参考 django-blastplus 里的 protein sensitivity 预设：
# NORMAL: gapopen=11, gapextend=1, matrix=BLOSUM62
# NEAR MATCH: gapopen=10, gapextend=1, matrix=BLOSUM90
# DISTANT: gapopen=14, gapextend=2, matrix=BLOSUM45
PROTEIN_SENSITIVITY_PRESETS: Dict[str, Dict[str, object]] = {
    "normal": {"gapopen": 11, "gapextend": 1, "matrix": "BLOSUM62", "word_size": 3},
    "near": {"gapopen": 10, "gapextend": 1, "matrix": "BLOSUM90", "word_size": 3},
    "distant": {"gapopen": 14, "gapextend": 2, "matrix": "BLOSUM45", "word_size": 3},
}


@dataclass
class BlastpConfig:
    data_path: str
    pair_file: str
    pdb_root: str
    tmalign_path: str
    work_dir: str = "./blastp_baseline"
    blastp_bin: str = "blastp"
    makeblastdb_bin: str = "makeblastdb"
    alphabet: str = DEFAULT_ALPHABET
    topk: int = 12
    eval_search_k: int = 200
    evalue: float = 1e-3
    sensitivity: str = "normal"  # normal / near / distant
    word_size: Optional[int] = None
    matrix: Optional[str] = None
    gapopen: Optional[int] = None
    gapextend: Optional[int] = None
    num_threads: int = 8
    max_target_seqs: Optional[int] = None
    random_state: int = 42
    test_size: int = 1024
    tmalign_reference: int = 1
    tmalign_workers: Optional[int] = None
    keep_xml: bool = False
    keep_temp: bool = True
    verbose: bool = True
    exclude_self_hit: bool = True
    ungapped: bool = False
    seg: str = "yes"
    comp_based_stats: Optional[int] = None


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def normalize_label(label) -> str:
    s = str(label).strip()
    if s.endswith(".pdb"):
        s = s[:-4]
    # BLAST may emit ids like lcl|4A27-A, pdb|4A27|A, 4A27_A, or 4A27:A.
    if s.startswith("lcl|"):
        s = s[4:]
    m = re.fullmatch(r"pdb\|([0-9A-Za-z]{4})\|([A-Za-z0-9])", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.fullmatch(r"([0-9A-Za-z]{4})[_: ]([A-Za-z0-9])", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return s


def decode_token_sequence(
    seq_tokens: Sequence[int],
    alphabet: str = DEFAULT_ALPHABET,
    pad_idx: int = 0,
) -> str:
    seq = []
    max_token = len(alphabet)
    for tok in seq_tokens:
        tok = int(tok)
        if tok == pad_idx:
            continue
        if 1 <= tok <= max_token:
            seq.append(alphabet[tok - 1])
        else:
            raise ValueError(
                f"Found token={tok}, but alphabet only defines 1..{max_token}. "
                f"Please check --alphabet against your dataset encoding."
            )
    return "".join(seq)


def write_fasta(records: Iterable[Tuple[str, str]], fasta_path: str | Path) -> None:
    fasta_path = Path(fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as f:
        for rid, seq in records:
            f.write(f">{rid}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def export_library_fasta(
    dataset: ProteinDataset,
    fasta_path: str | Path,
    alphabet: str = DEFAULT_ALPHABET,
) -> Dict[str, int]:
    label_to_local_idx: Dict[str, int] = {}
    records = []
    for local_idx in range(len(dataset)):
        item = dataset[local_idx]
        label = normalize_label(item["lab"])
        seq = decode_token_sequence(item["seq"], alphabet=alphabet)
        if not seq:
            continue
        if label in label_to_local_idx:
            raise ValueError(f"Duplicate FASTA id detected: {label}")
        label_to_local_idx[label] = local_idx
        records.append((label, seq))
    write_fasta(records, fasta_path)
    return label_to_local_idx


def export_query_fasta(
    query_set: QueryHomologyDataset,
    protein_dataset: ProteinDataset,
    fasta_path: str | Path,
    alphabet: str = DEFAULT_ALPHABET,
) -> List[str]:
    query_ids: List[str] = []
    records = []
    for query_idx in range(len(query_set)):
        lib_idx = int(query_set[query_idx]["idx1"])
        item = protein_dataset[lib_idx]
        label = normalize_label(item["lab"])
        seq = decode_token_sequence(item["seq"], alphabet=alphabet)
        if not seq:
            continue
        query_ids.append(label)
        records.append((label, seq))
    write_fasta(records, fasta_path)
    return query_ids


def ensure_executable(name_or_path: str) -> str:
    if shutil.which(name_or_path):
        return name_or_path
    p = Path(name_or_path)
    if p.is_file():
        return str(p)
    raise FileNotFoundError(f"Executable not found: {name_or_path}")


def make_blast_db(
    fasta_path: str | Path,
    db_prefix: str | Path,
    makeblastdb_bin: str = "makeblastdb",
    title: str = "protein_db",
    verbose: bool = True,
) -> None:
    makeblastdb_bin = ensure_executable(makeblastdb_bin)
    fasta_path = str(Path(fasta_path).resolve())
    db_prefix = str(Path(db_prefix).resolve())
    cmd = [
        makeblastdb_bin,
        "-in", fasta_path,
        "-dbtype", "prot",
        "-parse_seqids",
        "-out", db_prefix,
        "-title", title,
    ]
    log("[makeblastdb] " + " ".join(cmd), verbose)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "makeblastdb failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def build_blastp_params(cfg: BlastpConfig) -> Dict[str, object]:
    preset = PROTEIN_SENSITIVITY_PRESETS[cfg.sensitivity]
    params = dict(preset)
    if cfg.word_size is not None:
        params["word_size"] = cfg.word_size
    if cfg.matrix is not None:
        params["matrix"] = cfg.matrix
    if cfg.gapopen is not None:
        params["gapopen"] = cfg.gapopen
    if cfg.gapextend is not None:
        params["gapextend"] = cfg.gapextend
    return params


def run_blastp_subprocess(
    query_fasta: str | Path,
    db_prefix: str | Path,
    out_path: str | Path,
    cfg: BlastpConfig,
) -> None:
    blastp_bin = ensure_executable(cfg.blastp_bin)
    params = build_blastp_params(cfg)
    max_target_seqs = cfg.max_target_seqs or max(cfg.topk + 1, cfg.eval_search_k)

    outfmt_fields = [
        "qseqid", "sseqid", "bitscore", "evalue", "pident",
        "length", "qlen", "slen", "qcovs"
    ]
    outfmt = "6 " + " ".join(outfmt_fields)

    cmd = [
        blastp_bin,
        "-query", str(Path(query_fasta).resolve()),
        "-db", str(Path(db_prefix).resolve()),
        "-out", str(Path(out_path).resolve()),
        "-outfmt", outfmt,
        "-evalue", str(cfg.evalue),
        "-num_threads", str(cfg.num_threads),
        "-max_target_seqs", str(max_target_seqs),
        "-matrix", str(params["matrix"]),
        "-gapopen", str(params["gapopen"]),
        "-gapextend", str(params["gapextend"]),
        "-word_size", str(params["word_size"]),
        "-seg", str(cfg.seg),
    ]
    if cfg.ungapped:
        cmd.extend(["-ungapped"])
    if cfg.comp_based_stats is not None:
        cmd.extend(["-comp_based_stats", str(cfg.comp_based_stats)])

    log("[blastp] " + " ".join(cmd), cfg.verbose)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "blastp failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def run_blastp_xml_biopython_compatible(
    query_fasta: str | Path,
    db_prefix: str | Path,
    xml_out_path: str | Path,
    cfg: BlastpConfig,
) -> None:
    """
    更接近 django-blastplus 的写法：
    - 使用本地 blastp
    - 输出 XML（outfmt=5）
    但为了兼容新版环境，默认不依赖 Biopython 的 commandline wrapper，
    仍用 subprocess 直接拼命令。
    """
    blastp_bin = ensure_executable(cfg.blastp_bin)
    params = build_blastp_params(cfg)
    max_target_seqs = cfg.max_target_seqs or max(cfg.topk + 1, cfg.eval_search_k)

    cmd = [
        blastp_bin,
        "-query", str(Path(query_fasta).resolve()),
        "-db", str(Path(db_prefix).resolve()),
        "-out", str(Path(xml_out_path).resolve()),
        "-outfmt", "5",
        "-evalue", str(cfg.evalue),
        "-num_threads", str(cfg.num_threads),
        "-max_target_seqs", str(max_target_seqs),
        "-matrix", str(params["matrix"]),
        "-gapopen", str(params["gapopen"]),
        "-gapextend", str(params["gapextend"]),
        "-word_size", str(params["word_size"]),
        "-seg", str(cfg.seg),
    ]
    if cfg.comp_based_stats is not None:
        cmd.extend(["-comp_based_stats", str(cfg.comp_based_stats)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "blastp XML run failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def parse_blast_tabular(
    blast_out_path: str | Path,
    label_to_local_idx: Dict[str, int],
    query_ids: Sequence[str],
    topk: int,
    exclude_self_hit: bool = True,
) -> Tuple[List[List[int]], Dict[str, float]]:
    hits_by_query: Dict[str, List[int]] = {normalize_label(qid): [] for qid in query_ids}
    seen_by_query: Dict[str, set] = {normalize_label(qid): set() for qid in query_ids}
    self_hits = 0
    unknown_subject_ids = 0
    unknown_subject_examples = []

    with Path(blast_out_path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            qid, sid = row[0], row[1]
            qid = normalize_label(qid)
            sid = normalize_label(sid)
            if qid not in hits_by_query:
                continue
            if exclude_self_hit and qid == sid:
                self_hits += 1
                continue
            if sid not in label_to_local_idx:
                unknown_subject_ids += 1
                if len(unknown_subject_examples) < 5:
                    unknown_subject_examples.append(sid)
                continue
            idx = int(label_to_local_idx[sid])
            if idx in seen_by_query[qid]:
                continue
            hits_by_query[qid].append(idx)
            seen_by_query[qid].add(idx)

    indices: List[List[int]] = []
    no_hit_queries = 0
    for qid_raw in query_ids:
        qid = normalize_label(qid_raw)
        row = hits_by_query[qid][:topk]
        if not row:
            no_hit_queries += 1
        indices.append(row)

    stats = {
        "self_hit_count": float(self_hits),
        "self_hit_rate": float(self_hits / max(len(query_ids), 1)),
        "no_hit_queries": float(no_hit_queries),
        "no_hit_rate": float(no_hit_queries / max(len(query_ids), 1)),
        "unknown_subject_ids": float(unknown_subject_ids),
        "unknown_subject_examples": unknown_subject_examples,
    }
    return indices, stats


def load_eval_datasets(cfg: BlastpConfig):
    data = pt.load(cfg.data_path, weights_only=False)
    lib_data = ProteinDataset(data, mode="graph")
    pdb2idx = {normalize_label(lib_data[i]["lab"]): i for i in range(len(lib_data))}

    queryhomo = QueryHomologyDataset(None, cfg.pair_file, pdb2idx)
    all_query_map = np.arange(len(queryhomo), dtype=np.int64)
    _, test_map = train_test_split(
        all_query_map,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    test_map = np.sort(test_map)
    test_set = QueryHomologyDataset(queryhomo, mapping=test_map)
    return lib_data, test_set


def evaluate_blastp(cfg: BlastpConfig) -> Dict[str, object]:
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    lib_data, test_set = load_eval_datasets(cfg)
    log(f"Library proteins: {len(lib_data)}", cfg.verbose)
    log(f"Test queries: {len(test_set)}", cfg.verbose)

    db_fasta = work_dir / "library.fasta"
    db_prefix = work_dir / "blastdb" / "protein_db"
    query_fasta = work_dir / "queries.fasta"
    blast_out = work_dir / "blastp.tsv"
    blast_xml = work_dir / "blastp.xml"

    label_to_local_idx = export_library_fasta(lib_data, db_fasta, alphabet=cfg.alphabet)
    query_ids = export_query_fasta(test_set, lib_data, query_fasta, alphabet=cfg.alphabet)

    make_blast_db(
        fasta_path=db_fasta,
        db_prefix=db_prefix,
        makeblastdb_bin=cfg.makeblastdb_bin,
        title="protein_retrieval_library",
        verbose=cfg.verbose,
    )

    if cfg.keep_xml:
        run_blastp_xml_biopython_compatible(
            query_fasta=query_fasta,
            db_prefix=db_prefix,
            xml_out_path=blast_xml,
            cfg=cfg,
        )

    run_blastp_subprocess(
        query_fasta=query_fasta,
        db_prefix=db_prefix,
        out_path=blast_out,
        cfg=cfg,
    )

    candidate_pool_k = max(int(cfg.topk), int(cfg.eval_search_k))
    idx_all, parse_stats = parse_blast_tabular(
        blast_out_path=blast_out,
        label_to_local_idx=label_to_local_idx,
        query_ids=query_ids,
        topk=candidate_pool_k,
        exclude_self_hit=cfg.exclude_self_hit,
    )

    valid_query_pos = [i for i, row in enumerate(idx_all) if len(row) > 0]
    valid_idx = [idx_all[i][:cfg.topk] for i in valid_query_pos]

    metrics = {
        "candidate_pool_k": int(candidate_pool_k),
        "num_test_queries": int(len(test_set)),
        "num_queries_with_hits": int(len(valid_query_pos)),
        "num_queries_without_hits": int(len(test_set) - len(valid_query_pos)),
        "hit_query_fraction": float(len(valid_query_pos) / max(len(test_set), 1)),
        **parse_stats,
    }

    if len(valid_query_pos) == 0:
        raise RuntimeError(
            "BLASTp produced no evaluable non-self hits, so no TM-align tasks could be generated.\n"
            f"parse_stats={metrics}\n"
            "Most likely causes:\n"
            "1) BLAST only returned self-hits and --exclude_self_hit filtered them all out;\n"
            "2) the exported amino-acid alphabet does not match your token encoding;\n"
            "3) the search is too strict for remote homologs.\n"
            "Try: --sensitivity distant --evalue 1000 --seg no --comp_based_stats 0\n"
            "If you only want to verify the pipeline first, temporarily add --include_self_hit."
        )


    valid_test_set = QueryHomologyDataset(test_set, mapping=np.asarray(valid_query_pos, dtype=np.int64))

    sorted_results, avg_top1_score, avg_topk_score, avg_top1_tm, avg_top1_seqid = calculate_remote_homology_score(
        query=valid_test_set,
        database=lib_data,
        idx=valid_idx,
        k=cfg.topk,
        pdb_root=cfg.pdb_root,
        tmalign_path=cfg.tmalign_path,
        reference=cfg.tmalign_reference,
        num_workers=cfg.tmalign_workers,
    )

    metrics.update({
        "avg_top1_score": float(avg_top1_score),
        "avg_topk_score": float(avg_topk_score),
        "avg_top1_tm": float(avg_top1_tm),
        "avg_top1_seqid": float(avg_top1_seqid),
    })

    log("================ BLASTp baseline ================", cfg.verbose)
    for k, v in metrics.items():
        if isinstance(v, float):
            log(f"{k}: {v:.6f}", cfg.verbose)
        else:
            log(f"{k}: {v}", cfg.verbose)

    return {
        "sorted_results": sorted_results,
        "metrics": metrics,
        "idx": valid_idx,
        "idx_all": idx_all,
        "valid_query_pos": valid_query_pos,
        "work_dir": str(work_dir.resolve()),
        "blast_out": str(blast_out.resolve()),
        "blast_xml": str(blast_xml.resolve()) if cfg.keep_xml else None,
    }


def parse_args() -> BlastpConfig:
    parser = argparse.ArgumentParser(
        description="BLASTp baseline for the current protein retrieval project"
    )
    parser.add_argument("--data_path", type=str, default='./data/sorted_1300_p0_h1.pt')
    parser.add_argument("--pair_file", type=str, default='./data/tmalign.out')
    parser.add_argument("--pdb_root", type=str, default='../../data/pdb')
    parser.add_argument("--tmalign_path", type=str, default='./TMalign')

    parser.add_argument("--work_dir", type=str, default="./blastp_baseline")
    parser.add_argument("--blastp_bin", type=str, default="blastp")
    parser.add_argument("--makeblastdb_bin", type=str, default="makeblastdb")
    parser.add_argument("--alphabet", type=str, default=DEFAULT_ALPHABET)

    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--eval_search_k", type=int, default=200)
    parser.add_argument("--evalue", type=float, default=1e-3)
    parser.add_argument("--sensitivity", type=str, default="normal", choices=list(PROTEIN_SENSITIVITY_PRESETS.keys()))
    parser.add_argument("--word_size", type=int, default=None)
    parser.add_argument("--matrix", type=str, default=None)
    parser.add_argument("--gapopen", type=int, default=None)
    parser.add_argument("--gapextend", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--max_target_seqs", type=int, default=None)

    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=1024)
    parser.add_argument("--tmalign_reference", type=int, default=1)
    parser.add_argument("--tmalign_workers", type=int, default=None)

    parser.add_argument("--keep_xml", action="store_true")
    parser.add_argument("--keep_temp", action="store_true", default=True)
    parser.add_argument("--exclude_self_hit", action="store_true", default=True)
    parser.add_argument("--include_self_hit", action="store_false", dest="exclude_self_hit")
    parser.add_argument("--ungapped", action="store_true")
    parser.add_argument("--seg", type=str, default="yes", choices=["yes", "no"])
    parser.add_argument("--comp_based_stats", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    return BlastpConfig(
        data_path=args.data_path,
        pair_file=args.pair_file,
        pdb_root=args.pdb_root,
        tmalign_path=args.tmalign_path,
        work_dir=args.work_dir,
        blastp_bin=args.blastp_bin,
        makeblastdb_bin=args.makeblastdb_bin,
        alphabet=args.alphabet,
        topk=args.topk,
        eval_search_k=args.eval_search_k,
        evalue=args.evalue,
        sensitivity=args.sensitivity,
        word_size=args.word_size,
        matrix=args.matrix,
        gapopen=args.gapopen,
        gapextend=args.gapextend,
        num_threads=args.num_threads,
        max_target_seqs=args.max_target_seqs,
        random_state=args.random_state,
        test_size=args.test_size,
        tmalign_reference=args.tmalign_reference,
        tmalign_workers=args.tmalign_workers,
        keep_xml=args.keep_xml,
        keep_temp=args.keep_temp,
        verbose=not args.quiet,
        exclude_self_hit=args.exclude_self_hit,
        ungapped=args.ungapped,
        seg=args.seg,
        comp_based_stats=args.comp_based_stats,
    )


if __name__ == "__main__":
    config = parse_args()
    evaluate_blastp(config)
