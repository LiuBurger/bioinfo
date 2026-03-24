from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
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
except ImportError:  # standalone / flat-file mode
    from data import ProteinDataset, QueryHomologyDataset
    from tools import calculate_remote_homology_score


# 假设 0 是 padding，1..20 依次对应这 20 个天然氨基酸。
# 如果你的 token 编码顺序不同，请通过 --alphabet 覆盖。
DEFAULT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# 沿用你前面 BLASTp baseline 的敏感度预设，便于横向比较。
PROTEIN_SENSITIVITY_PRESETS: Dict[str, Dict[str, object]] = {
    "normal": {"gapopen": 11, "gapextend": 1, "matrix": "BLOSUM62", "word_size": 3},
    "near": {"gapopen": 10, "gapextend": 1, "matrix": "BLOSUM90", "word_size": 3},
    "distant": {"gapopen": 14, "gapextend": 2, "matrix": "BLOSUM45", "word_size": 3},
}


@dataclass
class PsiBlastConfig:
    data_path: str
    pair_file: str
    pdb_root: str
    tmalign_path: str
    work_dir: str = "./psiblast_baseline"
    psiblast_bin: str = "psiblast"
    makeblastdb_bin: str = "makeblastdb"
    alphabet: str = DEFAULT_ALPHABET
    topk: int = 12
    eval_search_k: int = 200
    # 对检索 baseline，保存更多候选通常更稳，所以默认放宽输出 e-value。
    evalue: float = 1000.0
    # profile inclusion threshold，默认用经典的 0.005。
    inclusion_ethresh: float = 0.005
    num_iterations: int = 3
    sensitivity: str = "normal"
    word_size: Optional[int] = None
    matrix: Optional[str] = None
    gapopen: Optional[int] = None
    gapextend: Optional[int] = None
    pseudocount: Optional[int] = None
    num_threads: int = 8
    max_target_seqs: Optional[int] = None
    max_hsps: int = 1
    random_state: int = 42
    test_size: int = 1024
    tmalign_reference: int = 1
    tmalign_workers: Optional[int] = None
    keep_xml: bool = False
    keep_temp: bool = True
    verbose: bool = True
    exclude_self_hit: bool = True
    seg: str = "yes"
    comp_based_stats: Optional[int] = None
    parse_deflines: bool = False


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def normalize_label(label) -> str:
    s = str(label).strip()
    if s.endswith(".pdb"):
        s = s[:-4]
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
    cmd = [
        makeblastdb_bin,
        "-in", str(Path(fasta_path).resolve()),
        "-dbtype", "prot",
        "-parse_seqids",
        "-out", str(Path(db_prefix).resolve()),
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


def build_search_params(cfg: PsiBlastConfig) -> Dict[str, object]:
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


def run_psiblast_subprocess(
    query_fasta: str | Path,
    db_prefix: str | Path,
    out_path: str | Path,
    cfg: PsiBlastConfig,
) -> None:
    psiblast_bin = ensure_executable(cfg.psiblast_bin)
    params = build_search_params(cfg)
    max_target_seqs = cfg.max_target_seqs or max(cfg.topk + 1, cfg.eval_search_k)

    outfmt_fields = [
        "qseqid", "sseqid", "bitscore", "evalue", "pident",
        "length", "qlen", "slen", "qcovs"
    ]
    outfmt = "6 " + " ".join(outfmt_fields)

    cmd = [
        psiblast_bin,
        "-query", str(Path(query_fasta).resolve()),
        "-db", str(Path(db_prefix).resolve()),
        "-out", str(Path(out_path).resolve()),
        "-outfmt", outfmt,
        "-evalue", str(cfg.evalue),
        "-num_iterations", str(cfg.num_iterations),
        "-inclusion_ethresh", str(cfg.inclusion_ethresh),
        "-num_threads", str(cfg.num_threads),
        "-max_target_seqs", str(max_target_seqs),
        "-max_hsps", str(cfg.max_hsps),
        "-matrix", str(params["matrix"]),
        "-gapopen", str(params["gapopen"]),
        "-gapextend", str(params["gapextend"]),
        "-word_size", str(params["word_size"]),
        "-seg", str(cfg.seg),
    ]
    if cfg.comp_based_stats is not None:
        cmd.extend(["-comp_based_stats", str(cfg.comp_based_stats)])
    if cfg.pseudocount is not None:
        cmd.extend(["-pseudocount", str(cfg.pseudocount)])
    if cfg.parse_deflines:
        cmd.append("-parse_deflines")

    log("[psiblast] " + " ".join(cmd), cfg.verbose)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "psiblast failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def run_psiblast_xml(
    query_fasta: str | Path,
    db_prefix: str | Path,
    xml_out_path: str | Path,
    cfg: PsiBlastConfig,
) -> None:
    psiblast_bin = ensure_executable(cfg.psiblast_bin)
    params = build_search_params(cfg)
    max_target_seqs = cfg.max_target_seqs or max(cfg.topk + 1, cfg.eval_search_k)

    cmd = [
        psiblast_bin,
        "-query", str(Path(query_fasta).resolve()),
        "-db", str(Path(db_prefix).resolve()),
        "-out", str(Path(xml_out_path).resolve()),
        "-outfmt", "5",
        "-evalue", str(cfg.evalue),
        "-num_iterations", str(cfg.num_iterations),
        "-inclusion_ethresh", str(cfg.inclusion_ethresh),
        "-num_threads", str(cfg.num_threads),
        "-max_target_seqs", str(max_target_seqs),
        "-max_hsps", str(cfg.max_hsps),
        "-matrix", str(params["matrix"]),
        "-gapopen", str(params["gapopen"]),
        "-gapextend", str(params["gapextend"]),
        "-word_size", str(params["word_size"]),
        "-seg", str(cfg.seg),
    ]
    if cfg.comp_based_stats is not None:
        cmd.extend(["-comp_based_stats", str(cfg.comp_based_stats)])
    if cfg.pseudocount is not None:
        cmd.extend(["-pseudocount", str(cfg.pseudocount)])
    if cfg.parse_deflines:
        cmd.append("-parse_deflines")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "psiblast XML run failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def parse_blast_tabular(
    blast_out_path,
    label_to_local_idx,
    query_ids,
    topk,
    exclude_self_hit=True,
):
    hits_by_query = {normalize_label(qid): [] for qid in query_ids}
    seen_by_query = {normalize_label(qid): set() for qid in query_ids}

    self_hits = 0
    unknown_subject_ids = 0
    unknown_subject_examples = []

    malformed_lines = 0
    malformed_examples = []

    with Path(blast_out_path).open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()

            # 空行直接跳过
            if not line:
                continue

            # 跳过注释/提示/警告行
            if (
                line.startswith("#")
                or line.startswith("WARNING")
                or line.startswith("Warning")
                or line.startswith("Search has CONVERGED")
                or line.startswith("Results from round")
                or line.startswith("Psi-BLAST")
                or line.startswith("BLAST")
            ):
                continue

            # 优先按 tab 切；如果不是 tab，再退回按任意空白切
            row = line.split("\t")
            if len(row) < 2:
                row = line.split()

            if len(row) < 2:
                malformed_lines += 1
                if len(malformed_examples) < 10:
                    malformed_examples.append((lineno, line))
                continue

            qid = normalize_label(row[0])
            sid = normalize_label(row[1])

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

    indices = []
    no_hit_queries = 0
    for qid_raw in query_ids:
        qid = normalize_label(qid_raw)
        row = hits_by_query[qid][:topk]
        if not row:
            no_hit_queries += 1
        indices.append(row)

    stats = {
        "self_hit_count": float(self_hits),
        "unknown_subject_ids": float(unknown_subject_ids),
        "unknown_subject_examples": unknown_subject_examples,
        "num_queries_with_hits": float(len(query_ids) - no_hit_queries),
        "num_queries_without_hits": float(no_hit_queries),
        "no_hit_rate": float(no_hit_queries / max(len(query_ids), 1)),
        "malformed_lines": float(malformed_lines),
        "malformed_examples": malformed_examples,
    }
    return indices, stats


def load_eval_datasets(cfg: PsiBlastConfig):
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


def evaluate_psiblast(cfg: PsiBlastConfig) -> Dict[str, object]:
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    lib_data, test_set = load_eval_datasets(cfg)
    log(f"Library proteins: {len(lib_data)}", cfg.verbose)
    log(f"Test queries: {len(test_set)}", cfg.verbose)

    db_fasta = work_dir / "library.fasta"
    db_prefix = work_dir / "blastdb" / "protein_db"
    query_fasta = work_dir / "queries.fasta"
    psiblast_out = work_dir / "psiblast.tsv"
    psiblast_xml = work_dir / "psiblast.xml"

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
        run_psiblast_xml(
            query_fasta=query_fasta,
            db_prefix=db_prefix,
            xml_out_path=psiblast_xml,
            cfg=cfg,
        )

    run_psiblast_subprocess(
        query_fasta=query_fasta,
        db_prefix=db_prefix,
        out_path=psiblast_out,
        cfg=cfg,
    )

    candidate_pool_k = max(int(cfg.topk), int(cfg.eval_search_k))
    idx_all, parse_stats = parse_blast_tabular(
        blast_out_path=psiblast_out,
        label_to_local_idx=label_to_local_idx,
        query_ids=query_ids,
        topk=candidate_pool_k,
        exclude_self_hit=cfg.exclude_self_hit,
    )

    valid_query_pos = [i for i, row in enumerate(idx_all) if len(row) > 0]
    valid_idx = [idx_all[i][:cfg.topk] for i in valid_query_pos]

    metrics: Dict[str, object] = {
        "candidate_pool_k": int(candidate_pool_k),
        "num_test_queries": int(len(test_set)),
        "num_queries_with_hits": int(len(valid_query_pos)),
        "num_queries_without_hits": int(len(test_set) - len(valid_query_pos)),
        "hit_query_fraction": float(len(valid_query_pos) / max(len(test_set), 1)),
        "num_iterations": int(cfg.num_iterations),
        "inclusion_ethresh": float(cfg.inclusion_ethresh),
        **parse_stats,
    }

    if len(valid_query_pos) == 0:
        raise RuntimeError(
            "PSI-BLAST produced no evaluable non-self hits, so no TM-align tasks could be generated.\n"
            f"parse_stats={metrics}\n"
            "Most likely causes:\n"
            "1) the executable path is wrong or not actually the PSI-BLAST / PSI-BLASTexB binary;\n"
            "2) BLAST ids were returned in a format not covered by normalize_label;\n"
            "3) the exported amino-acid alphabet does not match your token encoding;\n"
            "4) the search is too strict or iterations are too few.\n"
            "Try: --num_iterations 5 --inclusion_ethresh 0.005 --evalue 1000 --seg no --comp_based_stats 0\n"
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

    log("================ PSI-BLAST baseline ================", cfg.verbose)
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
        "psiblast_out": str(psiblast_out.resolve()),
        "psiblast_xml": str(psiblast_xml.resolve()) if cfg.keep_xml else None,
    }


def parse_args() -> PsiBlastConfig:
    parser = argparse.ArgumentParser(
        description="PSI-BLAST / PSI-BLASTexB baseline for the current protein retrieval project"
    )
    parser.add_argument("--data_path", type=str, default='./data/sorted_1300_p0_h1.pt')
    parser.add_argument("--pair_file", type=str, default='./data/tmalign.out')
    parser.add_argument("--pdb_root", type=str, default='../../data/pdb')
    parser.add_argument("--tmalign_path", type=str, default='./TMalign')

    parser.add_argument("--work_dir", type=str, default="./psiblast_baseline")
    parser.add_argument("--psiblast_bin", type=str, default="psiblast")
    parser.add_argument("--makeblastdb_bin", type=str, default="makeblastdb")
    parser.add_argument("--alphabet", type=str, default=DEFAULT_ALPHABET)

    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--eval_search_k", type=int, default=200)
    parser.add_argument("--evalue", type=float, default=1000.0)
    parser.add_argument("--inclusion_ethresh", type=float, default=0.005)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--sensitivity", type=str, default="normal", choices=list(PROTEIN_SENSITIVITY_PRESETS.keys()))
    parser.add_argument("--word_size", type=int, default=None)
    parser.add_argument("--matrix", type=str, default=None)
    parser.add_argument("--gapopen", type=int, default=None)
    parser.add_argument("--gapextend", type=int, default=None)
    parser.add_argument("--pseudocount", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--max_target_seqs", type=int, default=None)
    parser.add_argument("--max_hsps", type=int, default=1)

    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=1024)
    parser.add_argument("--tmalign_reference", type=int, default=1)
    parser.add_argument("--tmalign_workers", type=int, default=None)

    parser.add_argument("--keep_xml", action="store_true")
    parser.add_argument("--keep_temp", action="store_true", default=True)
    parser.add_argument("--exclude_self_hit", action="store_true", default=True)
    parser.add_argument("--include_self_hit", action="store_false", dest="exclude_self_hit")
    parser.add_argument("--seg", type=str, default="yes", choices=["yes", "no"])
    parser.add_argument("--comp_based_stats", type=int, default=None)
    parser.add_argument("--parse_deflines", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    return PsiBlastConfig(
        data_path=args.data_path,
        pair_file=args.pair_file,
        pdb_root=args.pdb_root,
        tmalign_path=args.tmalign_path,
        work_dir=args.work_dir,
        psiblast_bin=args.psiblast_bin,
        makeblastdb_bin=args.makeblastdb_bin,
        alphabet=args.alphabet,
        topk=args.topk,
        eval_search_k=args.eval_search_k,
        evalue=args.evalue,
        inclusion_ethresh=args.inclusion_ethresh,
        num_iterations=args.num_iterations,
        sensitivity=args.sensitivity,
        word_size=args.word_size,
        matrix=args.matrix,
        gapopen=args.gapopen,
        gapextend=args.gapextend,
        pseudocount=args.pseudocount,
        num_threads=args.num_threads,
        max_target_seqs=args.max_target_seqs,
        max_hsps=args.max_hsps,
        random_state=args.random_state,
        test_size=args.test_size,
        tmalign_reference=args.tmalign_reference,
        tmalign_workers=args.tmalign_workers,
        keep_xml=args.keep_xml,
        keep_temp=args.keep_temp,
        verbose=not args.quiet,
        exclude_self_hit=args.exclude_self_hit,
        seg=args.seg,
        comp_based_stats=args.comp_based_stats,
        parse_deflines=args.parse_deflines,
    )


if __name__ == "__main__":
    config = parse_args()
    evaluate_psiblast(config)
