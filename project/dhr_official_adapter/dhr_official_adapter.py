from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from dhr_official_model import OfficialDHRCheckpoint, SequenceRecord


DEFAULT_TOKEN_MAP = {
    0: "",
    1: "A",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "P",
    14: "Q",
    15: "R",
    16: "S",
    17: "T",
    18: "V",
    19: "W",
    20: "Y",
    21: "X",
}
ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


@dataclass
class PairGroup:
    idx1: int
    idx2_list: List[int]
    tmscore_list: List[float]
    seqid_list: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official DHR checkpoint direct-inference adapter for project data / TSV / FASTA.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_export = subparsers.add_parser("export-project-tsv", help="Convert project .pt data to DHR TSV format.")
    add_project_inputs(p_export)
    p_export.add_argument("--output_tsv", required=True, type=str)

    p_build = subparsers.add_parser("build-db", help="Embed a sequence DB with official candidate encoder and build a FAISS index.")
    add_input_source_args(p_build)
    add_common_inference_args(p_build)
    p_build.add_argument("--output_dir", required=True, type=str)
    p_build.add_argument("--metric", choices=["l2", "ip"], default="l2")

    p_ret = subparsers.add_parser("retrieve", help="Retrieve top-k candidates for TSV / FASTA / project queries.")
    add_input_source_args(p_ret, prefix="query")
    add_common_inference_args(p_ret)
    p_ret.add_argument("--database_dir", required=True, type=str)
    p_ret.add_argument("--output_tsv", required=True, type=str)
    p_ret.add_argument("--topk", type=int, default=200)

    p_eval = subparsers.add_parser("project-eval", help="One-command official DHR retrieval + TM-align evaluation on your project.")
    add_project_inputs(p_eval)
    add_common_inference_args(p_eval)
    p_eval.add_argument("--pair_file", required=True, type=str)
    p_eval.add_argument("--work_dir", required=True, type=str)
    p_eval.add_argument("--metric", choices=["l2", "ip"], default="l2")
    p_eval.add_argument("--topk", type=int, default=12)
    p_eval.add_argument("--search_k", type=int, default=200)
    p_eval.add_argument("--pdb_root", required=True, type=str)
    p_eval.add_argument("--tmalign_path", required=True, type=str)
    p_eval.add_argument("--tmalign_reference", type=int, default=1)
    p_eval.add_argument("--tmalign_workers", type=int, default=None)
    p_eval.add_argument("--eval_subset", choices=["all", "test_split", "train_split"], default="test_split")
    p_eval.add_argument("--test_size", type=int, default=1024)
    p_eval.add_argument("--random_state", type=int, default=42)

    return parser.parse_args()


def add_common_inference_args(parser: argparse.ArgumentParser):
    parser.add_argument("--ckpt_dir", required=True, type=str, default='/home/burger/bioinfo/project/dhr_official_adapter')
    parser.add_argument("--device", default=("cuda:6" if torch.cuda.is_available() else "cpu"), type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")


def add_project_inputs(parser: argparse.ArgumentParser):
    parser.add_argument("--project_pt", required=True, type=str, help="Torch-saved project dataset object.")
    parser.add_argument(
        "--token_map_json",
        type=str,
        default=None,
        help="Optional JSON file mapping integer token ids to amino-acid letters.",
    )
    parser.add_argument(
        "--token_map_preset",
        type=str,
        default="acdefghiklmnpqrstvwy",
        choices=["acdefghiklmnpqrstvwy", "none"],
        help="Fallback token order when project sequences are integer ids.",
    )


def add_input_source_args(parser: argparse.ArgumentParser, prefix: str = "input"):
    parser.add_argument(f"--{prefix}_tsv", type=str, default=None, help="TSV file with columns: id<TAB>sequence")
    parser.add_argument(f"--{prefix}_fasta", type=str, default=None, help="FASTA file with amino-acid sequences")
    parser.add_argument(f"--{prefix}_project_pt", type=str, default=None, help="Project .pt file as alternative input source")
    parser.add_argument(f"--{prefix}_token_map_json", type=str, default=None)
    parser.add_argument(
        f"--{prefix}_token_map_preset",
        type=str,
        default="acdefghiklmnpqrstvwy",
        choices=["acdefghiklmnpqrstvwy", "none"],
    )


def read_tsv_records(tsv_path: str) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"TSV line {line_no} should contain at least 2 columns: {line!r}")
            seq_id = str(parts[0]).strip()
            seq = clean_sequence(parts[1])
            if line_no == 1 and seq_id.lower() == "id" and seq.lower() == "sequence":
                continue
            records.append(SequenceRecord(id=seq_id, sequence=seq, global_idx=len(records)))
    if not records:
        raise ValueError(f"No sequence records found in {tsv_path}")
    return records


def read_fasta_records(fasta_path: str) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []
    cur_id: Optional[str] = None
    cur_seq: List[str] = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    records.append(SequenceRecord(id=cur_id, sequence=clean_sequence("".join(cur_seq)), global_idx=len(records)))
                cur_id = line[1:].strip().split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        records.append(SequenceRecord(id=cur_id, sequence=clean_sequence("".join(cur_seq)), global_idx=len(records)))
    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    return records


def clean_sequence(seq: str) -> str:
    seq = str(seq).upper().replace(" ", "").replace("\n", "")
    cleaned = []
    for ch in seq:
        if ch in ALLOWED_AA:
            cleaned.append(ch)
        elif ch in {"*", "-", "."}:
            continue
        else:
            cleaned.append("X")
    out = "".join(cleaned)
    if not out:
        raise ValueError("Encountered an empty sequence after cleaning.")
    return out


def load_token_map(json_path: Optional[str], preset: str) -> Dict[int, str]:
    if json_path is not None:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        mapping: Dict[int, str] = {}
        for k, v in raw.items():
            mapping[int(k)] = str(v).upper()
        if 0 not in mapping:
            mapping[0] = ""
        return mapping
    if preset == "acdefghiklmnpqrstvwy":
        return dict(DEFAULT_TOKEN_MAP)
    if preset == "none":
        raise ValueError("Integer token sequences require --token_map_json or a non-none preset.")
    raise ValueError(f"Unsupported token map preset: {preset}")


def _tensor_to_list(x) -> List[int]:
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.detach().cpu().view(-1).tolist()]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.reshape(-1).tolist()]
    return [int(v) for v in list(x)]


def integer_tokens_to_sequence(tokens, token_map: Dict[int, str]) -> str:
    ints = _tensor_to_list(tokens)
    chars: List[str] = []
    for tok in ints:
        if tok == 0:
            continue
        ch = token_map.get(tok, token_map.get(int(tok), "X"))
        if not ch:
            continue
        chars.append(ch)
    seq = clean_sequence("".join(chars))
    return seq


def _extract_project_arrays(obj) -> Tuple[Sequence, Sequence]:
    if isinstance(obj, tuple):
        if len(obj) < 2:
            raise ValueError("Project tuple must contain at least (seq, lab)")
        seqs = obj[0]
        labs = obj[1]
        return seqs, labs
    if hasattr(obj, "seq") and hasattr(obj, "lab"):
        return obj.seq, obj.lab
    raise TypeError("Unsupported project object. Expected tuple(seq, lab, ...) or object with .seq and .lab")


def normalize_label(label) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8")
    if isinstance(label, np.bytes_):
        return label.decode("utf-8")
    return str(label)


def load_project_records(project_pt: str, token_map_json: Optional[str], token_map_preset: str) -> List[SequenceRecord]:
    obj = torch.load(project_pt, map_location="cpu", weights_only=False)
    seqs, labs = _extract_project_arrays(obj)
    token_map = load_token_map(token_map_json, token_map_preset)

    records: List[SequenceRecord] = []
    for idx in range(len(labs)):
        label = normalize_label(labs[idx])
        seq_item = seqs[idx]
        if isinstance(seq_item, str):
            seq = clean_sequence(seq_item)
        elif isinstance(seq_item, (bytes, np.bytes_)):
            seq = clean_sequence(seq_item.decode("utf-8"))
        else:
            seq = integer_tokens_to_sequence(seq_item, token_map)
        records.append(SequenceRecord(id=label, sequence=seq, global_idx=idx))
    if not records:
        raise ValueError(f"No records extracted from {project_pt}")
    return records


def load_records_from_args(args: argparse.Namespace, prefix: str = "input") -> List[SequenceRecord]:
    tsv_path = getattr(args, f"{prefix}_tsv", None)
    fasta_path = getattr(args, f"{prefix}_fasta", None)
    project_pt = getattr(args, f"{prefix}_project_pt", None)
    chosen = [x is not None for x in [tsv_path, fasta_path, project_pt]]
    if sum(chosen) != 1:
        raise ValueError(f"Exactly one of --{prefix}_tsv / --{prefix}_fasta / --{prefix}_project_pt must be provided.")
    if tsv_path is not None:
        return read_tsv_records(tsv_path)
    if fasta_path is not None:
        return read_fasta_records(fasta_path)
    return load_project_records(
        project_pt=project_pt,
        token_map_json=getattr(args, f"{prefix}_token_map_json", None),
        token_map_preset=getattr(args, f"{prefix}_token_map_preset", "acdefghiklmnpqrstvwy"),
    )


def save_records_tsv(records: Sequence[SequenceRecord], output_tsv: str):
    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for rec in records:
            writer.writerow([rec.id, rec.sequence])


def build_faiss_index(embeddings: np.ndarray, metric: str):
    import faiss

    dim = int(embeddings.shape[1])
    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
    elif metric == "ip":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    index.add(embeddings.astype(np.float32))
    return index


def build_database(records: Sequence[SequenceRecord], args: argparse.Namespace) -> Dict[str, str]:
    model = OfficialDHRCheckpoint(args.ckpt_dir, device=args.device)
    embeddings = model.encode_records(records, tower="candidate", batch_size=args.batch_size, progress=not args.quiet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "row_idx": np.arange(len(records), dtype=np.int64),
        "global_idx": [rec.global_idx if rec.global_idx is not None else i for i, rec in enumerate(records)],
        "id": [rec.id for rec in records],
        "sequence": [rec.sequence for rec in records],
    })
    df_path = output_dir / "df-ebd.pkl"
    emb_path = output_dir / "embeddings.npy"
    index_path = output_dir / "index-ebd.index"
    metric_meta = output_dir / "meta.json"

    df.to_pickle(df_path)
    np.save(emb_path, embeddings)
    index = build_faiss_index(embeddings, metric=args.metric)
    import faiss

    faiss.write_index(index, str(index_path))
    with open(metric_meta, "w", encoding="utf-8") as f:
        json.dump({"metric": args.metric, "count": len(records), "dim": int(embeddings.shape[1])}, f, indent=2)

    return {
        "output_dir": str(output_dir),
        "df_path": str(df_path),
        "emb_path": str(emb_path),
        "index_path": str(index_path),
        "metric_meta": str(metric_meta),
    }


def retrieve_against_database(query_records: Sequence[SequenceRecord], args: argparse.Namespace) -> pd.DataFrame:
    db_dir = Path(args.database_dir)
    index_path = db_dir / "index-ebd.index"
    df_path = db_dir / "df-ebd.pkl"
    if not index_path.is_file():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not df_path.is_file():
        raise FileNotFoundError(f"DB metadata not found: {df_path}")

    model = OfficialDHRCheckpoint(args.ckpt_dir, device=args.device)
    query_embeddings = model.encode_records(query_records, tower="query", batch_size=args.batch_size, progress=not args.quiet)

    import faiss

    index = faiss.read_index(str(index_path))
    db_df = pd.read_pickle(df_path).reset_index(drop=True)
    scores, indices = index.search(query_embeddings.astype(np.float32), int(args.topk))

    rows: List[Dict[str, object]] = []
    for q_i, rec in enumerate(query_records):
        for rank, (db_row_idx, score) in enumerate(zip(indices[q_i].tolist(), scores[q_i].tolist()), start=1):
            db_row = db_df.iloc[int(db_row_idx)]
            rows.append({
                "query_id": rec.id,
                "query_global_idx": rec.global_idx,
                "rank": rank,
                "candidate_row_idx": int(db_row_idx),
                "candidate_global_idx": int(db_row["global_idx"]),
                "candidate_id": str(db_row["id"]),
                "faiss_score": float(score),
            })
    return pd.DataFrame(rows)


def load_pair_groups(pair_file: str, label_to_idx: Dict[str, int]) -> List[PairGroup]:
    df = pd.read_csv(pair_file, names=["name1", "name2", "tmscore", "seqid"], sep="\t")
    df["name1"] = df["name1"].astype(str).str.split("/").str[-1].str.removesuffix(".pdb")
    df["name2"] = df["name2"].astype(str).str.split("/").str[-1].str.removesuffix(".pdb")
    df["idx1"] = df["name1"].map(label_to_idx)
    df["idx2"] = df["name2"].map(label_to_idx)
    df = df.dropna(subset=["idx1", "idx2"]).copy()
    df["idx1"] = df["idx1"].astype(np.int64)
    df["idx2"] = df["idx2"].astype(np.int64)

    groups: List[PairGroup] = []
    for idx1, g in df.groupby("idx1", sort=False):
        groups.append(PairGroup(
            idx1=int(idx1),
            idx2_list=g["idx2"].to_numpy(dtype=np.int64).tolist(),
            tmscore_list=g["tmscore"].to_numpy(dtype=np.float32).tolist(),
            seqid_list=g["seqid"].to_numpy(dtype=np.float32).tolist(),
        ))
    return groups


def choose_eval_groups(groups: Sequence[PairGroup], eval_subset: str, test_size: int, random_state: int) -> List[PairGroup]:
    if eval_subset == "all":
        return list(groups)
    n = len(groups)
    if n == 0:
        return []
    test_size = min(max(int(test_size), 1), n)
    try:
        from sklearn.model_selection import train_test_split

        all_idx = np.arange(n, dtype=np.int64)
        train_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=random_state)
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)
    except Exception:
        rng = np.random.RandomState(random_state)
        all_idx = np.arange(n, dtype=np.int64)
        rng.shuffle(all_idx)
        test_idx = np.sort(all_idx[:test_size])
        train_idx = np.sort(all_idx[test_size:])

    chosen_idx = test_idx if eval_subset == "test_split" else train_idx
    return [groups[int(i)] for i in chosen_idx.tolist()]


def make_pdb_path(pdb_root: str, pdb_name: str) -> Path:
    return Path(pdb_root) / pdb_name[1:3] / f"{pdb_name}.pdb"


def run_tmalign(task):
    q_idx, c_idx, pdb1, pdb2, tmalign_path, reference = task
    cmd = [tmalign_path, str(pdb1), str(pdb2)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"TMalign failed. cmd={' '.join(cmd)}\nreturncode={result.returncode}\nstderr={result.stderr}"
        )
    output = result.stdout
    seqid_match = re.search(r"Seq_ID=.*?=\s*([0-9.]+)", output)
    tm_scores = re.findall(r"TM-score=\s*([0-9.]+)", output)
    if seqid_match is None:
        raise ValueError(f"Failed to parse Seq_ID from TMalign output.\n{output}")
    if not (1 <= reference <= len(tm_scores)):
        raise ValueError(f"Invalid reference={reference}, only {len(tm_scores)} TM-scores parsed.")
    seqid = float(seqid_match.group(1))
    tm_score = float(tm_scores[reference - 1])
    remote_score = tm_score - 0.6 + min(0.4 - seqid, 0.0)
    return q_idx, c_idx, {"score": remote_score, "tm_score": tm_score, "seqid": seqid}


def calculate_remote_homology_score(
    query_groups: Sequence[PairGroup],
    records: Sequence[SequenceRecord],
    candidate_rows: Sequence[Sequence[int]],
    topk: int,
    pdb_root: str,
    tmalign_path: str,
    reference: int,
    num_workers: Optional[int],
):
    tasks = []
    tmalign_path = str(Path(tmalign_path).resolve())
    if not Path(tmalign_path).is_file():
        raise FileNotFoundError(f"TMalign binary not found: {tmalign_path}")

    for q_local, group in enumerate(query_groups):
        q_name = records[int(group.idx1)].id
        q_file = make_pdb_path(pdb_root, q_name)
        if not q_file.is_file():
            raise FileNotFoundError(f"Query PDB file not found: {q_file}")
        for cand_global_idx in list(candidate_rows[q_local])[:topk]:
            c_name = records[int(cand_global_idx)].id
            c_file = make_pdb_path(pdb_root, c_name)
            if not c_file.is_file():
                raise FileNotFoundError(f"Candidate PDB file not found: {c_file}")
            tasks.append((q_local, int(cand_global_idx), q_file, c_file, tmalign_path, int(reference)))

    if not tasks:
        raise ValueError("No TM-align tasks were generated.")
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, min(len(tasks), cpu_count // 2 if cpu_count > 1 else 1))

    results_dict: Dict[int, List[Dict[str, float]]] = {i: [] for i in range(len(query_groups))}
    avg_topk_score = 0.0
    success_cnt = 0
    error_messages: List[str] = []
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_tmalign, task) for task in tasks]
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="TM-align", leave=False)
        for future in iterator:
            try:
                q_idx, c_idx, result = future.result()
                results_dict[q_idx].append({
                    "c_idx": c_idx,
                    "score": float(result["score"]),
                    "tm_score": float(result["tm_score"]),
                    "seqid": float(result["seqid"]),
                })
                avg_topk_score += float(result["score"])
                success_cnt += 1
            except Exception as exc:
                error_messages.append(str(exc))

    if success_cnt == 0:
        raise RuntimeError("All TM-align tasks failed." + (f"\nFirst error:\n{error_messages[0]}" if error_messages else ""))
    avg_topk_score /= success_cnt

    sorted_results: Dict[int, List[Dict[str, float]]] = {}
    avg_top1_score = 0.0
    avg_top1_tm = 0.0
    avg_top1_seqid = 0.0
    valid_queries = 0
    for q_idx in range(len(query_groups)):
        row = results_dict[q_idx]
        row.sort(key=lambda x: x["score"], reverse=True)
        sorted_results[q_idx] = row
        if row:
            avg_top1_score += row[0]["score"]
            avg_top1_tm += row[0]["tm_score"]
            avg_top1_seqid += row[0]["seqid"]
            valid_queries += 1
    if valid_queries > 0:
        avg_top1_score /= valid_queries
        avg_top1_tm /= valid_queries
        avg_top1_seqid /= valid_queries

    return {
        "sorted_results": sorted_results,
        "avg_top1_score": float(avg_top1_score),
        "avg_topk_score": float(avg_topk_score),
        "avg_top1_tm": float(avg_top1_tm),
        "avg_top1_seqid": float(avg_top1_seqid),
        "valid_queries": int(valid_queries),
        "failed_tmalign_tasks": len(error_messages),
        "first_error": error_messages[0] if error_messages else None,
    }


def filter_self_hits(
    raw_indices: np.ndarray,
    raw_scores: np.ndarray,
    db_df: pd.DataFrame,
    query_groups: Sequence[PairGroup],
    records: Sequence[SequenceRecord],
    keep_topk: int,
) -> Tuple[List[List[int]], List[List[float]], float]:
    filtered_global_idx: List[List[int]] = []
    filtered_scores: List[List[float]] = []
    self_hits = 0

    for q_local, (row_idxes, row_scores) in enumerate(zip(raw_indices, raw_scores)):
        q_global_idx = int(query_groups[q_local].idx1)
        q_id = records[q_global_idx].id
        cur_idxes: List[int] = []
        cur_scores: List[float] = []
        hit_self = False
        for db_row_idx, score in zip(row_idxes.tolist(), row_scores.tolist()):
            meta = db_df.iloc[int(db_row_idx)]
            cand_global_idx = int(meta["global_idx"])
            cand_id = str(meta["id"])
            if cand_global_idx == q_global_idx or cand_id == q_id:
                hit_self = True
                continue
            cur_idxes.append(cand_global_idx)
            cur_scores.append(float(score))
            if len(cur_idxes) >= keep_topk:
                break
        filtered_global_idx.append(cur_idxes)
        filtered_scores.append(cur_scores)
        self_hits += int(hit_self)

    self_hit_rate = self_hits / max(len(query_groups), 1)
    return filtered_global_idx, filtered_scores, float(self_hit_rate)


def command_export_project_tsv(args: argparse.Namespace):
    records = load_project_records(args.project_pt, args.token_map_json, args.token_map_preset)
    save_records_tsv(records, args.output_tsv)
    print(f"Exported {len(records)} sequences to {args.output_tsv}")


def command_build_db(args: argparse.Namespace):
    records = load_records_from_args(args, prefix="input")
    paths = build_database(records, args)
    print(json.dumps(paths, indent=2, ensure_ascii=False))


def command_retrieve(args: argparse.Namespace):
    records = load_records_from_args(args, prefix="query")
    df = retrieve_against_database(records, args)
    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved retrieval results to {out_path}")


def command_project_eval(args: argparse.Namespace):
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    records = load_project_records(args.project_pt, args.token_map_json, args.token_map_preset)
    label_to_idx = {rec.id: int(rec.global_idx) for rec in records}
    groups_all = load_pair_groups(args.pair_file, label_to_idx)
    groups = choose_eval_groups(groups_all, args.eval_subset, args.test_size, args.random_state)
    if not groups:
        raise ValueError("No query groups selected for evaluation.")

    db_dir = work_dir / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_args = argparse.Namespace(**vars(args))
    db_args.output_dir = str(db_dir)
    if not (db_dir / "index-ebd.index").is_file() or not (db_dir / "df-ebd.pkl").is_file():
        build_database(records, db_args)

    model = OfficialDHRCheckpoint(args.ckpt_dir, device=args.device)
    query_records = [records[int(g.idx1)] for g in groups]
    query_embeddings = model.encode_records(query_records, tower="query", batch_size=args.batch_size, progress=not args.quiet)

    import faiss

    index = faiss.read_index(str(db_dir / "index-ebd.index"))
    db_df = pd.read_pickle(db_dir / "df-ebd.pkl").reset_index(drop=True)
    search_k = max(int(args.search_k), int(args.topk) + 1)
    raw_scores, raw_indices = index.search(query_embeddings.astype(np.float32), search_k)
    candidate_rows, candidate_scores, self_hit_rate = filter_self_hits(
        raw_indices=raw_indices,
        raw_scores=raw_scores,
        db_df=db_df,
        query_groups=groups,
        records=records,
        keep_topk=int(args.topk),
    )

    metrics = calculate_remote_homology_score(
        query_groups=groups,
        records=records,
        candidate_rows=candidate_rows,
        topk=int(args.topk),
        pdb_root=args.pdb_root,
        tmalign_path=args.tmalign_path,
        reference=int(args.tmalign_reference),
        num_workers=args.tmalign_workers,
    )

    summary = {
        "query_count": len(groups),
        "eval_subset": args.eval_subset,
        "search_k": search_k,
        "topk": int(args.topk),
        "metric": args.metric,
        "self_hit_rate": self_hit_rate,
        "avg_top1_score": metrics["avg_top1_score"],
        "avg_topk_score": metrics["avg_topk_score"],
        "avg_top1_tm": metrics["avg_top1_tm"],
        "avg_top1_seqid": metrics["avg_top1_seqid"],
        "valid_queries": metrics["valid_queries"],
        "failed_tmalign_tasks": metrics["failed_tmalign_tasks"],
        "first_error": metrics["first_error"],
    }
    with open(work_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    rows: List[Dict[str, object]] = []
    for q_local, group in enumerate(groups):
        q_rec = records[int(group.idx1)]
        for rank, (cand_idx, faiss_score) in enumerate(zip(candidate_rows[q_local], candidate_scores[q_local]), start=1):
            c_rec = records[int(cand_idx)]
            rows.append({
                "query_rank_group": q_local,
                "query_global_idx": int(group.idx1),
                "query_id": q_rec.id,
                "rank": rank,
                "candidate_global_idx": int(cand_idx),
                "candidate_id": c_rec.id,
                "faiss_score": float(faiss_score),
            })
    pd.DataFrame(rows).to_csv(work_dir / "retrieval_rows.tsv", sep="\t", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary to {work_dir / 'summary.json'}")
    print(f"Saved retrieval rows to {work_dir / 'retrieval_rows.tsv'}")


def main():
    args = parse_args()
    if args.command == "export-project-tsv":
        command_export_project_tsv(args)
    elif args.command == "build-db":
        command_build_db(args)
    elif args.command == "retrieve":
        command_retrieve(args)
    elif args.command == "project-eval":
        command_project_eval(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
