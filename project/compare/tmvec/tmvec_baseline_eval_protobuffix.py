#!/usr/bin/env python3
"""
TM-Vec baseline evaluator for the user's remote-homology retrieval project.

Design goal:
- Keep the user's existing evaluation protocol (FAISS retrieval + TM-align re-ranking metric)
- Replace ONLY the encoder with the official TM-Vec encoder/checkpoint
- Produce apples-to-apples comparison numbers for a thesis/paper

This script assumes the local project already contains the user's dataset utilities.
It tries both import styles:
    from utils.data / utils.tools
and
    from data / tools
"""

from __future__ import annotations

import argparse
import csv
import json
import gc
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch as pt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- project imports -------------------------------------------------------
try:
    from utils.data import ProteinDataset, QueryHomologyDataset
    from utils.tools import build_idx, calculate_remote_homology_score
except ImportError:
    from data import ProteinDataset, QueryHomologyDataset
    from tools import build_idx, calculate_remote_homology_score

# --- tm-vec imports --------------------------------------------------------
from transformers import T5EncoderModel, T5Tokenizer
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import encode as tmvec_encode


DEFAULT_ALPHABET = "ARNDCQEGHILKMFPSTWYV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate official TM-Vec embeddings under the user's existing retrieval protocol."
    )

    # data / split
    parser.add_argument("--data_path", type=str, required=True, help="Path to your saved protein dataset (.pt).")
    parser.add_argument("--pair_file", type=str, required=True, help="TM-align pair file used to build QueryHomologyDataset.")
    parser.add_argument("--test_size", type=int, default=1024)
    parser.add_argument("--random_state", type=int, default=42)

    # evaluation
    parser.add_argument("--topk", type=int, default=12, help="Final evaluation top-k after self-hit filtering.")
    parser.add_argument("--eval_search_k", type=int, default=200, help="Initial FAISS candidate count before self-hit filtering.")
    parser.add_argument("--pdb_root", type=str, required=True, help="Root directory of PDB files used by TM-align.")
    parser.add_argument("--tmalign_path", type=str, required=True, help="Path to TM-align binary.")
    parser.add_argument("--tmalign_reference", type=int, default=1)
    parser.add_argument("--tmalign_workers", type=int, default=None)

    # tm-vec
    parser.add_argument("--tmvec_model", type=str, required=True, help="TM-Vec checkpoint (.ckpt).")
    parser.add_argument("--tmvec_config", type=str, required=True, help="TM-Vec config json.")
    parser.add_argument(
        "--protrans_model",
        type=str,
        default=None,
        help="Local ProtT5 model directory. If omitted, HuggingFace path Rostlab/prot_t5_xl_uniref50 is used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="cpu | gpu | CUDA index like 0/1/2. Matches official tm-vec script semantics.",
    )

    # sequence decoding
    parser.add_argument(
        "--alphabet",
        type=str,
        default=DEFAULT_ALPHABET,
        help="Token-to-amino-acid mapping for indices 1..20. Padding is assumed to be 0.",
    )
    parser.add_argument(
        "--unknown_token",
        type=str,
        default="X",
        help="Fallback amino acid symbol for invalid / unseen indices.",
    )

    # cache / output
    parser.add_argument("--output_dir", type=str, default="./tmvec_eval_outputs")
    parser.add_argument("--cache_lib_embeddings", action="store_true", default=True)
    parser.add_argument("--no_cache_lib_embeddings", action="store_false", dest="cache_lib_embeddings")
    parser.add_argument("--cache_query_embeddings", action="store_true", default=True)
    parser.add_argument("--no_cache_query_embeddings", action="store_false", dest="cache_query_embeddings")
    parser.add_argument("--write_fasta", action="store_true", default=True)
    parser.add_argument("--no_write_fasta", action="store_false", dest="write_fasta")

    return parser.parse_args()


def resolve_device(device_arg: str) -> pt.device:
    device_arg = str(device_arg).strip().lower()
    if device_arg == "cpu":
        return pt.device("cpu")
    if pt.cuda.is_available():
        if device_arg == "gpu":
            return pt.device("cuda:6")
        if device_arg.isdigit():
            return pt.device(f"cuda:{int(device_arg)}")
    return pt.device("cpu")


def tensor_tokens_to_sequence(seq_tensor, alphabet: str, unknown_token: str = "X") -> str:
    """
    Convert integer-coded sequence to amino-acid string.

    Assumptions consistent with the user's current code:
    - 0 is padding
    - valid residue indices are 1..len(alphabet)
    """
    if hasattr(seq_tensor, "detach"):
        values = seq_tensor.detach().cpu().tolist()
    else:
        values = list(seq_tensor)

    chars: List[str] = []
    for token in values:
        token = int(token)
        if token == 0:
            continue
        if 1 <= token <= len(alphabet):
            chars.append(alphabet[token - 1])
        else:
            chars.append(unknown_token)
    return "".join(chars)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, a_min=1e-12, a_max=None)


def write_fasta(path: Path, names: Sequence[str], seqs: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name, seq in zip(names, seqs):
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def load_tmvec_models(
    tmvec_model_path: str,
    tmvec_config_path: str,
    protrans_model_path: str | None,
    device: pt.device,
):
    hf_source = protrans_model_path or "Rostlab/prot_t5_xl_uniref50"

    # Prefer the slow SentencePiece tokenizer directly from spiece.model.
    # This avoids newer Transformers trying to convert SentencePiece -> fast tokenizer
    # through protobuf/tiktoken paths that often break in offline HPC environments.
    try:
        spiece_path = Path(hf_source) / "spiece.model"
        if spiece_path.is_file():
            tokenizer = T5Tokenizer(
                vocab_file=str(spiece_path),
                do_lower_case=False,
                legacy=True,
            )
        else:
            tokenizer = T5Tokenizer.from_pretrained(
                hf_source,
                do_lower_case=False,
                legacy=True,
                use_fast=False,
                local_files_only=protrans_model_path is not None,
            )
    except TypeError:
        # Some transformers versions do not accept use_fast for T5Tokenizer.from_pretrained.
        tokenizer = T5Tokenizer.from_pretrained(
            hf_source,
            do_lower_case=False,
            legacy=True,
            local_files_only=protrans_model_path is not None,
        )

    prot_model = T5EncoderModel.from_pretrained(
        hf_source,
        local_files_only=protrans_model_path is not None,
    )
    prot_model = prot_model.to(device).eval()

    tmvec_config = trans_basic_block_Config.from_json(tmvec_config_path)
    tmvec_model = trans_basic_block.load_from_checkpoint(
        tmvec_model_path,
        config=tmvec_config,
        map_location=device,
    )
    tmvec_model = tmvec_model.to(device).eval()
    return tokenizer, prot_model, tmvec_model


@pt.no_grad()
def encode_sequences_tmvec(
    sequences: Sequence[str],
    tmvec_model,
    prot_model,
    tokenizer,
    device: pt.device,
    out_path: Path | None = None,
) -> np.ndarray:
    if out_path is not None and out_path.is_file():
        return np.load(out_path)

    # official tm_vec.tm_vec_utils.encode performs single-sequence encoding
    # using ProtT5 + official TM-Vec head; we keep it to stay close to the repo.
    embeddings = tmvec_encode(list(sequences), tmvec_model, prot_model, tokenizer, device)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embeddings)
    return embeddings


def filter_self_hits(
    indices: np.ndarray,
    scores: np.ndarray,
    query_map: np.ndarray,
    lib_map: np.ndarray,
    topk: int,
):
    filtered_indices = []
    filtered_scores = []
    self_hits = 0
    for row_idx, (row, row_scores, q_global_idx) in enumerate(zip(indices, scores, query_map)):
        row_filtered_idx = []
        row_filtered_score = []
        hit_self = False
        for j_local, score in zip(row, row_scores):
            j_local = int(j_local)
            cand_global_idx = int(lib_map[j_local])
            if cand_global_idx == int(q_global_idx):
                hit_self = True
                continue
            row_filtered_idx.append(j_local)
            row_filtered_score.append(float(score))
            if len(row_filtered_idx) >= topk:
                break
        filtered_indices.append(row_filtered_idx)
        filtered_scores.append(row_filtered_score)
        self_hits += int(hit_self)
    self_hit_rate = self_hits / max(len(query_map), 1)
    return filtered_indices, filtered_scores, self_hit_rate


def save_rank_table(path: Path, test_set, database, filtered_indices, faiss_scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_name", "rank", "candidate_name", "faiss_score"])
        for i, row in enumerate(filtered_indices):
            query_name = str(database[int(test_set[i]["idx1"])] ["lab"])
            for rank, j_local in enumerate(row, start=1):
                candidate_name = str(database[int(j_local)]["lab"])
                writer.writerow([query_name, rank, candidate_name, float(faiss_scores[i][rank - 1])])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    data = pt.load(args.data_path, weights_only=False)
    lib_data = ProteinDataset(data, mode="graph")
    lib_map = np.arange(len(lib_data), dtype=np.int64)
    pdb2idx = {str(lib_data[i]["lab"]): i for i in range(len(lib_data))}

    queryhomo = QueryHomologyDataset(None, args.pair_file, pdb2idx)
    all_query_map = np.arange(len(queryhomo), dtype=np.int64)
    _, test_map = train_test_split(
        all_query_map,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    test_map = np.sort(test_map)
    test_set = QueryHomologyDataset(queryhomo, mapping=test_map)
    test_global_idx = np.array([int(test_set[i]["idx1"]) for i in range(len(test_set))], dtype=np.int64)

    print(f"Library proteins: {len(lib_data)}")
    print(f"Test queries: {len(test_set)}")

    print("Decoding amino-acid sequences from integer tokens...")
    lib_names = [str(lib_data[i]["lab"]) for i in range(len(lib_data))]
    lib_seqs = [
        tensor_tokens_to_sequence(lib_data[i]["seq"], args.alphabet, args.unknown_token)
        for i in tqdm(range(len(lib_data)), desc="library sequences")
    ]
    query_names = [lib_names[i] for i in test_global_idx]
    query_seqs = [lib_seqs[i] for i in test_global_idx]

    if args.write_fasta:
        write_fasta(output_dir / "library.fasta", lib_names, lib_seqs)
        write_fasta(output_dir / "queries.fasta", query_names, query_seqs)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    tokenizer, prot_model, tmvec_model = load_tmvec_models(
        tmvec_model_path=args.tmvec_model,
        tmvec_config_path=args.tmvec_config,
        protrans_model_path=args.protrans_model,
        device=device,
    )

    lib_cache = output_dir / "tmvec_lib_embeddings.npy" if args.cache_lib_embeddings else None
    query_cache = output_dir / "tmvec_query_embeddings.npy" if args.cache_query_embeddings else None

    print("Encoding library sequences with official TM-Vec...")
    embs_lib = encode_sequences_tmvec(
        lib_seqs,
        tmvec_model=tmvec_model,
        prot_model=prot_model,
        tokenizer=tokenizer,
        device=device,
        out_path=lib_cache,
    )

    print("Encoding query sequences with official TM-Vec...")
    embs_query = encode_sequences_tmvec(
        query_seqs,
        tmvec_model=tmvec_model,
        prot_model=prot_model,
        tokenizer=tokenizer,
        device=device,
        out_path=query_cache,
    )

    # Official TM-Vec search uses cosine similarity, so we normalize before IP search.
    embs_lib = l2_normalize(embs_lib)
    embs_query = l2_normalize(embs_query)

    search_k = max(args.topk + 1, args.eval_search_k)
    print(f"Searching top-{search_k} neighbors with FAISS inner product (cosine after L2 norm)...")
    indices, scores = build_idx(embs_lib, embs_query, topk=search_k)

    filtered_indices, filtered_scores, self_hit_rate = filter_self_hits(
        indices=indices,
        scores=scores,
        query_map=test_global_idx,
        lib_map=lib_map,
        topk=args.topk,
    )

    save_rank_table(
        output_dir / "tmvec_faiss_candidates.tsv",
        test_set=test_set,
        database=lib_data,
        filtered_indices=filtered_indices,
        faiss_scores=filtered_scores,
    )

    print("Running TM-align-based evaluation on retrieved candidates...")
    sorted_results, avg_top1_score, avg_topk_score, avg_top1_tm, avg_top1_seqid = calculate_remote_homology_score(
        query=test_set,
        database=lib_data,
        idx=filtered_indices,
        k=args.topk,
        pdb_root=args.pdb_root,
        tmalign_path=args.tmalign_path,
        reference=args.tmalign_reference,
        num_workers=args.tmalign_workers,
    )

    metrics = {
        "method": "TM-Vec (official encoder) + user evaluation pipeline",
        "tmvec_model": str(args.tmvec_model),
        "tmvec_config": str(args.tmvec_config),
        "protrans_model": str(args.protrans_model) if args.protrans_model is not None else "Rostlab/prot_t5_xl_uniref50",
        "device": str(device),
        "library_size": int(len(lib_data)),
        "test_queries": int(len(test_set)),
        "search_k": int(search_k),
        "topk": int(args.topk),
        "self_hit_rate": float(self_hit_rate),
        "avg_top1_score": float(avg_top1_score),
        "avg_topk_score": float(avg_topk_score),
        "avg_top1_tm": float(avg_top1_tm),
        "avg_top1_seqid": float(avg_top1_seqid),
        "alphabet": args.alphabet,
    }

    with open(output_dir / "tmvec_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "tmvec_top1_results.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_idx", "query_name", "candidate_idx", "candidate_name", "remote_score", "tm_score", "seqid"])
        for q_idx, rows in sorted_results.items():
            if not rows:
                continue
            query_name = str(lib_data[int(test_set[q_idx]["idx1"])] ["lab"])
            top1 = rows[0]
            cand_name = str(lib_data[int(top1["c_idx"])] ["lab"])
            writer.writerow([
                int(q_idx),
                query_name,
                int(top1["c_idx"]),
                cand_name,
                float(top1["score"]),
                float(top1["tm_score"]),
                float(top1["seqid"]),
            ])

    print("=" * 70)
    print(f"Self-hit rate before filtering: {metrics['self_hit_rate']:.4f}")
    print(f"Average Top-1 Remote homologous score: {metrics['avg_top1_score']:.6f}")
    print(f"Average Top-{args.topk} Remote homologous score: {metrics['avg_topk_score']:.6f}")
    print(f"Average Top-1 TM-score: {metrics['avg_top1_tm']:.6f}")
    print(f"Average Top-1 SeqID: {metrics['avg_top1_seqid']:.6f}")
    print(f"Saved outputs to: {output_dir.resolve()}")

    del tmvec_model, prot_model, tokenizer
    gc.collect()
    if pt.cuda.is_available():
        pt.cuda.empty_cache()


if __name__ == "__main__":
    main()
