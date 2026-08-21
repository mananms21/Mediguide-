"""
MEDIGUIDE — FAISS RAG Index Builder

Usage:
  # Download pre-built index from HF Hub (recommended, ~30 seconds):
  python rag/build_index.py --mode download

  # Build from scratch using MedQuAD (~10 minutes on CPU):
  python rag/build_index.py --mode build

The index is saved to rag/index/ by default.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def download_from_hub(
    hf_dataset_id: str = "Shriyanshml/mediguide-rag-index",
    save_dir: str = "rag/index",
) -> None:
    """Download pre-built FAISS index from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading RAG index from {hf_dataset_id}…")

    for filename in ["faiss_index.bin", "medquad_docs.pkl"]:
        path = hf_hub_download(
            repo_id=hf_dataset_id,
            filename=filename,
            repo_type="dataset",
            local_dir=save_dir,
            local_dir_use_symlinks=False,
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"   ✅ {filename} ({size_mb:.1f} MB) → {path}")

    print(f"\n✅ Index ready at {save_dir}/")
    _print_index_stats(save_dir)


def build_from_scratch(
    save_dir: str = "rag/index",
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """Build FAISS index locally from the MedQuAD HuggingFace dataset."""
    import faiss
    import pandas as pd
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("📦 Loading MedQuAD from HuggingFace Hub…")
    raw = load_dataset(
        "pythonafroz/medquad-medical-question-answer-for-ai-research",
        trust_remote_code=True,
    )
    frames = [raw[s].to_pandas() for s in raw.keys()]
    df = pd.concat(frames, ignore_index=True)

    # Clean
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].str.strip()
    df["answer"]   = df["answer"].str.strip()
    df = df[df["answer"].str.len() > 50]
    df = df[df["question"].str.len() > 5]
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    print(f"   Loaded {len(df):,} clean Q&A pairs")

    # Build document list
    docs = [
        {
            "question":   str(r.question),
            "answer":     str(r.answer),
            "source":     str(getattr(r, "source", "MedQuAD")),
            "focus_area": str(getattr(r, "focus_area", "General")),
        }
        for r in df.itertuples()
    ]

    # Encode
    print(f"\n🔢 Encoding {len(docs):,} questions with {encoder_model}…")
    print("   (This takes ~5–10 minutes on CPU)")
    encoder    = SentenceTransformer(encoder_model)
    questions  = [d["question"] for d in docs]
    embeddings = encoder.encode(
        questions,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Build FAISS index
    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    index_path = os.path.join(save_dir, "faiss_index.bin")
    docs_path  = os.path.join(save_dir, "medquad_docs.pkl")
    faiss.write_index(index, index_path)
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)

    print(f"\n✅ Index saved to {save_dir}/")
    _print_index_stats(save_dir)


def _print_index_stats(save_dir: str) -> None:
    """Print basic index statistics."""
    import faiss

    index = faiss.read_index(os.path.join(save_dir, "faiss_index.bin"))
    with open(os.path.join(save_dir, "medquad_docs.pkl"), "rb") as f:
        docs = pickle.load(f)

    print(f"\n📊 Index statistics:")
    print(f"   Vectors  : {index.ntotal:,}")
    print(f"   Documents: {len(docs):,}")
    print(f"   Dimension: {index.d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or download the MEDIGUIDE FAISS RAG index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["download", "build"],
        default="download",
        help="'download' from HF Hub (fast) or 'build' from scratch (slow)",
    )
    parser.add_argument(
        "--save-dir",
        default="rag/index",
        help="Directory to save the index (default: rag/index)",
    )
    args = parser.parse_args()

    if args.mode == "download":
        download_from_hub(save_dir=args.save_dir)
    else:
        build_from_scratch(save_dir=args.save_dir)
