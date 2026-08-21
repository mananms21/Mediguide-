"""
MEDIGUIDE — RAG Retriever

Retrieves relevant MedQuAD Q&A pairs via FAISS cosine similarity search
and injects them as grounding context into the model prompt.

Quick-start:
    from rag.retriever import MedRAGRetriever

    retriever = MedRAGRetriever()                   # lazy-loads on first use
    context   = retriever.format_context("What is Type 2 diabetes?")
    docs      = retriever.retrieve("chest pain symptoms", top_k=5)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


class MedRAGRetriever:
    """
    FAISS-based retriever backed by the MedQuAD dataset.

    The index is loaded lazily on first use so that importing this module
    has no side-effects (important for Streamlit's module caching).
    """

    # Default index location relative to the project root
    _DEFAULT_INDEX_DIR = str(Path(__file__).parent / "index")

    def __init__(
        self,
        index_dir: Optional[str] = None,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.index_dir     = index_dir or self._DEFAULT_INDEX_DIR
        self._encoder_name = encoder_model

        # Lazily populated
        self._encoder = None
        self._index   = None
        self._docs: list[dict] | None = None

    # ── Public API ────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the top-k most relevant MedQuAD documents for *query*.

        Returns a list of dicts, each with keys:
            question, answer, source, focus_area, relevance_score
        """
        self._lazy_load()

        embedding = self._encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        import faiss as _faiss
        _faiss.normalize_L2(embedding)

        scores, indices = self._index.search(embedding, top_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                doc = dict(self._docs[idx])
                doc["relevance_score"] = round(float(score), 4)
                results.append(doc)
        return results

    def format_context(self, query: str, top_k: int = 3) -> str:
        """
        Return a formatted context string for prompt injection.

        Example output:
            Relevant medical references:

            [Ref 1] Topic: Endocrinology
            Q: What are the symptoms of Type 2 diabetes?
            A: Common symptoms include frequent urination…
        """
        docs = self.retrieve(query, top_k)
        if not docs:
            return ""

        parts = ["Relevant medical references:\n"]
        for i, doc in enumerate(docs, 1):
            answer_preview = doc["answer"]
            if len(answer_preview) > 350:
                answer_preview = answer_preview[:347] + "…"
            parts.append(
                f"[Ref {i}] Topic: {doc.get('focus_area', 'General')}\n"
                f"Q: {doc['question']}\n"
                f"A: {answer_preview}"
            )
        return "\n\n".join(parts)

    def build_phi3_prompt(
        self,
        question: str,
        system_prompt: str,
        use_rag: bool = True,
        top_k: int = 3,
    ) -> str:
        """
        Build a complete Phi-3 instruct prompt, optionally with RAG context.
        """
        if use_rag and self.is_available:
            context = self.format_context(question, top_k)
            user_message = f"{context}\n\nBased on the references above, answer:\n{question}"
        else:
            user_message = question

        return (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_message}<|end|>\n"
            f"<|assistant|>\n"
        )

    def build_falcon_prompt(
        self,
        question: str,
        use_rag: bool = True,
        top_k: int = 3,
    ) -> str:
        """
        Build a Falcon-style chat prompt, optionally with RAG context.
        """
        if use_rag and self.is_available:
            context  = self.format_context(question, top_k)
            question = f"{context}\n\nBased on the references above, answer: {question}"
        return f": {question}?\n: "

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True if the FAISS index files exist on disk."""
        return os.path.exists(os.path.join(self.index_dir, "faiss_index.bin"))

    @property
    def num_documents(self) -> int:
        """Number of documents in the index (loads index if needed)."""
        self._lazy_load()
        return len(self._docs or [])

    # ── Internal ──────────────────────────────────────────────────

    def _lazy_load(self) -> None:
        """Load FAISS index and sentence encoder on first use."""
        if self._index is not None:
            return  # Already loaded

        index_path = os.path.join(self.index_dir, "faiss_index.bin")
        docs_path  = os.path.join(self.index_dir, "medquad_docs.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'.\n"
                "Run:  python rag/build_index.py --mode download\n"
                "or:   python rag/build_index.py --mode build"
            )

        import faiss as _faiss
        from sentence_transformers import SentenceTransformer

        self._index   = _faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self._docs = pickle.load(f)
        self._encoder = SentenceTransformer(self._encoder_name)

    def __repr__(self) -> str:
        status = f"{self.num_documents:,} docs" if self._index else "not loaded"
        return f"MedRAGRetriever(index_dir='{self.index_dir}', {status})"
