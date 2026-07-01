# ─────────────────────────────────────────────────────────────────────────────
# FAISS Vector Store — Per-candidate index + Cross-candidate index
# ─────────────────────────────────────────────────────────────────────────────
# Two index types:
#   1. CandidateVectorStore: Per-candidate FAISS index.
#      Used in Step 5 criterion Q&A to retrieve the most relevant
#      chunks for a given criterion question from ONE candidate's resume.
#
#   2. CrossCandidateIndex: Aggregated index across ALL candidates.
#      Used in Step 4 shortlisting to rank ALL candidates by rubric similarity.
#      Each candidate is represented by their MEAN chunk embedding.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from schemas.candidate import ResumeChunk
from core.config import settings

logger = logging.getLogger(__name__)


class CandidateVectorStore:
    """
    FAISS index for a single candidate's resume chunks.
    
    Uses IndexFlatIP (inner product) — with L2-normalized vectors,
    inner product equals cosine similarity.
    """

    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self.dimension = settings.embedding_dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: list[ResumeChunk] = []

    def add_chunks(self, chunks: list[ResumeChunk], embeddings: np.ndarray) -> None:
        """
        Add embedded chunks to the index.
        
        Args:
            chunks: List of ResumeChunk objects
            embeddings: float32 array of shape (len(chunks), dimension), L2-normalized
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        if len(chunks) == 0:
            return

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.debug(f"[{self.candidate_id}] Added {len(chunks)} chunks to index")

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_n: int = 5,
    ) -> list[tuple[ResumeChunk, float]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query_embedding: float32 array of shape (1, dimension), L2-normalized
            top_n: Number of chunks to retrieve
        
        Returns:
            List of (ResumeChunk, similarity_score) tuples, sorted by score desc
        """
        if self.index.ntotal == 0:
            logger.warning(f"[{self.candidate_id}] Index is empty — no chunks to retrieve")
            return []

        actual_top_n = min(top_n, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, actual_top_n)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            results.append((self.chunks[idx], float(score)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """
        Returns the mean of all chunk embeddings.
        Used by CrossCandidateIndex to represent this candidate as a single vector.
        """
        if self.index.ntotal == 0:
            return None
        # Reconstruct all vectors from the flat index
        all_vecs = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
        for i in range(self.index.ntotal):
            self.index.reconstruct(i, all_vecs[i])
        return all_vecs.mean(axis=0)

    def save(self, path: str | Path) -> None:
        """Serialize index and chunks to disk for reuse across runs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str | Path, candidate_id: str) -> "CandidateVectorStore":
        """Load a previously serialized index."""
        path = Path(path)
        store = cls(candidate_id)
        store.index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".chunks.pkl"), "rb") as f:
            store.chunks = pickle.load(f)
        return store


class CrossCandidateIndex:
    """
    Cross-candidate FAISS index for Step 4 shortlisting.
    
    Each candidate is indexed by their MEAN resume embedding.
    Query vector = composite rubric embedding.
    Returns top-K candidates by cosine similarity to the rubric.
    
    This is a RECALL step — deliberately generous (err toward keeping
    borderline candidates) since real scoring happens in Step 5.
    """

    def __init__(self):
        self.dimension = settings.embedding_dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.candidate_ids: list[str] = []

    def add_candidate(self, candidate_id: str, mean_embedding: np.ndarray) -> None:
        """Add a single candidate's representative embedding."""
        vec = mean_embedding.astype(np.float32).reshape(1, -1)
        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self.index.add(vec)
        self.candidate_ids.append(candidate_id)

    def shortlist(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        similarity_floor: float = 0.30,
    ) -> list[tuple[str, float]]:
        """
        Return top-K candidates by cosine similarity to query.
        
        Args:
            query_embedding: Rubric composite embedding, shape (1, dimension)
            top_k: Max candidates to return
            similarity_floor: Minimum similarity score; below this = filtered out
        
        Returns:
            List of (candidate_id, similarity_score) sorted by score desc
        """
        if self.index.ntotal == 0:
            return []

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < similarity_floor:
                continue
            results.append((self.candidate_ids[idx], float(score)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def total_candidates(self) -> int:
        return self.index.ntotal
