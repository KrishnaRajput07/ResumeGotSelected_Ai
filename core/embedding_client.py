# ─────────────────────────────────────────────────────────────────────────────
# Embedding Client — BAAI/bge-large-en-v1.5 via sentence-transformers
# ─────────────────────────────────────────────────────────────────────────────
# Engineering notes:
#   - Model is loaded ONCE at startup (lazy singleton pattern)
#   - BGE-large requires a query prefix for asymmetric retrieval:
#       * Passage (resume chunks): no prefix needed
#       * Query (criterion questions): prepend "Represent this sentence for searching..."
#   - Embeddings are L2-normalized before storage (enables cosine via inner product)
#   - Disk cache keyed by (model_name, text_hash) avoids re-embedding unchanged docs
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import settings

logger = logging.getLogger(__name__)

# BGE-large asymmetric retrieval prefix (for queries, not passages)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingClient:
    """
    Local sentence-transformer embedding client.
    
    Handles:
    - Lazy model loading (model downloads on first use, ~1.3 GB for bge-large)
    - Batched encoding for efficiency
    - L2 normalization (required for cosine similarity via FAISS inner product)
    - Disk-based embedding cache to avoid re-embedding unchanged text
    """

    def __init__(self):
        self._model = None          # Lazy loaded
        self._cache_dir = Path(settings.embeddings_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
            )
            logger.info("Embedding model loaded.")
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """
        Embed resume text chunks (passages).
        NO query prefix — passages are embedded as-is.
        
        Returns:
            float32 array of shape (len(texts), embedding_dimension), L2-normalized
        """
        if not texts:
            return np.zeros((0, settings.embedding_dimension), dtype=np.float32)

        # Check cache for individual texts
        embeddings = []
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cached = self._load_from_cache(text, prefix="passage")
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch encode uncached texts
        if uncached_texts:
            new_embeddings = self._encode_batch(uncached_texts, query_prefix=False)
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb
                self._save_to_cache(texts[idx], emb, prefix="passage")

        result = np.stack(embeddings, axis=0).astype(np.float32)
        return self._normalize(result)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a retrieval query (criterion text, rubric profile).
        Prepends BGE query prefix for asymmetric retrieval.
        
        Returns:
            float32 array of shape (1, embedding_dimension), L2-normalized
        """
        prefixed_text = BGE_QUERY_PREFIX + text

        cached = self._load_from_cache(prefixed_text, prefix="query")
        if cached is not None:
            return self._normalize(cached.reshape(1, -1))

        emb = self._encode_batch([prefixed_text], query_prefix=False)
        self._save_to_cache(prefixed_text, emb[0], prefix="query")
        return self._normalize(emb[0].reshape(1, -1))

    def embed_queries_batch(self, texts: list[str]) -> np.ndarray:
        """Batch embed multiple queries (e.g., all rubric criteria at once)."""
        prefixed = [BGE_QUERY_PREFIX + t for t in texts]
        embs = self._encode_batch(prefixed, query_prefix=False)
        return self._normalize(embs)

    def _encode_batch(self, texts: list[str], query_prefix: bool = False) -> np.ndarray:
        """Internal batched encoding."""
        result = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,  # We normalize manually
            convert_to_numpy=True,
        )
        return result.astype(np.float32)

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings for cosine similarity via inner product."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms

    def _cache_key(self, text: str, prefix: str) -> str:
        text_hash = hashlib.sha256(
            f"{settings.embedding_model}:{prefix}:{text}".encode()
        ).hexdigest()[:16]
        return text_hash

    def _cache_path(self, text: str, prefix: str) -> Path:
        key = self._cache_key(text, prefix)
        return self._cache_dir / f"{key}.pkl"

    def _load_from_cache(self, text: str, prefix: str) -> Optional[np.ndarray]:
        path = self._cache_path(text, prefix)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                path.unlink(missing_ok=True)
        return None

    def _save_to_cache(self, text: str, embedding: np.ndarray, prefix: str) -> None:
        path = self._cache_path(text, prefix)
        try:
            with open(path, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")


# Singleton
embedding_client = EmbeddingClient()
