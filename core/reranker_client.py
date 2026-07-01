import logging

from core.config import settings

logger = logging.getLogger(__name__)


class RerankerClient:
    """Local BGE cross-encoder reranker for improving FAISS retrieval precision."""

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model: %s", settings.reranker_model)
            self._model = CrossEncoder(
                settings.reranker_model,
                device=settings.reranker_device,
            )
            logger.info("Reranker model loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        passages: list[tuple[object, float]],
        top_n: int | None = None,
    ) -> list[tuple[object, float]]:
        if not settings.reranker_enabled or not passages:
            return passages[: top_n or len(passages)]

        top_n = top_n or settings.reranker_top_n
        pairs = [(query, chunk.text) for chunk, _ in passages]
        scores = self.model.predict(pairs)
        reranked = [
            (chunk, float(score))
            for (chunk, _embedding_score), score in zip(passages, scores)
        ]
        return sorted(reranked, key=lambda item: item[1], reverse=True)[:top_n]


reranker_client = RerankerClient()
