"""
Provider-agnostic embedding service for PulsePal.
Supports configurable embedding providers and dimensions.
"""

import os
import logging
import time
import random
from typing import List, Optional
from abc import ABC, abstractmethod
import google.generativeai as genai

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def get_dimensions(self) -> int:
        pass

    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        pass


class GoogleEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding provider."""

    def __init__(
        self, api_key: str, model: str = "models/embedding-001", dimensions: int = 768
    ):
        self.dimensions = dimensions
        self.model = model
        genai.configure(api_key=api_key)

    def get_dimensions(self) -> int:
        return self.dimensions

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding with exponential backoff for rate limiting."""
        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    output_dimensionality=self.dimensions,
                )
                return result["embedding"]

            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                elif attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise
                else:
                    raise

        raise Exception(f"Failed after {max_retries} attempts")

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        Google's API has strict payload limits (~36KB), so we process individually.
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.create_embedding(text)
                embeddings.append(embedding)
                if i < len(texts) - 1:
                    time.sleep(0.1)  # Rate limiting between requests
            except Exception as e:
                logger.error(f"Failed to create embedding for text {i}: {e}")
                # Re-raise to fail fast rather than continue with partial results
                raise Exception(f"Embedding failed for batch item {i}: {e}")
        return embeddings


class KeywordFallbackProvider(EmbeddingProvider):
    """Fallback when API unavailable."""

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions
        logger.warning("Using keyword fallback - degraded search")

    def get_dimensions(self) -> int:
        return self.dimensions

    def create_embedding(self, text: str) -> List[float]:
        return [0.0] * self.dimensions

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.create_embedding(text) for text in texts]


class EmbeddingService:
    """Main embedding service."""

    def __init__(self):
        self.provider = self._initialize_provider()

    def _initialize_provider(self) -> EmbeddingProvider:
        provider_name = os.getenv("EMBEDDING_PROVIDER", "google").lower()

        if provider_name == "google":
            api_key = os.getenv("GOOGLE_API_KEY_EMBEDDING")
            if not api_key:
                logger.error("GOOGLE_API_KEY_EMBEDDING not found, using fallback")
                return KeywordFallbackProvider()

            dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
            model = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

            logger.info(f"Using Google embeddings: {dimensions}D")
            return GoogleEmbeddingProvider(api_key, model, dimensions)

        return KeywordFallbackProvider()

    def get_dimensions(self) -> int:
        return self.provider.get_dimensions()

    def create_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.get_dimensions()

        try:
            return self.provider.create_embedding(text)
        except Exception as e:
            logger.error(f"Falling back: {e}")
            fallback = KeywordFallbackProvider(self.get_dimensions())
            return fallback.create_embedding(text)

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of texts.
        In migration scenarios, we want to fail fast rather than fall back to keywords.
        """
        try:
            return self.provider.create_embeddings_batch(texts)
        except Exception as e:
            # Check if this is being called from migration context
            import inspect

            frame = inspect.currentframe()
            try:
                # Look up the call stack for migration-related functions
                caller_names = []
                current = frame
                for _ in range(10):  # Check up to 10 frames up
                    current = current.f_back
                    if current is None:
                        break
                    caller_names.append(current.f_code.co_name)

                is_migration = any("migrat" in name.lower() for name in caller_names)

                if is_migration:
                    logger.error(f"Migration embedding failed: {e}")
                    raise e  # Fail fast during migration
                else:
                    logger.error(f"Batch failed, using fallback: {e}")
                    fallback = KeywordFallbackProvider(self.get_dimensions())
                    return fallback.create_embeddings_batch(texts)

            finally:
                del frame


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def create_embedding(text: str) -> List[float]:
    return get_embedding_service().create_embedding(text)


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    return get_embedding_service().create_embeddings_batch(texts)


def get_embedding_dimensions() -> int:
    return get_embedding_service().get_dimensions()
