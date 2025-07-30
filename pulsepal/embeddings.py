"""
Local embedding service for Pulsepal using BAAI-bge-large-en-v1.5 model.

This module provides embedding generation capabilities for RAG search functionality,
using a locally stored BGE model for vector creation.
"""

import os
import gc
import asyncio
from typing import List, Optional, Any
import time
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Lazy imports to avoid slow startup
_torch = None
_AutoModel = None
_AutoTokenizer = None

def _lazy_imports():
    """Lazy load heavy dependencies only when needed."""
    global _torch, _AutoModel, _AutoTokenizer
    if _torch is None:
        logger.info("Loading torch and transformers libraries...")
        import torch
        from transformers import AutoModel, AutoTokenizer
        _torch = torch
        _AutoModel = AutoModel
        _AutoTokenizer = AutoTokenizer
        logger.info("Libraries loaded successfully")
    return _torch, _AutoModel, _AutoTokenizer


class LocalEmbeddingService:
    """
    Service for generating embeddings using the local BAAI-bge-large-en-v1.5 model.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the local embedding service.

        Args:
            model_path: Path to the local model directory. If None, uses environment variable.
        """
        # Don't load dependencies in init - wait until model loading
        self.model = None
        self.tokenizer = None
        self.device = None

        # Determine model path
        if model_path is None:
            model_path = os.getenv("BGE_MODEL_PATH")
            
            if model_path is None:
                # Default to the known location on this system
                model_path = "/mnt/c/Users/Robert Moskwa/huggingface_models/hub/models--BAAI--bge-large-en-v1.5/snapshots"
                logger.info(f"Using default BGE model path: {model_path}")

        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the BGE model and tokenizer from local path only."""
        try:
            # Load dependencies if not already loaded
            torch, AutoModel, AutoTokenizer = _lazy_imports()
            
            # Set device after torch is loaded
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info(f"Loading BGE model from local path: {self.model_path}")

            # If path points to snapshots directory, find the latest snapshot
            actual_model_path = self.model_path
            if os.path.exists(self.model_path) and self.model_path.endswith("snapshots"):
                snapshots = [
                    d
                    for d in os.listdir(self.model_path)
                    if os.path.isdir(os.path.join(self.model_path, d))
                ]
                if snapshots:
                    # Use the first (and likely only) snapshot
                    actual_model_path = os.path.join(self.model_path, snapshots[0])
                    logger.info(f"Found snapshot: {actual_model_path}")

            # Check if local path exists
            if not os.path.exists(actual_model_path):
                raise FileNotFoundError(
                    f"Local model path does not exist: {actual_model_path}"
                )

            # Load from local path only
            logger.info(f"Loading model from: {actual_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                actual_model_path, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                actual_model_path, local_files_only=True
            )

            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info(
                f"BGE model loaded successfully from local path on device: {self.device}"
            )

        except Exception as e:
            logger.error(f"Error loading BGE model from local path: {e}")
            raise RuntimeError(f"Failed to load BGE model from {self.model_path}: {e}")

    def _encode_texts(self, texts: List[str]) -> Any:
        """
        Encode texts into embeddings using the BGE model.

        Args:
            texts: List of texts to encode

        Returns:
            Tensor of embeddings
        """
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # BGE model max length
            return_tensors="pt",
        )

        # Move inputs to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Get torch module
        torch = _torch
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

            # Perform mean pooling on the token embeddings
            # Mask padding tokens for proper mean pooling
            attention_mask = encoded_input["attention_mask"]
            token_embeddings = model_output.last_hidden_state

            # Expand attention mask to match token embeddings dimensions
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )

            # Sum embeddings and divide by actual length (excluding padding)
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # Normalize embeddings for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings

    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.

        Args:
            text: Text to create an embedding for

        Returns:
            List of floats representing the embedding
        """
        try:
            if not text or not text.strip():
                # Return zero embedding for empty text
                return [0.0] * 1024  # BGE-large embedding dimension

            embeddings = self._encode_texts([text])
            return embeddings[0].cpu().tolist()

        except Exception as e:
            logger.error(f"Error creating single embedding: {e}")
            # Return zero embedding as fallback
            return [0.0] * 1024

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts in a batch.

        Args:
            texts: List of texts to create embeddings for

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []

        max_retries = 3
        retry_delay = 1.0

        for retry in range(max_retries):
            try:
                # Filter out empty texts and keep track of original indices
                non_empty_texts = []
                text_indices = []

                for i, text in enumerate(texts):
                    if text and text.strip():
                        non_empty_texts.append(text)
                        text_indices.append(i)

                if not non_empty_texts:
                    # All texts are empty, return zero embeddings
                    return [[0.0] * 1024 for _ in texts]

                # Process in smaller batches to avoid memory issues
                batch_size = 16
                all_embeddings = []

                for i in range(0, len(non_empty_texts), batch_size):
                    batch_texts = non_empty_texts[i : i + batch_size]
                    batch_embeddings = self._encode_texts(batch_texts)
                    all_embeddings.extend(batch_embeddings.cpu().tolist())

                # Reconstruct full results array with zero embeddings for empty texts
                results = []
                non_empty_idx = 0

                for i, text in enumerate(texts):
                    if text and text.strip():
                        results.append(all_embeddings[non_empty_idx])
                        non_empty_idx += 1
                    else:
                        results.append([0.0] * 1024)

                return results

            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(
                        f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(
                        f"Failed to create batch embeddings after {max_retries} attempts: {e}"
                    )
                    # Try creating embeddings one by one as fallback
                    logger.info("Attempting to create embeddings individually...")
                    embeddings = []
                    successful_count = 0

                    for text in texts:
                        try:
                            embedding = self.create_embedding(text)
                            embeddings.append(embedding)
                            successful_count += 1
                        except Exception as individual_error:
                            logger.error(
                                f"Failed to create individual embedding: {individual_error}"
                            )
                            # Add zero embedding as fallback
                            embeddings.append([0.0] * 1024)

                    logger.info(
                        f"Successfully created {successful_count}/{len(texts)} embeddings individually"
                    )
                    return embeddings

    async def create_embeddings_batch_async(
        self, 
        texts: List[str], 
        batch_size: int = 16,
        max_workers: int = 2
    ) -> List[List[float]]:
        """
        Async batch embedding generation with memory management and parallel processing.
        
        Args:
            texts: List of texts to create embeddings for
            batch_size: Size of batches for processing
            max_workers: Maximum number of worker threads
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []

        try:
            loop = asyncio.get_event_loop()
            
            # Process in chunks to manage memory
            all_embeddings = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                
                # Create tasks for batch processing
                tasks = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    task = loop.run_in_executor(
                        executor, 
                        self.create_embeddings_batch, 
                        batch
                    )
                    tasks.append(task)
                
                # Process batches and collect results
                for i, task in enumerate(asyncio.as_completed(tasks)):
                    batch_embeddings = await task
                    all_embeddings.extend(batch_embeddings)
                    
                    # Memory management - garbage collect every few batches
                    if i % 4 == 0:
                        gc.collect()
                        
            logger.info(f"Async batch processing completed: {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Async batch embedding failed: {e}")
            # Fallback to synchronous processing
            return self.create_embeddings_batch(texts)

    def create_embeddings_optimized_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        enable_memory_optimization: bool = True
    ) -> List[List[float]]:
        """
        Optimized batch processing with adaptive batching and memory management.
        
        Args:
            texts: List of texts to create embeddings for
            batch_size: Initial batch size (will be adapted based on performance)
            enable_memory_optimization: Whether to enable memory optimization
            
        Returns:
            List of embeddings with optimized processing
        """
        if not texts:
            return []

        # Adaptive batch sizing based on text length
        avg_length = sum(len(text) for text in texts) / len(texts)
        if avg_length > 1000:  # Long texts
            batch_size = max(8, batch_size // 2)
        elif avg_length < 200:  # Short texts
            batch_size = min(64, batch_size * 2)

        logger.debug(f"Using optimized batch size: {batch_size} for avg text length: {avg_length}")

        try:
            results = []
            processed_count = 0
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Process batch
                batch_start = time.time()
                batch_embeddings = self._encode_texts(batch).cpu().tolist()
                batch_time = time.time() - batch_start
                
                results.extend(batch_embeddings)
                processed_count += len(batch)
                
                # Memory optimization
                if enable_memory_optimization and i % (batch_size * 4) == 0:
                    gc.collect()
                    # Clear GPU cache if using CUDA
                    torch = _torch
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Performance monitoring
                if processed_count % (batch_size * 4) == 0:
                    rate = len(batch) / batch_time
                    logger.debug(f"Processed {processed_count}/{len(texts)} texts at {rate:.1f} texts/sec")
            
            logger.info(f"Optimized batch processing completed: {len(results)} embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Optimized batch processing failed: {e}")
            # Fallback to original batch processing
            return self.create_embeddings_batch(texts)


# Global instance
_embedding_service: Optional[LocalEmbeddingService] = None


def get_embedding_service() -> LocalEmbeddingService:
    """
    Get the global embedding service instance.

    Returns:
        LocalEmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = LocalEmbeddingService()
    return _embedding_service


def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using the local BGE model.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    service = get_embedding_service()
    return service.create_embedding(text)


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a batch using the local BGE model.

    Args:
        texts: List of texts to create embeddings for

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    service = get_embedding_service()
    return service.create_embeddings_batch(texts)


async def create_embeddings_batch_async(
    texts: List[str], 
    batch_size: int = 16,
    max_workers: int = 2
) -> List[List[float]]:
    """
    Create embeddings for multiple texts asynchronously with optimized processing.

    Args:
        texts: List of texts to create embeddings for
        batch_size: Size of batches for processing
        max_workers: Maximum number of worker threads

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    service = get_embedding_service()
    return await service.create_embeddings_batch_async(texts, batch_size, max_workers)


def create_embeddings_optimized_batch(
    texts: List[str], 
    batch_size: int = 32,
    enable_memory_optimization: bool = True
) -> List[List[float]]:
    """
    Create embeddings with optimized batch processing and memory management.

    Args:
        texts: List of texts to create embeddings for
        batch_size: Initial batch size (adaptive)
        enable_memory_optimization: Whether to enable memory optimization

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    service = get_embedding_service()
    return service.create_embeddings_optimized_batch(texts, batch_size, enable_memory_optimization)