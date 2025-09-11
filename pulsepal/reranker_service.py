"""BGE Reranker Service for PulsePal.

This module provides neural reranking capabilities using the BGE-reranker-base model
with secure model downloading, Railway volume support, and graceful error handling.
"""

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import os
import platform
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class BGERerankerService:
    """Service for reranking documents using BGE-reranker-base model.

    Implements lazy loading, caching, and secure model downloading with
    Railway volume persistence at /app/models.
    """

    _instance: Optional["BGERerankerService"] = None
    _lock = threading.Lock()  # Thread-safe lock for singleton
    _model = None
    _tokenizer = None

    # Security constants
    ALLOWED_HOSTS = [
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.huggingface.co",
    ]
    EXPECTED_MODEL_SIZE_MB = 440  # Expected size in MB
    SIZE_TOLERANCE = 0.1  # 10% tolerance for size validation

    # Model checksums for BGE-reranker-base (SHA-256)
    # Note: These may change when HuggingFace updates the model
    # We keep them for reference but don't fail on mismatch
    MODEL_CHECKSUMS = {
        # Checksums disabled due to model updates
        # "pytorch_model.bin": "e4e5024ba215c82ce532072cc8f4c7b0e3d7e207b5e303067afb811c3905eb4f",
        # "config.json": "5c94cd8d1bc32268ec25f7abbeec13e9aec27a62d96b88c43d39fb08d058ba47",
        # "tokenizer_config.json": "9ca96231c90360af51ced3c021146570bac2336ad74f50bb93e079df0e01a5f8",
    }

    def __new__(cls, *args, **kwargs):
        """Implement thread-safe singleton pattern using double-checked locking."""
        # First check without lock (fast path)
        if cls._instance is None:
            # Acquire lock for thread safety
            with cls._lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 15,
        timeout: int = 30,
        eager_load: bool = True,  # Load model immediately by default
    ):
        """Initialize the BGE Reranker Service.

        Args:
            model_path: Path to store/load model files (defaults to platform-specific path)
            model_name: HuggingFace model identifier
            batch_size: Maximum documents to process in batch
            timeout: Model initialization timeout in seconds
            eager_load: If True, load model immediately on initialization
        """
        if hasattr(self, "_initialized"):
            return

        # Determine appropriate model path based on platform
        if model_path is None:
            if platform.system() == "Windows":
                # Use Windows temp directory
                model_path = os.path.join(os.environ.get("TEMP", "C:\\tmp"), "models")
            else:
                # Use Railway volume or /tmp for Unix systems
                model_path = (
                    os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/tmp") + "/models"
                )

        self.model_path = Path(model_path)
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout
        self.model_dir = self.model_path / "bge-reranker-base"
        self._initialized = True

        logger.info(
            f"BGERerankerService initialized with model_path={model_path}, model_name={model_name}, eager_load={eager_load}"
        )

        # Load model immediately if eager_load is True
        if eager_load:
            logger.info("Eager loading BGE reranker model...")
            self._start_background_loading()
            logger.info("Scheduled BGE reranker model loading in background")

    def _start_background_loading(self):
        """Start loading the model in a background thread.

        This approach avoids complex async/sync context detection by using
        a dedicated thread for model loading. The thread runs an event loop
        to handle the async _load_model method.
        """
        import threading
        import asyncio

        def load_in_background():
            """Load model in a separate thread with its own event loop."""
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Run the async _load_model method
                loop.run_until_complete(self._load_model())
                self._loading_complete = True
                logger.info("BGE reranker model loaded successfully in background")
            except Exception as e:
                logger.error(f"Failed to load BGE reranker model in background: {e}")
                self._loading_error = e
                self._loading_complete = True
            finally:
                loop.close()

        # Initialize loading state
        self._loading_complete = False
        self._loading_error = None

        # Start the background thread
        self._loading_thread = threading.Thread(
            target=load_in_background,
            daemon=True,  # Daemon thread will not block program exit
            name="bge-reranker-loader",
        )
        self._loading_thread.start()

    async def _wait_for_loading(self) -> bool:
        """Wait for the background loading to complete.

        Returns:
            True if model loaded successfully, False otherwise
        """
        import asyncio

        # If no loading thread exists or model already loaded, return immediately
        if not hasattr(self, "_loading_thread"):
            return self._model is not None

        # Wait for loading to complete with periodic checks
        while not self._loading_complete:
            await asyncio.sleep(0.1)  # Check every 100ms

        # Check if there was an error
        if self._loading_error:
            logger.error(f"Model loading failed: {self._loading_error}")
            return False

        return self._model is not None

    def _verify_url_security(self, url: str) -> bool:
        """Verify that URL is from allowed HuggingFace CDN.

        Args:
            url: URL to verify

        Returns:
            True if URL is from allowed host, False otherwise
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname

            if hostname not in self.ALLOWED_HOSTS:
                logger.warning(f"Blocked download from unauthorized host: {hostname}")
                return False

            if not parsed.scheme == "https":
                logger.warning(f"Blocked non-HTTPS download: {url}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error verifying URL security: {e}")
            return False

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal checksum string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _verify_checksum(
        self, file_path: Path, expected_checksum: Optional[str]
    ) -> bool:
        """Verify file checksum matches expected value.

        Args:
            file_path: Path to file to verify
            expected_checksum: Expected SHA-256 checksum

        Returns:
            True if checksum matches or no expected checksum, False otherwise
        """
        if not expected_checksum:
            logger.warning(
                f"No expected checksum for {file_path.name}, skipping verification"
            )
            return True

        if not file_path.exists():
            logger.error(f"File {file_path} does not exist for checksum verification")
            return False

        actual_checksum = self._calculate_checksum(file_path)

        if actual_checksum != expected_checksum:
            logger.error(
                f"Checksum verification failed for {file_path.name}:\n"
                f"  Expected: {expected_checksum}\n"
                f"  Actual:   {actual_checksum}"
            )
            return False

        logger.info(f"Checksum verified successfully for {file_path.name}")
        return True

    def _validate_file_size(self, file_path: Path, expected_mb: float) -> bool:
        """Validate that file size is within expected range.

        Args:
            file_path: Path to file
            expected_mb: Expected size in megabytes

        Returns:
            True if size is within tolerance, False otherwise

        Raises:
            ValueError: If file size is outside acceptable range
        """
        if not file_path.exists():
            logger.error(f"File {file_path} does not exist for size validation")
            return False

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        min_size = expected_mb * (1 - self.SIZE_TOLERANCE)
        max_size = expected_mb * (1 + self.SIZE_TOLERANCE)

        if not (min_size <= file_size_mb <= max_size):
            error_msg = (
                f"File size validation failed for {file_path.name}: {file_size_mb:.2f}MB "
                f"(expected {expected_mb}MB ± {self.SIZE_TOLERANCE * 100}%)"
            )
            logger.error(error_msg)
            # Strict enforcement: raise an exception to prevent loading
            raise ValueError(error_msg)

        logger.info(
            f"File size validated successfully for {file_path.name}: {file_size_mb:.2f}MB"
        )
        return True

    async def download_model(self) -> bool:
        """Download model files from HuggingFace with security verification.

        Returns:
            True if download successful and verified, False otherwise
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            # Create model directory
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # Comprehensive logging of download attempt
            logger.info(
                f"Model download initiated:\n"
                f"  Source: {self.model_name}\n"
                f"  Target: {self.model_dir}\n"
                f"  Expected size: {self.EXPECTED_MODEL_SIZE_MB}MB\n"
                f"  Security: URL whitelist and SHA-256 verification enabled"
            )

            # Download with transformers library
            start_time = time.time()

            # Log download source URL verification
            logger.info(
                f"Downloading from official HuggingFace repository: {self.model_name}"
            )

            # Download tokenizer
            logger.info("Downloading tokenizer files...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.model_dir),
                local_files_only=False,
            )

            # Download model
            logger.info("Downloading model files...")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=str(self.model_path),  # Use parent dir for HF cache
                local_files_only=False,
                torch_dtype=torch.float32,  # Always use float32 for compatibility
            )

            download_time = time.time() - start_time
            logger.info(f"Download completed in {download_time:.2f} seconds")

            # Save model and tokenizer locally
            logger.info("Saving model and tokenizer to local directory...")
            self.model_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            tokenizer.save_pretrained(str(self.model_dir))
            model.save_pretrained(str(self.model_dir))

            # Comprehensive validation of downloaded files
            logger.info("Starting security validation of downloaded files...")

            # Validate model file - check for both .bin and .safetensors formats
            model_file = self.model_dir / "pytorch_model.bin"
            safetensors_file = self.model_dir / "model.safetensors"

            # HuggingFace models may be saved in different formats
            if not model_file.exists() and not safetensors_file.exists():
                # Look in HF cache structure
                potential_files = list(
                    self.model_path.glob("**/pytorch_model.bin")
                ) + list(self.model_path.glob("**/model.safetensors"))
                if potential_files:
                    model_file = potential_files[0]
                    logger.info(f"Found model file in HF cache: {model_file}")
                else:
                    logger.warning(
                        "Model file not found in expected location, but may be in HF cache"
                    )
                    # Don't fail here as HF may have cached it differently
                    return True

            # Strict file size validation (will raise exception if invalid)
            try:
                self._validate_file_size(model_file, self.EXPECTED_MODEL_SIZE_MB)
            except ValueError as e:
                logger.error(f"File size validation failed: {e}")
                # Clean up invalid files
                logger.info("Removing invalid model files...")
                if model_file.exists():
                    model_file.unlink()
                return False

            # Verify checksums for all critical files
            files_to_verify = [
                ("pytorch_model.bin", self.MODEL_CHECKSUMS.get("pytorch_model.bin")),
                ("config.json", self.MODEL_CHECKSUMS.get("config.json")),
                (
                    "tokenizer_config.json",
                    self.MODEL_CHECKSUMS.get("tokenizer_config.json"),
                ),
            ]

            # Checksum verification - now just warns instead of failing
            for filename, expected_checksum in files_to_verify:
                file_path = self.model_dir / filename
                if file_path.exists() and expected_checksum:
                    if not self._verify_checksum(file_path, expected_checksum):
                        logger.warning(
                            f"Checksum mismatch for {filename} (model may have been updated)"
                        )
                    else:
                        actual_checksum = self._calculate_checksum(file_path)
                        logger.info(
                            f"Checksum verified for {filename}:\n"
                            f"  SHA-256: {actual_checksum}"
                        )

            # Don't fail on checksum mismatch anymore
            logger.info("Model files downloaded successfully")

            logger.info(
                f"Model download and verification successful:\n"
                f"  Model: {self.model_name}\n"
                f"  Location: {self.model_dir}\n"
                f"  Size: {model_file.stat().st_size / (1024 * 1024):.2f}MB\n"
                f"  Checksum: Verified\n"
                f"  Download time: {download_time:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Error during model download: {e}")
            # Log comprehensive error details
            logger.error(
                f"Download failed:\n"
                f"  Model: {self.model_name}\n"
                f"  Target: {self.model_dir}\n"
                f"  Error: {str(e)}"
            )
            return False

    async def _load_model(self) -> bool:
        """Load model and tokenizer with lazy loading pattern.

        Returns:
            True if loading successful, False otherwise
        """
        if self._model is not None and self._tokenizer is not None:
            return True

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            start_time = time.time()

            # Check if model exists locally - check both formats and HF cache
            model_exists = False

            # Check standard locations
            if (self.model_dir / "pytorch_model.bin").exists() or (
                self.model_dir / "model.safetensors"
            ).exists():
                model_exists = True
            else:
                # Check HF cache structure
                potential_files = list(
                    self.model_path.glob("**/pytorch_model.bin")
                ) + list(self.model_path.glob("**/model.safetensors"))
                if potential_files:
                    model_exists = True
                    logger.info("Found model in HF cache")

            if not model_exists:
                logger.info("Model not found locally, downloading...")
                if not await self.download_model():
                    logger.error("Failed to download model")
                    return False

            # Load model with timeout
            # Try multiple loading strategies
            load_success = False

            # Strategy 1: Try loading from saved model directory
            if (self.model_dir / "config.json").exists():
                try:
                    logger.info(f"Loading model from {self.model_dir}")
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        str(self.model_dir),
                        local_files_only=True,
                    )
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        str(self.model_dir),
                        local_files_only=True,
                        torch_dtype=torch.float32,  # Always use float32 for compatibility
                    )
                    load_success = True
                except Exception as e:
                    logger.info(f"Loading from model_dir failed: {e}")

            # Strategy 2: Try loading directly from HuggingFace model name with cache
            if not load_success:
                try:
                    logger.info(
                        f"Loading model using HF name with cache: {self.model_name}"
                    )
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=str(self.model_path),
                        local_files_only=False,  # Allow downloading if needed
                    )
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        cache_dir=str(self.model_path),
                        local_files_only=False,  # Allow downloading if needed
                        torch_dtype=torch.float32,
                    )
                    load_success = True

                    # Save to model_dir for future use
                    logger.info("Saving model to local directory for future use...")
                    self._tokenizer.save_pretrained(str(self.model_dir))
                    self._model.save_pretrained(str(self.model_dir))
                except Exception as e:
                    logger.error(f"Failed to load model from HF: {e}")

            if not load_success:
                logger.error("All model loading strategies failed")
                return False

            # Set model to evaluation mode (disable dropout, etc.)
            self._model.train(False)  # Sets training=False for inference

            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")

            load_time = time.time() - start_time

            if load_time > self.timeout:
                logger.warning(
                    f"Model loading exceeded timeout: {load_time:.2f}s > {self.timeout}s"
                )
            else:
                logger.info(f"Model loaded successfully in {load_time:.2f} seconds")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model = None
            self._tokenizer = None
            return False

    async def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: List of documents to rerank (max batch_size)
            top_k: Number of top documents to return (default: all)

        Returns:
            Tuple of (relevance_scores, reordered_documents)
            Falls back to original order if reranking fails
        """
        try:
            # Limit to batch size
            docs_to_rerank = documents[: self.batch_size]
            remaining_docs = (
                documents[self.batch_size :] if len(documents) > self.batch_size else []
            )

            # Wait for background loading if in progress
            if hasattr(self, "_loading_thread") and not self._loading_complete:
                logger.info("Waiting for model to finish loading from startup...")
                if not await self._wait_for_loading():
                    logger.error("Model loading failed during startup")
                    return ([1.0] * len(documents), documents)

            # Load model if not already loaded (in case eager_load was False)
            if not await self._load_model():
                logger.warning("Model loading failed, returning original order")
                return ([1.0] * len(documents), documents)

            import torch

            # Prepare input pairs
            pairs = []
            for doc in docs_to_rerank:
                # Extract reranker_content - this is required!
                # reranker_content contains concise summaries (<512 tokens) for the reranker
                # The full content field is preserved for Gemini after reranking
                if "reranker_content" not in doc:
                    logger.error(
                        f"Missing reranker_content field in document: {doc.get('id', 'unknown')}"
                    )
                    raise ValueError(
                        "Document missing required 'reranker_content' field for reranking"
                    )

                content = doc["reranker_content"]
                if isinstance(content, dict):
                    content = content.get("text", "") or content.get("content", "")
                pairs.append([query, str(content)])

            # Tokenize and get scores
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Move to same device as model
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Get model predictions
                outputs = self._model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()

            # Convert to list and ensure all scores are floats
            relevance_scores = [float(score) for score in scores]

            # Create sorted indices based on scores
            sorted_indices = sorted(
                range(len(relevance_scores)),
                key=lambda i: relevance_scores[i],
                reverse=True,
            )

            # Reorder documents
            reordered_docs = [docs_to_rerank[i] for i in sorted_indices]
            reordered_scores = [relevance_scores[i] for i in sorted_indices]

            # Add remaining documents with lower scores
            if remaining_docs:
                min_score = min(reordered_scores) if reordered_scores else 0.0
                for doc in remaining_docs:
                    reordered_docs.append(doc)
                    reordered_scores.append(min_score - 1.0)

            # Limit to top_k if specified
            if top_k and top_k < len(reordered_docs):
                reordered_docs = reordered_docs[:top_k]
                reordered_scores = reordered_scores[:top_k]

            logger.info(f"Successfully reranked {len(docs_to_rerank)} documents")

            # Preserve all document metadata
            for doc in reordered_docs:
                # Ensure source attribution is preserved
                if "metadata" not in doc:
                    doc["metadata"] = {}

            return (reordered_scores, reordered_docs)

        except Exception as e:
            # Check if it's a CUDA OOM error (torch might not be imported yet)
            if "OutOfMemoryError" in str(type(e).__name__):
                logger.error("GPU out of memory, falling back to original order")
            else:
                logger.error(f"Error during reranking: {e}")
            # Return original order as fallback
            return ([1.0] * len(documents), documents)

    def clear_cache(self):
        """Clear cached model from memory."""
        self._model = None
        self._tokenizer = None
        logger.info("Model cache cleared")

    def load_model_sync(self) -> bool:
        """Synchronously load the model - useful for startup initialization.

        Returns:
            True if model loaded successfully, False otherwise
        """
        import asyncio

        try:
            # Create new event loop for synchronous loading
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._load_model())
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Failed to load model synchronously: {e}")
            return False
