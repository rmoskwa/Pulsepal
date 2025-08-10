"""
Semantic Query Router for PulsePal - Intelligent RAG Triggering

This module implements semantic query routing to intelligently determine when RAG
(Retrieval-Augmented Generation) searches are mandatory for Pulseq-specific queries.
Uses free, local embeddings (all-MiniLM-L6-v2) for classification.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import comprehensive function list from function_index
from .function_index import MATLAB_FUNCTIONS

logger = logging.getLogger(__name__)


class QueryRoute(Enum):
    """Routing decisions for query processing."""

    FORCE_RAG = "force_rag"  # Must search Pulseq knowledge
    NO_RAG = "no_rag"  # Skip RAG (pure physics)
    GEMINI_CHOICE = "gemini_choice"  # Let Gemini decide


@dataclass
class RoutingDecision:
    """Container for routing decision with metadata."""

    route: QueryRoute
    confidence: float
    reasoning: str
    search_hints: List[str] = field(default_factory=list)
    semantic_scores: Dict[str, float] = field(default_factory=dict)
    trigger_type: str = "unknown"  # 'semantic', 'keyword', 'code_detection', 'fallback'


class ThresholdManager:
    """Manages configurable thresholds for semantic similarity."""

    def __init__(self):
        """Initialize thresholds from environment or defaults."""
        self.thresholds = {
            "sequence_similarity": float(os.getenv("THRESHOLD_SEQUENCE", "0.7")),
            "implementation_similarity": float(
                os.getenv("THRESHOLD_IMPLEMENTATION", "0.65")
            ),
            "function_similarity": float(os.getenv("THRESHOLD_FUNCTION", "0.75")),
            "physics_similarity": float(os.getenv("THRESHOLD_PHYSICS", "0.7")),
        }
        logger.info(f"Initialized thresholds: {self.thresholds}")

    def update_threshold(self, category: str, new_value: float):
        """
        Update a threshold dynamically.

        Args:
            category: Threshold category to update
            new_value: New threshold value (0-1)
        """
        if category in self.thresholds:
            old_value = self.thresholds[category]
            self.thresholds[category] = new_value
            logger.info(f"Updated {category} threshold: {old_value} -> {new_value}")
        else:
            logger.warning(f"Unknown threshold category: {category}")

    def get_threshold(self, category: str) -> float:
        """Get threshold for a category."""
        return self.thresholds.get(category, 0.7)


class SemanticRouter:
    """Semantic router for intelligent query classification."""

    # Build comprehensive Pulseq function set from function_index (all 150 functions)
    PULSEQ_FUNCTIONS = set()

    # Add all direct call functions (like makeAdc, makeTrapezoid, etc.)
    PULSEQ_FUNCTIONS.update(MATLAB_FUNCTIONS.get("direct_calls", set()))

    # Add all class methods from Sequence, EventLibrary, SeqPlot, TransformFOV
    for class_name, methods in MATLAB_FUNCTIONS.get("class_methods", {}).items():
        PULSEQ_FUNCTIONS.update(methods)

    # Add eve.* functions (EventLibrary)
    PULSEQ_FUNCTIONS.update(MATLAB_FUNCTIONS.get("eve_functions", set()))

    # Add tra.* functions (TransformFOV)
    PULSEQ_FUNCTIONS.update(MATLAB_FUNCTIONS.get("tra_functions", set()))

    # Add mr.aux.* functions
    PULSEQ_FUNCTIONS.update(MATLAB_FUNCTIONS.get("mr_aux_functions", set()))

    # Add mr.aux.quat.* functions
    PULSEQ_FUNCTIONS.update(MATLAB_FUNCTIONS.get("mr_aux_quat_functions", set()))

    # All 150+ MATLAB Pulseq functions!
    # This provides complete coverage for function detection

    # Sequence type keywords
    SEQUENCE_TYPES = {
        "epi",
        "echo planar",
        "spin echo",
        "gradient echo",
        "gre",
        "tse",
        "turbo spin",
        "mprage",
        "ute",
        "haste",
        "trufisp",
        "press",
        "spiral",
        "diffusion",
        "dwi",
        "flair",
        "stir",
        "fiesta",
        "ssfp",
        "flash",
        "dess",
        "cest",
        "bold",
    }

    # Implementation keywords
    IMPLEMENTATION_KEYWORDS = {
        "implement",
        "create",
        "build",
        "develop",
        "write",
        "program",
        "code",
        "example",
        "sample",
        "template",
        "how to",
        "how do i",
        "show me",
        "demonstrate",
        "tutorial",
        "guide",
    }

    # Pure physics keywords
    PHYSICS_KEYWORDS = {
        "t1",
        "t2",
        "relaxation",
        "magnetization",
        "precession",
        "resonance",
        "larmor",
        "flip angle",
        "ernst angle",
        "signal equation",
        "bloch equation",
        "fourier",
        "shimming",
        "field homogeneity",
        "susceptibility",
        "chemical shift",
        "j-coupling",
        "dipolar coupling",
        "noe",
        "exchange",
    }

    def __init__(self):
        """Initialize the semantic router with embeddings model."""
        self.threshold_manager = ThresholdManager()
        self.encoder = None
        self._initialize_encoder()
        self._load_concept_embeddings()
        logger.info("Semantic router initialized successfully")

    def _initialize_encoder(self):
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            import platform
            from pathlib import Path

            # Use all-MiniLM-L6-v2 for efficient, free embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            # Determine cache directory based on environment
            if os.getenv("TRANSFORMERS_CACHE"):
                # Use explicit env var if set
                cache_dir = os.getenv("TRANSFORMERS_CACHE")
            elif platform.system() == "Windows":
                # Windows local development - use user's home directory
                cache_dir = str(Path.home() / ".cache" / "huggingface")
            else:
                # Linux/Railway deployment - use app directory
                cache_dir = "/app/.cache/huggingface"
            
            # Ensure cache directory exists
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading embedding model: {model_name}")
            logger.info(f"Using cache directory: {cache_dir}")
            start_time = time.time()

            self.encoder = SentenceTransformer(model_name, cache_folder=cache_dir)

            elapsed = time.time() - start_time
            logger.info(f"Embedding model loaded in {elapsed:.2f}s")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {e}")
            raise

    def _load_concept_embeddings(self):
        """Pre-compute embeddings for concept clusters."""
        if not self.encoder:
            logger.warning("Encoder not initialized, skipping concept embeddings")
            return

        # Sequence example concepts
        self.sequence_concepts = [
            "EPI sequence example implementation code",
            "spin echo sequence MATLAB code example",
            "gradient echo GRE sequence example implementation",
            "MPRAGE sequence implementation code",
            "turbo spin echo TSE sequence example",
            "diffusion weighted imaging DWI sequence",
            "FLASH sequence implementation example",
            "balanced SSFP TrueFISP sequence code",
            "spiral imaging sequence implementation",
            "HASTE sequence example code",
            "show me sequence example code implementation",
            "demonstrate MRI pulse sequence",
        ]

        # Implementation/how-to concepts
        self.implementation_concepts = [
            "how to implement MRI pulse sequence",
            "create pulse sequence code program",
            "build MRI sequence implementation",
            "develop scanner sequence programming",
            "write Pulseq sequence code",
            "program MRI pulse sequence",
            "code example for pulse sequence",
            "implementation guide tutorial",
            "step by step sequence creation",
            "sequence programming tutorial",
        ]

        # Pure physics concepts
        self.physics_concepts = [
            "T1 T2 relaxation time physics theory",
            "MRI physics principles fundamentals",
            "k-space theory mathematics",
            "Bloch equations physics",
            "magnetization vector dynamics",
            "RF pulse flip angle calculation",
            "signal equation derivation",
            "contrast mechanisms physics",
            "field homogeneity shimming",
            "chemical shift artifact physics",
        ]

        # Pre-compute embeddings
        logger.info("Pre-computing concept embeddings...")
        self.sequence_embeddings = self.encoder.encode(self.sequence_concepts)
        self.implementation_embeddings = self.encoder.encode(
            self.implementation_concepts
        )
        self.physics_embeddings = self.encoder.encode(self.physics_concepts)
        logger.info("Concept embeddings ready")

    def classify_query(self, query: str) -> RoutingDecision:
        """
        Classify a query to determine routing.

        Args:
            query: User query to classify

        Returns:
            RoutingDecision with route, confidence, and metadata
        """
        try:
            # Try multiple classification methods

            # 1. Check for explicit Pulseq functions
            function_check = self._check_pulseq_functions(query)
            if function_check:
                return function_check

            # 2. Check for code snippets
            code_check = self._check_code_presence(query)
            if code_check:
                return code_check

            # 3. Semantic classification
            semantic_result = self._semantic_classify(query)
            if semantic_result.confidence > 0.8:
                return semantic_result

            # 4. Keyword-based fallback
            keyword_result = self._keyword_classify(query)
            if keyword_result:
                return keyword_result

            # 5. Return semantic result even with lower confidence
            return semantic_result

        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            # Conservative fallback: assume RAG needed
            return RoutingDecision(
                route=QueryRoute.FORCE_RAG,
                confidence=0.5,
                reasoning="Classification failed - defaulting to RAG search for safety",
                trigger_type="fallback",
            )

    def _check_pulseq_functions(self, query: str) -> Optional[RoutingDecision]:
        """Check for explicit Pulseq function mentions."""
        query_lower = query.lower()

        # Check for function patterns
        patterns = [
            r"\b(mr|seq|opt|eve|tra)\.\w+",  # Namespace patterns
            r"\bmake[A-Z]\w+",  # makeXxx patterns
            r"\bcalc[A-Z]\w+",  # calcXxx patterns
        ]

        for pattern in patterns:
            if re.search(pattern, query):
                return RoutingDecision(
                    route=QueryRoute.FORCE_RAG,
                    confidence=1.0,
                    reasoning="Explicit Pulseq function pattern detected",
                    trigger_type="keyword",
                    search_hints=self._extract_function_names(query),
                )

        # Check for known function names
        for func in self.PULSEQ_FUNCTIONS:
            if func.lower() in query_lower:
                return RoutingDecision(
                    route=QueryRoute.FORCE_RAG,
                    confidence=1.0,
                    reasoning=f"Pulseq function '{func}' mentioned",
                    trigger_type="keyword",
                    search_hints=[func],
                )

        return None

    def _check_code_presence(self, query: str) -> Optional[RoutingDecision]:
        """Check if query contains code snippets."""
        code_indicators = [
            "```",  # Code blocks
            "def ",  # Python function
            "function ",  # MATLAB function
            "= mr.",  # Pulseq assignment
            "= seq.",  # Sequence operations
            ".addBlock(",  # Method calls
            ";",  # Statement terminator
            "import pypulseq",  # Python import
        ]

        for indicator in code_indicators:
            if indicator in query:
                return RoutingDecision(
                    route=QueryRoute.FORCE_RAG,
                    confidence=0.95,
                    reasoning="Code snippet detected in query",
                    trigger_type="code_detection",
                )

        return None

    def _semantic_classify(self, query: str) -> RoutingDecision:
        """Perform semantic classification using embeddings."""
        if not self.encoder:
            return RoutingDecision(
                route=QueryRoute.GEMINI_CHOICE,
                confidence=0.5,
                reasoning="Encoder not available",
                trigger_type="fallback",
            )

        # Encode the query
        query_embedding = self.encoder.encode([query])[0]

        # Calculate similarities
        sequence_sim = self._calculate_max_similarity(
            query_embedding, self.sequence_embeddings
        )
        implementation_sim = self._calculate_max_similarity(
            query_embedding, self.implementation_embeddings
        )
        physics_sim = self._calculate_max_similarity(
            query_embedding, self.physics_embeddings
        )

        semantic_scores = {
            "sequence": float(sequence_sim),
            "implementation": float(implementation_sim),
            "physics": float(physics_sim),
        }

        # Decision logic based on similarities
        seq_threshold = self.threshold_manager.get_threshold("sequence_similarity")
        impl_threshold = self.threshold_manager.get_threshold(
            "implementation_similarity"
        )
        phys_threshold = self.threshold_manager.get_threshold("physics_similarity")

        # Strong sequence or implementation match -> FORCE_RAG
        if sequence_sim > seq_threshold or implementation_sim > impl_threshold:
            return RoutingDecision(
                route=QueryRoute.FORCE_RAG,
                confidence=max(sequence_sim, implementation_sim),
                reasoning=f"Semantic match: {'sequence example' if sequence_sim > implementation_sim else 'implementation question'}",
                trigger_type="semantic",
                semantic_scores=semantic_scores,
                search_hints=self._generate_search_hints(query),
            )

        # Strong physics match with weak Pulseq match -> NO_RAG
        if physics_sim > phys_threshold and max(sequence_sim, implementation_sim) < 0.5:
            return RoutingDecision(
                route=QueryRoute.NO_RAG,
                confidence=physics_sim,
                reasoning="Pure physics question detected",
                trigger_type="semantic",
                semantic_scores=semantic_scores,
            )

        # Ambiguous case -> GEMINI_CHOICE
        return RoutingDecision(
            route=QueryRoute.GEMINI_CHOICE,
            confidence=0.6,
            reasoning="Ambiguous query - letting Gemini decide",
            trigger_type="semantic",
            semantic_scores=semantic_scores,
        )

    def _keyword_classify(self, query: str) -> Optional[RoutingDecision]:
        """Fallback keyword-based classification."""
        query_lower = query.lower()

        # Check for sequence type mentions
        for seq_type in self.SEQUENCE_TYPES:
            if seq_type in query_lower:
                # Check if it's asking for an example/implementation
                for impl_keyword in {"example", "show", "code", "implement", "create"}:
                    if impl_keyword in query_lower:
                        return RoutingDecision(
                            route=QueryRoute.FORCE_RAG,
                            confidence=0.85,
                            reasoning=f"Sequence example request: {seq_type}",
                            trigger_type="keyword",
                            search_hints=[seq_type],
                        )

        # Check for implementation questions
        impl_count = sum(1 for kw in self.IMPLEMENTATION_KEYWORDS if kw in query_lower)
        if impl_count >= 2:  # Multiple implementation keywords
            return RoutingDecision(
                route=QueryRoute.FORCE_RAG,
                confidence=0.8,
                reasoning="Implementation question detected",
                trigger_type="keyword",
            )

        # Check for pure physics
        physics_count = sum(1 for kw in self.PHYSICS_KEYWORDS if kw in query_lower)
        pulseq_indicators = {"pulseq", "pypulseq", "sequence", "mr.", "seq."}
        has_pulseq = any(ind in query_lower for ind in pulseq_indicators)

        if physics_count >= 2 and not has_pulseq:
            return RoutingDecision(
                route=QueryRoute.NO_RAG,
                confidence=0.75,
                reasoning="Pure physics question (no Pulseq context)",
                trigger_type="keyword",
            )

        return None

    def _calculate_max_similarity(
        self, query_embedding: np.ndarray, concept_embeddings: np.ndarray
    ) -> float:
        """Calculate maximum cosine similarity between query and concepts."""
        if len(concept_embeddings) == 0:
            return 0.0

        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        concept_norms = concept_embeddings / np.linalg.norm(
            concept_embeddings, axis=1, keepdims=True
        )

        # Calculate cosine similarities
        similarities = np.dot(concept_norms, query_norm)

        return float(np.max(similarities))

    def _extract_function_names(self, query: str) -> List[str]:
        """Extract potential function names from query."""
        functions = []

        # Extract namespace.function patterns
        patterns = re.findall(r"((?:mr|seq|opt|eve|tra)\.[\w]+)", query)
        functions.extend(patterns)

        # Extract makeXxx and calcXxx patterns
        patterns = re.findall(r"\b(make[A-Z]\w+|calc[A-Z]\w+)\b", query)
        functions.extend(patterns)

        return list(set(functions))  # Remove duplicates

    def _generate_search_hints(self, query: str) -> List[str]:
        """Generate search hints based on query content."""
        hints = []
        query_lower = query.lower()

        # Add sequence types mentioned
        for seq_type in self.SEQUENCE_TYPES:
            if seq_type in query_lower:
                hints.append(seq_type)

        # Add function names
        hints.extend(self._extract_function_names(query))

        # Limit hints
        return hints[:5]

    def log_routing_decision(
        self,
        session_id: str,
        query: str,
        decision: RoutingDecision,
        conversation_logger=None,
    ):
        """
        Log routing decision for analysis.

        Args:
            session_id: Current session ID
            query: Original query
            decision: Routing decision made
            conversation_logger: Optional conversation logger instance
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": query[:200],  # Truncate long queries
            "route": decision.route.value,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "semantic_scores": decision.semantic_scores,
            "triggered_by": decision.trigger_type,
            "search_hints": decision.search_hints,
        }

        # Log to standard logger
        logger.info(
            f"Routing decision: {decision.route.value} "
            f"(confidence: {decision.confidence:.2f}, trigger: {decision.trigger_type})"
        )

        # Log to conversation logger if available
        if conversation_logger:
            try:
                conversation_logger.log_routing_event(session_id, log_entry)
            except Exception as e:
                logger.warning(f"Failed to log to conversation logger: {e}")


def initialize_semantic_router() -> SemanticRouter:
    """
    Initialize the semantic router at application startup.

    Returns:
        Initialized SemanticRouter instance
    """
    logger.info("Initializing semantic router at startup...")
    start_time = time.time()

    try:
        router = SemanticRouter()
        elapsed = time.time() - start_time
        logger.info(f"Semantic router ready ({elapsed:.2f}s)")
        return router
    except Exception as e:
        logger.error(f"Failed to initialize semantic router: {e}")
        raise
