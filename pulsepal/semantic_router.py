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
from difflib import get_close_matches
from enum import Enum
from typing import Dict, List, Optional

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
    detected_functions: List[Dict] = field(
        default_factory=list
    )  # For direct function lookup
    validation_errors: List[str] = field(
        default_factory=list
    )  # Namespace/function validation errors


class ThresholdManager:
    """Manages configurable thresholds for semantic similarity."""

    def __init__(self):
        """Initialize thresholds from environment or defaults."""
        self.thresholds = {
            "sequence_similarity": float(os.getenv("THRESHOLD_SEQUENCE", "0.7")),
            "implementation_similarity": float(
                os.getenv("THRESHOLD_IMPLEMENTATION", "0.65"),
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

    def __init__(self, lazy_load: bool = False):
        """Initialize the semantic router with embeddings model.

        Args:
            lazy_load: If True, defer model loading until first use (better for startup time)
        """
        self.threshold_manager = ThresholdManager()
        self.encoder = None
        self._embeddings_loaded = False

        if not lazy_load:
            self._initialize_encoder()
            self._load_concept_embeddings()
            logger.info("Semantic router initialized successfully")
        else:
            logger.info("Semantic router initialized (lazy loading enabled)")

    def _initialize_encoder(self):
        """Initialize the sentence transformer model."""
        try:
            import platform
            from pathlib import Path

            from sentence_transformers import SentenceTransformer

            # Use all-MiniLM-L6-v2 for efficient, free embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

            # Determine cache directory based on environment with proper permission handling
            cache_dir = None

            if os.getenv("TRANSFORMERS_CACHE"):
                # Use explicit env var if set
                cache_dir = os.getenv("TRANSFORMERS_CACHE")
            elif platform.system() == "Windows":
                # Windows local development - use user's home directory
                cache_dir = str(Path.home() / ".cache" / "huggingface")
            elif os.path.exists("/app") and os.access("/app", os.W_OK):
                # Railway deployment - use app directory only if writable
                cache_dir = "/app/.cache/huggingface"
            else:
                # WSL2 or other Linux - use home directory
                cache_dir = str(Path.home() / ".cache" / "huggingface")

            # Ensure cache directory exists with proper error handling
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = Path(cache_dir) / ".write_test"
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot write to cache directory {cache_dir}: {e}")
                # Fallback to temp directory
                import tempfile

                cache_dir = os.path.join(tempfile.gettempdir(), "huggingface_cache")
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                logger.info(f"Using fallback cache directory: {cache_dir}")

            logger.info(f"Loading embedding model: {model_name}")
            logger.info(f"Using cache directory: {cache_dir}")
            start_time = time.time()

            self.encoder = SentenceTransformer(model_name, cache_folder=cache_dir)

            elapsed = time.time() - start_time
            logger.info(f"Embedding model loaded in {elapsed:.2f}s")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. Install with: pip install sentence-transformers",
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
            self.implementation_concepts,
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
        """Check for explicit Pulseq function mentions with multi-level namespace support."""
        from .function_index import validate_namespace

        query_lower = query.lower()
        detected_functions = []
        validation_errors = []

        # Multi-level namespace patterns (most specific first)
        namespace_patterns = [
            (r"\b(mr\.aux\.quat)\.([\w]+)", "mr.aux.quat"),  # Three-level namespace
            (r"\b(mr\.aux)\.([\w]+)", "mr.aux"),  # Two-level namespace
            (r"\b(eve)\.([\w]+)", "eve"),  # EventLibrary
            (r"\b(tra)\.([\w]+)", "tra"),  # TransformFOV
            (r"\b(mr|seq)\.([\w]+)", None),  # Single-level (common)
        ]

        # Check namespace patterns
        for pattern, namespace_type in namespace_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                namespace = match.group(1)
                func_name = match.group(2)

                # Validate namespace is correct
                validation = validate_namespace(func_name, namespace)

                if validation["correct_form"]:
                    # Function exists (may have wrong namespace)
                    detected_functions.append(
                        {
                            "name": func_name,
                            "namespace": namespace,
                            "full_match": match.group(0),
                            "confidence": 1.0,
                            "type": "explicit_namespace",
                            "correct_form": validation["correct_form"],
                            "is_valid": validation["is_valid"],
                        }
                    )

                    if not validation["is_valid"]:
                        validation_errors.append(validation["error"])

        # Check for makeXxx and calcXxx patterns
        make_calc_patterns = [
            r"\bmake[A-Z]\w+",  # makeXxx patterns
            r"\bcalc[A-Z]\w+",  # calcXxx patterns
        ]

        for pattern in make_calc_patterns:
            for match in re.finditer(pattern, query):
                func_name = match.group(0)
                if func_name in self.PULSEQ_FUNCTIONS and not any(
                    d["name"].lower() == func_name.lower() for d in detected_functions
                ):
                    detected_functions.append(
                        {
                            "name": func_name,
                            "namespace": None,
                            "full_match": func_name,
                            "confidence": 1.0,
                            "type": "pattern_match",
                        }
                    )

        # If we found namespace/pattern matches, return early with high confidence
        if detected_functions:
            # Include validation errors in reasoning if any
            reasoning = f"Detected {len(detected_functions)} Pulseq function(s) with explicit patterns"
            if validation_errors:
                reasoning += f" (namespace issues: {'; '.join(validation_errors)})"

            return RoutingDecision(
                route=QueryRoute.FORCE_RAG,
                confidence=1.0,
                reasoning=reasoning,
                trigger_type="keyword",
                search_hints=[f["name"] for f in detected_functions],
                detected_functions=detected_functions,
                validation_errors=validation_errors,  # Pass validation errors
            )

        # Check for known function names with fuzzy matching
        for func in self.PULSEQ_FUNCTIONS:
            # Exact match (case-insensitive)
            if func.lower() in query_lower:
                if not any(
                    d["name"].lower() == func.lower() for d in detected_functions
                ):
                    # Validate the function
                    validation = validate_namespace(func, None)
                    if validation["correct_form"]:  # Function exists
                        detected_functions.append(
                            {
                                "name": func,
                                "namespace": None,
                                "full_match": func,
                                "confidence": 1.0,
                                "type": "exact_match",
                                "correct_form": validation["correct_form"],
                                "is_valid": validation["is_valid"],
                            }
                        )

            # Fuzzy match: Convert CamelCase to spaced words
            # EventLibrary -> "event library"
            spaced_func = self._camelcase_to_spaced(func).lower()
            if spaced_func in query_lower:
                if not any(
                    d["name"].lower() == func.lower() for d in detected_functions
                ):
                    detected_functions.append(
                        {
                            "name": func,
                            "namespace": None,
                            "full_match": spaced_func,
                            "confidence": 0.95,
                            "type": "fuzzy_camelcase",
                        }
                    )

            # Also check with underscores (common variation)
            # EventLibrary -> "event_library"
            underscored = self._camelcase_to_underscored(func).lower()
            if underscored in query_lower:
                if not any(
                    d["name"].lower() == func.lower() for d in detected_functions
                ):
                    detected_functions.append(
                        {
                            "name": func,
                            "namespace": None,
                            "full_match": underscored,
                            "confidence": 0.95,
                            "type": "fuzzy_underscore",
                        }
                    )

        # Check for misspellings using Levenshtein distance
        misspelling_result = self._check_for_misspellings(query)
        if misspelling_result:
            # Merge detected functions from misspelling with others if any
            if detected_functions:
                # Add misspelling detection to existing detections
                misspelling_result.detected_functions.extend(detected_functions)
            return misspelling_result

        # If we detected any functions through other methods, return them
        if detected_functions:
            # Find the highest confidence among detections
            max_confidence = max(f["confidence"] for f in detected_functions)
            return RoutingDecision(
                route=QueryRoute.FORCE_RAG,
                confidence=max_confidence,
                reasoning=f"Detected {len(detected_functions)} Pulseq function(s)",
                trigger_type="keyword",
                search_hints=[f["name"] for f in detected_functions],
                detected_functions=detected_functions,
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
        # Lazy load encoder if needed
        if not self.encoder:
            try:
                self._initialize_encoder()
                if not self._embeddings_loaded:
                    self._load_concept_embeddings()
                    self._embeddings_loaded = True
            except Exception as e:
                logger.warning(
                    f"Failed to initialize encoder for semantic classification: {e}",
                )
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
            query_embedding,
            self.sequence_embeddings,
        )
        implementation_sim = self._calculate_max_similarity(
            query_embedding,
            self.implementation_embeddings,
        )
        physics_sim = self._calculate_max_similarity(
            query_embedding,
            self.physics_embeddings,
        )

        semantic_scores = {
            "sequence": float(sequence_sim),
            "implementation": float(implementation_sim),
            "physics": float(physics_sim),
        }

        # Decision logic based on similarities
        seq_threshold = self.threshold_manager.get_threshold("sequence_similarity")
        impl_threshold = self.threshold_manager.get_threshold(
            "implementation_similarity",
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
                for impl_keyword in ("example", "show", "code", "implement", "create"):
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
        self,
        query_embedding: np.ndarray,
        concept_embeddings: np.ndarray,
    ) -> float:
        """Calculate maximum cosine similarity between query and concepts."""
        if len(concept_embeddings) == 0:
            return 0.0

        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        concept_norms = concept_embeddings / np.linalg.norm(
            concept_embeddings,
            axis=1,
            keepdims=True,
        )

        # Calculate cosine similarities
        similarities = np.dot(concept_norms, query_norm)

        return float(np.max(similarities))

    def _check_for_misspellings(self, query: str) -> Optional[RoutingDecision]:
        """Check for misspelled Pulseq function names using Levenshtein distance.

        Optimized version that:
        1. Caches the function list as a sorted list for faster searching
        2. Only checks words that look like function names (camelCase, underscored)
        3. Uses early exit for performance
        """
        # Extract potential function names from the query
        # Only check words that look like they could be function names
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", query)

        # Pre-filter words that are likely function names
        potential_functions = []
        for word in words:
            # Skip very short words (likely not function names)
            if len(word) < 4:
                continue

            # Check if word looks like a function name:
            # - Contains uppercase (camelCase): makeAdc, EventLibrary
            # - Contains underscore: make_adc, event_library
            # - Starts with common prefixes: make, calc, get, set
            if (
                any(c.isupper() for c in word[1:])  # Has uppercase after first char
                or "_" in word  # Has underscore
                or word.startswith(("make", "calc", "get", "set", "add", "create"))
            ):
                potential_functions.append(word)

        # Only process if we have potential function names
        if not potential_functions:
            return None

        # Check each potential function for misspellings
        for word in potential_functions:
            # Use get_close_matches with optimized parameters
            # n=1 for just the best match, cutoff=0.75 for reasonable threshold
            matches = get_close_matches(
                word,
                self.PULSEQ_FUNCTIONS,
                n=1,  # Get only best match (faster)
                cutoff=0.75,  # 75% similarity threshold
            )

            if matches:
                matched_func = matches[0]
                # Quick similarity check (no need for complex calculation)
                # get_close_matches already validated similarity

                return RoutingDecision(
                    route=QueryRoute.FORCE_RAG,
                    confidence=0.85,  # Fixed confidence for misspellings
                    reasoning=f"Likely misspelling: '{word}' → '{matched_func}'",
                    trigger_type="misspelling",
                    search_hints=[matched_func],  # Use correct spelling for search
                    detected_functions=[
                        {
                            "name": matched_func,  # Corrected spelling
                            "namespace": None,
                            "full_match": word,  # Original misspelling
                            "confidence": 0.85,
                            "type": "misspelling_correction",
                        }
                    ],
                )

        return None

    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words (0 to 1)."""
        # Simple character-based similarity
        longer = max(len(word1), len(word2))
        if longer == 0:
            return 1.0

        # Count matching characters in same positions
        matches = sum(c1 == c2 for c1, c2 in zip(word1.lower(), word2.lower()))
        # Add partial credit for length similarity
        length_similarity = min(len(word1), len(word2)) / longer

        return (matches / longer + length_similarity) / 2

    def _camelcase_to_spaced(self, name: str) -> str:
        """Convert CamelCase to spaced words.
        EventLibrary -> 'event library'
        makeTrapezoid -> 'make trapezoid'
        """
        # Add space before capital letters (except first)
        result = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
        return result.lower()

    def _camelcase_to_underscored(self, name: str) -> str:
        """Convert CamelCase to underscored.
        EventLibrary -> 'event_library'
        makeTrapezoid -> 'make_trapezoid'
        """
        # Add underscore before capital letters (except first)
        result = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
        return result.lower()

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
            f"(confidence: {decision.confidence:.2f}, trigger: {decision.trigger_type})",
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
