"""
RAG Performance Monitoring for PulsePal.

This module provides comprehensive performance monitoring and metrics collection
for RAG search operations, including query performance, result quality, and
system resource usage.
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single RAG query."""

    query: str
    duration: float
    result_count: int
    avg_similarity: float
    max_similarity: float
    min_similarity: float
    search_type: str
    timestamp: float
    rerank_enabled: bool = False
    rerank_duration: float = 0.0
    hybrid_search: bool = False
    error: Optional[str] = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    total_queries: int = 0
    avg_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    avg_result_count: float = 0.0
    avg_similarity: float = 0.0
    error_rate: float = 0.0
    queries_per_second: float = 0.0
    cache_hit_rate: float = 0.0


class RAGPerformanceMonitor:
    """
    Monitor RAG search performance and quality metrics.

    This class tracks query performance, result quality, and provides
    aggregated statistics for system monitoring and optimization.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.

        Args:
            max_history: Maximum number of queries to keep in history
        """
        self.max_history = max_history
        self.query_history: deque = deque(maxlen=max_history)
        self.query_stats: Dict[str, List[QueryMetrics]] = defaultdict(list)
        self.cache_stats = {"hits": 0, "misses": 0}
        self._lock = threading.Lock()

        # Performance buckets for percentile calculations
        self.duration_buckets = deque(maxlen=max_history)
        self.similarity_buckets = deque(maxlen=max_history)

        logger.info(
            f"RAG Performance Monitor initialized with history size: {max_history}",
        )

    def start_query(self, query: str, search_type: str = "documents") -> Dict[str, Any]:
        """
        Start timing a query operation.

        Args:
            query: The search query
            search_type: Type of search operation

        Returns:
            Context dictionary for tracking the query
        """
        return {
            "query": query,
            "search_type": search_type,
            "start_time": time.time(),
            "rerank_enabled": False,
            "hybrid_search": False,
        }

    def record_query_completion(
        self,
        context: Dict[str, Any],
        results: List[Dict[str, Any]],
        error: Optional[str] = None,
        rerank_duration: float = 0.0,
    ) -> QueryMetrics:
        """
        Record completion of a query operation.

        Args:
            context: Query context from start_query
            results: Search results
            error: Optional error message
            rerank_duration: Time spent on reranking

        Returns:
            QueryMetrics object with collected metrics
        """
        end_time = time.time()
        duration = end_time - context["start_time"]

        # Calculate similarity statistics
        similarities = [r.get("similarity", 0) for r in results if "similarity" in r]
        avg_similarity = statistics.mean(similarities) if similarities else 0.0
        max_similarity = max(similarities) if similarities else 0.0
        min_similarity = min(similarities) if similarities else 0.0

        # Create metrics object
        metrics = QueryMetrics(
            query=context["query"],
            duration=duration,
            result_count=len(results),
            avg_similarity=avg_similarity,
            max_similarity=max_similarity,
            min_similarity=min_similarity,
            search_type=context["search_type"],
            timestamp=end_time,
            rerank_enabled=context.get("rerank_enabled", False),
            rerank_duration=rerank_duration,
            hybrid_search=context.get("hybrid_search", False),
            error=error,
        )

        # Store metrics
        with self._lock:
            self.query_history.append(metrics)
            self.query_stats[context["query"]].append(metrics)
            self.duration_buckets.append(duration)
            if similarities:
                self.similarity_buckets.extend(similarities)

        # Log performance info
        if error:
            logger.warning(f"Query failed: {context['query'][:50]}... Error: {error}")
        else:
            logger.debug(
                f"Query completed: {duration:.3f}s, {len(results)} results, "
                f"avg_sim: {avg_similarity:.3f}",
            )

        return metrics

    def record_cache_hit(self):
        """Record a cache hit."""
        with self._lock:
            self.cache_stats["hits"] += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        with self._lock:
            self.cache_stats["misses"] += 1

    def get_performance_stats(
        self, window_minutes: Optional[int] = None,
    ) -> PerformanceStats:
        """
        Get aggregated performance statistics.

        Args:
            window_minutes: Optional time window for statistics (None for all time)

        Returns:
            PerformanceStats object with aggregated metrics
        """
        with self._lock:
            # Filter by time window if specified
            if window_minutes:
                cutoff_time = time.time() - (window_minutes * 60)
                recent_queries = [
                    q for q in self.query_history if q.timestamp >= cutoff_time
                ]
            else:
                recent_queries = list(self.query_history)

            if not recent_queries:
                return PerformanceStats()

            # Calculate statistics
            durations = [q.duration for q in recent_queries]
            result_counts = [q.result_count for q in recent_queries]
            similarities = [
                q.avg_similarity for q in recent_queries if q.avg_similarity > 0
            ]
            errors = [q for q in recent_queries if q.error is not None]

            # Time span for QPS calculation
            if len(recent_queries) > 1:
                time_span = recent_queries[-1].timestamp - recent_queries[0].timestamp
                qps = len(recent_queries) / max(time_span, 1.0)
            else:
                qps = 0.0

            # Cache hit rate
            total_cache_ops = self.cache_stats["hits"] + self.cache_stats["misses"]
            cache_hit_rate = (
                self.cache_stats["hits"] / total_cache_ops
                if total_cache_ops > 0
                else 0.0
            )

            return PerformanceStats(
                total_queries=len(recent_queries),
                avg_duration=statistics.mean(durations),
                min_duration=min(durations),
                max_duration=max(durations),
                avg_result_count=statistics.mean(result_counts),
                avg_similarity=statistics.mean(similarities) if similarities else 0.0,
                error_rate=len(errors) / len(recent_queries),
                queries_per_second=qps,
                cache_hit_rate=cache_hit_rate,
            )

    def get_percentiles(self, metric: str = "duration") -> Dict[str, float]:
        """
        Get percentile statistics for a specific metric.

        Args:
            metric: Metric to calculate percentiles for ("duration" or "similarity")

        Returns:
            Dictionary with percentile values
        """
        with self._lock:
            if metric == "duration":
                values = list(self.duration_buckets)
            elif metric == "similarity":
                values = list(self.similarity_buckets)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if not values:
                return {}

            return {
                "p50": statistics.median(values),
                "p75": statistics.quantiles(values, n=4)[2]
                if len(values) >= 4
                else values[-1],
                "p90": statistics.quantiles(values, n=10)[8]
                if len(values) >= 10
                else values[-1],
                "p95": statistics.quantiles(values, n=20)[18]
                if len(values) >= 20
                else values[-1],
                "p99": statistics.quantiles(values, n=100)[98]
                if len(values) >= 100
                else values[-1],
            }

    def get_slow_queries(
        self, threshold_seconds: float = 1.0, limit: int = 10,
    ) -> List[QueryMetrics]:
        """
        Get slowest queries above threshold.

        Args:
            threshold_seconds: Minimum duration threshold
            limit: Maximum number of queries to return

        Returns:
            List of slow queries sorted by duration
        """
        with self._lock:
            slow_queries = [
                q for q in self.query_history if q.duration >= threshold_seconds
            ]

            return sorted(slow_queries, key=lambda x: x.duration, reverse=True)[:limit]

    def get_failed_queries(self, limit: int = 10) -> List[QueryMetrics]:
        """
        Get recent failed queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of failed queries sorted by timestamp
        """
        with self._lock:
            failed_queries = [q for q in self.query_history if q.error is not None]

            return sorted(failed_queries, key=lambda x: x.timestamp, reverse=True)[
                :limit
            ]

    def get_query_pattern_analysis(self) -> Dict[str, Any]:
        """
        Analyze query patterns and provide insights.

        Returns:
            Dictionary with pattern analysis results
        """
        with self._lock:
            if not self.query_history:
                return {}

            # Analyze search types
            search_types = defaultdict(int)
            for q in self.query_history:
                search_types[q.search_type] += 1

            # Analyze query lengths
            query_lengths = [len(q.query) for q in self.query_history]

            # Analyze reranking usage
            rerank_usage = sum(1 for q in self.query_history if q.rerank_enabled)
            hybrid_usage = sum(1 for q in self.query_history if q.hybrid_search)

            return {
                "search_type_distribution": dict(search_types),
                "avg_query_length": statistics.mean(query_lengths),
                "reranking_usage_rate": rerank_usage / len(self.query_history),
                "hybrid_search_usage_rate": hybrid_usage / len(self.query_history),
                "total_unique_queries": len(self.query_stats),
                "repeat_query_rate": 1
                - (len(self.query_stats) / len(self.query_history)),
            }

    def reset_stats(self):
        """Reset all performance statistics."""
        with self._lock:
            self.query_history.clear()
            self.query_stats.clear()
            self.cache_stats = {"hits": 0, "misses": 0}
            self.duration_buckets.clear()
            self.similarity_buckets.clear()

        logger.info("Performance statistics reset")

    def export_metrics(self, format: str = "dict") -> Any:
        """
        Export metrics in specified format.

        Args:
            format: Export format ("dict", "json", or "csv")

        Returns:
            Metrics in requested format
        """
        with self._lock:
            if format == "dict":
                return {
                    "performance_stats": self.get_performance_stats().__dict__,
                    "duration_percentiles": self.get_percentiles("duration"),
                    "similarity_percentiles": self.get_percentiles("similarity"),
                    "query_patterns": self.get_query_pattern_analysis(),
                    "slow_queries": [q.__dict__ for q in self.get_slow_queries()],
                    "failed_queries": [q.__dict__ for q in self.get_failed_queries()],
                }
            if format == "json":
                import json

                return json.dumps(self.export_metrics("dict"), indent=2, default=str)
            if format == "csv":
                import csv
                import io

                output = io.StringIO()
                writer = csv.writer(output)

                # Write headers
                writer.writerow(
                    [
                        "timestamp",
                        "query",
                        "duration",
                        "result_count",
                        "avg_similarity",
                        "search_type",
                        "rerank_enabled",
                        "hybrid_search",
                        "error",
                    ],
                )

                # Write data
                for q in self.query_history:
                    writer.writerow(
                        [
                            q.timestamp,
                            q.query[:100],
                            q.duration,
                            q.result_count,
                            q.avg_similarity,
                            q.search_type,
                            q.rerank_enabled,
                            q.hybrid_search,
                            q.error or "",
                        ],
                    )

                return output.getvalue()
            raise ValueError(f"Unsupported format: {format}")


# Global instance
_performance_monitor: Optional[RAGPerformanceMonitor] = None


def get_performance_monitor() -> RAGPerformanceMonitor:
    """
    Get the global performance monitor instance.

    Returns:
        RAGPerformanceMonitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = RAGPerformanceMonitor()
    return _performance_monitor


def monitor_query(func):
    """
    Decorator to automatically monitor query performance.

    Usage:
        @monitor_query
        def my_search_function(query, **kwargs):
            # search implementation
            return results
    """

    def wrapper(*args, **kwargs):
        monitor = get_performance_monitor()

        # Extract query and search type from arguments
        query = args[0] if args else kwargs.get("query", "unknown")
        search_type = kwargs.get("search_type", "documents")

        # Start monitoring
        context = monitor.start_query(query, search_type)

        try:
            # Execute function
            results = func(*args, **kwargs)

            # Record successful completion
            monitor.record_query_completion(context, results or [])
            return results

        except Exception as e:
            # Record failure
            monitor.record_query_completion(context, [], error=str(e))
            raise

    return wrapper
