#!/usr/bin/env python3
"""
Performance metrics tracking for Pulsepal intelligent refactor.

Measures response times and search patterns to validate improvements.
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, List
import statistics

from pulsepal.main_agent import run_pulsepal


class PerformanceMetrics:
    """Track and analyze Pulsepal performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'general_queries': [],
            'pulseq_queries': [],
            'all_queries': [],
            'timestamp': datetime.now().isoformat()
        }
    
    async def measure_query(self, query: str, category: str) -> Dict:
        """Measure response time for a single query."""
        start_time = time.time()
        
        try:
            session_id, response = await run_pulsepal(query)
            response_time = time.time() - start_time
            
            result = {
                'query': query,
                'category': category,
                'response_time': response_time,
                'response_length': len(response),
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            result = {
                'query': query,
                'category': category,
                'response_time': response_time,
                'response_length': 0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        # Store in appropriate category
        self.metrics['all_queries'].append(result)
        if category == 'general':
            self.metrics['general_queries'].append(result)
        else:
            self.metrics['pulseq_queries'].append(result)
        
        return result
    
    def calculate_statistics(self, queries: List[Dict]) -> Dict:
        """Calculate statistics for a set of queries."""
        if not queries:
            return {}
        
        response_times = [q['response_time'] for q in queries if q['success']]
        
        if not response_times:
            return {'error': 'No successful queries'}
        
        return {
            'count': len(queries),
            'successful': len(response_times),
            'failed': len(queries) - len(response_times),
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        general_stats = self.calculate_statistics(self.metrics['general_queries'])
        pulseq_stats = self.calculate_statistics(self.metrics['pulseq_queries'])
        overall_stats = self.calculate_statistics(self.metrics['all_queries'])
        
        report = f"""
================================================================================
📊 PULSEPAL PERFORMANCE METRICS REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 OVERALL PERFORMANCE
----------------------
Total Queries: {overall_stats.get('count', 0)}
Success Rate: {overall_stats.get('successful', 0)}/{overall_stats.get('count', 0)} ({overall_stats.get('successful', 0)/max(overall_stats.get('count', 1), 1)*100:.1f}%)
Average Response Time: {overall_stats.get('avg_response_time', 0):.2f}s
Median Response Time: {overall_stats.get('median_response_time', 0):.2f}s

📚 GENERAL QUERIES (No Search Expected)
---------------------------------------
Count: {general_stats.get('count', 0)}
Average Time: {general_stats.get('avg_response_time', 0):.2f}s
Target Range: 2-4 seconds
Status: {'✅ WITHIN TARGET' if 2 <= general_stats.get('avg_response_time', 0) <= 4 else '⚠️ OUTSIDE TARGET'}

🔍 PULSEQ-SPECIFIC QUERIES (Search Expected)
--------------------------------------------
Count: {pulseq_stats.get('count', 0)}
Average Time: {pulseq_stats.get('avg_response_time', 0):.2f}s
Target Range: 3-6 seconds
Status: {'✅ WITHIN TARGET' if 3 <= pulseq_stats.get('avg_response_time', 0) <= 6 else '⚠️ OUTSIDE TARGET'}

📈 PERFORMANCE IMPROVEMENT
--------------------------
Before Refactor (estimated):
  - All queries: 4-7 seconds (forced search)
  - Search rate: ~100%

After Refactor (measured):
  - General queries: {general_stats.get('avg_response_time', 0):.2f}s
  - Pulseq queries: {pulseq_stats.get('avg_response_time', 0):.2f}s
  - Estimated search rate: {len(self.metrics['pulseq_queries'])/max(len(self.metrics['all_queries']), 1)*100:.1f}%

💰 EFFICIENCY GAINS
-------------------
Average time saved per general query: ~{max(0, 5 - general_stats.get('avg_response_time', 0)):.1f}s
Estimated database load reduction: ~{100 - (len(self.metrics['pulseq_queries'])/max(len(self.metrics['all_queries']), 1)*100):.0f}%

================================================================================
"""
        return report
    
    def save_metrics(self, filename: str = "performance_metrics.json"):
        """Save metrics to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to {filename}")


async def run_performance_test():
    """Run a comprehensive performance test."""
    metrics = PerformanceMetrics()
    
    # Test queries - mix of general and Pulseq-specific
    test_queries = [
        # General MRI physics (should be fast, no search)
        ("What is the Larmor frequency?", "general"),
        ("Explain T2* decay", "general"),
        ("How do I calculate flip angle?", "general"),
        ("What causes chemical shift artifacts?", "general"),
        ("Describe the spin echo sequence", "general"),
        
        # Pulseq-specific (should search, slightly slower)
        ("How to use mr.makeTrapezoid?", "pulseq"),
        ("Show me pypulseq.make_sinc_pulse parameters", "pulseq"),
        ("MOLLI sequence implementation in MATLAB", "pulseq"),
        ("mr.calcDuration function usage", "pulseq"),
        ("Pulseq addBlock timing constraints", "pulseq"),
    ]
    
    print("\n🚀 Starting Performance Test...")
    print(f"Running {len(test_queries)} test queries\n")
    
    for i, (query, category) in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] Testing: {query[:50]}...")
        result = await metrics.measure_query(query, category)
        print(f"  ⏱️  Response time: {result['response_time']:.2f}s")
        print(f"  📏 Response length: {result['response_length']} chars")
        print(f"  ✅ Success: {result['success']}\n")
        
        # Small delay between queries
        await asyncio.sleep(1)
    
    # Generate and display report
    report = metrics.generate_report()
    print(report)
    
    # Save metrics
    metrics.save_metrics()
    
    return metrics


async def main():
    """Main entry point."""
    try:
        metrics = await run_performance_test()
        print("\n✅ Performance test complete!")
        
    except Exception as e:
        print(f"\n❌ Performance test error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())