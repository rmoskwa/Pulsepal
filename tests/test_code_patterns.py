"""
Test file for the simplified code_patterns module.
Focuses on function clustering only - no sequence detection or hard-coded suggestions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pulsepal.code_patterns import FunctionClusterAnalyzer
from pulsepal.debug_analyzer import PulseqDebugAnalyzer


def test_function_clustering():
    """Test that function clustering correctly identifies related functions."""

    analyzer = FunctionClusterAnalyzer()

    # Test with RF and gradient functions
    used_functions = ["mr.makeSincPulse", "mr.makeTrapezoid", "seq.addBlock"]

    results = analyzer.analyze_functions(used_functions)

    print("Function Clustering Test:")
    print("-" * 50)
    print(f"Functions analyzed: {results['functions_analyzed']}")
    print(f"Clusters detected: {results['clusters_detected']}")
    print(f"Related functions: {results['related_functions']}")
    print(f"Potentially missing: {results['potentially_missing']}")

    # Verify that it detects the right clusters
    assert "rf_excitation" in results["clusters_detected"], "Should detect RF cluster"

    # Check that it identifies related functions
    assert len(results["related_functions"]) > len(
        used_functions
    ), "Should find related functions"

    print("\n✅ Function clustering test passed!")
    return results


def test_missing_core_functions():
    """Test detection of missing core functions from clusters."""

    analyzer = FunctionClusterAnalyzer()

    # Code that has RF but missing ADC for readout
    used_functions = [
        "mr.makeSincPulse",
        "mr.makeTrapezoid",  # Gradient but no ADC
    ]

    results = analyzer.analyze_functions(used_functions)
    clusters = results["clusters_detected"]

    # Check for missing core functions
    if "readout_pattern" in clusters:
        missing_core = analyzer.find_missing_core_functions(used_functions, clusters)

        print("\nMissing Core Functions Test:")
        print("-" * 50)
        print(f"Clusters: {clusters}")
        print("Missing core functions:")
        for func, reason in missing_core:
            print(f"  - {func}: {reason}")

        # Should identify missing ADC
        assert any(
            func == "makeAdc" for func, _ in missing_core
        ), "Should identify missing ADC"

    print("\n✅ Missing core functions test passed!")
    return results


def test_integration_with_debug_analyzer():
    """Test that debug_analyzer properly uses function clustering."""

    analyzer = PulseqDebugAnalyzer()

    test_code = """
    seq = mr.Sequence();

    % RF pulse
    rf = mr.makeSincPulse(pi/2, 'duration', 3e-3);

    % Gradient
    gx = mr.makeTrapezoid('x', 'amplitude', 20);

    % Missing: ADC for readout

    seq.addBlock(rf, gx);
    """

    results = analyzer.analyze_code(test_code)

    print("\nDebug Analyzer Integration Test:")
    print("-" * 50)

    # Check that function clustering is included
    assert "function_clusters" in results, "Should include function cluster analysis"

    clusters = results["function_clusters"]
    print(f"Clusters detected: {clusters.get('clusters_detected', [])}")
    print(
        f"Potentially missing: {clusters.get('potentially_missing', [])[:5]}"
    )  # First 5

    # Should detect RF and gradient clusters
    assert "rf_excitation" in clusters["clusters_detected"], "Should detect RF cluster"

    # Should identify missing functions
    assert "makeAdc" in clusters["potentially_missing"], "Should identify missing ADC"

    print("\n✅ Integration test passed!")
    return results


def test_cluster_info():
    """Test getting information about specific clusters."""

    analyzer = FunctionClusterAnalyzer()

    print("\nCluster Information Test:")
    print("-" * 50)

    # Get info about RF excitation cluster
    rf_info = analyzer.get_cluster_info("rf_excitation")
    print("RF Excitation Cluster:")
    print(f"  Description: {rf_info.get('description')}")
    print(f"  Functions: {rf_info.get('functions')[:3]}...")  # First 3

    # Get info about readout pattern
    readout_info = analyzer.get_cluster_info("readout_pattern")
    print("\nReadout Pattern Cluster:")
    print(f"  Description: {readout_info.get('description')}")
    print(f"  Functions: {readout_info.get('functions')}")

    assert rf_info["exists"], "RF excitation cluster should exist"
    assert readout_info["exists"], "Readout pattern cluster should exist"

    print("\n✅ Cluster info test passed!")
    return rf_info, readout_info


def compare_with_llm_capabilities():
    """
    Demonstrate why function clustering is valuable even though
    the LLM can identify sequences and provide suggestions.
    """

    print("\nWhy Function Clustering is Valuable:")
    print("=" * 60)

    print("\n1. FINITE DOMAIN KNOWLEDGE")
    print("   - Pulseq has ~150 functions that rarely change")
    print("   - Function relationships are stable and well-defined")
    print("   - This is domain-specific knowledge worth encoding")

    print("\n2. LIGHTWEIGHT & FAST")
    print("   - Simple dictionary lookups, no LLM inference needed")
    print("   - Provides immediate feedback about missing functions")
    print("   - No risk of hallucination about function relationships")

    print("\n3. COMPLEMENTS LLM REASONING")
    print("   - LLM handles: sequence identification, physics reasoning, suggestions")
    print("   - Function clustering handles: which functions work together")
    print("   - Together: comprehensive debugging and assistance")

    print("\n4. WHAT WE DON'T DO (because LLM is better):")
    print("   ❌ Sequence type detection - LLM understands context better")
    print("   ❌ Hard-coded suggestions - LLM provides flexible, contextual advice")
    print("   ❌ Pattern matching sequences - Too many variations, LLM is smarter")

    print("\n5. PERFECT DIVISION OF LABOR:")
    print("   - Function clustering: Deterministic, finite, stable knowledge")
    print("   - LLM: Dynamic reasoning, context understanding, flexible suggestions")


if __name__ == "__main__":
    print("Testing Simplified Code Patterns Module")
    print("=" * 60)
    print("Focus: Function clustering only (no sequence detection)")
    print("=" * 60)

    try:
        # Run tests
        test_function_clustering()
        test_missing_core_functions()
        test_integration_with_debug_analyzer()
        test_cluster_info()

        # Explain the design philosophy
        compare_with_llm_capabilities()

        print("\n" + "=" * 60)
        print("🎉 All tests passed!")
        print("\nThe simplified code_patterns.py module is working correctly.")
        print("It focuses on what's valuable: function clustering for a finite,")
        print("stable set of ~150 Pulseq functions.")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
