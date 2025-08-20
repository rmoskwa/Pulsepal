"""
Test suite ensuring PulsePal handles both common and novel problems.
"""

from pulsepal.debug_analyzer import PulseqDebugAnalyzer
from pulsepal.concept_mapper import ConceptMapper


class TestNovelProblemHandling:
    """Test that novel problems are handled gracefully."""

    def test_novel_problem_detection(self):
        """Test that novel problems are identified correctly."""
        mapper = ConceptMapper()

        # Test a novel problem
        novel_problem = "The RF amplifier is producing harmonics"
        result = mapper.map_problem_to_code(novel_problem)

        assert result["approach"] == "systematic_analysis"
        assert result["is_novel"] is True
        assert "general physics reasoning" in result["note"].lower()

    def test_common_problem_detection(self):
        """Test that common problems get specific hints."""
        mapper = ConceptMapper()

        # Test a common problem
        common_problem = "My k-space trajectory is wrong"
        result = mapper.map_problem_to_code(common_problem)

        assert not result.get("is_novel", False)
        assert "gradient" in str(result).lower()

    def test_novel_problem_debugging(self):
        """Test that novel problems still get analyzed."""
        analyzer = PulseqDebugAnalyzer()

        novel_problem = "The scanner makes a strange clicking noise during my sequence"
        code = "gx = mr.makeTrapezoid('x', 'amplitude', 40000);"

        results = analyzer.debug_concept(novel_problem, code)

        # Should still provide analysis even for novel problem
        assert "reasoning" in results
        assert (
            "systematic" in results["reasoning"].lower()
            or "novel" in results["reasoning"].lower()
        )
        assert len(results["issues"]) > 0

    def test_physics_reasoning_independence(self):
        """Test that physics reasoning works without pattern matching."""
        analyzer = PulseqDebugAnalyzer()

        # Completely novel problem
        problem = "The liquid helium consumption increased after running my sequence"
        code = "% Some sequence code"

        results = analyzer.debug_concept(problem, code)

        # Should provide systematic guidance
        assert results["reasoning"]
        assert (
            "novel" in results["reasoning"].lower()
            or "systematic" in results["reasoning"].lower()
        )

        # Should still identify that analysis is needed
        assert any(
            "systematic" in issue.solution.lower() for issue in results["issues"]
        )


class TestSyntaxDebugging:
    """Test Category 1 debugging remains deterministic."""

    def test_syntax_checking_unchanged(self):
        """Ensure syntax checking still works deterministically."""
        analyzer = PulseqDebugAnalyzer()

        code = """
        seq = mr.Sequence();
        mr.write('test.seq');  # Wrong
        seq.addBlock(rf, gz);   # Correct
        """

        results = analyzer.debug_syntax(code)

        # Should find the mr.write error
        assert len(results) >= 1
        assert any("mr.write" in str(issue.incorrect_usage) for issue in results)
        assert any("seq.write" in str(issue.correct_usage) for issue in results)
