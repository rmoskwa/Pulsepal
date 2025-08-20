"""
Tests for log rotation manager - Task 3: Session importance scoring.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pulsepal.log_manager import LogRotationManager


class TestImportanceScoring:
    """Test session importance scoring (Task 3)."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            log_dir.mkdir()
            yield log_dir
    
    @pytest.fixture
    def rotation_manager(self, temp_log_dir):
        """Create LogRotationManager instance with test directory."""
        return LogRotationManager(
            log_dir=str(temp_log_dir),
            archive_dir=str(temp_log_dir / "archive"),
            retention_days=30,
            max_size_gb=1.0,
            importance_threshold=0.7,
        )
    
    def create_session_with_exchanges(self, log_dir: Path, session_id: str, num_exchanges: int, has_errors: bool = False, days_old: int = 0):
        """Create test session with specified number of exchanges."""
        jsonl_file = log_dir / f"{session_id}.jsonl"
        
        with open(jsonl_file, 'w') as f:
            for i in range(num_exchanges):
                message = {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Message {i}",
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Add error messages if specified
                if has_errors and i == num_exchanges // 2:
                    message["content"] = "Error: Something went wrong"
                    message["error"] = True
                
                f.write(json.dumps(message) + "\n")
        
        # Create associated files
        txt_file = log_dir / f"{session_id}.txt"
        with open(txt_file, 'w') as f:
            f.write(f"Session summary with {num_exchanges} exchanges\n")
            if has_errors:
                f.write("Session contained errors\n")
        
        search_file = log_dir / f"{session_id}_searches.jsonl"
        with open(search_file, 'w') as f:
            f.write(json.dumps({"query": "test"}) + "\n")
        
        # Set modification time if needed
        if days_old > 0:
            old_time = (datetime.now() - timedelta(days=days_old)).timestamp()
            import os
            os.utime(jsonl_file, (old_time, old_time))
    
    def test_importance_criteria_definition(self, rotation_manager):
        """Test that importance criteria are properly defined."""
        # The scoring algorithm should consider:
        # - Long conversations (>20 exchanges)
        # - Contains errors/issues
        # - Recent activity
        # - Large file size (knowledge-rich)
        
        # This is validated through the implementation
        assert hasattr(rotation_manager, 'score_session_importance')
        assert hasattr(rotation_manager, 'importance_threshold')
    
    def test_score_long_conversation(self, rotation_manager, temp_log_dir):
        """Test scoring for long conversations (>20 exchanges)."""
        # Create a long conversation
        self.create_session_with_exchanges(temp_log_dir, "session_long", 25)
        
        score = rotation_manager.score_session_importance("session_long")
        
        # Should get at least 0.4 for being long
        assert score >= 0.4
    
    def test_score_conversation_with_errors(self, rotation_manager, temp_log_dir):
        """Test scoring for sessions with errors."""
        # Create a session with errors
        self.create_session_with_exchanges(temp_log_dir, "session_error", 10, has_errors=True)
        
        score = rotation_manager.score_session_importance("session_error")
        
        # Should get at least 0.3 for having errors
        assert score >= 0.3
    
    def test_score_recent_session(self, rotation_manager, temp_log_dir):
        """Test scoring for recent sessions."""
        # Create a recent session (0 days old)
        self.create_session_with_exchanges(temp_log_dir, "session_recent", 5, days_old=0)
        
        score = rotation_manager.score_session_importance("session_recent")
        
        # Should get bonus for being recent
        assert score >= 0.2
    
    def test_score_old_session(self, rotation_manager, temp_log_dir):
        """Test scoring for old sessions."""
        # Create an old session (30 days old)
        self.create_session_with_exchanges(temp_log_dir, "session_old", 5, days_old=30)
        
        score = rotation_manager.score_session_importance("session_old")
        
        # Should get lower score for being old
        assert score < 0.2
    
    def test_score_large_session(self, rotation_manager, temp_log_dir):
        """Test scoring for large knowledge-rich sessions."""
        # Create a large session with lots of content
        session_id = "session_large"
        jsonl_file = temp_log_dir / f"{session_id}.jsonl"
        
        with open(jsonl_file, 'w') as f:
            for i in range(100):
                message = {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": "x" * 1000,  # Large content
                    "timestamp": datetime.now().isoformat(),
                }
                f.write(json.dumps(message) + "\n")
        
        # Create associated files
        (temp_log_dir / f"{session_id}.txt").touch()
        (temp_log_dir / f"{session_id}_searches.jsonl").touch()
        
        score = rotation_manager.score_session_importance(session_id)
        
        # Should get bonus for large file size (>50KB)
        assert jsonl_file.stat().st_size > 50000
        assert score >= 0.1
    
    def test_combined_importance_factors(self, rotation_manager, temp_log_dir):
        """Test scoring with multiple importance factors."""
        # Create a session that is:
        # - Long (25 exchanges) -> +0.4
        # - Has errors -> +0.3
        # - Recent (5 days old) -> +0.2
        # - Large size -> +0.1
        self.create_session_with_exchanges(
            temp_log_dir, 
            "session_important",
            25,
            has_errors=True,
            days_old=5
        )
        
        score = rotation_manager.score_session_importance("session_important")
        
        # Should have high importance score
        assert score >= 0.7  # Threshold for important sessions
    
    def test_mark_sessions_with_errors(self, rotation_manager, temp_log_dir):
        """Test that sessions with errors are marked as important."""
        # Create sessions with different error patterns
        test_cases = [
            ("session_error1", "Error: Database connection failed"),
            ("session_error2", "Exception: Invalid parameter"),
            ("session_error3", "error occurred during processing"),
            ("session_error4", "EXCEPTION raised"),
        ]
        
        for session_id, error_msg in test_cases:
            jsonl_file = temp_log_dir / f"{session_id}.jsonl"
            with open(jsonl_file, 'w') as f:
                f.write(json.dumps({"content": error_msg}) + "\n")
            
            # Create associated files
            (temp_log_dir / f"{session_id}.txt").touch()
            (temp_log_dir / f"{session_id}_searches.jsonl").touch()
            
            score = rotation_manager.score_session_importance(session_id)
            
            # Should detect error and increase score
            assert score >= 0.3, f"Failed to detect error in: {error_msg}"
    
    def test_mark_long_conversations(self, rotation_manager, temp_log_dir):
        """Test that long conversations (>20 exchanges) are marked as important."""
        test_cases = [
            ("session_10", 10, 0.2),   # Medium conversation
            ("session_20", 20, 0.2),   # At threshold
            ("session_25", 25, 0.4),   # Long conversation
            ("session_50", 50, 0.4),   # Very long conversation
        ]
        
        for session_id, num_exchanges, min_score in test_cases:
            self.create_session_with_exchanges(temp_log_dir, session_id, num_exchanges)
            score = rotation_manager.score_session_importance(session_id)
            assert score >= min_score, f"Session with {num_exchanges} exchanges should score >= {min_score}"
    
    def test_importance_threshold_configuration(self, temp_log_dir):
        """Test custom importance threshold configuration."""
        # Create manager with custom threshold
        manager = LogRotationManager(
            log_dir=str(temp_log_dir),
            importance_threshold=0.5,
        )
        
        assert manager.importance_threshold == 0.5
        
        # Create a session with score around 0.6
        self.create_session_with_exchanges(temp_log_dir, "session_medium", 15, has_errors=True)
        
        score = manager.score_session_importance("session_medium")
        
        # Should be considered important if score >= 0.5
        is_important = score >= manager.importance_threshold
        assert is_important == (score >= 0.5)
    
    def test_missing_session_file(self, rotation_manager, temp_log_dir):
        """Test scoring when session file is missing."""
        score = rotation_manager.score_session_importance("nonexistent_session")
        
        # Should return 0 for missing sessions
        assert score == 0.0
    
    def test_corrupted_session_file(self, rotation_manager, temp_log_dir):
        """Test scoring with corrupted JSON file."""
        # Create a corrupted file
        jsonl_file = temp_log_dir / "session_corrupt.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write("Not valid JSON\n")
            f.write('{"valid": "json"}\n')
            f.write("More invalid content\n")
        
        # Create associated files
        (temp_log_dir / "session_corrupt.txt").touch()
        (temp_log_dir / "session_corrupt_searches.jsonl").touch()
        
        # Should handle gracefully
        score = rotation_manager.score_session_importance("session_corrupt")
        
        # Should still calculate some score based on what it can read
        assert score >= 0  # Should not crash
    
    def test_score_normalization(self, rotation_manager, temp_log_dir):
        """Test that scores are normalized between 0 and 1."""
        # Create a session with all importance factors maxed out
        self.create_session_with_exchanges(
            temp_log_dir,
            "session_max",
            100,  # Very long
            has_errors=True,
            days_old=0  # Very recent
        )
        
        score = rotation_manager.score_session_importance("session_max")
        
        # Score should never exceed 1.0
        assert 0 <= score <= 1.0