"""
Tests for log rotation manager - Task 1: Date-based rotation logic.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pulsepal.log_manager import LogRotationManager


class TestDateBasedRotation:
    """Test date-based rotation logic (Task 1)."""
    
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
        )
    
    def create_test_session(self, log_dir: Path, session_id: str, days_old: int = 0):
        """Create test session files with specified age."""
        # Create main JSONL file
        jsonl_file = log_dir / f"{session_id}.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps({"message": "test", "timestamp": datetime.now().isoformat()}) + "\n")
        
        # Create text summary file
        txt_file = log_dir / f"{session_id}.txt"
        with open(txt_file, 'w') as f:
            f.write("Test session summary\n")
        
        # Create searches file
        search_file = log_dir / f"{session_id}_searches.jsonl"
        with open(search_file, 'w') as f:
            f.write(json.dumps({"query": "test search"}) + "\n")
        
        # Set modification time to simulate age
        if days_old > 0:
            old_time = (datetime.now() - timedelta(days=days_old)).timestamp()
            import os
            os.utime(jsonl_file, (old_time, old_time))
            os.utime(txt_file, (old_time, old_time))
            os.utime(search_file, (old_time, old_time))
    
    def test_parse_timestamp_format(self, rotation_manager):
        """Test parsing of timestamp-based session filenames."""
        # Test standard timestamp format
        timestamp = rotation_manager._parse_session_timestamp("session_20240115_143022.jsonl")
        assert timestamp is not None
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 15
        assert timestamp.hour == 14
        assert timestamp.minute == 30
        assert timestamp.second == 22
    
    def test_parse_hex_id_format(self, rotation_manager, temp_log_dir):
        """Test parsing of hex ID session filenames using modification time."""
        # Create a file with hex ID format
        test_file = temp_log_dir / "session_abc123.jsonl"
        test_file.touch()
        
        # Should fall back to modification time
        timestamp = rotation_manager._parse_session_timestamp("session_abc123.jsonl")
        assert timestamp is not None
        # Should be very recent (within last minute)
        assert (datetime.now() - timestamp).total_seconds() < 60
    
    def test_calculate_file_age(self, rotation_manager, temp_log_dir):
        """Test calculation of session file age."""
        # Create sessions with different ages
        self.create_test_session(temp_log_dir, "session_20240101_120000", days_old=45)
        self.create_test_session(temp_log_dir, "session_20240201_120000", days_old=15)
        
        # Parse timestamps and verify ages
        old_timestamp = rotation_manager._parse_session_timestamp("session_20240101_120000.jsonl")
        new_timestamp = rotation_manager._parse_session_timestamp("session_20240201_120000.jsonl")
        
        assert old_timestamp < new_timestamp
        
    def test_identify_old_sessions(self, rotation_manager, temp_log_dir):
        """Test identification of sessions older than retention period."""
        # Create sessions with different ages
        self.create_test_session(temp_log_dir, "session_old1", days_old=35)
        self.create_test_session(temp_log_dir, "session_old2", days_old=40)
        self.create_test_session(temp_log_dir, "session_recent1", days_old=10)
        self.create_test_session(temp_log_dir, "session_recent2", days_old=20)
        
        # Identify old sessions
        old_sessions = rotation_manager.identify_old_sessions()
        
        # Should identify only sessions older than 30 days
        assert len(old_sessions) == 2
        assert "session_old1" in old_sessions
        assert "session_old2" in old_sessions
        assert "session_recent1" not in old_sessions
        assert "session_recent2" not in old_sessions
    
    def test_retention_period_configuration(self, temp_log_dir):
        """Test custom retention period configuration."""
        # Create manager with 7-day retention
        manager = LogRotationManager(
            log_dir=str(temp_log_dir),
            retention_days=7,
        )
        
        # Create sessions with different ages
        self.create_test_session(temp_log_dir, "session_old", days_old=10)
        self.create_test_session(temp_log_dir, "session_recent", days_old=5)
        
        # Identify old sessions
        old_sessions = manager.identify_old_sessions()
        
        # Should identify only sessions older than 7 days
        assert len(old_sessions) == 1
        assert "session_old" in old_sessions
        assert "session_recent" not in old_sessions
    
    def test_files_eligible_for_deletion(self, rotation_manager, temp_log_dir):
        """Test creating list of files eligible for deletion."""
        # Create old session
        self.create_test_session(temp_log_dir, "session_old", days_old=35)
        
        # Get old sessions
        old_sessions = rotation_manager.identify_old_sessions()
        assert len(old_sessions) == 1
        
        # Verify all associated files are identified
        session_id = old_sessions[0]
        expected_files = [
            temp_log_dir / f"{session_id}.jsonl",
            temp_log_dir / f"{session_id}.txt",
            temp_log_dir / f"{session_id}_searches.jsonl",
        ]
        
        for file_path in expected_files:
            assert file_path.exists()
    
    @pytest.mark.asyncio
    async def test_dry_run_mode(self, rotation_manager, temp_log_dir):
        """Test dry-run mode for safety."""
        # Create old session
        self.create_test_session(temp_log_dir, "session_old", days_old=35)
        
        # Run cleanup in dry-run mode
        stats = await rotation_manager.cleanup_old_sessions(dry_run=True)
        
        # Verify statistics
        assert stats["sessions_checked"] == 1
        assert stats["sessions_removed"] == 1
        assert stats["space_freed_bytes"] > 0
        
        # Verify files still exist (dry-run should not delete)
        assert (temp_log_dir / "session_old.jsonl").exists()
        assert (temp_log_dir / "session_old.txt").exists()
        assert (temp_log_dir / "session_old_searches.jsonl").exists()
    
    def test_edge_cases(self, rotation_manager, temp_log_dir):
        """Test edge cases in date parsing."""
        # Test various filename formats
        test_cases = [
            ("session_20240115.jsonl", True),  # Date only
            ("session_20240115_14.jsonl", True),  # Partial time
            ("session_abc123def.jsonl", False),  # Hex ID (needs file)
            ("session_2024.jsonl", False),  # Incomplete date
            ("notasession.jsonl", False),  # Wrong prefix
        ]
        
        for filename, should_parse in test_cases:
            if not should_parse:
                # Create file for modification time fallback
                (temp_log_dir / filename).touch()
            
            timestamp = rotation_manager._parse_session_timestamp(filename)
            if should_parse:
                assert timestamp is not None
            # Even hex IDs should get modification time
            elif (temp_log_dir / filename).exists():
                assert timestamp is not None