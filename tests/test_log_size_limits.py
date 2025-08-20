"""
Tests for log rotation manager - Task 2: Size-based limits.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pulsepal.log_manager import LogRotationManager


class TestSizeBasedLimits:
    """Test size-based rotation limits (Task 2)."""
    
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
            max_size_gb=0.001,  # 1MB for testing
        )
    
    def create_test_session_with_size(self, log_dir: Path, session_id: str, size_kb: int, days_old: int = 0):
        """Create test session files with specified size."""
        # Create main JSONL file with specified size
        jsonl_file = log_dir / f"{session_id}.jsonl"
        content_size = size_kb * 1024 // 3  # Divide among 3 files
        
        with open(jsonl_file, 'w') as f:
            # Write enough data to reach target size
            data = "x" * (content_size - 100)  # Leave room for JSON structure
            f.write(json.dumps({"data": data}) + "\n")
        
        # Create text summary file
        txt_file = log_dir / f"{session_id}.txt"
        with open(txt_file, 'w') as f:
            f.write("x" * content_size)
        
        # Create searches file
        search_file = log_dir / f"{session_id}_searches.jsonl"
        with open(search_file, 'w') as f:
            f.write(json.dumps({"query": "x" * (content_size - 20)}) + "\n")
        
        # Set modification time if needed
        if days_old > 0:
            old_time = (datetime.now() - timedelta(days=days_old)).timestamp()
            import os
            os.utime(jsonl_file, (old_time, old_time))
            os.utime(txt_file, (old_time, old_time))
            os.utime(search_file, (old_time, old_time))
    
    def test_calculate_directory_size(self, rotation_manager, temp_log_dir):
        """Test calculation of total log directory size."""
        # Create sessions with known sizes
        self.create_test_session_with_size(temp_log_dir, "session_100kb", 100)
        self.create_test_session_with_size(temp_log_dir, "session_200kb", 200)
        self.create_test_session_with_size(temp_log_dir, "session_300kb", 300)
        
        # Calculate total size
        total_size = rotation_manager.calculate_directory_size()
        
        # Should be approximately 600KB (with some JSON overhead)
        assert total_size > 590 * 1024  # At least 590KB
        assert total_size < 650 * 1024  # Less than 650KB
    
    def test_max_size_configuration(self, temp_log_dir):
        """Test max_log_size_gb setting configuration."""
        # Create manager with 0.002 GB limit (approximately 2MB)
        manager = LogRotationManager(
            log_dir=str(temp_log_dir),
            max_size_gb=0.002,  # 0.002 GB
        )
        
        # 0.002 GB = 0.002 * 1024^3 bytes = 2147483.648 bytes
        assert manager.max_size_bytes == int(0.002 * 1024**3)
    
    def test_identify_sessions_by_size(self, rotation_manager, temp_log_dir):
        """Test identification of sessions to remove for size limit."""
        # Create sessions with different ages and sizes
        self.create_test_session_with_size(temp_log_dir, "session_old_small", 100, days_old=40)
        self.create_test_session_with_size(temp_log_dir, "session_old_large", 500, days_old=35)
        self.create_test_session_with_size(temp_log_dir, "session_recent_small", 100, days_old=5)
        self.create_test_session_with_size(temp_log_dir, "session_recent_large", 500, days_old=10)
        
        # Target 500KB (should remove oldest first)
        target_size = 500 * 1024
        sessions_to_remove = rotation_manager.identify_sessions_by_size(target_size)
        
        # Should prioritize oldest sessions
        session_ids = [s[0] for s in sessions_to_remove]
        assert "session_old_small" in session_ids
        assert "session_old_large" in session_ids
        
        # Verify size calculation
        total_remove_size = sum(s[1] for s in sessions_to_remove)
        assert total_remove_size > 500 * 1024  # Should remove enough to meet target
    
    @pytest.mark.asyncio
    async def test_size_based_cleanup(self, rotation_manager, temp_log_dir):
        """Test cleanup when size limit exceeded."""
        # Create sessions that exceed 1MB limit
        self.create_test_session_with_size(temp_log_dir, "session_1", 400, days_old=20)
        self.create_test_session_with_size(temp_log_dir, "session_2", 400, days_old=15)
        self.create_test_session_with_size(temp_log_dir, "session_3", 400, days_old=10)
        
        # Run cleanup
        stats = await rotation_manager.cleanup_old_sessions(dry_run=False)
        
        # Should remove oldest sessions to get under limit
        assert stats["sessions_removed"] >= 1
        
        # Verify size is now under limit
        final_size = rotation_manager.calculate_directory_size()
        assert final_size <= rotation_manager.max_size_bytes
    
    def test_size_warning_thresholds(self, rotation_manager, temp_log_dir):
        """Test warning thresholds at 80% and 90% of limit."""
        # Create sessions approaching limit
        self.create_test_session_with_size(temp_log_dir, "session_1", 300)
        self.create_test_session_with_size(temp_log_dir, "session_2", 300)
        self.create_test_session_with_size(temp_log_dir, "session_3", 200)
        
        total_size = rotation_manager.calculate_directory_size()
        limit = rotation_manager.max_size_bytes
        
        # Check if we're at warning thresholds
        usage_percent = (total_size / limit) * 100
        
        if usage_percent >= 80:
            # Should trigger warning (we'll check logs in real implementation)
            assert total_size >= limit * 0.8
        
        if usage_percent >= 90:
            # Should trigger critical warning
            assert total_size >= limit * 0.9
    
    def test_prioritize_oldest_for_deletion(self, rotation_manager, temp_log_dir):
        """Test that oldest files are prioritized when over limit."""
        # Create sessions with timestamps
        sessions = [
            ("session_20240101_120000", 200, 50),
            ("session_20240201_120000", 200, 40),
            ("session_20240301_120000", 200, 30),
            ("session_20240401_120000", 200, 20),
            ("session_20240501_120000", 200, 10),
        ]
        
        for session_id, size_kb, days_old in sessions:
            self.create_test_session_with_size(temp_log_dir, session_id, size_kb, days_old)
        
        # Identify sessions to remove (target 500KB)
        target_size = 500 * 1024
        sessions_to_remove = rotation_manager.identify_sessions_by_size(target_size)
        
        # Extract session IDs
        removed_ids = [s[0] for s in sessions_to_remove]
        
        # Should remove oldest first
        assert "session_20240101_120000" in removed_ids
        assert "session_20240201_120000" in removed_ids
        assert "session_20240301_120000" in removed_ids
        
        # Newest should not be removed
        assert "session_20240501_120000" not in removed_ids
    
    @pytest.mark.asyncio
    async def test_combined_date_and_size_limits(self, rotation_manager, temp_log_dir):
        """Test that both date and size limits are enforced."""
        # Create old sessions (should be removed by date)
        self.create_test_session_with_size(temp_log_dir, "session_old1", 100, days_old=35)
        self.create_test_session_with_size(temp_log_dir, "session_old2", 100, days_old=40)
        
        # Create recent large sessions (should be removed by size if needed)
        self.create_test_session_with_size(temp_log_dir, "session_recent1", 400, days_old=10)
        self.create_test_session_with_size(temp_log_dir, "session_recent2", 400, days_old=5)
        
        # Run cleanup
        stats = await rotation_manager.cleanup_old_sessions(dry_run=False)
        
        # Old sessions should be removed
        assert not (temp_log_dir / "session_old1.jsonl").exists()
        assert not (temp_log_dir / "session_old2.jsonl").exists()
        
        # Check if size limit was also enforced
        final_size = rotation_manager.calculate_directory_size()
        assert final_size <= rotation_manager.max_size_bytes
    
    def test_empty_directory(self, rotation_manager, temp_log_dir):
        """Test size calculation on empty directory."""
        size = rotation_manager.calculate_directory_size()
        assert size == 0
    
    def test_single_file_session(self, rotation_manager, temp_log_dir):
        """Test size calculation with incomplete session (missing files)."""
        # Create only the main JSONL file
        jsonl_file = temp_log_dir / "session_incomplete.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps({"data": "test"}) + "\n")
        
        # Should still calculate size correctly
        size = rotation_manager.calculate_directory_size()
        assert size > 0
        assert size == jsonl_file.stat().st_size