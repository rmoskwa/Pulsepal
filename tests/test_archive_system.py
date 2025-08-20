"""
Tests for log rotation manager - Task 4: Archive system.
"""

import asyncio
import json
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pulsepal.log_manager import LogRotationManager


class TestArchiveSystem:
    """Test archive system functionality (Task 4)."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            log_dir = base_dir / "logs"
            archive_dir = base_dir / "archive"
            log_dir.mkdir()
            archive_dir.mkdir()
            yield log_dir, archive_dir
    
    @pytest.fixture
    def rotation_manager(self, temp_dirs):
        """Create LogRotationManager instance with test directories."""
        log_dir, archive_dir = temp_dirs
        return LogRotationManager(
            log_dir=str(log_dir),
            archive_dir=str(archive_dir),
            retention_days=30,
            max_size_gb=1.0,
        )
    
    def create_test_session(self, log_dir: Path, session_id: str, content: str = "test", days_old: int = 0):
        """Create test session files."""
        # Create main JSONL file
        jsonl_file = log_dir / f"{session_id}.jsonl"
        with open(jsonl_file, 'w') as f:
            for i in range(5):
                f.write(json.dumps({
                    "message": f"{content} message {i}",
                    "timestamp": datetime.now().isoformat()
                }) + "\n")
        
        # Create text summary file
        txt_file = log_dir / f"{session_id}.txt"
        with open(txt_file, 'w') as f:
            f.write(f"{content} session summary\n")
        
        # Create searches file
        search_file = log_dir / f"{session_id}_searches.jsonl"
        with open(search_file, 'w') as f:
            f.write(json.dumps({"query": f"{content} search"}) + "\n")
        
        # Set modification time if needed
        if days_old > 0:
            old_time = (datetime.now() - timedelta(days=days_old)).timestamp()
            import os
            os.utime(jsonl_file, (old_time, old_time))
            os.utime(txt_file, (old_time, old_time))
            os.utime(search_file, (old_time, old_time))
        
        return jsonl_file, txt_file, search_file
    
    def test_archive_directory_structure(self, rotation_manager):
        """Test that archive directory structure is created properly."""
        assert rotation_manager.archive_dir.exists()
        assert rotation_manager.archive_dir.is_dir()
    
    @pytest.mark.asyncio
    async def test_archive_session_compression(self, rotation_manager, temp_dirs):
        """Test that sessions are compressed when archived."""
        log_dir, archive_dir = temp_dirs
        
        # Create a test session
        session_id = "session_20240115_120000"
        files = self.create_test_session(log_dir, session_id, "compress_test")
        
        # Archive the session
        success = await rotation_manager.archive_session(session_id)
        assert success
        
        # Check that archive was created in year-month subdirectory
        archive_subdir = archive_dir / "2024-01"
        assert archive_subdir.exists()
        
        # Check that tar.gz file was created
        archive_file = archive_subdir / f"{session_id}.tar.gz"
        assert archive_file.exists()
        
        # Verify it's a valid tar.gz file
        with tarfile.open(archive_file, "r:gz") as tar:
            members = tar.getnames()
            assert f"{session_id}.jsonl" in members
            assert f"{session_id}.txt" in members
            assert f"{session_id}_searches.jsonl" in members
    
    @pytest.mark.asyncio
    async def test_archive_metadata_creation(self, rotation_manager, temp_dirs):
        """Test that archive metadata is created with session info."""
        log_dir, archive_dir = temp_dirs
        
        # Create a test session
        session_id = "session_20240215_140000"
        self.create_test_session(log_dir, session_id)
        
        # Archive the session
        await rotation_manager.archive_session(session_id)
        
        # Check metadata file
        metadata_file = archive_dir / "2024-02" / f"{session_id}_metadata.json"
        assert metadata_file.exists()
        
        # Verify metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata["session_id"] == session_id
        assert "archived_at" in metadata
        assert "importance_score" in metadata
        assert "files" in metadata
        assert len(metadata["files"]) == 3
        assert "original_size" in metadata
        assert "compressed_size" in metadata
        assert metadata["compressed_size"] < metadata["original_size"]  # Compression worked
    
    @pytest.mark.asyncio
    async def test_restore_from_archive(self, rotation_manager, temp_dirs):
        """Test restoring a session from archive."""
        log_dir, archive_dir = temp_dirs
        
        # Create and archive a session
        session_id = "session_20240315_100000"
        original_files = self.create_test_session(log_dir, session_id, "restore_test")
        
        # Read original content for comparison
        with open(original_files[0], 'r') as f:
            original_jsonl = f.read()
        
        # Archive the session
        await rotation_manager.archive_session(session_id)
        
        # Delete the original files
        for file_path in original_files:
            file_path.unlink()
        
        # Verify files are gone
        assert not (log_dir / f"{session_id}.jsonl").exists()
        
        # Restore from archive
        success = await rotation_manager.restore_from_archive(session_id)
        assert success
        
        # Verify files are restored
        assert (log_dir / f"{session_id}.jsonl").exists()
        assert (log_dir / f"{session_id}.txt").exists()
        assert (log_dir / f"{session_id}_searches.jsonl").exists()
        
        # Verify content matches
        with open(log_dir / f"{session_id}.jsonl", 'r') as f:
            restored_jsonl = f.read()
        assert restored_jsonl == original_jsonl
    
    @pytest.mark.asyncio
    async def test_archive_year_month_subdirectories(self, rotation_manager, temp_dirs):
        """Test that archives are organized by year-month."""
        log_dir, archive_dir = temp_dirs
        
        # Create sessions from different months
        sessions = [
            "session_20240115_120000",  # January 2024
            "session_20240215_120000",  # February 2024
            "session_20240315_120000",  # March 2024
            "session_20230615_120000",  # June 2023
        ]
        
        for session_id in sessions:
            self.create_test_session(log_dir, session_id)
            await rotation_manager.archive_session(session_id)
        
        # Check subdirectory structure
        assert (archive_dir / "2024-01").exists()
        assert (archive_dir / "2024-02").exists()
        assert (archive_dir / "2024-03").exists()
        assert (archive_dir / "2023-06").exists()
        
        # Verify files are in correct subdirectories
        assert (archive_dir / "2024-01" / "session_20240115_120000.tar.gz").exists()
        assert (archive_dir / "2024-02" / "session_20240215_120000.tar.gz").exists()
        assert (archive_dir / "2024-03" / "session_20240315_120000.tar.gz").exists()
        assert (archive_dir / "2023-06" / "session_20230615_120000.tar.gz").exists()
    
    @pytest.mark.asyncio
    async def test_archive_hex_id_sessions(self, rotation_manager, temp_dirs):
        """Test archiving sessions with hex ID format (no timestamp)."""
        log_dir, archive_dir = temp_dirs
        
        # Create session with hex ID
        session_id = "session_abc123def"
        self.create_test_session(log_dir, session_id)
        
        # Archive the session
        success = await rotation_manager.archive_session(session_id)
        assert success
        
        # Should go to "unknown" subdirectory for sessions without timestamps
        unknown_dir = archive_dir / "unknown"
        assert unknown_dir.exists()
        assert (unknown_dir / f"{session_id}.tar.gz").exists()
    
    @pytest.mark.asyncio
    async def test_archive_format_documentation(self, rotation_manager, temp_dirs):
        """Test that archive format is properly documented in metadata."""
        log_dir, archive_dir = temp_dirs
        
        session_id = "session_test_format"
        self.create_test_session(log_dir, session_id)
        
        await rotation_manager.archive_session(session_id)
        
        # Find the metadata file
        metadata_files = list(archive_dir.rglob(f"{session_id}_metadata.json"))
        assert len(metadata_files) == 1
        
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        
        # Metadata should document the archive format
        assert "session_id" in metadata
        assert "archived_at" in metadata
        assert "files" in metadata
        assert isinstance(metadata["files"], list)
        assert all(isinstance(f, str) for f in metadata["files"])
    
    @pytest.mark.asyncio
    async def test_concurrent_archive_operations(self, rotation_manager, temp_dirs):
        """Test thread-safe concurrent archive operations."""
        log_dir, _ = temp_dirs
        
        # Create multiple sessions
        session_ids = [f"session_concurrent_{i}" for i in range(5)]
        for session_id in session_ids:
            self.create_test_session(log_dir, session_id)
        
        # Archive them concurrently
        tasks = [rotation_manager.archive_session(sid) for sid in session_ids]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)
        
        # Verify all archives were created
        for session_id in session_ids:
            archive_files = list(rotation_manager.archive_dir.rglob(f"{session_id}.tar.gz"))
            assert len(archive_files) == 1
    
    @pytest.mark.asyncio
    async def test_archive_missing_files(self, rotation_manager, temp_dirs):
        """Test archiving when some session files are missing."""
        log_dir, _ = temp_dirs
        
        session_id = "session_incomplete"
        
        # Create only the main JSONL file (missing .txt and _searches.jsonl)
        jsonl_file = log_dir / f"{session_id}.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps({"message": "test"}) + "\n")
        
        # Should still archive what exists
        success = await rotation_manager.archive_session(session_id)
        assert success
        
        # Check that archive contains only the existing file
        archive_files = list(rotation_manager.archive_dir.rglob(f"{session_id}.tar.gz"))
        assert len(archive_files) == 1
        
        with tarfile.open(archive_files[0], "r:gz") as tar:
            members = tar.getnames()
            assert f"{session_id}.jsonl" in members
            assert f"{session_id}.txt" not in members
            assert f"{session_id}_searches.jsonl" not in members
    
    @pytest.mark.asyncio
    async def test_archive_nonexistent_session(self, rotation_manager):
        """Test archiving a session that doesn't exist."""
        success = await rotation_manager.archive_session("nonexistent_session")
        assert not success
    
    @pytest.mark.asyncio
    async def test_restore_nonexistent_archive(self, rotation_manager):
        """Test restoring a session that wasn't archived."""
        success = await rotation_manager.restore_from_archive("nonexistent_session")
        assert not success
    
    @pytest.mark.asyncio
    async def test_restore_to_custom_directory(self, rotation_manager, temp_dirs):
        """Test restoring a session to a custom directory."""
        log_dir, _ = temp_dirs
        
        # Create and archive a session
        session_id = "session_custom_restore"
        self.create_test_session(log_dir, session_id)
        await rotation_manager.archive_session(session_id)
        
        # Delete original files
        for file_path in log_dir.glob(f"{session_id}*"):
            file_path.unlink()
        
        # Create custom restore directory
        restore_dir = log_dir.parent / "restored"
        restore_dir.mkdir()
        
        # Restore to custom directory
        success = await rotation_manager.restore_from_archive(session_id, str(restore_dir))
        assert success
        
        # Verify files are in custom directory
        assert (restore_dir / f"{session_id}.jsonl").exists()
        assert (restore_dir / f"{session_id}.txt").exists()
        assert (restore_dir / f"{session_id}_searches.jsonl").exists()
    
    @pytest.mark.asyncio
    async def test_archive_compression_ratio(self, rotation_manager, temp_dirs):
        """Test that compression actually reduces file size."""
        log_dir, _ = temp_dirs
        
        session_id = "session_compression"
        
        # Create session with repetitive content (compresses well)
        jsonl_file = log_dir / f"{session_id}.jsonl"
        with open(jsonl_file, 'w') as f:
            for i in range(100):
                f.write(json.dumps({"message": "This is a repetitive message " * 10}) + "\n")
        
        txt_file = log_dir / f"{session_id}.txt"
        with open(txt_file, 'w') as f:
            f.write("Repetitive content " * 1000)
        
        (log_dir / f"{session_id}_searches.jsonl").touch()
        
        # Calculate original size
        original_size = sum(f.stat().st_size for f in log_dir.glob(f"{session_id}*"))
        
        # Archive the session
        await rotation_manager.archive_session(session_id)
        
        # Find archive and check compression
        archive_files = list(rotation_manager.archive_dir.rglob(f"{session_id}.tar.gz"))
        assert len(archive_files) == 1
        
        compressed_size = archive_files[0].stat().st_size
        
        # Compression should significantly reduce size
        compression_ratio = compressed_size / original_size
        assert compression_ratio < 0.5  # At least 50% compression for repetitive content