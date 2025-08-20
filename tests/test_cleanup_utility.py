"""
Tests for cleanup utility - Task 5: Build cleanup utility.
"""

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


class TestCleanupUtility:
    """Test cleanup utility CLI functionality (Task 5)."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            log_dir.mkdir()
            yield log_dir

    @pytest.fixture
    def cleanup_script(self):
        """Path to cleanup script."""
        return Path(__file__).parent.parent / "scripts" / "cleanup.py"

    def create_test_sessions(self, log_dir: Path):
        """Create test session files with various ages."""
        sessions = [
            ("session_old1", 40, 100),  # 40 days old, 100KB
            ("session_old2", 35, 200),  # 35 days old, 200KB
            ("session_recent1", 10, 150),  # 10 days old, 150KB
            ("session_recent2", 5, 100),  # 5 days old, 100KB
        ]

        for session_id, days_old, size_kb in sessions:
            # Create files
            jsonl_file = log_dir / f"{session_id}.jsonl"
            txt_file = log_dir / f"{session_id}.txt"
            search_file = log_dir / f"{session_id}_searches.jsonl"

            # Write content to achieve target size
            content_size = size_kb * 1024 // 3

            with open(jsonl_file, "w") as f:
                f.write(json.dumps({"data": "x" * content_size}) + "\n")

            with open(txt_file, "w") as f:
                f.write("x" * content_size)

            with open(search_file, "w") as f:
                f.write(json.dumps({"query": "x" * (content_size - 20)}) + "\n")

            # Set modification time
            if days_old > 0:
                old_time = (datetime.now() - timedelta(days=days_old)).timestamp()
                import os

                os.utime(jsonl_file, (old_time, old_time))
                os.utime(txt_file, (old_time, old_time))
                os.utime(search_file, (old_time, old_time))

    def test_cleanup_script_exists(self, cleanup_script):
        """Test that cleanup script exists."""
        assert cleanup_script.exists()
        assert cleanup_script.suffix == ".py"

    def test_dry_run_flag(self, cleanup_script, temp_log_dir):
        """Test --dry-run flag functionality."""
        self.create_test_sessions(temp_log_dir)

        # Run cleanup with dry-run
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--dry-run",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "[DRY RUN]" in result.stdout

        # Verify no files were deleted
        assert len(list(temp_log_dir.glob("session_*.jsonl"))) == 4

    def test_force_flag(self, cleanup_script, temp_log_dir):
        """Test --force flag skips confirmation."""
        self.create_test_sessions(temp_log_dir)

        # Run cleanup with force flag
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--force",
                "--log-dir",
                str(temp_log_dir),
                "--no-archive",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Should not have confirmation prompt in output
        assert "Proceed with cleanup?" not in result.stdout

    def test_archive_flag(self, cleanup_script, temp_log_dir):
        """Test --archive flag functionality."""
        self.create_test_sessions(temp_log_dir)

        # Run cleanup with archive flag
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--force",
                "--archive",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Check that archive directory was created
        archive_dir = temp_log_dir / "archive"
        assert archive_dir.exists()

    def test_no_archive_flag(self, cleanup_script, temp_log_dir):
        """Test --no-archive flag skips archiving."""
        self.create_test_sessions(temp_log_dir)

        # Run cleanup without archiving
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--force",
                "--no-archive",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Archive directory should not be created
        archive_dir = temp_log_dir / "archive"
        assert not archive_dir.exists() or len(list(archive_dir.rglob("*.tar.gz"))) == 0

    def test_older_than_parameter(self, cleanup_script, temp_log_dir):
        """Test --older-than parameter for custom retention."""
        self.create_test_sessions(temp_log_dir)

        # Run cleanup with custom retention (37 days)
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--force",
                "--no-archive",
                "--older-than",
                "37",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Only session_old1 (40 days) should be deleted
        remaining_sessions = list(temp_log_dir.glob("session_*.jsonl"))
        remaining_sessions = [
            f for f in remaining_sessions if "_searches" not in f.name
        ]
        assert len(remaining_sessions) == 3  # old2, recent1, recent2 remain

    def test_json_output(self, cleanup_script, temp_log_dir):
        """Test --json flag for JSON output."""
        self.create_test_sessions(temp_log_dir)

        # Run cleanup with JSON output
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--dry-run",
                "--json",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Parse JSON output
        output_lines = result.stdout.strip().split("\n")
        # Find the JSON output (might have some log messages before it)
        json_output = None
        for i, line in enumerate(output_lines):
            if line.strip().startswith("{"):
                # Found start of JSON, concatenate remaining lines
                json_output = "\n".join(output_lines[i:])
                break

        assert json_output is not None
        data = json.loads(json_output)

        assert "sessions_checked" in data
        assert "sessions_removed" in data
        assert "space_freed_bytes" in data

    def test_verbose_logging(self, cleanup_script, temp_log_dir):
        """Test --verbose flag enables detailed logging."""
        self.create_test_sessions(temp_log_dir)

        # Run with verbose flag
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--dry-run",
                "--verbose",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Verbose mode should show more detailed output
        assert len(result.stdout) > 0

    def test_custom_directories(self, cleanup_script):
        """Test --log-dir and --archive-dir parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_log_dir = Path(tmpdir) / "custom_logs"
            custom_archive_dir = Path(tmpdir) / "custom_archive"
            custom_log_dir.mkdir()

            self.create_test_sessions(custom_log_dir)

            # Run with custom directories
            result = subprocess.run(
                [
                    sys.executable,
                    str(cleanup_script),
                    "--force",
                    "--archive",
                    "--log-dir",
                    str(custom_log_dir),
                    "--archive-dir",
                    str(custom_archive_dir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert custom_archive_dir.exists()

    def test_max_size_parameter(self, cleanup_script, temp_log_dir):
        """Test --max-size parameter for size-based cleanup."""
        # Create large sessions
        for i in range(5):
            session_id = f"session_large_{i}"
            jsonl_file = temp_log_dir / f"{session_id}.jsonl"

            # Create 200KB file
            with open(jsonl_file, "w") as f:
                f.write(json.dumps({"data": "x" * 200000}) + "\n")

            # Set different ages
            old_time = (datetime.now() - timedelta(days=i * 5)).timestamp()
            import os

            os.utime(jsonl_file, (old_time, old_time))

        # Run with size limit (0.0005 GB = ~500KB)
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--force",
                "--no-archive",
                "--max-size",
                "0.0005",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Should have removed oldest files to get under limit
        remaining = list(temp_log_dir.glob("session_*.jsonl"))
        assert len(remaining) < 5

    def test_importance_threshold(self, cleanup_script, temp_log_dir):
        """Test --importance-threshold parameter."""
        # Create sessions with errors (high importance)
        session_id = "session_important"
        jsonl_file = temp_log_dir / f"{session_id}.jsonl"

        with open(jsonl_file, "w") as f:
            for i in range(25):  # Long conversation
                msg = {"content": "Error occurred" if i == 10 else f"Message {i}"}
                f.write(json.dumps(msg) + "\n")

        # Make it old
        old_time = (datetime.now() - timedelta(days=40)).timestamp()
        import os

        os.utime(jsonl_file, (old_time, old_time))

        # Run with low importance threshold
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--force",
                "--archive",
                "--importance-threshold",
                "0.3",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Should archive the important session
        assert "archived" in result.stdout.lower()

    def test_cleanup_report_format(self, cleanup_script, temp_log_dir):
        """Test that cleanup report is properly formatted."""
        self.create_test_sessions(temp_log_dir)

        # Run cleanup
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--dry-run",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Check report format
        assert "Session Log Cleanup Report" in result.stdout
        assert "Sessions checked:" in result.stdout
        assert "Sessions archived:" in result.stdout
        assert "Sessions removed:" in result.stdout
        assert "Space freed:" in result.stdout

    def test_no_cleanup_needed(self, cleanup_script, temp_log_dir):
        """Test behavior when no cleanup is needed."""
        # Create only recent sessions
        for i in range(3):
            session_id = f"session_recent_{i}"
            jsonl_file = temp_log_dir / f"{session_id}.jsonl"
            jsonl_file.write_text(json.dumps({"data": "test"}) + "\n")

        # Run cleanup
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--force",
                "--log-dir",
                str(temp_log_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "No cleanup needed" in result.stdout

    def test_error_handling(self, cleanup_script):
        """Test error handling for invalid parameters."""
        # Test with non-existent directory
        result = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--log-dir",
                "/nonexistent/directory",
            ],
            capture_output=True,
            text=True,
        )

        # Should handle gracefully (might still return 0 if directory is created)
        assert result.returncode in [0, 1]

    def test_help_message(self, cleanup_script):
        """Test --help displays usage information."""
        result = subprocess.run(
            [sys.executable, str(cleanup_script), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "PulsePal Session Log Cleanup Utility" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--force" in result.stdout
        assert "--archive" in result.stdout
        assert "Examples:" in result.stdout
