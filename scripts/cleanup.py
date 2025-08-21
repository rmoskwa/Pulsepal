#!/usr/bin/env python3
"""
PulsePal Session Log Cleanup Utility

A command-line tool for managing and cleaning up PulsePal conversation logs.
Provides manual control over log rotation, archival, and cleanup operations.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pulsepal.log_manager import LogRotationManager


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def print_report(stats: Dict, verbose: bool = False):
    """Print a formatted cleanup report."""
    print("\n" + "=" * 60)
    print("PulsePal Log Cleanup Report")
    print("=" * 60)

    print(f"Sessions checked:  {stats.get('sessions_checked', 0)}")
    print(f"Sessions archived: {stats.get('sessions_archived', 0)}")
    print(f"Sessions removed:  {stats.get('sessions_removed', 0)}")
    print(f"Space freed:       {format_size(stats.get('space_freed_bytes', 0))}")

    if stats.get("errors"):
        print(f"\nErrors encountered: {len(stats['errors'])}")
        if verbose:
            for error in stats["errors"]:
                print(f"  - {error}")

    print("=" * 60)


def print_json_report(stats: Dict):
    """Print report in JSON format for scripting."""
    # Add formatted size for convenience
    stats["space_freed_formatted"] = format_size(stats.get("space_freed_bytes", 0))
    stats["timestamp"] = datetime.now().isoformat()
    print(json.dumps(stats, indent=2))


async def analyze_logs(
    log_dir: str = "conversationLogs",
    retention_days: int = 30,
    importance_threshold: float = 0.7,
) -> Dict:
    """Analyze log directory without making changes."""
    manager = LogRotationManager(
        log_dir=log_dir,
        retention_days=retention_days,
        importance_threshold=importance_threshold,
    )

    # Get current state
    current_size = manager.calculate_directory_size()
    old_sessions = manager.identify_old_sessions()

    # Analyze sessions
    analysis = {
        "total_size": current_size,
        "total_size_formatted": format_size(current_size),
        "retention_days": retention_days,
        "old_sessions": len(old_sessions),
        "sessions_by_importance": {"high": 0, "medium": 0, "low": 0},
        "space_to_free": 0,
        "sessions_to_archive": [],
        "sessions_to_remove": [],
    }

    # Score each old session
    for session_id in old_sessions:
        score = manager.score_session_importance(session_id)

        # Calculate session size
        session_size = 0
        for pattern in [
            f"{session_id}.jsonl",
            f"{session_id}.txt",
            f"{session_id}_searches.jsonl",
        ]:
            file_path = Path(log_dir) / pattern
            if file_path.exists():
                session_size += file_path.stat().st_size

        analysis["space_to_free"] += session_size

        if score >= importance_threshold:
            analysis["sessions_by_importance"]["high"] += 1
            analysis["sessions_to_archive"].append(
                {"id": session_id, "score": score, "size": session_size}
            )
        elif score >= 0.5:
            analysis["sessions_by_importance"]["medium"] += 1
            analysis["sessions_to_remove"].append(
                {"id": session_id, "score": score, "size": session_size}
            )
        else:
            analysis["sessions_by_importance"]["low"] += 1
            analysis["sessions_to_remove"].append(
                {"id": session_id, "score": score, "size": session_size}
            )

    return analysis


async def main():
    """Main entry point for the cleanup utility."""
    parser = argparse.ArgumentParser(
        description="PulsePal Session Log Cleanup Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be cleaned
  python scripts/cleanup.py --dry-run

  # Clean logs older than 30 days
  python scripts/cleanup.py

  # Clean logs older than 7 days
  python scripts/cleanup.py --older-than 7

  # Force cleanup without confirmation
  python scripts/cleanup.py --force

  # Archive important sessions before removal
  python scripts/cleanup.py --archive

  # Analyze logs without cleaning
  python scripts/cleanup.py --analyze

  # Set custom importance threshold
  python scripts/cleanup.py --importance-threshold 0.8

  # Clean and output JSON report
  python scripts/cleanup.py --json
        """,
    )

    # Operation modes
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate cleanup without making changes"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze logs without cleaning (show statistics)",
    )

    # Cleanup options
    parser.add_argument(
        "--older-than",
        type=int,
        default=30,
        metavar="DAYS",
        help="Remove sessions older than DAYS (default: 30)",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=1.0,
        metavar="GB",
        help="Maximum log directory size in GB (default: 1.0)",
    )
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=0.7,
        metavar="SCORE",
        help="Importance score threshold for archiving (0.0-1.0, default: 0.7)",
    )

    # Behavior flags
    parser.add_argument(
        "--archive",
        action="store_true",
        default=True,
        help="Archive important sessions before removal (default: enabled)",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip archiving, remove all old sessions",
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    # Output options
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress all output except errors"
    )

    # Directory options
    parser.add_argument(
        "--log-dir",
        type=str,
        default="conversationLogs",
        help="Log directory path (default: conversationLogs)",
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default="conversationLogs/archive",
        help="Archive directory path (default: conversationLogs/archive)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.importance_threshold < 0 or args.importance_threshold > 1:
        parser.error("Importance threshold must be between 0.0 and 1.0")

    # Handle no-archive flag
    archive_important = args.archive and not args.no_archive

    try:
        # Analyze mode
        if args.analyze:
            if not args.quiet:
                print("Analyzing PulsePal conversation logs...")

            analysis = await analyze_logs(
                log_dir=args.log_dir,
                retention_days=args.older_than,
                importance_threshold=args.importance_threshold,
            )

            if args.json:
                print(json.dumps(analysis, indent=2))
            elif not args.quiet:
                print("\n" + "=" * 60)
                print("PulsePal Log Analysis")
                print("=" * 60)
                print(f"Total size:          {analysis['total_size_formatted']}")
                print(f"Retention period:    {analysis['retention_days']} days")
                print(f"Old sessions:        {analysis['old_sessions']}")
                print(f"Space to free:       {format_size(analysis['space_to_free'])}")
                print("\nSessions by importance:")
                print(
                    f"  High (archive):    {analysis['sessions_by_importance']['high']}"
                )
                print(
                    f"  Medium:            {analysis['sessions_by_importance']['medium']}"
                )
                print(
                    f"  Low:               {analysis['sessions_by_importance']['low']}"
                )
                print("=" * 60)

            return

        # Create rotation manager
        manager = LogRotationManager(
            log_dir=args.log_dir,
            archive_dir=args.archive_dir,
            retention_days=args.older_than,
            max_size_gb=args.max_size,
            importance_threshold=args.importance_threshold,
        )

        # Check what would be done
        old_sessions = manager.identify_old_sessions()
        current_size = manager.calculate_directory_size()

        if not old_sessions and current_size <= manager.max_size_bytes:
            if not args.quiet:
                print(
                    "No cleanup needed. All sessions are within retention period and size limits."
                )
            return

        # Show what will be done
        if not args.quiet and not args.force:
            print(
                f"\nFound {len(old_sessions)} sessions older than {args.older_than} days"
            )
            print(f"Current log size: {format_size(current_size)}")

            if args.dry_run:
                print("\n[DRY RUN MODE - No changes will be made]")
            elif not args.force:
                response = input("\nProceed with cleanup? [y/N]: ")
                if response.lower() != "y":
                    print("Cleanup cancelled.")
                    return

        # Perform cleanup
        if not args.quiet and not args.dry_run:
            print("\nPerforming cleanup...")

        stats = await manager.cleanup_old_sessions(
            dry_run=args.dry_run, archive_important=archive_important
        )

        # Output results
        if args.json:
            print_json_report(stats)
        elif not args.quiet:
            print_report(stats, verbose=args.verbose)

            if args.dry_run:
                print("\n[DRY RUN COMPLETE - No actual changes were made]")

        # Exit with error code if there were errors
        if stats.get("errors"):
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nCleanup interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
