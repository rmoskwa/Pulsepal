"""
Monitoring and alerting for PulsePal log rotation system.

Provides health checks, metrics tracking, and alerting capabilities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .log_manager import LogRotationManager
from .settings import get_settings

logger = logging.getLogger(__name__)


class LogMonitor:
    """Monitor log rotation system health and metrics."""
    
    def __init__(self, rotation_manager: Optional[LogRotationManager] = None):
        """
        Initialize the log monitor.
        
        Args:
            rotation_manager: Optional LogRotationManager instance
        """
        self.settings = get_settings()
        
        if rotation_manager:
            self.rotation_manager = rotation_manager
        else:
            self.rotation_manager = LogRotationManager(
                retention_days=self.settings.log_retention_days,
                max_size_gb=self.settings.max_log_size_gb,
            )
        
        self.metrics_file = Path("conversationLogs") / ".metrics.json"
        self.alerts: List[Dict] = []
        self._last_check = datetime.now()
    
    def check_health(self) -> Dict:
        """
        Perform a health check on the log system.
        
        Returns:
            Dictionary with health status and any issues found
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "issues": [],
        }
        
        # Check directory exists and is writable
        try:
            log_dir = self.rotation_manager.log_dir
            if not log_dir.exists():
                health["checks"]["directory_exists"] = False
                health["issues"].append("Log directory does not exist")
                health["status"] = "unhealthy"
            else:
                health["checks"]["directory_exists"] = True
                
                # Check write permissions
                test_file = log_dir / ".write_test"
                try:
                    test_file.touch()
                    test_file.unlink()
                    health["checks"]["directory_writable"] = True
                except Exception as e:
                    health["checks"]["directory_writable"] = False
                    health["issues"].append(f"Cannot write to log directory: {e}")
                    health["status"] = "unhealthy"
        except Exception as e:
            health["checks"]["directory_access"] = False
            health["issues"].append(f"Cannot access log directory: {e}")
            health["status"] = "unhealthy"
        
        # Check disk space
        try:
            current_size = self.rotation_manager.calculate_directory_size()
            max_size = self.rotation_manager.max_size_bytes
            usage_percent = (current_size / max_size) * 100 if max_size > 0 else 0
            
            health["checks"]["disk_usage"] = {
                "current_bytes": current_size,
                "max_bytes": max_size,
                "usage_percent": usage_percent,
            }
            
            if usage_percent >= 95:
                health["issues"].append(f"Critical: Disk usage at {usage_percent:.1f}%")
                health["status"] = "critical"
            elif usage_percent >= 90:
                health["issues"].append(f"Warning: Disk usage at {usage_percent:.1f}%")
                if health["status"] == "healthy":
                    health["status"] = "warning"
        except Exception as e:
            health["checks"]["disk_usage"] = None
            health["issues"].append(f"Cannot calculate disk usage: {e}")
        
        # Check archive directory
        try:
            archive_dir = self.rotation_manager.archive_dir
            if archive_dir.exists():
                health["checks"]["archive_directory"] = True
            else:
                health["checks"]["archive_directory"] = False
                health["issues"].append("Archive directory does not exist")
        except Exception as e:
            health["checks"]["archive_directory"] = None
            health["issues"].append(f"Cannot check archive directory: {e}")
        
        # Check for old sessions
        try:
            old_sessions = self.rotation_manager.identify_old_sessions()
            health["checks"]["old_sessions"] = len(old_sessions)
            
            if len(old_sessions) > 50:
                health["issues"].append(f"Many old sessions pending cleanup: {len(old_sessions)}")
                if health["status"] == "healthy":
                    health["status"] = "warning"
        except Exception as e:
            health["checks"]["old_sessions"] = None
            health["issues"].append(f"Cannot check old sessions: {e}")
        
        return health
    
    def get_metrics(self) -> Dict:
        """
        Get current metrics for the log system.
        
        Returns:
            Dictionary with various metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "sessions": {},
            "storage": {},
            "rotation": {},
        }
        
        try:
            # Count sessions
            all_sessions = list(self.rotation_manager.log_dir.glob("session_*.jsonl"))
            all_sessions = [s for s in all_sessions if "_searches" not in s.name]
            
            metrics["sessions"]["total"] = len(all_sessions)
            
            # Categorize by age
            now = datetime.now()
            recent = 0
            old = 0
            very_old = 0
            
            for session_file in all_sessions:
                timestamp = self.rotation_manager._parse_session_timestamp(session_file.name)
                if timestamp:
                    age_days = (now - timestamp).days
                    if age_days <= 7:
                        recent += 1
                    elif age_days <= self.rotation_manager.retention_days:
                        old += 1
                    else:
                        very_old += 1
            
            metrics["sessions"]["recent_7d"] = recent
            metrics["sessions"]["old_in_retention"] = old
            metrics["sessions"]["pending_cleanup"] = very_old
            
            # Storage metrics
            current_size = self.rotation_manager.calculate_directory_size()
            metrics["storage"]["current_bytes"] = current_size
            metrics["storage"]["current_mb"] = current_size / (1024**2)
            metrics["storage"]["max_bytes"] = self.rotation_manager.max_size_bytes
            metrics["storage"]["usage_percent"] = (current_size / self.rotation_manager.max_size_bytes * 100) if self.rotation_manager.max_size_bytes > 0 else 0
            
            # Archive metrics
            if self.rotation_manager.archive_dir.exists():
                archive_files = list(self.rotation_manager.archive_dir.rglob("*.tar.gz"))
                metrics["storage"]["archived_sessions"] = len(archive_files)
                metrics["storage"]["archive_size_bytes"] = sum(f.stat().st_size for f in archive_files)
            
            # Rotation settings
            metrics["rotation"]["retention_days"] = self.rotation_manager.retention_days
            metrics["rotation"]["importance_threshold"] = self.rotation_manager.importance_threshold
            metrics["rotation"]["last_check"] = self._last_check.isoformat()
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def save_metrics(self):
        """Save current metrics to file."""
        try:
            metrics = self.get_metrics()
            
            # Load existing metrics history
            history = []
            if self.metrics_file.exists():
                try:
                    with open(self.metrics_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            history = data
                except Exception:
                    pass
            
            # Add current metrics
            history.append(metrics)
            
            # Keep only last 30 days of metrics (1 per hour = 720 entries)
            if len(history) > 720:
                history = history[-720:]
            
            # Save updated history
            with open(self.metrics_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def check_alerts(self) -> List[Dict]:
        """
        Check for conditions that should trigger alerts.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        now = datetime.now()
        
        # Check disk usage
        try:
            current_size = self.rotation_manager.calculate_directory_size()
            max_size = self.rotation_manager.max_size_bytes
            usage_percent = (current_size / max_size) * 100 if max_size > 0 else 0
            
            if usage_percent >= 95:
                alerts.append({
                    "level": "critical",
                    "type": "disk_usage",
                    "message": f"Log directory at {usage_percent:.1f}% capacity",
                    "timestamp": now.isoformat(),
                    "value": usage_percent,
                })
            elif usage_percent >= 90:
                alerts.append({
                    "level": "warning",
                    "type": "disk_usage",
                    "message": f"Log directory at {usage_percent:.1f}% capacity",
                    "timestamp": now.isoformat(),
                    "value": usage_percent,
                })
        except Exception as e:
            alerts.append({
                "level": "error",
                "type": "monitoring_error",
                "message": f"Cannot check disk usage: {e}",
                "timestamp": now.isoformat(),
            })
        
        # Check for too many old sessions
        try:
            old_sessions = self.rotation_manager.identify_old_sessions()
            if len(old_sessions) > 100:
                alerts.append({
                    "level": "warning",
                    "type": "pending_cleanup",
                    "message": f"{len(old_sessions)} sessions pending cleanup",
                    "timestamp": now.isoformat(),
                    "value": len(old_sessions),
                })
        except Exception as e:
            alerts.append({
                "level": "error",
                "type": "monitoring_error",
                "message": f"Cannot check old sessions: {e}",
                "timestamp": now.isoformat(),
            })
        
        self.alerts = alerts
        return alerts
    
    def get_dashboard_data(self) -> Dict:
        """
        Get comprehensive data for a monitoring dashboard.
        
        Returns:
            Dictionary with all monitoring data
        """
        return {
            "health": self.check_health(),
            "metrics": self.get_metrics(),
            "alerts": self.check_alerts(),
            "timestamp": datetime.now().isoformat(),
        }
    
    def log_rotation_event(self, event_type: str, details: Dict):
        """
        Log a rotation event for audit trail.
        
        Args:
            event_type: Type of event (cleanup, archive, error, etc.)
            details: Event details
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details,
        }
        
        # Log to file
        event_log_file = Path("conversationLogs") / ".rotation_events.jsonl"
        try:
            with open(event_log_file, 'a') as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to log rotation event: {e}")
        
        # Also log to standard logger
        logger.info(f"Rotation event: {event_type} - {details}")


# Global monitor instance
_monitor: Optional[LogMonitor] = None


def get_monitor() -> LogMonitor:
    """Get or create the global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = LogMonitor()
    return _monitor