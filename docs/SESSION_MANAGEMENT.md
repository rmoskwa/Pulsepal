# PulsePal Session Management

## Overview

PulsePal implements sophisticated session management to maintain conversation context, track user preferences, and provide a seamless experience across interactions. Sessions are automatically created and managed to optimize both performance and user experience.

## Session Architecture

### Core Components

1. **SessionManager** (`dependencies.py`)
   - Creates and manages session lifecycles
   - Handles session rotation and cleanup
   - Maintains session persistence

2. **ConversationContext**
   - Stores conversation history
   - Tracks language preferences (MATLAB/Python)
   - Maintains code examples within session
   - Records session metadata

3. **Session Storage**
   - File-based storage in `sessions/` directory
   - JSON format for easy debugging
   - Automatic archival of old sessions

## Session Lifecycle

### Session Creation

Sessions are created automatically when:
- A new user starts a conversation (CLI or web)
- No session ID is provided
- An expired session ID is used

```python
# Automatic session creation
from pulsepal.main_agent import run_pulsepal
session_id, response = await run_pulsepal("Your query here")

# Manual session creation
from pulsepal.dependencies import SessionManager
session_manager = SessionManager()
session_id = session_manager.create_session()
```

### Session Persistence

Sessions persist for:
- **Default Duration**: 24 hours (configurable)
- **Maximum History**: 100 messages (configurable)
- **Inactive Timeout**: 2 hours of inactivity

### Session Rotation Policy

The rotation policy ensures optimal performance:

1. **Active Sessions** (< 24 hours old)
   - Kept in `sessions/active/`
   - Fast access for ongoing conversations
   - Automatically loaded on request

2. **Archived Sessions** (> 24 hours old)
   - Moved to `sessions/archive/`
   - Compressed for storage efficiency
   - Available for historical reference

3. **Cleanup Process**
   - Runs automatically every hour
   - Archives expired sessions
   - Removes sessions older than 7 days (configurable)

## Configuration Options

### Environment Variables

Configure session behavior in `.env`:

```bash
# Session Duration
MAX_SESSION_DURATION_HOURS=24      # How long sessions remain active
SESSION_INACTIVE_TIMEOUT_HOURS=2   # Timeout for inactive sessions

# Session Limits
MAX_CONVERSATION_HISTORY=100       # Maximum messages per session
MAX_ACTIVE_SESSIONS=1000          # Maximum concurrent active sessions

# Archive Settings
SESSION_ARCHIVE_DAYS=7             # Days to keep archived sessions
SESSION_COMPRESSION=true           # Compress archived sessions

# Session Storage
SESSION_STORAGE_PATH=./sessions    # Where to store session files
```

### Programmatic Configuration

```python
from pulsepal.settings import Settings

settings = Settings(
    max_session_duration_hours=48,
    max_conversation_history=200,
    session_storage_path="/custom/path"
)
```

## Archive Process

### Automatic Archival

Sessions are automatically archived when:
- They exceed the maximum duration
- The cleanup process runs (hourly)
- Manual cleanup is triggered

### Archive Structure

```
sessions/
├── active/                 # Current sessions
│   ├── session_abc123.json
│   └── session_def456.json
├── archive/               # Archived sessions
│   ├── 2025-01-18/       # Organized by date
│   │   ├── session_old1.json.gz
│   │   └── session_old2.json.gz
│   └── 2025-01-17/
└── metadata.json          # Session index
```

### Manual Archive Management

```bash
# Archive all expired sessions
python -c "from pulsepal.dependencies import SessionManager; SessionManager().archive_expired_sessions()"

# Archive specific session
python -c "from pulsepal.dependencies import SessionManager; SessionManager().archive_session('session_id')"

# List archived sessions
python -c "from pulsepal.dependencies import SessionManager; SessionManager().list_archived_sessions()"
```

## CLI Commands for Cleanup

### Basic Cleanup Commands

```bash
# Clean up old sessions (removes > 7 days)
python scripts/cleanup_sessions.py

# Clean up with custom retention
python scripts/cleanup_sessions.py --days 30

# Dry run (show what would be deleted)
python scripts/cleanup_sessions.py --dry-run

# Force cleanup of all sessions
python scripts/cleanup_sessions.py --force --all
```

### Session Management CLI

```bash
# List active sessions
python -m pulsepal.cli sessions list

# Show session details
python -m pulsepal.cli sessions show <session_id>

# Delete specific session
python -m pulsepal.cli sessions delete <session_id>

# Export session history
python -m pulsepal.cli sessions export <session_id> --output history.json

# Import session (restore)
python -m pulsepal.cli sessions import history.json
```

### Scheduled Cleanup

For production environments, schedule cleanup with cron:

```bash
# Add to crontab
0 * * * * python /path/to/pulsepal/scripts/cleanup_sessions.py

# Or use systemd timer
[Unit]
Description=PulsePal Session Cleanup

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

## Monitoring Guidelines

### Health Metrics

Monitor these session metrics:

1. **Active Session Count**
   ```python
   session_manager.get_active_session_count()
   ```

2. **Session Age Distribution**
   ```python
   session_manager.get_session_age_stats()
   ```

3. **Storage Usage**
   ```python
   session_manager.get_storage_usage()
   ```

4. **Session Activity**
   ```python
   session_manager.get_activity_metrics()
   ```

### Monitoring Dashboard

```python
# Simple monitoring script
from pulsepal.dependencies import SessionManager

sm = SessionManager()
stats = sm.get_session_stats()

print(f"Active Sessions: {stats['active_count']}")
print(f"Archived Sessions: {stats['archived_count']}")
print(f"Storage Used: {stats['storage_mb']} MB")
print(f"Avg Session Age: {stats['avg_age_hours']} hours")
print(f"Oldest Session: {stats['oldest_session_id']}")
```

### Alerts

Set up alerts for:
- High active session count (> 900)
- Storage usage exceeding limits
- Failed cleanup processes
- Corrupted session files

## Recovery Procedures

### Session Recovery

If sessions are corrupted or lost:

1. **Check Backups**
   ```bash
   ls sessions/archive/backups/
   ```

2. **Restore from Archive**
   ```bash
   python scripts/restore_session.py --session-id <id> --date 2025-01-18
   ```

3. **Rebuild from Logs**
   ```bash
   python scripts/rebuild_sessions.py --from-logs conversationLogs/
   ```

### Common Issues and Solutions

1. **Session Not Found**
   - Check if archived: `ls sessions/archive/*/session_*.json.gz`
   - Restore if needed: `gunzip < session.json.gz > sessions/active/session.json`

2. **Session Corruption**
   - Validate JSON: `python -m json.tool session.json`
   - Remove corrupted: `rm sessions/active/corrupted_session.json`
   - Recreate: User starts new conversation

3. **Storage Full**
   - Run immediate cleanup: `python scripts/cleanup_sessions.py --force`
   - Reduce retention: `--days 3`
   - Move archives: `mv sessions/archive /backup/location/`

4. **Performance Issues**
   - Check session count: Too many active sessions
   - Archive old sessions: `python scripts/cleanup_sessions.py`
   - Optimize storage: Enable compression

## Best Practices

### Session Management

1. **Regular Cleanup** - Schedule hourly cleanup in production
2. **Monitor Storage** - Set up disk usage alerts
3. **Backup Archives** - Keep backups of important sessions
4. **Test Recovery** - Regularly test recovery procedures
5. **Document Sessions** - Keep metadata about important sessions

### Performance Optimization

1. **Limit History** - Keep conversation history reasonable (100-200 messages)
2. **Archive Promptly** - Don't keep old sessions active
3. **Compress Archives** - Enable compression for archived sessions
4. **Clean Storage** - Remove very old archives periodically

### Security Considerations

1. **Sanitize Sessions** - Remove sensitive data before archiving
2. **Encrypt Archives** - Consider encrypting archived sessions
3. **Access Control** - Limit access to session storage
4. **Audit Logs** - Log session access and modifications

## Integration Examples

### Web Interface (Chainlit)

```python
# Chainlit automatically manages sessions
@cl.on_chat_start
async def start():
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
```

### CLI Interface

```python
# CLI with persistent session
import argparse
from pulsepal.main_agent import run_pulsepal

parser = argparse.ArgumentParser()
parser.add_argument("--session-id", help="Continue existing session")
args = parser.parse_args()

session_id, response = await run_pulsepal(
    query="Your query",
    session_id=args.session_id
)
print(f"Session: {session_id}")
print(f"Response: {response}")
```

### API Integration

```python
# REST API with session support
from fastapi import FastAPI, Header
from typing import Optional

app = FastAPI()

@app.post("/query")
async def query(
    text: str,
    session_id: Optional[str] = Header(None)
):
    session_id, response = await run_pulsepal(text, session_id)
    return {
        "session_id": session_id,
        "response": response
    }
```

## Troubleshooting

For session-related issues, check:

1. Session storage permissions
2. Disk space availability
3. Session file validity (valid JSON)
4. Environment variable configuration
5. Cleanup process logs

For additional support, contact: rmoskwa@wisc.edu
