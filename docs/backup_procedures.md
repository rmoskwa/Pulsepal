# PulsePal Backup and Recovery Procedures

## Quick Reference

### Create Backup
```bash
python scripts/backup_pulsepal.py backup
```

### Verify Backup
```bash
python scripts/backup_pulsepal.py verify --backup-path backups/backup_YYYYMMDD_HHMMSS
```

### Restore from Backup
```bash
python scripts/backup_pulsepal.py restore --backup-path backups/backup_YYYYMMDD_HHMMSS
```

## Detailed Procedures

### 1. Creating a Backup

**When to backup:**
- Before any major refactoring
- Before module renaming operations
- Before database schema changes
- Daily during active development

**Steps:**
1. Navigate to project root
2. Run: `python scripts/backup_pulsepal.py backup`
3. Note the backup location printed (e.g., `backups/backup_20250118_143022`)
4. Verify the backup: `python scripts/backup_pulsepal.py verify --backup-path <backup_dir>`

**What gets backed up:**
- All source code in `pulsepal/` directory
- Test files in `tests/` directory
- Documentation in `docs/` directory
- Scripts in `scripts/` directory
- Configuration files (requirements.txt, setup.py, etc.)
- Conversation logs in `conversationLogs/` directory
- Environment configuration structure (without secrets)
- Database schema export instructions

### 2. Verifying Backup Integrity

**Always verify after creating a backup:**
```bash
python scripts/backup_pulsepal.py verify --backup-path backups/backup_20250118_143022
```

**Verification checks:**
- All files present
- Checksums match manifest
- Directory structure intact

### 3. Restoring from Backup

**WARNING:** Restoration will overwrite current files!

**Steps:**
1. Verify the backup first:
   ```bash
   python scripts/backup_pulsepal.py verify --backup-path backups/backup_20250118_143022
   ```

2. Perform restoration:
   ```bash
   python scripts/backup_pulsepal.py restore --backup-path backups/backup_20250118_143022
   ```

3. The script will:
   - Create a safety backup of current state
   - Restore all files from the backup
   - Report success/failure

**Post-restore checklist:**
- [ ] Verify application starts correctly
- [ ] Check database connections
- [ ] Run test suite
- [ ] Verify API keys are loaded properly

### 4. Database Schema Backup

**Manual process required for full database backup:**

1. Install Supabase CLI (if not installed):
   ```bash
   npm install -g supabase
   ```

2. Login to Supabase:
   ```bash
   supabase login
   ```

3. Link to PulsePal project:
   ```bash
   supabase link --project-ref mnbvsrsivuuuwbtkmumt
   ```

4. Export schema:
   ```bash
   supabase db dump --schema public > backups/database_schema_$(date +%Y%m%d).sql
   ```

### 5. Backup Directory Structure

```
backups/
└── backup_20250118_143022/
    ├── codebase/           # All source code
    │   ├── pulsepal/
    │   ├── tests/
    │   ├── docs/
    │   └── ...
    ├── conversationLogs/   # Session data
    ├── environment/        # Config files (sanitized)
    │   ├── .env.example
    │   └── .env.keys      # Keys only, no values
    ├── database_schema.sql # DB schema instructions
    ├── manifest.json       # File checksums
    └── backup_metadata.json # Backup details
```

## Emergency Recovery

### If backup restoration fails:

1. **Check safety backup:**
   - Look for `backups/safety_<timestamp>` directory
   - This contains state before failed restore

2. **Manual recovery:**
   ```bash
   # Copy files manually from backup
   cp -r backups/backup_YYYYMMDD_HHMMSS/codebase/* .
   ```

3. **Git recovery (if available):**
   ```bash
   git stash  # Save any uncommitted changes
   git reset --hard HEAD  # Reset to last commit
   ```

## Best Practices

1. **Regular Backups:**
   - Daily during active development
   - Before any risky operations
   - After completing major features

2. **Backup Retention:**
   - Keep last 7 daily backups
   - Keep all pre-refactoring backups
   - Archive monthly backups

3. **Testing Restores:**
   - Test restore procedure monthly
   - Verify in a separate directory first:
     ```bash
     python scripts/backup_pulsepal.py restore \
       --backup-path backups/backup_YYYYMMDD_HHMMSS \
       --target-path /tmp/pulsepal_test
     ```

4. **Documentation:**
   - Log all backup/restore operations
   - Document any issues encountered
   - Update procedures as needed

## Troubleshooting

### Common Issues:

**Permission Denied:**
- Ensure write permissions in project directory
- Run with appropriate user privileges

**Disk Space:**
- Each backup uses ~10-50MB (depending on logs)
- Clean old backups regularly

**Missing Files:**
- Check backup manifest.json for file list
- Verify source files exist before backup

**Checksum Failures:**
- File corruption during backup
- Retry backup operation
- Check disk health

## Quick Backup Checklist

Before major changes:
- [ ] Create backup: `python scripts/backup_pulsepal.py backup`
- [ ] Verify backup: `python scripts/backup_pulsepal.py verify --backup-path <path>`
- [ ] Note backup location in change log
- [ ] Export database schema (if needed)
- [ ] Test restore in separate directory (optional but recommended)