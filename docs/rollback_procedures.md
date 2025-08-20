# PulsePal Rollback Procedures

## Quick Reference Guide

### Before Any Refactoring
```bash
# Create checkpoint
python scripts/rollback_manager.py checkpoint --story "story_name"
```

### Quick Rollback Options
```bash
# Rollback to previous commit
python scripts/rollback_manager.py git

# Rollback to specific commit
python scripts/rollback_manager.py git --commit abc123

# Rollback module renames
python scripts/rollback_manager.py module

# Full rollback from backup
python scripts/rollback_manager.py full --backup-path backups/backup_YYYYMMDD_HHMMSS

# List checkpoints
python scripts/rollback_manager.py list

# Rollback to checkpoint
python scripts/rollback_manager.py rollback --index 1
```

## Detailed Rollback Procedures

### 1. Git-Based Rollback

**When to use:**
- Code changes need to be reverted
- No database changes involved
- Git history is intact

**Procedure:**
```bash
# Rollback to previous commit
python scripts/rollback_manager.py git

# Rollback to specific commit
python scripts/rollback_manager.py git --commit <commit_hash>

# Rollback to branch
python scripts/rollback_manager.py git --branch main
```

**What happens:**
1. Checks for uncommitted changes (stashes if needed)
2. Performs git reset --hard to target commit
3. Logs rollback operation

**Post-rollback:**
- Run tests to verify functionality
- Check that application starts correctly
- Review stashed changes if any

### 2. Module Rename Rollback

**When to use:**
- After failed module renaming operation
- Import errors after refactoring
- Need to revert file structure changes

**Procedure:**
```bash
# Auto-detect latest rename mapping
python scripts/rollback_manager.py module

# Use specific mapping file
python scripts/rollback_manager.py module --mapping-file rename_mappings_20250118.json
```

**Mapping file format:**
```json
{
  "pulsepal/old_name.py": "pulsepal/new_name.py",
  "tests/test_old.py": "tests/test_new.py"
}
```

**What happens:**
1. Reads rename mappings
2. Reverses all file moves
3. Cleans up empty directories
4. Reports any failures

### 3. Database Schema Rollback

**When to use:**
- After failed database migration
- Schema changes causing issues
- Need to revert to previous DB state

**Manual procedure required:**

1. **Via Supabase Dashboard:**
   - Go to https://supabase.com/dashboard
   - Navigate to project: mnbvsrsivuuuwbtkmumt
   - Go to Database → Migrations
   - Identify problematic migration
   - Run rollback SQL

2. **Via Supabase CLI:**
   ```bash
   # Reset database (CAUTION: Loses all data)
   supabase db reset --linked
   
   # Apply migrations up to specific point
   supabase db push --linked
   ```

3. **Using backup schema:**
   ```bash
   # If you have a schema backup
   supabase db push backups/database_schema_20250118.sql --linked
   ```

### 4. Full System Rollback

**When to use:**
- Multiple components affected
- Need complete restoration
- Other rollback methods insufficient

**Procedure:**
```bash
# List available backups
ls -la backups/

# Perform full rollback
python scripts/rollback_manager.py full --backup-path backups/backup_20250118_143022
```

**What happens:**
1. Verifies backup integrity
2. Creates safety backup of current state
3. Restores all files from backup
4. Reports success/failure

### 5. Checkpoint-Based Rollback

**Creating checkpoints:**
```bash
# Before starting any story
python scripts/rollback_manager.py checkpoint --story "1.0.risk-mitigation"
```

**Using checkpoints:**
```bash
# List available checkpoints
python scripts/rollback_manager.py list

# Rollback to most recent checkpoint
python scripts/rollback_manager.py rollback

# Rollback to specific checkpoint (by index from list)
python scripts/rollback_manager.py rollback --index 2
```

## Rollback Decision Tree

```
Problem Detected
│
├─> Only code changes?
│   └─> Use Git Rollback
│
├─> Module/import errors?
│   └─> Use Module Rename Rollback
│
├─> Database issues?
│   └─> Use Database Rollback (manual)
│
├─> Multiple components affected?
│   └─> Use Full Backup Rollback
│
└─> Have checkpoint?
    └─> Use Checkpoint Rollback
```

## Story-Specific Rollback Checklists

### Story 1.0: Risk Mitigation Setup
- [ ] No rollback needed (infrastructure only)

### Story 2.0: Module Renaming (_v2 suffix)
- [ ] Create checkpoint before starting
- [ ] Save rename_mappings.json
- [ ] If fails: `python scripts/rollback_manager.py module`
- [ ] Verify imports work after rollback

### Story 3.0: Import Path Updates
- [ ] Create checkpoint before starting
- [ ] If fails: `python scripts/rollback_manager.py git`
- [ ] Run import validation tests

### Story 4.0: Configuration Cleanup
- [ ] Backup .env file manually
- [ ] If fails: Restore .env from backup
- [ ] Verify API keys work

### Story 5.0: Testing Updates
- [ ] Create checkpoint before starting
- [ ] If fails: `python scripts/rollback_manager.py git`
- [ ] Ensure test suite runs

## Emergency Recovery

### If all rollbacks fail:

1. **Fresh clone from git:**
   ```bash
   cd ..
   mv pulsePal pulsePal_broken
   git clone <repository_url> pulsePal
   cd pulsePal
   ```

2. **Restore from safety backup:**
   ```bash
   # Find safety backups
   ls -la backups/safety_*
   
   # Restore manually
   cp -r backups/safety_YYYYMMDD_HHMMSS/codebase/* .
   ```

3. **Contact team:**
   - Document what went wrong
   - Save all error logs
   - Don't make further changes

## Best Practices

### Before Any Refactoring:
1. Create checkpoint
2. Commit current changes
3. Run full test suite
4. Create backup
5. Document current state

### During Refactoring:
1. Make incremental changes
2. Test after each step
3. Commit working states
4. Keep rename mappings

### After Problems:
1. Stop making changes
2. Assess damage scope
3. Choose appropriate rollback
4. Verify restoration
5. Document issue

## Rollback Log

All rollback operations are logged to `.rollback_history.json`:

```json
{
  "type": "git",
  "timestamp": "2025-01-18T14:30:00",
  "details": {
    "commit": "abc123",
    "branch": null
  }
}
```

Review this log to understand rollback history.

## Testing Rollback Procedures

### Test in safe environment:
```bash
# Create test backup
python scripts/backup_pulsepal.py backup

# Make test changes
echo "test" > test_file.txt

# Test rollback
python scripts/rollback_manager.py git

# Verify test file removed
ls test_file.txt  # Should not exist
```

## Troubleshooting

### Common Issues:

**"Uncommitted changes detected"**
- Changes will be auto-stashed
- Review stash after rollback: `git stash list`

**"No backup found"**
- Check backup directory exists
- Verify backup path is correct
- Create new backup if needed

**"Import errors after rollback"**
- Clear Python cache: `find . -type d -name __pycache__ -exec rm -r {} +`
- Restart Python interpreter
- Check PYTHONPATH

**"Database out of sync"**
- Manual intervention required
- Check Supabase dashboard
- Consider full reset if development environment