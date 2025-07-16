# Clean Git Workflow
Complete git workflow for $ARGUMENTS.

## Steps
1. **Review changes:** `git_status()` and `git_diff_unstaged()`
2. **Update CLAUDE.md** with all architectural changes
3. **Verify tests** written and passing
4. **Stage systematically:** `git_add(files=["CLAUDE.md", "src/", "tests/"])`
5. **Commit:** `git_commit("feat: $ARGUMENTS - TASK-ID")`
6. **Verify clean:** `git_status()` shows clean state

Never leave uncommitted changes.
