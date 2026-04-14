---
name: git-sync
description: Commit all changes and push to origin/main
scope: [/Users/tomhoyt/Desktop/firm_project/unigemm_test]
---

# Git Sync

Automatically stage all changes, commit, and push to `origin/main`.

## Steps

1. Run `git status` and `git diff` to see all changes
2. Run `git log --oneline -5` to follow existing commit message style
3. Draft a concise commit message summarizing the changes (in English, follow the style of recent commits)
4. Run `git add` on the specific changed files (do NOT use `git add -A`)
5. Run `git commit` with the message ending with the co-author line
6. Run `git push`

## Rules

- Stage specific files, never `git add -A` or `git add .`
- Do NOT stage `.claude/settings.local.json`
- Commit message format: short summary line, then blank line, then bullet points if needed, then `Co-Authored-By:` line
- If there are no changes (nothing to commit), inform the user and skip
- Always push to `origin/main`
