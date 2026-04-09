#!/bin/bash
# Bidirectional sync between Hugging Face Space and GitHub
set -euo pipefail

echo "Fetching updates from both remotes..."
git fetch --prune origin
git fetch --prune github

# 1. Detect divergent histories FIRST
if ! git merge-base --is-ancestor origin/main github/main && \
   ! git merge-base --is-ancestor github/main origin/main; then
    echo "❌ WARNING: Divergent histories detected! Both remotes have different new commits."
    echo "Manual merge required."
    echo "HF main: $(git rev-parse origin/main)"
    echo "GitHub main: $(git rev-parse github/main)"
    exit 1
fi

# 2. Count commits to determine exactly who is ahead
HF_AHEAD=$(git rev-list --count github/main..origin/main)
GH_AHEAD=$(git rev-list --count origin/main..github/main)

if [ "$HF_AHEAD" -gt 0 ]; then
    echo "HF has $HF_AHEAD new commit(s). Pushing HF → GitHub..."
    git push github "refs/remotes/origin/main:refs/heads/main"

elif [ "$GH_AHEAD" -gt 0 ]; then
    echo "GitHub has $GH_AHEAD new commit(s). Pushing GitHub → HF..."
    git push origin "refs/remotes/github/main:refs/heads/main"

else
    echo "✅ No changes detected. Both remotes are perfectly in sync."
fi

echo "Synchronization complete."