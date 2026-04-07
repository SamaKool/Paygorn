#!/bin/bash
# Bidirectional sync between Hugging Face Space and GitHub
set -euo pipefail

echo "Fetching updates from both remotes..."
git fetch --prune origin
git fetch --prune github

# Check for HF → GitHub changes
echo "Checking for Hugging Face → GitHub changes..."
if ! git diff --quiet origin/main github/main; then
    echo "HF has changes GitHub doesn't have. Pushing HF → GitHub..."
    git push github "refs/remotes/origin/main:refs/heads/main"
else
    echo "No HF → GitHub changes detected."
fi

# Check for GitHub → HF changes
echo "Checking for GitHub → Hugging Face changes..."
if ! git diff --quiet github/main origin/main; then
    echo "GitHub has changes HF doesn't have. Pushing GitHub → HF..."
    git push origin "refs/remotes/github/main:refs/heads/main"
else
    echo "No GitHub → HF changes detected."
fi

# Detect divergent histories (both sides have new commits)
if ! git merge-base --is-ancestor origin/main github/main && \
   ! git merge-base --is-ancestor github/main origin/main; then
    echo "WARNING: Divergent histories detected! Manual merge required."
    echo "HF main: $(git rev-parse origin/main)"
    echo "GitHub main: $(git rev-parse github/main)"
    exit 1
fi

echo "Synchronization complete."