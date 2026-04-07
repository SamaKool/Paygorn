#!/bin/bash
# Bidirectional sync between Hugging Face Space and GitHub

echo "Fetching updates from both remotes..."
git fetch origin
git fetch github

# Check for HF → GitHub changes
echo "Checking for Hugging Face → GitHub changes..."
if ! git diff --quiet origin/main github/main; then
    echo "HF has changes GitHub doesn't have. Pushing HF → GitHub..."
    git push github origin/main
else
    echo "No HF → GitHub changes detected."
fi

# Check for GitHub → HF changes
echo "Checking for GitHub → Hugging Face changes..."
if ! git diff --quiet github/main origin/main; then
    echo "GitHub has changes HF doesn't have. Pushing GitHub → HF..."
    git push origin github/main
else
    echo "No GitHub → HF changes detected."
fi

echo "Synchronization complete."
