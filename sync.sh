#!/bin/bash

# ==============================================================================
# PAYGORN GIT COMMAND CENTER
# ==============================================================================

# Colors for UI
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default Configurations
DEFAULT_GH_URL="github.com/SamaKool/Paygorn.git"
DEFAULT_HF_URL="huggingface.co/spaces/TwoBraincells/Elite-Trade-Sentry.git"
TOKEN_FILE=".git_tokens"

# Ensure token file is gitignored
if ! grep -q "^${TOKEN_FILE}$" .gitignore 2>/dev/null; then
    echo "${TOKEN_FILE}" >> .gitignore
fi

# Load saved tokens
if [ -f "$TOKEN_FILE" ]; then
    source "$TOKEN_FILE"
fi

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

get_token() {
    local provider=$1
    local var_name="${provider}_TOKEN"
    local token="${!var_name}"

    # Loop until the user actually provides a non-empty string
    while [ -z "$token" ]; do
        echo -e "${YELLOW}No token found for ${provider}.${NC}"
        read -s -p "Please paste your ${provider} token: " token
        echo ""
        
        if [ -n "$token" ]; then
            # Save for future
            echo "${var_name}=\"${token}\"" >> "$TOKEN_FILE"
            export ${var_name}="${token}"
        else
            echo -e "${RED}Token cannot be blank. Try again.${NC}"
        fi
    done
}

configure_remote() {
    local remote_name=$1
    local base_url=$2
    local token=$3

    # Inject token into URL for password-less auth
    local auth_url="https://oauth2:${token}@${base_url}"

    # Check if remote exists, modify if it does, add if it doesn't
    if git remote | grep -q "^${remote_name}$"; then
        git remote set-url "$remote_name" "$auth_url"
    else
        git remote add "$remote_name" "$auth_url"
    fi
}

check_uncommitted() {
    if [[ -n $(git status -s) ]]; then
        echo -e "${YELLOW}Wait! You have uncommitted local changes.${NC}"
        read -p "Do you want me to automatically commit them for you? (y/n): " auto_commit
        if [[ "${auto_commit,,}" == "y" ]]; then
            git add .
            git commit -m "chore: auto-sync update $(date +'%Y-%m-%d %H:%M:%S')"
            echo -e "${GREEN}Changes committed successfully.${NC}"
        else
            echo -e "${RED}Warning: Continuing without committing your latest local changes!${NC}"
        fi
    fi
}

compare_branches() {
    local remote_name=$1
    echo -e "${BLUE}Fetching latest data from ${remote_name}...${NC}"
    git fetch "$remote_name" main 2>/dev/null
    
    local divergence=$(git rev-list --left-right --count HEAD...${remote_name}/main 2>/dev/null)
    local ahead=$(echo $divergence | awk '{print $1}')
    local behind=$(echo $divergence | awk '{print $2}')

    echo -e "Status: You are ${GREEN}${ahead} commits ahead${NC}, and ${RED}${behind} commits behind${NC} ${remote_name}/main."
}

execute_push() {
    local remote_name=$1
    check_uncommitted
    compare_branches "$remote_name"

    echo -e "${BLUE}Initiating push to ${remote_name}...${NC}"
    # Push active local branch (HEAD) to remote main
    if git push "$remote_name" HEAD:main; then
        echo -e "${GREEN}Push successful!${NC}"
    else
        echo -e "${RED}Push rejected by remote.${NC}"
        read -p "Do you want to FORCE push and overwrite the remote? (y/n): " force
        if [[ "${force,,}" == "y" ]]; then
            echo -e "${YELLOW}Force pushing...${NC}"
            git push -f "$remote_name" HEAD:main
            echo -e "${GREEN}Force push complete.${NC}"
        else
            echo "Operation aborted."
        fi
    fi
}

execute_pull() {
    local remote_name=$1
    echo -e "${BLUE}Pulling from ${remote_name}...${NC}"
    
    if git pull "$remote_name" main --rebase; then
        echo -e "${GREEN}Pull successful! Your local repo is up to date.${NC}"
    else
        echo -e "${RED}Pull failed due to conflicts.${NC}"
        echo "Please resolve conflicts manually, or run 'git rebase --abort'."
    fi
}

# ==============================================================================
# MAIN FLOW
# ==============================================================================

echo -e "${BLUE}=======================================${NC}"
echo -e "${GREEN}      PAYGORN SYNC COMMAND CENTER      ${NC}"
echo -e "${BLUE}=======================================${NC}"

# STEP 1: Action Selection
echo "What do you want to do?"
echo "  A) Push"
echo "  B) Pull"
echo "  C) Sync (Zero-Touch Cloud Sync)"
read -p "Select (A/B/C): " action

# STEP 2: Target Selection
if [[ "${action,,}" == "a" || "${action,,}" == "b" ]]; then
    echo ""
    echo "Target provider?"
    echo "  A) GitHub (SamaKool/Paygorn)"
    echo "  B) HuggingFace (TwoBraincells/Elite-Trade-Sentry)"
    read -p "Select (A/B): " target
    
    if [[ "${target,,}" == "a" ]]; then
        get_token "GITHUB"
        configure_remote "github" "$DEFAULT_GH_URL" "$GITHUB_TOKEN"
        if [[ "${action,,}" == "a" ]]; then execute_push "github"; else execute_pull "github"; fi
    elif [[ "${target,,}" == "b" ]]; then
        get_token "HF"
        configure_remote "hf" "$DEFAULT_HF_URL" "$HF_TOKEN"
        if [[ "${action,,}" == "a" ]]; then execute_push "hf"; else execute_pull "hf"; fi
    else
        echo -e "${RED}Invalid target selected. Exiting.${NC}"
        exit 1
    fi

# SYNC LOGIC (Automated Zero-Touch)
elif [[ "${action,,}" == "c" ]]; then
    echo -e "\n${BLUE}Initiating Zero-Touch Synchronization...${NC}"
    
    # Check if user forgot to commit locally before syncing clouds
    check_uncommitted

    get_token "GITHUB"
    get_token "HF"
    configure_remote "github" "$DEFAULT_GH_URL" "$GITHUB_TOKEN"
    configure_remote "hf" "$DEFAULT_HF_URL" "$HF_TOKEN"

    echo "Fetching metadata from both remotes..."
    git fetch --prune github 2>/dev/null
    git fetch --prune hf 2>/dev/null

    # 1. The Disaster Guard (Divergent History Check)
    if ! git merge-base --is-ancestor hf/main github/main >/dev/null 2>&1 && \
       ! git merge-base --is-ancestor github/main hf/main >/dev/null 2>&1; then
        echo -e "${RED}❌ WARNING: Divergent histories detected!${NC}"
        echo "Both remotes have different, conflicting new commits."
        echo "Manual merge required to prevent data loss."
        exit 1
    fi

    # 2. The Zero-Touch Direction Detector
    HF_AHEAD=$(git rev-list --count github/main..hf/main 2>/dev/null || echo 0)
    GH_AHEAD=$(git rev-list --count hf/main..github/main 2>/dev/null || echo 0)

    if [ "$HF_AHEAD" -gt 0 ]; then
        echo -e "${YELLOW}HuggingFace is ahead by $HF_AHEAD commit(s). Pushing HF → GitHub...${NC}"
        git push github "refs/remotes/hf/main:refs/heads/main"
        echo -e "${GREEN}Sync complete.${NC}"

    elif [ "$GH_AHEAD" -gt 0 ]; then
        echo -e "${YELLOW}GitHub is ahead by $GH_AHEAD commit(s). Pushing GitHub → HuggingFace...${NC}"
        git push hf "refs/remotes/github/main:refs/heads/main"
        echo -e "${GREEN}Sync complete.${NC}"

    else
        echo -e "${GREEN}✅ No changes detected. Both remotes are perfectly in sync.${NC}"
    fi

# INVALID ACTION
else
    echo -e "${RED}Invalid action selected. Run script again.${NC}"
    exit 1
fi

echo -e "${GREEN}All tasks completed. Over and out.${NC}"