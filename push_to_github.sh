#!/usr/bin/env bash
set -e

# ==============================================================================
# Push Pedagogical Vision Backend to GitHub
# ==============================================================================

REPO_NAME="${1:-EduSphere-AI-production-build}"
GITHUB_USER="kuganryde"

echo "=== Publishing Pedagogical Vision Backend to GitHub ==="
echo "Target Repository: https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

if [ -n "$GITHUB_TOKEN" ]; then
    echo "Pushing using provided GITHUB_TOKEN..."
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
    git push -u origin main --force
    echo "Successfully pushed to https://github.com/${GITHUB_USER}/${REPO_NAME}!"
    exit 0
fi

# Check if authenticated via gh CLI with write access
if gh auth status 2>&1 | grep -q "Logged in to github.com"; then
    echo "Attempting to create and push repo via GitHub CLI..."
    gh repo create "${GITHUB_USER}/${REPO_NAME}" --public --source=. --remote=origin --push || {
        echo ""
        echo "Note: If the current gh token is read-only, run:"
        echo "  gh auth login --scopes repo"
        echo "  ./push_to_github.sh ${REPO_NAME}"
        echo ""
        echo "Or push using a GitHub Personal Access Token (PAT):"
        echo "  GITHUB_TOKEN=<your_token> ./push_to_github.sh ${REPO_NAME}"
    }
else
    echo "Please provide a GITHUB_TOKEN or log in via gh CLI:"
    echo "  GITHUB_TOKEN=<your_token> ./push_to_github.sh ${REPO_NAME}"
fi
