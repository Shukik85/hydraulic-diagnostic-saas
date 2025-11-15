#!/bin/bash
# CI/CD Setup Script
# Configures GitHub repository for automated CI/CD

set -e

echo "======================================"
echo "🚀 CI/CD Setup Wizard"
echo "======================================"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo "Install: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
gh auth status || {
    echo "Please authenticate with GitHub CLI:"
    gh auth login
}

echo -e "\n1. Setting up branch protection rules..."

# Master branch protection
gh api repos/:owner/:repo/branches/master/protection \
  --method PUT \
  --field required_status_checks[strict]=true \
  --field 'required_status_checks[contexts][]=code-quality' \
  --field 'required_status_checks[contexts][]=security' \
  --field 'required_status_checks[contexts][]=backend-tests' \
  --field required_pull_request_reviews[required_approving_review_count]=1 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field enforce_admins=false \
  --field restrictions=null \
  || echo "Branch protection setup failed (may need admin access)"

echo -e "\n2. Enabling GitHub Actions..."
gh api repos/:owner/:repo/actions/permissions \
  --method PUT \
  --field enabled=true \
  --field allowed_actions=all

echo -e "\n3. Setting up secrets (interactive)..."

read -p "Enter staging server host (or skip): " STAGING_HOST
if [ -n "$STAGING_HOST" ]; then
    gh secret set STAGING_HOST --body "$STAGING_HOST"
fi

read -p "Path to staging SSH key (or skip): " SSH_KEY_PATH
if [ -n "$SSH_KEY_PATH" ] && [ -f "$SSH_KEY_PATH" ]; then
    gh secret set STAGING_DEPLOY_KEY < "$SSH_KEY_PATH"
fi

read -p "Enter Codecov token (or skip): " CODECOV_TOKEN
if [ -n "$CODECOV_TOKEN" ]; then
    gh secret set CODECOV_TOKEN --body "$CODECOV_TOKEN"
fi

echo -e "\n4. Enabling Dependabot..."
echo "Dependabot configuration is in .github/dependabot.yml"
echo "It will automatically create PRs for dependency updates"

echo -e "\n======================================"
echo -e "✅ CI/CD Setup Complete!"
echo -e "======================================"

echo -e "\nNext steps:"
echo "1. Push your code to trigger first CI run"
echo "2. Check Actions tab: https://github.com/:owner/:repo/actions"
echo "3. Review and configure branch protection rules"
echo "4. Set up staging environment"

echo -e "\nFor more info, see: docs/CI_CD.md"
