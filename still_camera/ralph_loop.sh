#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${PROMPT_FILE:-tasks.md}"
BRANCH="${BRANCH:-feature/rgbd-snapshot-realsense}"
MAX_ITERS="${MAX_ITERS:-30}"
TEST_CMD="${TEST_CMD:-python -m unittest discover}"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "ERROR: $PROMPT_FILE not found in $(pwd)"
  exit 1
fi

# Ensure git repo
if [[ ! -d ".git" ]]; then
  git init
  git add "$PROMPT_FILE" || true
  git commit -m "chore: add initial task plan" || true
fi

# Ensure branch
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git checkout "$BRANCH"
  else
    git checkout -b "$BRANCH"
  fi
fi

# Aider config
cat > .aider.conf.yml <<'YAML'
auto-commits: true
commit-language: en
attribute-co-authored-by: true
YAML

echo "== Ralph loop starting =="
echo "Branch: $BRANCH"
echo "Prompt:  $PROMPT_FILE"
echo "Iters:   $MAX_ITERS"
echo

for ((i=1; i<=MAX_ITERS; i++)); do
  echo "---- Iteration $i/$MAX_ITERS ----"

  aider \
    --model gemini/gemini-2.5-pro \
    --yes-always \
    --auto-commits \
    --commit-language en \
    --message-file "$PROMPT_FILE" \
    --auto-test \
    --test-cmd "$TEST_CMD" \
    --exit || true

  echo
done

echo "== Ralph loop finished =="
