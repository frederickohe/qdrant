#!/usr/bin/env bash
# Deploy rag-api source and sync into the production autobus-rag stack.
set -euo pipefail

BRANCH="${DEPLOY_BRANCH:-main}"
RAG_STACK_PATH="${RAG_STACK_PATH:-/var/www/autobus-rag}"

git fetch origin "$BRANCH"
git reset --hard "origin/$BRANCH"

if [ ! -d "$RAG_STACK_PATH" ]; then
  echo "Production RAG stack not found at $RAG_STACK_PATH" >&2
  exit 1
fi

rsync -a --delete ./rag-api/ "$RAG_STACK_PATH/rag-api/"

cd "$RAG_STACK_PATH"
docker network inspect greenbrain_rag >/dev/null 2>&1 || docker network create greenbrain_rag
docker network inspect caddy >/dev/null 2>&1 || docker network create caddy

docker compose -f docker-compose.yaml build rag-api
docker compose -f docker-compose.yaml up -d qdrant embeddings rag-api

docker compose -f docker-compose.yaml ps
