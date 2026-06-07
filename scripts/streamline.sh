#!/usr/bin/env bash
# Run LLM-Streamline across all models and prune budgets (see scripts/_common.sh).
source "$(dirname "$0")/_common.sh"
run_method "streamline"
