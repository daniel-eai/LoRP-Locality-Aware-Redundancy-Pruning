#!/usr/bin/env bash
# Run LoRP (ours) across all models and prune budgets (see scripts/_common.sh).
source "$(dirname "$0")/_common.sh"
run_method "lorp"
