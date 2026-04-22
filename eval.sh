#!/usr/bin/env bash
set -uo pipefail

/opt/venv/bin/python scripts/eval_interval_guardrail_metrics.py \
  --dirs ./results/nemotron_nano_v2_vl_safewatch_datav2/eval_json_full \
  --tiou-thresholds 0.5 \
  --unmatched-policy strict
