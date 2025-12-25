#!/bin/bash
# YouTube Study Notes Generator - Unified Workflow
# Usage: ./run.sh [URL]

cd "$(dirname "$0")"
source venv/bin/activate
python main.py "$@"

