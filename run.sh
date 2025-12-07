#!/bin/bash
# YouTube Study Notes Generator - Quick Run Script
# Usage: ./run.sh [URL] [--prompt PROMPT]

cd "$(dirname "$0")"
source venv/bin/activate
python app.py "$@"

