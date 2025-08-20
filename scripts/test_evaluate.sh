#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

vulguard evaluating \
    -model lr \
    -repo_name ffmpeg \
    -repo_language C \
    -dg_save_folder . \
    -test_set $PARENT_DIR/sample/sample_test.jsonl