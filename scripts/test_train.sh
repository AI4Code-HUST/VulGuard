#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

vulguard training \
    -model lr \
    -repo_name ffmpeg \
    -repo_language C \
    -dg_save_folder . \
    -train_set $PARENT_DIR/sample/sample_train.jsonl