#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

vulguard inferencing \
    -model lr \
    -repo_name ffmpeg \
    -repo_language C \
    -model_path dg_cache/save/libssh2 \
    -dg_save_folder . \
    -infer_set $PARENT_DIR/sample/sample_test.jsonl 