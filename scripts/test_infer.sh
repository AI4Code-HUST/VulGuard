#!/bin/bash

vulguard inferencing \
    -repo_name libssh2 \
    -model deepjit \
    -model_path dg_cache/save/libssh2 \
    -infer_set dg_cache/dataset/libssh2/data/test_merge_libssh2.jsonl \
    -dictionary dg_cache/dataset/libssh2/dict_libssh2.jsonl \
    -dg_cache . \
    -device cuda