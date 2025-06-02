#!/bin/bash

vulguard training \
    -model deepjit \
    -train_set dg_cache/dataset/libssh2/data/train_merge_libssh2.jsonl \
    -val_set dg_cache/dataset/libssh2/data/val_merge_libssh2.jsonl \
    -dictionary dg_cache/dataset/libssh2/dict_libssh2.jsonl \
    -repo_name libssh2 \
    -repo_path cloned \
    -repo_language C \
    -dg_cache . \
    -device cuda \
    -epoch 30