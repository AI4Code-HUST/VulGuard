#!/bin/bash

defectguard evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/Idelized/SETUP1-features-test-FFmpeg.jsonl" \
    -commit_test_set "dataset/FFmpeg/Idelized/SETUP1-simcom-test-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/Idelized/dict-FFmpeg.jsonl" \
    -dg_save_folder Idealized \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C