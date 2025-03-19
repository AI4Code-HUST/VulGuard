#!/bin/bash

defectguard training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/Idealized/SETUP1-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/Idealized/SETUP1-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/Idealized/SETUP1-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/Idealized/dict-FFmpeg.jsonl" \
    -dg_save_folder Idealized \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30