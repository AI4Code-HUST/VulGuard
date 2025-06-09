#!/bin/bash

vulguard training \
    -model lr \
    -repo_name libssh2 \
    -repo_path cloned \
    -repo_language C \
    -dg_save_folder . \
    -epoch 1