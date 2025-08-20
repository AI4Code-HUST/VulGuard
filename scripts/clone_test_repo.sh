#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
CLONE_DIR=$PARENT_DIR/clone

mkdir -p $CLONE_DIR

git clone https://github.com/libssh2/libssh2 $CLONE_DIR/libssh2