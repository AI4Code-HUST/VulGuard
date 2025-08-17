#!/bin/bash
set -e

pip install -e /app

exec "$@"
