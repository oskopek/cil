#!/bin/bash
set -e

source venv/bin/activate

echo "Training..."
exec python -m cil.train

