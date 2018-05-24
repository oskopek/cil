#!/bin/bash
set -e
module load python_gpu/3.6.4

source venv/bin/activate

echo "Training..."
exec python -m cil.train $@ 2>&1
