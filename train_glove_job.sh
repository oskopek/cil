#/bin/bash
set -e
cd ..
mkdir -p "/scratch/$USER/"
cp -r cil "/scratch/$USER/"
cd "/scratch/$USER/cil/glove"
./run.sh
