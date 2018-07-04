#/bin/bash
set -e
cd ..
dir="/tmp/$USER"
mkdir -p "$dir"
cp -r cil "$dir/"
cd "$dir/cil/glove"
./run.sh
