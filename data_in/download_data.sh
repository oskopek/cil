#!/bin/bash

set -e

echo "Make sure you are on the ETHZ VPN."
echo ""

wget "http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip"
unzip twitter-datasets.zip
rm twitter-datasets.zip

python3 train_test_split.py
