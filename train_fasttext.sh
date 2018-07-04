#!/bin/bash
set -e
dir="/scratch/$USER"

rm -rf "$dir"
mkdir -p "$dir"
cp -r data_in/ "$dir"

cp "fasttext/transform_fasttext.py" "$dir/data_in/twitter-datasets/"
cp "fasttext/eval_fasttext.py" "$dir/data_in/"
cd "$dir/data_in/twitter-datasets"
python transform_fasttext.py
cd ..
wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
unzip v0.1.0.zip
cd fastText-0.1.0
make
cd ..

echo ''
echo "Outputs are in: $dir/data_in"
echo ''

for i in `seq 1 5`; do
    echo "Run: $i"
    ./fastText-0.1.0/fasttext supervised -input twitter-datasets/fasttext_train_data.txt -output "model_$i"
    ./fastText-0.1.0/fasttext predict "./model_$i.bin" twitter-datasets/fasttext_eval_data.txt > pred_eval.txt
    ./fastText-0.1.0/fasttext predict "./model_$i.bin" twitter-datasets/fasttext_test_data.txt > pred_test.txt
    python eval_fasttext.py >> eval_accs.txt
    mv pred_eval.txt pred_eval_"$i".txt
    mv pred_test.txt pred_test_"$i".txt
    mv pred_test.txt.kaggle.csv pred_test_"$i".txt.kaggle.csv
done
