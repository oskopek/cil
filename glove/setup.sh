#Adapted version of the solution to excercise 6
#Note: make sure to have data in data_in folder before running

set -e

echo 'Splitting combined data to make it easier to read data in program'
cat ../data_in/twitter-datasets/train_data.txt | cut -c 3- > train_sent.txt
cat ../data_in/twitter-datasets/train_data.txt | cut -c -1 > train_label.txt
cat ../data_in/twitter-datasets/eval_data.txt | cut -c 3- > eval_sent.txt
cat ../data_in/twitter-datasets/eval_data.txt | cut -c -1 > eval_label.txt

echo 'Generating vocab, same as Ex. 6 solution'
cat train_sent.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt

cat vocab.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut.txt

echo 'Download relevant stanford embeddings, clean up temp files'
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
rm -rf glove.twitter.27B.zip glove.twitter.27B.25d.txt glove.twitter.27B.50d.txt glove.twitter.27B.100d.txt

echo 'Pickle the vocabulary'
python3 pickle_vocab.py
echo 'Done'
