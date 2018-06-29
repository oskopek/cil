#Adapted version of the solution to excercise 6
#Note: make sure to have data in data_in folder before running

#Splitting combined data to make it easier to read data in program
cat ../data_in/twitter-datasets/train_data.txt | cut -c 3- > train_sent.txt
cat ../data_in/twitter-datasets/train_data.txt | cut -c -1 > train_label.txt
cat ../data_in/twitter-datasets/eval_data.txt | cut -c 3- > eval_sent.txt
cat ../data_in/twitter-datasets/eval_data.txt | cut -c -1 > eval_label.txt

#Generating vocab, same as Ex. 6 solution
cat train_sent.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt

cat vocab.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut.txt

#Download relevant stanford embeddings, clean up rest
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
rm glove.twitter.27B.zip
rm glove.twitter.27B.25d.txt
rm glove.twitter.27B.50d.txt
rm glove.twitter.27B.100d.txt

#Pickle the vocabulary and build cooc matrix (also from Ex. 6 solution)
python3 pickle_vocab.py	
python3 cooc.py

#Run the glove embedding sentiment classification with random forest and stanford embeddings
python3 glove_embeddings.py 0 1

#Run the glove embedding sentiment classification with logistic regression and stanford embeddings
python3 glove_embeddings.py 1 1

#Run the glove embedding sentiment classification with random forest and self-computed embeddings
python3 glove_embeddings.py 0 0

#Run the glove embedding sentiment classification with logistic regression and self-computed embeddings
python3 glove_embeddings.py 1 0