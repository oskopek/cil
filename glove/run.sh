#Adapted version of the solution to excercise 6
#Note: make sure to have data in data_in folder before running

set -e

echo 'Build cooc matrix (also from Ex. 6 solution)'
python3 cooc.py

echo 'Run the glove embedding sentiment classification with random forest and stanford embeddings'
python3 glove_embeddings.py 0 1

echo 'Run the glove embedding sentiment classification with logistic regression and stanford embeddings'
python3 glove_embeddings.py 1 1

echo 'Run the glove embedding sentiment classification with random forest and self-computed embeddings'
python3 glove_embeddings.py 0 0

echo 'Run the glove embedding sentiment classification with logistic regression and self-computed embeddings'
python3 glove_embeddings.py 1 0

echo 'Done'
