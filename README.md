# Computational Intelligence Lab 2018 -- Text Sentiment Classification
### Larissa Laich, Lukas Jendele, Michael Wiegner, Ondrej Skopek
Department of Computer Science, ETH Zurich, Switzerland

## Project definition
The use of microblogging and text messaging as a media of communication has greatly increased over the past 10 years. Such large volumes of data amplifies the need for automatic methods to understand the opinion conveyed in a text.

### Resources
All the necessary resources (including training data) are available at https://inclass.kaggle.com/c/cil-text-classification-2018

### Training Data
For this problem, we have acquired 2.5M tweets classified as either positive or negative.

### Evaluation Metrics
*Classification Accuracy*

## File description

* `train_pos.txt` and `train_neg.txt` -- a small set of training tweets for each of the two classes.
* `train_pos_full.txt` and `train_neg_full.txt` -- a complete set of training tweets for each of the two classes, about 1M tweets per class.
* `test_data.txt` -- the test set, that is the tweets for which you have to predict the sentiment label.
* `sampleSubmission.csv` -- a sample submission file in the correct format, note that each test tweet is numbered.
  * Submission of predictions: -1 = negative prediction, 1 = positive prediction

Note that all tweets have been tokenized already, so that the words and punctuation are properly separated by a whitespace.

## Pre-processing Techniques


0.	~~Basic (Remove Unicode strings and noise), already done~~

---

1.	**Remove Numbers (Best)**
2.	**Replace Repetitions of Punctuation (Best)**
3.	~~Handling Capitalized Words (Poor)~~
4.	~~Lowercase, already done~~
5.	~~Replace Slang and Abbreviations (Poor)~~ FIX them instead.
6.	**Replace Elongated Words (Varying)**
7.	~~**Replace Contractions (Varying)**~~
8.	~~Replace negations with antonyms (Poor)~~
9.	**Handling Negations (High)**
10.	*Remove Stopwords (Poor)*
11.	**Stemming (Best)**
12.	**Lemmatizing (High)**
13.	~~Other (Replace urls and user mentions) (High), already done~~
14.	~~Spelling Correction (Poor)~~
15.	~~Remove Punctuation (Poor)~~


Source: https://link.springer.com/chapter/10.1007%2F978-3-319-67008-9_31 Effrosynidis D., Symeonidis S., Arampatzis A. (2017) A Comparison of Pre-processing Techniques for Twitter Sentiment Analysis. In: Kamps J., Tsakonas G., Manolopoulos Y., Iliadis L., Karydis I. (eds) Research and Advanced Technology for Digital Libraries. TPDL 2017. Lecture Notes in Computer Science, vol 10450. Springer, Cham

