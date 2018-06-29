# Computational Intelligence Lab 2018 -- Text Sentiment Classification
### Larissa Laich, Lukas Jendele, Michael Wiegner, Ondrej Skopek
Department of Computer Science, ETH Zurich, Switzerland

## Project definition

Perform text sentiment classification on Twitter data (tweets). The goal is to develop an automatic method to reveal the authors' opinion/sentiment.

## Report

The report is located at: `report/report.pdf`.

## Setup

The following two steps will prepare your environment to begin training and evaluating models.

Simply run

```
make setup # for Tensorflow on GPU
```

For Tensorflow on CPU, change the `tensorflow-gpu` line in `requirements.txt` to `tensorflow`.

## Training

To train, run `make train` from the main project folder.
The model specified by `cil/flags.py` will be trained.

You can also run the experiments from make directly:

```
make lstm128
make lstm128_ce
make lstm128_we
make gru256
make stacklstm
make cnn512
make glove_lr
make glove_rf
make fasttext
make transformer
```

### Special cases

For some special cases, we provide additional instructions for training.

#### Transformer

You need to install the TensorFlow serving server:

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server
```


#### Fasttext



#### Ensemble vote
This was computed offline using Excel. The Excel sheet we used is uploaded to `data_out` and named `prediction_ensemble.xlsx`



## Structure
* `cil/` - 
    * `data/` - data loading and preprocessing
    * `models/` - neural network definitions
* `data_in/` - input twitter data
* `data_out/` - output data for kaggle
* `glove` - glove embeddings with logistic regression or the random forest
* `report/` - the final report/paper
* `transformer/` - transformer model

TODO add fasttext data


### Training Data
2.5M tweets classified as either positive or negative.
* `train_pos.txt` and `train_neg.txt` -- a small set of training tweets for each of the two classes.
* `train_pos_full.txt` and `train_neg_full.txt` -- a complete set of training tweets for each of the two classes, about 1M tweets per class.
All tweets have been tokenized already, so that the words and punctuation are properly separated by a whitespace.

 
### Evaluation Metrics
The evaluation metrics is *Classification Accuracy*
* `test_data.txt` -- the test set, that is the tweets for which you have to predict the sentiment label.


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

## Status

Right now, we have a preprocessing class that hasn't been too rigorously tested, but should mostly work, except for TODOs.
We need to experiment with it a bit more to the find best settings for our data. Also, the model can be changed (the most obvious optimization might be to *remove* char embeddings).

## TODOs and ideas

* Add pre-trained embeddings (Glove, word2vec, fasttext, ELMO, etc)
  * https://www.tensorflow.org/hub/modules/google/elmo/1
* Experiment with architecture (CNNs, remove char embeddings, etc)
* Experiment with preprocessing and its flags, vocab size, etc
* Try sub-word embeddings
* Add TF-IDF weighting 
* Try http://spacy.io ?
* Add F1-score, prec/recall, embedding visualization
* Remove notebooks?
* Remove <URL> tokens?
* Look at RCNN, Lukas thinks it did exactly this thing on this dataset (custom LSTM cell)
* Optimize vocab size: Look at number of unique words (or actually, at the words themselves).
* ~~Gradient clipping~~
* Try different cells (current -- GRU)
* Remove retweets from dataset (RT and existing tweet after it)
