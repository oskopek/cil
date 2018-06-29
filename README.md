# Computational Intelligence Lab 2018 -- Text Sentiment Classification
### Larissa Laich, Lukas Jendele, Michael Wiegner, Ondrej Skopek
Department of Computer Science, ETH Zurich, Switzerland

## Project definition
The use of microblogging and text messaging as a media of communication has greatly increased over the past 10 years. Such large volumes of data amplifies the need for automatic methods to understand the opinion conveyed in a text.

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

#### Glove



#### Fasttext



#### Ensemble vote





## Structure

* `datasets/` - all data sources required for training/validation/testing.
* `notebooks/` - visualization notebooks.
* `outputs/` - any output for a model will be placed here, including logs, summaries, checkpoints, and Kaggle submission `.csv` files.
* `report/` - report sources and PDF.
* `src/` - all source code.
    * `core/` - base classes
    * `datasources/` - routines for reading and preprocessing entries for training and testing
    * `models/` - neural network definitions
    * `schedules/` - training schedules
    * `synthgan/` - synthetic refinement of UnityEyes using CycleGan.
    * `util/` - utility methods
    * `main.sh` - main training script to run
    * `train.py` - training script for VGG19
* `setup/` - environment setup scripts and configuration files.
* `venv/` - the virtualenv.

### Outputs

When your model has completed training, it will perform a full evaluation on the test set. For class `ExampleNet`, this output can be found in the folder `outputs/ExampleNet/` as `to_submit_to_kaggle_XXXXXXXXX.csv`.

Submit this `csv` file to our page on [Kaggle](https://www.kaggle.com/c/mp18-eye-gaze-estimation/submissions).


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
