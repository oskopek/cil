# Computational Intelligence Lab 2018 -- Text Sentiment Classification
### Larissa Laich, Lukas Jendele, Ondrej Skopek, Michael Wiegner
Department of Computer Science, ETH Zurich, Switzerland

## 1. Project definition

Perform text sentiment classification on Twitter data (tweets). The goal is to develop an automatic method to reveal the authors' opinion/sentiment.

## 2. Report

The report is located at: `report/report.pdf`.

## 3. Setup

The following steps will prepare your environment to begin training and evaluating models.

* For Tensorflow on CPU, change the `tensorflow-gpu` line in `requirements.txt` to `tensorflow`.
  * Do note that we expect the usual Linux utilities to be installed, f.e. `bash`, `wget`, etc.
  * We also expect the executables `python` and `python3` to point to a Python 3.6 or newer version of Python, and `python2` to point to a Python 2.7+ version of Python 2.
* For downloading the data (part of the setup process), please make sure to be on the ETH network.
* After everything is ready, simply run: `make setup` in a terminal from the project root directory.

## 4. Training

To train, run `make train` from the main project folder.
To train on Leonhard (submit a job), run `make job` from the main project folder.
The model specified by `cil/flags.py` will be trained.

### 4.1. Neural network experiments

You can setup the training to use a specific experiment by running one of the following commands
prior to `make train` or `make job`:

```
make lstm128
make lstm128_ce
make lstm128_we
make gru256
make stacklstm
make cnn512
```

### 4.2. Baselines and external models

Use the following commands to directly run the baselines and external models.
You may need to change paths in the beginning of scripts (especially for fasttext),
if you are not running the models on Leonhard.

```
# GloVe
make glove

# FastText
make fasttext

# Transformer
make transformer-train-serve # in one terminal
make transformer-predict # after training finishes, in second terminal
```

To run Transformer and FastText, please read on for some preconditions and requirements.

#### 4.2.1. Transformer

You need to install the TensorFlow serving server:

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server
```

#### 4.2.2. FastText

Please make sure you have `make`, `gcc`/`g++`, and other standard Linux build tools installed to build FastText.


### 4.3. Ensemble

The ensemble was computed offline using Excel, classifying based on the majority
vote on the test set predictions.
The Excel sheet we used is uploaded to `data_out` and named `prediction_ensemble.xlsx`


## 5. Project structure

* `cil/` - root Python package for all our neural models, data loading, ...
    * `data/` - data loading and preprocessing
    * `experiments/` - flag files for reproducing experiments from the report
    * `models/` - neural network architecture implementations
* `data_in/` - input twitter data and preprocessed data + some preprocessing scripts
* `data_out/` - output predictions, TensorBoard events, ensemble spreadsheet, ...
* `fasttext/` - scripts to run the FastText classifier
* `glove` - glove embeddings with logistic regression or random forests
* `report/` - the final report
* `transformer/` - the Transformer model


## 6. Training Data

The training data consists of 2.5M tweets classified as either positive or negative.
All tweets have been tokenized already, so that the words and punctuation are properly separated by a whitespace.
All our data files are in `data_in/twitter-datasets/`.

The original files were:
* `train_pos.txt` and `train_neg.txt` -- a small set of training tweets for each of the two classes.
* `train_pos_full.txt` and `train_neg_full.txt` -- a complete set of training tweets for each of the two classes, about 1M tweets per class.

For all our experiments, we used a fixed train/eval/test split, computed by the `data_in/train_test_split.py`
(with a fixed random seed) script upon running the setup using `make setup`.
The data split (after running setup) is available at:
* `data_in/twitter-datasets/train_data.txt`
* `data_in/twitter-datasets/eval_data.txt`

