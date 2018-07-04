from collections import Counter

import nltk
import numpy as np

from . import flags
from .models.cnn import CNN
from .models.rnn import RNN
from .models.rnn_we import RNNWE
from .models.rnn_ce import RNNCE
from .models.stackrnn import StackRNN
from .data.preprocessing import Preprocessing
from .data.datasets import Datasets

# Parse arguments
flags.define_flags()
FLAGS = flags.FLAGS

# Fix random seed
np.random.seed(FLAGS.seed)

# Preprocessing settings
preprocessing = Preprocessing()
if FLAGS.enable_preprocessing:
    stemming = getattr(nltk.stem, FLAGS.stemmer)() if FLAGS.stemmer else None
    lemmatization = getattr(nltk.stem, FLAGS.lemmatizer)() if FLAGS.lemmatizer else None
    preprocessing = Preprocessing(
        standardize=FLAGS.standardize,
        segment_hashtags=FLAGS.segment_hashtags,
        contractions=FLAGS.contractions,
        rem_numbers=FLAGS.rem_numbers,
        punct_squash=FLAGS.punct_squash,
        fix_slang=FLAGS.fix_slang,
        word_squash=FLAGS.word_squash,
        rem_stopwords=FLAGS.rem_stopwords,
        stemming=stemming,
        lemmatization=lemmatization,
    )

# Load the data
PREFIX = "data_in/twitter-datasets"
data = Datasets(
    train_file=f"{PREFIX}/train_data.txt",
    eval_file=f"{PREFIX}/eval_data.txt",
    test_file=f"{PREFIX}/test_data.txt",
    preprocessing=preprocessing,
    vocab_size=FLAGS.vocab_size,
    padding_size=FLAGS.padding_size)
data.load()

idx = 100
idx2 = 1
print("vocab\t\t", len(data.word_vocab))
print("vocab words tf\t", len(data.train.vocabulary('words')))
print("vocab chars tf\t", len(data.train.vocabulary('chars')))
print("vocab sent tf \t", data.train.vocabulary('labels'))


def unk_percentage(X_words):
    UNK = 1
    counts = Counter()
    for line in X_words:
        counts.update(line)
    return counts[UNK] / sum(counts.values())


def cut_percentage(X_words):
    padding_size = FLAGS.padding_size
    if padding_size is None:
        return 0
    cut = 0
    words = 0
    for line in X_words:
        cut += max(0, len(line) - padding_size)
        words += len(line)
    return cut / words


print("X_train\t\t", data.X_train[idx][idx2])
# print("X_train_word\t", data.X_train_word[idx, idx2])
print(f"X_train_wordUNK\t {unk_percentage(data.train._word_ids)}")
print(f"X_train_wordCUT\t {cut_percentage(data.train._word_ids)}")
print("y_train\t\t", data.y_train[idx])

print("X_eval\t\t", data.X_eval[idx][idx2])
# print("X_eval_word\t", data.X_eval_word[idx, idx2])
print(f"X_eval_wordUNK\t {unk_percentage(data.eval._word_ids)}")
print(f"X_eval_wordCUT\t {cut_percentage(data.train._word_ids)}")
print("y_eval\t\t", data.y_eval[idx])

print("X_test\t\t", data.X_test[idx][idx2])
# print("X_test_word\t", data.X_test_word[idx, idx2])
print(f"X_test_wordUNK\t {unk_percentage(data.test._word_ids)}")
print(f"X_test_wordCUT\t {cut_percentage(data.train._word_ids)}")


def print_data(data, strr):
    print(strr, "dataX", len(data._word_ids), len(data._charseq_ids))
    if hasattr(data, '_labels'):
        print(strr, "dataY", len(data._labels))
    print(strr, "lens", len(data._sentence_lens))


print_data(data.train, "train")
print_data(data.eval, "eval")
print_data(data.test, "test")

# Construct the network
print("Constructing the network.", flush=True)

if FLAGS.model == "RNN":
    net_class = RNN
elif FLAGS.model == "RNNWE":
    net_class = RNNWE
elif FLAGS.model == "RNNCE":
    net_class = RNNCE
elif FLAGS.model == "CNN":
    net_class = CNN
elif FLAGS.model == "StackRNN":
    net_class = StackRNN
else:
    raise ValueError(f"Unknown model {FLAGS.model}.")

expname = f'epochs{FLAGS.epochs}-bs{FLAGS.batch_size}{"-" + str(FLAGS.exp) if FLAGS.exp else ""}'
expname = f'{expname}_ttfull'

network = net_class(
    rnn_cell=FLAGS.rnn_cell,
    rnn_cell_dim=FLAGS.rnn_cell_dim,
    attention=FLAGS.attention,
    attention_size=FLAGS.attention_size,
    num_words=len(data.train.vocabulary('words')),
    num_chars=len(data.train.vocabulary('chars')),
    logdir=FLAGS.logdir,
    expname=expname,
    threads=FLAGS.threads,
    word_embedding=FLAGS.word_embedding,
    char_embedding=FLAGS.char_embedding,
    keep_prob=FLAGS.keep_prob,
    learning_rate=FLAGS.learning_rate,
    seed=FLAGS.seed)

# Train
network.train(data, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)
print("End.")
