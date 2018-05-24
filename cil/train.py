from collections import Counter

import nltk
import numpy as np

from . import flags
from .models.rnn import RNN
from .data.preprocessing import Preprocessing
from .data.datasets import Datasets

# Parse arguments
flags.define_flags()
FLAGS = flags.FLAGS

# Fix random seed
np.random.seed(FLAGS.seed)

# Load the data
PREFIX = "data_in/twitter-datasets/"
stemming = getattr(nltk.stem, FLAGS.stemmer)() if FLAGS.stemmer else None
lemmatization = getattr(nltk.stem, FLAGS.lemmatizer)() if FLAGS.lemmatizer else None
data = Datasets(
    # train_pos_file=PREFIX + "train_pos_full.txt",
    train_pos_file=PREFIX + "train_pos.txt",
    # train_neg_file=PREFIX + "train_neg_full.txt",
    train_neg_file=PREFIX + "train_neg.txt",
    test_file=PREFIX + "test_data.txt",
    preprocessing=Preprocessing(
        standardize=FLAGS.standardize,
        segment_hashtags=FLAGS.segment_hashtags,
        contractions=FLAGS.contractions,
        rem_numbers=FLAGS.rem_numbers,
        punct_squash=FLAGS.punct_squash,
        fix_slang=FLAGS.fix_slang,
        word_squash=FLAGS.word_squash,
        expl_negations=FLAGS.expl_negations,
        rem_stopwords=FLAGS.rem_stopwords,
        stemming=stemming,
        lemmatization=lemmatization,
        padding_size=FLAGS.padding_size),
    eval_size=FLAGS.eval_size,
    vocab_size=FLAGS.vocab_size)
data.load()

idx = 100
idx2 = 1
print("vocab\t\t", len(data.word_vocab))
print("vocab words tf\t", len(data.train.vocabulary('words')))
print("vocab chars tf\t", len(data.train.vocabulary('chars')))
print("vocab sent tf \t", data.train.vocabulary('sentiments'))


def unk_percentage(X_words):
    UNK = 1
    counts = Counter()
    for line in X_words:
        counts.update(line)
    return counts[UNK] / sum(counts.values())


print("X_train\t\t", data.X_train[idx][idx2])
# print("X_train_word\t", data.X_train_word[idx, idx2])
print(f"X_train_wordUNK\t {unk_percentage(data.train._word_ids)}")
print("y_train\t\t", data.y_train[idx])

print("X_eval\t\t", data.X_eval[idx][idx2])
# print("X_eval_word\t", data.X_eval_word[idx, idx2])
print(f"X_eval_wordUNK\t {unk_percentage(data.eval._word_ids)}")
print("y_eval\t\t", data.y_eval[idx])

print("X_test\t\t", data.X_test[idx][idx2])
# print("X_test_word\t", data.X_test_word[idx, idx2])
print(f"X_test_wordUNK\t {unk_percentage(data.test._word_ids)}")


def print_data(data, strr):
    print(strr, "dataX", len(data._word_ids), len(data._charseq_ids))
    if hasattr(data, '_sentiments'):
        print(strr, "dataY", len(data._sentiments))
    print(strr, "lens", len(data._sentence_lens))


print_data(data.train, "train")
print_data(data.eval, "eval")
print_data(data.test, "test")

# Construct the network
print("Constructing the network.", flush=True)

network = RNN(
    rnn_cell=FLAGS.rnn_cell,
    rnn_cell_dim=FLAGS.rnn_cell_dim,
    num_words=len(data.train.vocabulary('words')),
    num_chars=len(data.train.vocabulary('chars')),
    logdir=FLAGS.logdir,
    expname=f'epochs{FLAGS.epochs}-bs{FLAGS.batch_size}{"-" + str(FLAGS.exp) if FLAGS.exp else ""}',
    threads=FLAGS.threads,
    word_embedding=FLAGS.word_embedding,
    char_embedding=FLAGS.char_embedding,
    keep_prob=FLAGS.keep_prob,
    learning_rate=FLAGS.learning_rate,
    seed=FLAGS.seed)

# Train
network.train(data, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)
print("End.")
