from collections import Counter

import numpy as np

from . import flags
from .models.rnn import RNN
from .data.preprocessing import Preprocessing
from .data.datasets import Datasets

# Fix random seed
SEED = 42
np.random.seed(SEED)

# Parse arguments
flags.define_flags()
FLAGS = flags.FLAGS

# Load the data
PREFIX = "data_in/twitter-datasets/"
EVAL_SIZE = 0.25
data = Datasets(
    train_pos_file=PREFIX + "train_pos_full.txt",
    # train_pos_file=PREFIX + "train_pos.txt",
    train_neg_file=PREFIX + "train_neg_full.txt",
    # train_neg_file=PREFIX + "train_neg.txt",
    test_file=PREFIX + "test_data.txt",
    preprocessing=Preprocessing(),
    eval_size=EVAL_SIZE,
    vocab_size=20000)
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
expname = f"{FLAGS.rnn_cell}{FLAGS.rnn_cell_dim}-bs{FLAGS.batch_size}-epochs{FLAGS.epochs}"
expname += f"-char{FLAGS.char_embedding}-word{FLAGS.word_embedding}"
network = RNN(
    rnn_cell=FLAGS.rnn_cell,
    rnn_cell_dim=FLAGS.rnn_cell_dim,
    num_words=len(data.train.vocabulary('words')),
    num_chars=len(data.train.vocabulary('chars')),
    logdir=FLAGS.logdir,
    expname=expname,
    threads=FLAGS.threads,
    word_embedding=FLAGS.word_embedding,
    char_embedding=FLAGS.char_embedding,
    keep_prob=FLAGS.keep_prob,
    learning_rate=FLAGS.learning_rate,
    seed=SEED)

# Train
best_eval_accuracy = 0
test_predictions = None

network.train(data, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)
print("End.")
