import numpy as np
import tensorflow as tf
import cil.flags as flags
from cil.rnn import Network
from cil.preprocessing import Datasets, Preprocessing
import sys
import datetime

from collections import Counter

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
    eval_size=EVAL_SIZE,
    vocab_size=20000)
data.load()

idx = 100
idx2 = 1
print("vocab\t\t", len(data.word_vocab))
print("vocab words tf\t", len(data.data_train.vocabulary('words')))
print("vocab chars tf\t", len(data.data_train.vocabulary('chars')))
print("vocab sent tf \t", data.data_train.vocabulary('sentiments'))


def unk_percentage(X_words):
    UNK = 1
    counts = Counter()
    for line in X_words:
        counts.update(line)
    return counts[UNK] / sum(counts.values())


print("X_train\t\t", data.X_train[idx][idx2])
# print("X_train_word\t", data.X_train_word[idx, idx2])
print(f"X_train_wordUNK\t {unk_percentage(data.data_train._word_ids)}")
print("y_train\t\t", data.y_train[idx])

print("X_eval\t\t", data.X_eval[idx][idx2])
# print("X_eval_word\t", data.X_eval_word[idx, idx2])
print(f"X_eval_wordUNK\t {unk_percentage(data.data_eval._word_ids)}")
print("y_eval\t\t", data.y_eval[idx])

print("X_test\t\t", data.X_test[idx][idx2])
# print("X_test_word\t", data.X_test_word[idx, idx2])
print(f"X_test_wordUNK\t {unk_percentage(data.data_test._word_ids)}")


def print_data(data, strr):
    print(strr, "dataX", len(data._word_ids), len(data._charseq_ids))
    if hasattr(data, '_sentiments'):
        print(strr, "dataY", len(data._sentiments))
    print(strr, "lens", len(data._sentence_lens))


print_data(data.data_train, "train")
print_data(data.data_eval, "eval")
print_data(data.data_test, "test")

# Construct the network
print("Constructing the network.", flush=True)
expname = "{}{}-bs{}-epochs{}-char{}-word{}".format(FLAGS.rnn_cell, FLAGS.rnn_cell_dim,
                                                    FLAGS.batch_size, FLAGS.epochs,
                                                    FLAGS.char_embedding, FLAGS.word_embedding)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
exp_id = "{}_{}".format(timestamp, expname)
network = Network(
    rnn_cell=FLAGS.rnn_cell,
    rnn_cell_dim=FLAGS.rnn_cell_dim,
    num_words=len(data.data_train.vocabulary('words')),
    num_chars=len(data.data_train.vocabulary('chars')),
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

for epoch in range(FLAGS.epochs):
    print("Training epoch {}".format(epoch + 1), flush=True)
    network.train_epoch(data.data_train)
    eval_accuracy, eval_loss = network.evaluate_epoch(data.data_eval, "eval")
    print(
        "Evaluation accuracy after epoch {} is {:.2f}. Eval loss is {:.2f}".format(
            epoch + 1, 100. * eval_accuracy, eval_loss),
        flush=True)

    if eval_accuracy > best_eval_accuracy:
        best_eval_accuracy = eval_accuracy
        test_predictions = network.predict_epoch(data.data_test, "test")

        # Print test predictions
        out_file = "data_out/predictions_{}_epoch_{}_{}.csv".format(exp_id, epoch, eval_accuracy)
        with open(out_file, "w+") as f:
            print("Id,Prediction", file=f)
            for i, prediction in enumerate(test_predictions):
                print(
                    "{},{}".format(
                        i + 1,
                        int(data.data_test.vocabulary('sentiments')[prediction]) * 2 - 1),
                    file=f)
        print("Exported predictions to", out_file, flush=True)
        print(flush=True)
print("End.")
