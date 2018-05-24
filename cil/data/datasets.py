import numpy as np
import sklearn

from .twitter_dataset import TwitterDataset


class Datasets:
    X_train = None
    X_train_word = None
    y_train = None
    X_eval = None
    X_eval_word = None
    y_eval = None
    X_test = None
    X_test_word = None

    word_vocab = None
    inv_word_vocab = None

    train = None
    eval = None
    test = None

    def __init__(self,
                 train_pos_file,
                 train_neg_file,
                 test_file,
                 preprocessing,
                 eval_size=0.33,
                 random_state=42,
                 vocab_size=20000):
        self.train_pos_file = train_pos_file
        self.train_neg_file = train_neg_file
        self.test_file = test_file
        self.eval_size = eval_size
        self.random_state = random_state
        self.preprocessing = preprocessing
        self.vocab_size = vocab_size

    @staticmethod
    def _read_lines(file):
        with open(file, "r") as f:
            lines = f.readlines()
        return lines

    def load(self):
        print("Loading data from disk...")
        X_train_pos = Datasets._read_lines(self.train_pos_file)
        X_train_neg = Datasets._read_lines(self.train_neg_file)
        y_train = [1] * len(X_train_pos) + [0] * len(X_train_neg)
        X_train = X_train_pos + X_train_neg
        del X_train_pos, X_train_neg

        X_test = Datasets._read_lines(self.test_file)
        X_test = [line.split(sep=',', maxsplit=1)[1] for line in X_test]  # remove numbers

        print("Splitting...")
        X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
            X_train, y_train, test_size=self.eval_size, random_state=self.random_state)

        print("Preprocessing...")
        X_train, y_train = self.preprocessing.transform(X_train, labels=y_train)
        X_eval, y_eval = self.preprocessing.transform(X_eval, labels=y_eval)
        X_test, _ = self.preprocessing.transform(X_test, labels=None)

        print("Generating vocabulary...")
        word_vocab, inv_word_vocab = self.preprocessing.vocab(
            X_train, vocab_downsize=self.vocab_size)
        # X_train_word = self.preprocessing.vocab(
        #     X_train, vocab_downsize=(word_vocab, inv_word_vocab))
        # X_eval_word = self.preprocessing.vocab(
        #     X_eval, vocab_downsize=(word_vocab, inv_word_vocab))
        # X_test_word = self.preprocessing.vocab(
        #     X_test, vocab_downsize=(word_vocab, inv_word_vocab))

        self.X_train = X_train
        # self.X_train_word = X_train_word
        self.y_train = y_train

        self.X_eval = X_eval
        # self.X_eval_word = X_eval_word
        self.y_eval = y_eval

        self.X_test = X_test
        # self.X_test_word = X_test_word

        self.word_vocab = word_vocab
        self.inv_word_vocab = inv_word_vocab

        print("Generating TF data...")
        self.train = TwitterDataset(X_train, y_train, word_vocab=self.word_vocab)
        self.eval = TwitterDataset(X_eval, y_eval, train=self.train)
        self.test = TwitterDataset(X_test, None, train=self.train)

    def batches_per_epoch_generator(self, batch_size, data=None, shuffle=True):
        if data is None:
            data = self.X_train_word

        n_rows = data.shape[0]
        if shuffle:
            train_permutation = np.random.permutation(n_rows)
        else:
            train_permutation = np.arange(n_rows)

        for i in range(0, n_rows, batch_size):
            batch = data[train_permutation[i:i + batch_size]]
            if len(batch) == 0:
                raise StopIteration
            else:
                yield batch
