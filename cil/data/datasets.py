import numpy as np

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
                 vocab_size=20000,
                 padding_size=40):
        self.train_pos_file = train_pos_file
        self.train_neg_file = train_neg_file
        self.test_file = test_file
        self.eval_size = eval_size
        self.random_state = random_state
        self.preprocessing = preprocessing
        self.vocab_size = vocab_size
        self.padding_size = padding_size

    @staticmethod
    def _read_lines(filename, quote='"'):
        with open(filename, "r") as f:
            X = []
            y = []
            reader = csv.reader(f, delimiter=',', quotechar=quote)
            for label, line in reader:
                X.append(line)
                y.append(label)
        return X, y

    def load(self):
        print("Loading data from disk...")
        X_train, y_train = Datasets._read_lines(self.train_file)
        X_eval, y_eval = Datasets._read_lines(self.eval_file)
        X_test, _ = Datasets._read_lines(self.test_file, quote=None)
        print(X_train[0], y_train[0])
        print(X_eval[0], y_eval[0])
        print(X_test[0]) # TODO: Debug

        print("Preprocessing...")
        X_train, y_train = self.preprocessing.transform(X_train, labels=y_train)
        X_eval, y_eval = self.preprocessing.transform(X_eval, labels=y_eval)
        X_test, _ = self.preprocessing.transform(X_test, labels=None)

        print("Generating vocabulary...")
        word_vocab, inv_word_vocab = self.preprocessing.vocab(
            X_train, vocab_downsize=self.vocab_size)

        self.X_train = X_train
        self.y_train = y_train

        self.X_eval = X_eval
        self.y_eval = y_eval

        self.X_test = X_test

        self.word_vocab = word_vocab
        self.inv_word_vocab = inv_word_vocab

        print("Generating TF data...")
        self.train = TwitterDataset(
            X_train, y_train, word_vocab=self.word_vocab, padding_size=self.padding_size)
        self.eval = TwitterDataset(X_eval, y_eval, train=self.train, padding_size=self.padding_size)
        self.test = TwitterDataset(X_test, None, train=self.train, padding_size=self.padding_size)

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
