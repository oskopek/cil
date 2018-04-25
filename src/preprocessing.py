from collections import Counter
import datetime
import os
import sys

import numpy as np
import nltk

SEED = 42

# Replace missing values with the default value, but do not insert them.
class missingdict(dict):

    def __init__(self, default_val=None, *args, **kwargs):
        super(missingdict, self).__init__(*args, **kwargs)
        self.default_val = default_val

    def __missing__(self, key):
        return self.default_val


class CILDataset:
    """Class capable of loading CIL Twitter dataset."""

    def __init__(self, lines, sentiments, word_vocab=None, train=None):
        """Load dataset from the given files.

        Arguments:
        train: If given, the vocabularies from the training data will be reused.
        """

        # Create vocabulary_maps
        if train:
            self._vocabulary_maps = train._vocabulary_maps
        else:
            self._vocabulary_maps = {'chars': {'$pad$': 0, '$unk$': 1},
                                     'sentiments': {0: 0, 1: 1}}
            if word_vocab:
                self._vocabulary_maps['words'] = word_vocab
            else:
                self._vocabulary_maps['words'] = {0: 0, 1: 1}, # pad = 0, unk = 1

        self._word_ids = []
        self._charseq_ids = []
        self._charseqs_map = {'$pad$': 0}
        self._charseqs = []
        self._sentiments = []

        # Load the sentences
        for idx, line in enumerate(lines):
            if sentiments: # if not test
                sentiment = sentiments[idx]
                assert sentiment in self._vocabulary_maps['sentiments']
                self._sentiments.append(self._vocabulary_maps['sentiments'][sentiment])

            self._word_ids.append([])
            self._charseq_ids.append([])
            for word in line:
                # Characters
                if word not in self._charseqs_map:
                    self._charseqs_map[word] = len(self._charseqs)
                    self._charseqs.append([])
                    for c in word:
                        if c not in self._vocabulary_maps['chars']:
                            if not train:
                                self._vocabulary_maps['chars'][c] = len(self._vocabulary_maps['chars'])
                            else:
                                c = '$unk$'
                        self._charseqs[-1].append(self._vocabulary_maps['chars'][c])
                self._charseq_ids[-1].append(self._charseqs_map[word])

                # Words -- missingdict handles unks automatically
                self._word_ids[-1].append(self._vocabulary_maps['words'][word])

        # Compute sentence lengths
        sentences = len(self._word_ids)
        self._sentence_lens = np.zeros([sentences], np.int32)
        for i in range(sentences):
            self._sentence_lens[i] = len(self._word_ids[i])

        # Create vocabularies
        if train:
            self._vocabularies = train._vocabularies
        else:
            self._vocabularies = {}
            for feature, words in self._vocabulary_maps.items():
                self._vocabularies[feature] = [""] * len(words)
                for word, id in words.items():
                    self._vocabularies[feature][id] = word

        self._permutation = np.random.permutation(len(self._sentence_lens))

        
    def vocabulary(self, feature):
        """Return vocabulary for required feature.

        The features are the following:
        words
        chars
        sentiments
        """
        return self._vocabularies[feature]

    def next_batch(self, batch_size):
        """Return the next batch.

        Arguments:
        Returns: (sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, sentiments)
        sequence_lens: batch of sentence_lens
        word_ids: batch of word_ids
        charseq_ids: batch of charseq_ids (the same shape as word_ids, but with the ids pointing into charseqs).
        charseqs: unique charseqs in the batch, indexable by charseq_ids;
          contain indices of characters from vocabulary('chars')
        charseq_lens: length of charseqs
        sentiments: batch of sentiments
        
        batch: [string]
        
        sequence_lens: tweet -> len([word_id]) == len([charseq_id]) # number of words per tweet
        word_ids: tweet -> [word_id] # 
        word_vocab: word -> word_id
        charseq_ids: tweet -> [charseq_id]
        charseqs: charseq_id -> [char_id]
        charseq_lens: word_id -> len([char_id])
        char_vocab: char -> char_id
        """

        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        return self._next_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens))
            return True
        return False

    def whole_data_as_batch(self):
        """Return the whole dataset in the same result as next_batch.

        Returns the same results as next_batch.
        """
        return self._next_batch(np.arange(len(self._sentence_lens)))

    def _next_batch(self, batch_perm):
        batch_size = len(batch_perm)

        # General data
        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        batch_word_ids = np.zeros([batch_size, max_sentence_len], np.int32)
        for i in range(batch_size):
            batch_word_ids[i, 0:batch_sentence_lens[i]] = self._word_ids[batch_perm[i]]
        
        batch_sentiments = np.zeros([batch_size], np.int32)
        for i in range(batch_size):
            batch_sentiments[i] = self._sentiments[batch_perm[i]]

        # Character-level data
        batch_charseq_ids = np.zeros([batch_size, max_sentence_len], np.int32)
        charseqs_map, charseqs, charseq_lens = {}, [], []
        for i in range(batch_size):
            for j, charseq_id in enumerate(self._charseq_ids[batch_perm[i]]):
                if charseq_id not in charseqs_map:
                    charseqs_map[charseq_id] = len(charseqs)
                    charseqs.append(self._charseqs[charseq_id])
                batch_charseq_ids[i, j] = charseqs_map[charseq_id]

        batch_charseq_lens = np.array([len(charseq) for charseq in charseqs], np.int32)
        batch_charseqs = np.zeros([len(charseqs), np.max(batch_charseq_lens)], np.int32)
        for i in range(len(charseqs)):
            batch_charseqs[i, 0:len(charseqs[i])] = charseqs[i]

        return batch_sentence_lens, batch_word_ids, batch_charseq_ids, batch_charseqs, batch_charseq_lens, batch_sentiments

class Preprocessing(object):
    methods = None

    PAD_SYMBOL = "$pad$"
    UNK_SYMBOL = "$unk$"
    BASE_VOCAB = {PAD_SYMBOL: 0, UNK_SYMBOL: 1}

    def __init__(self, standardize=True, normalize=True, rem_numbers=True, punct_squash=True, fix_slang=True, word_squash=True, expl_negations=True, rem_stopwords=True, stemming=nltk.stem.PorterStemmer, lemmatization=nltk.stem.WordNetLemmatizer, padding_size=40):
        self.padding_size = padding_size
        self.methods = [
                # line operations
                (self.standardize, standardize),
                (self.normalize, normalize),
                (self.rem_numbers, rem_numbers),
                (self.lines_to_matrix, True),
                # matrix operations
                (self.punct_squash, punct_squash),
                (self.fix_slang, fix_slang),
                (self.word_squash, word_squash),
                (self.expl_negations, expl_negations),
                (self.rem_stopwords, rem_stopwords),
                (self.stemming, stemming),
                (self.lemmatization, lemmatization)
        ]

    def transform(self, lines, labels=None): # labels == None => test transformation
        for fn, args in self.methods:
            # assert len(lines) == len(labels)
            if args:
                lines, labels = fn(lines, labels, args)
        return lines, labels

    def standardize(self, lines, labels, args):
        def _standardize(lines):
            for line in lines:
                newline = line.strip().split()
                newline = " ".join([w.strip().lower() for w in newline])
                yield newline

        return _standardize(lines), labels

    def normalize(self, lines, labels, args):
        return lines, labels

    def rem_numbers(self, lines, labels, args):
        return lines, labels

    def lines_to_matrix(self, lines, labels, args):
        lines = list(lines)
        for i, line in enumerate(lines):
            lines[i] = line.split()
        return lines, labels

    def punct_squash(self, lines, labels, args):
        return lines, labels

    def fix_slang(self, lines, labels, args):
        return lines, labels

    def word_squash(self, lines, labels, args):
        return lines, labels

    def expl_negations(self, lines, labels, args):
        return lines, labels

    def rem_stopwords(self, lines, labels, args):
        return lines, labels

    def stemming(self, lines, labels, args):
        return lines, labels

    def lemmatization(self, lines, labels, args):
        return lines, labels

    def _vocab_downsize_dict(self, lines, vocab, inv_vocab):
        lines = np.asarray(lines)
        data = np.full((len(lines), self.padding_size), "$pad$", dtype=object)
        cut_counter = 0
        for i, line in enumerate(lines):
            strs = np.asarray(line).astype(object)
            fin_len = min(self.padding_size, len(strs))
            data[i, :fin_len] = strs[:fin_len]
            if len(strs) > self.padding_size:
                cut_counter += 1
        if cut_counter > 0:
            print("WARNING: Cut {} sentences to length {}.".format(cut_counter, self.padding_size))

        data = np.vectorize(lambda word: inv_vocab[vocab[word]])(data)
        return data

    def _vocab_downsize_tosize(self, lines, vocab_size):
        counter = Counter()
        for line in lines:
            counter.update(line)

        vocab = dict(self.BASE_VOCAB)
        uid = len(self.BASE_VOCAB)
        
        for word, _ in counter.most_common(vocab_size - len(self.BASE_VOCAB)):
            assert word not in vocab
            vocab[word] = uid
            uid += 1

        return missingdict(vocab[self.UNK_SYMBOL], vocab)

    def vocab(self, lines, vocab_downsize):
        if isinstance(vocab_downsize, int):
            vocab = self._vocab_downsize_tosize(lines, vocab_downsize)
            inv_vocab = {v: k for k, v in vocab.items()}
            return vocab, inv_vocab
        else:
            return self._vocab_downsize_dict(lines, *vocab_downsize)

from sklearn.model_selection import train_test_split

class Datasets(object):
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

    data_train = None
    data_eval = None
    data_test = None

    def __init__(self, train_pos_file, train_neg_file, test_file, eval_size=0.33, random_state=42, preprocessing=Preprocessing(), vocab_size=20000):
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
        X_test = [line.split(sep=',', maxsplit=1)[1] for line in X_test] # remove numbers

        print("Splitting...")
        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=self.eval_size, random_state=self.random_state)

        print("Preprocessing...")
        X_train, y_train = self.preprocessing.transform(X_train, labels=y_train)
        X_eval, y_eval = self.preprocessing.transform(X_eval, labels=y_eval)
        X_test, _ = self.preprocessing.transform(X_test, labels=None)
        
        print("Generating vocabulary...")
        word_vocab, inv_word_vocab = self.preprocessing.vocab(X_train, vocab_downsize=self.vocab_size)
        # X_train_word = self.preprocessing.vocab(X_train, vocab_downsize=(word_vocab, inv_word_vocab))
        # X_eval_word = self.preprocessing.vocab(X_eval, vocab_downsize=(word_vocab, inv_word_vocab))
        # X_test_word = self.preprocessing.vocab(X_test, vocab_downsize=(word_vocab, inv_word_vocab))

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
        self.data_train = CILDataset(X_train, y_train, word_vocab=self.word_vocab)
        self.data_eval = CILDataset(X_eval, y_eval, train=self.data_train)
        self.data_test = CILDataset(X_test, None, train=self.data_train)

    def batches_per_epoch_generator(self, batch_size, data=None, shuffle=True):
        if data is None:
            data = self.X_train_word

        n_rows = data.shape[0]
        if shuffle:
            train_permutation = np.random.permutation(n_rows)
        else:
            train_permutation = np.arange(n_rows)

        for i in range(0, n_rows, batch_size):
            batch = data[train_permutation[i: i + batch_size]]
            if len(batch) == 0:
                raise StopIteration
            else:
                yield batch


if __name__ == "__main__":
    PREFIX = "../data_in/twitter-datasets/"
    EVAL_SIZE = 0.33
    # data = Datasets(train_pos_file=PREFIX + "train_pos_full.txt", train_neg_file=PREFIX + "train_pos_full.txt", test_file=PREFIX + "test_data.txt", eval_size=EVAL_SIZE, vocab_size=20000)
    data = Datasets(train_pos_file=PREFIX + "train_pos.txt", train_neg_file=PREFIX + "train_pos.txt", test_file=PREFIX + "test_data.txt", eval_size=EVAL_SIZE, vocab_size=20000)
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