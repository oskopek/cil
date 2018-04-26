from collections import Counter
import re

import numpy as np
import nltk

from sklearn.model_selection import train_test_split

# from nltk.stem.api import StemmerI
# from nltk.stem.regexp import RegexpStemmer
# from nltk.stem.lancaster import LancasterStemmer
# from nltk.stem.isri import ISRIStemmer
# from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.stem.rslp import RSLPStemmer

SEED = 42


# Replace missing values with the default value, but do not insert them.
class missingdict(dict):
    def __init__(self, *args, default_val=None, **kwargs):
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
        self.is_train = train is None  # if train is none, this dataset is the training one

        # Create vocabulary_maps
        if train:
            self._vocabulary_maps = train._vocabulary_maps
        else:
            self._vocabulary_maps = {'chars': {'$pad$': 0, '$unk$': 1}, 'sentiments': {0: 0, 1: 1}}
            if word_vocab:
                self._vocabulary_maps['words'] = word_vocab
            else:
                self._vocabulary_maps['words'] = {0: 0, 1: 1},  # pad = 0, unk = 1

        self._word_ids = []
        self._charseq_ids = []
        self._charseqs_map = {'$pad$': 0}
        self._charseqs = []
        if sentiments:
            self._sentiments = []

        # Load the sentences
        for idx, line in enumerate(lines):
            if sentiments:  # if not test
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
                                self._vocabulary_maps['chars'][c] = len(
                                    self._vocabulary_maps['chars'])
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

        self._new_permutation()

    def _new_permutation(self):
        if self.is_train:
            self._permutation = np.random.permutation(len(self._sentence_lens))
        else:
            self._permutation = np.arange(len(self._sentence_lens))

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
            self._new_permutation()
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

        if hasattr(self, '_sentiments'):  # not test
            batch_sentiments = np.zeros([batch_size], np.int32)
            for i in range(batch_size):
                batch_sentiments[i] = self._sentiments[batch_perm[i]]
        else:
            batch_sentiments = None

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

    def __init__(
            self,
            standardize=True,
            segment_hashtags=10,
            contractions=True,
            rem_numbers=True,
            punct_squash=True,
            fix_slang=True,
            word_squash=3,
            expl_negations=False,
            rem_stopwords=False,
            stemming=nltk.stem.PorterStemmer(),
            #  stemming=None,
            #  lemmatization=nltk.stem.WordNetLemmatizer(),
            lemmatization=None,
            padding_size=40):
        self.padding_size = padding_size
        self.methods = [
            # line operations
            (self.standardize, standardize),
            (self.word_squash, word_squash),
            (self.segment_hashtags, segment_hashtags),
            (self.fix_slang, fix_slang),
            (self.contractions, contractions),
            (self.rem_numbers, rem_numbers),
            (self.punct_squash, punct_squash),
            (self.lines_to_matrix, True),

            # matrix operations
            (self.expl_negations, expl_negations),
            (self.rem_stopwords, rem_stopwords),
            (self.stemming, stemming),
            (self.lemmatization, lemmatization)
        ]

    def transform(self, lines, labels=None):  # labels == None => test transformation
        for fn, args in self.methods:
            # assert len(lines) == len(labels)
            if args:
                lines, labels = fn(lines, labels, args)
        return lines, labels

    def contractions(self, lines, labels, args):
        re_map = [
            (r" ([a-z]+)'re ", r" \1 are "),
            (r" youre ", " you are "),
            (r" (it|that|he|she|what|there|who|here|where|how)'s ", r" \1 is "),
            (r" i[ ]?'[ ]?m ", " i am "),
            (r" can't ", " can not "),
            (r" ain't ", " am not "),
            (r" won't ", " will not "),
            (r" ([a-z]+)n't ", r" \1 not "),
            (r" ([a-z]+)'ll ", r" \1 will "),
            (r" ([a-z]+)'ve ", r" \1 have "),
            (r" (i|you|he|she|it|we|they|u)'d ", r" \1 would "),
            (r" (how|why|where|what)'d ", r" \1 did "),
            (r" ([a-z]+)'d ", r" \1 "),  # just remove it here "lol'd"
        ]

        re_map = [(re.compile(x), y) for x, y in re_map]

        def contraction_map(lines):
            for line in lines:
                for reg, subs in re_map:
                    line = re.sub(reg, subs, line)
                yield line

        return contraction_map(lines), labels

    def standardize(self, lines, labels, args):
        def _standardize(lines):
            for line in lines:
                newline = line.strip().split()
                newline = " ".join([w.strip().lower() for w in newline])
                yield newline

        return _standardize(lines), labels

    def segment_hashtags(self, lines, labels, limit):
        #https://stackoverflow.com/questions/38125281/split-sentence-without-space-in-python-nltk

        WORDS = nltk.corpus.sentence_polarity.words() + ["tweet"] * 2000 + ["tweets"] * 2000
        COUNTS = Counter(WORDS)

        def pdist(counter):
            "Make a probability distribution, given evidence from a Counter."
            N = sum(counter.values())
            return lambda x: counter[x] / N

        P = pdist(COUNTS)

        def Pwords(words):
            "Probability of words, assuming each word is independent of others."
            return product(P(w) for w in words)

        def product(nums):
            "Multiply the numbers together.  (Like `sum`, but with multiplication.)"
            result = 1
            for x in nums:
                result *= x
            return result

        def splits(text, start=0, L=20):
            "Return a list of all (first, rest) pairs; start <= len(first) <= L."
            return [(text[:i], text[i:]) for i in range(start, min(len(text), L) + 1)]

        def segment(text):
            "Return a list of words that is the most probable segmentation of text."
            if not text:
                return []
            else:
                candidates = ([first] + segment(rest) for (first, rest) in splits(text, 1))
                return max(candidates, key=Pwords)

        num_pattern = re.compile(r'# ')
        two_pattern = re.compile(r'([^0-9])2([^0-9])')
        four_pattern = re.compile(r'([^0-9])4([^0-9])')
        hashtag_pattern = re.compile(r'#[^ ]+')

        # TODO: debug this

        def build_hashtag_map(X, hmap={}):
            for line in X:
                line = re.sub(num_pattern, '$hashtag_symbol$', line)
                line = re.sub(two_pattern, r'\1to\2', line)
                line = re.sub(four_pattern, r'\1for\2', line)
                new_line = line
                for hashtag in re.finditer(hashtag_pattern, line):
                    hashtag = hashtag.group(0)[1:].strip()
                    # print(hashtag)
                    if hashtag not in hmap:
                        if len(hashtag) > limit:  # TODO: increase this number
                            #           print("Ignoring segmentation of '{}' (too long)".format(hashtag))
                            continue

                        segmented = segment(hashtag)
                        if len(segmented) >= len(hashtag) / 3 * 2:
                            # print("Ignoring segmentation '{}' of '{}'".format(segmented, hashtag))
                            hmap[hashtag] = hashtag
                        else:
                            hmap[hashtag] = segmented
                            # print("Segmented:", hashtag, hmap[hashtag])
                    new_line = new_line.replace('#' + hashtag, "# " + " ".join(hmap[hashtag]))
                yield new_line

        lines = build_hashtag_map(lines)
        return lines, labels

    def rem_numbers(self, lines, labels, args):
        re_map = [
            (r" [0-9]+ ", " "),
            (r"[0-9]+", " "),
        ]

        re_map = [(re.compile(x), y) for x, y in re_map]

        def num_map(lines):
            for line in lines:
                for reg, subs in re_map:
                    line = re.sub(reg, subs, line)
                yield line

        return num_map(lines), labels

    def lines_to_matrix(self, lines, labels, args):
        lines = list(lines)
        if labels:
            if len(lines) != len(labels):
                print("Lines", len(lines), "labels", len(labels))
                assert len(lines) != len(labels)
        for i, line in enumerate(lines):
            lines[i] = line.split()
        return lines, labels

    def punct_squash(self, lines, labels, args):
        pattern = re.compile(r"([^a-z0-9] ?)\1+")
        repl = r" \1 "

        def gen_punct_squash(lines):
            for line in lines:
                yield re.sub(pattern, repl, line)

        return gen_punct_squash(lines), labels

    def fix_slang(self, lines, labels, args):
        re_map = [
            (r"<[ ]*3", " $heart_emoji$ "),
            (r"<[ ]*[\\/][ ]*3", " $brokenheart_emoji$ "),
            (r"r[ ]*\.[ ]*i[ ]*\.[ ]*p", " $rest_in_peace$ "),
            (r" u r ", " you're "),
            (r" ur ", " your "),
            (r" gd ", " good "),
            (r" tht ", " that "),
            (r"[?] [!]", " ?! "),
            (r"[:=][ ]*[ -]*[(c@{<]", " $sad_emoji$ "),
            (r"[:=][ ]*'[ -]*[ (]+[ $]", " $crying_emoji$ "),
            (r"[:=][ ]*'[ -]*[ )]+[ $]", " $happy_tears_emoji$ "),
            (r"[:=][ ]*'[ -]*[ $]", " $cry_unk_emoji$ "),
            (r"[:=][ -]*[o0 ]+[ $]", " $surprise_emoji$ "),
            (r"\:[ -]*[*x ]+[ $]", " $kiss_emoji$ "),
            (r"[;*][ -]*[d) ]+[ $]", " $wink_emoji$ "),
            (r"[:=x][ -]*[p ]+[ $]", " $tongue_emoji$ "),
            (r"@", " at "),
            (r"&", " and "),
            (r"pl[sz]+", " please "),
            (r"\.\.\.", " $three_dots$ "),
            (r"\.\.", " $three_dots$ "),  # even though they are two
        ]

        re_map = [(re.compile(x), y) for x, y in re_map]

        emoji_map = {
            ":‑) :-)) :) :-] :] :-3 :3 :-> :> 8-) 8) :-} :} :o) :c) :^) =] =)": " $happy_emoji$ ",
            ":‑D :D 8‑D 8D x‑D xD X‑D XD =D =3 B^D": " $laugh_emoji$ ",
            "]:": " $sad_emoji$ ",
            '/: -_- :| :| :‑/ :/ :‑. >:\\ >:/ :\\ =/ =\\ :L =L :S': " $skeptical_emoji$ ",
            ":$": " $embarrassed_emoji$ ",
            ":‑X :X :‑# :# :‑& :&": " $tongue_tied_emoji$ ",
            "O:‑) O:) 0:‑3 0:3 0:‑) 0:) 0;^)": " $angel_emoji$ ",
            ">:‑) >:) }:‑) }:) 3:‑) 3:) >;)": " $evil_emoji$ ",
            "|;‑) 8-) B)": " $cool_emoji$ ",
            "|‑O": " $bored_emoji$ ",
            ":‑J": " $tongue_in_cheek_emoji$ ",
            "#‑)": " $partied_all_night_emoji$ ",
            "%‑) %)": " $confused_emoji$ ",
            ":‑# :#": " $sick_emoji$ ",
            "<:‑|": " $dumb_emoji$ ",
            "<_< >_>": " $guilty_emoji$ ",
            '\\o/ *\\0/*': " $cheer_emoji$ "
        }

        subs_map = []
        for key, val in emoji_map.items():
            for k2 in key.split():
                subs_map.append((f"{k2}", val))

        def mp(lines):
            for line in lines:
                for repl, wth in subs_map:
                    line = line.replace(repl, wth)
                for reg, subs in re_map:
                    line = re.sub(reg, subs, line)
                yield line

        return mp(lines), labels

    def word_squash(self, lines, labels, args):
        squash = str(args - 1)
        three_chars = re.compile(r"(\w)\1{" + squash + ",}")

        def squash_gen(lines):
            for line in lines:
                yield re.sub(three_chars, r"\1", line)

        return squash_gen(lines), labels

    def expl_negations(self, lines, labels, args):
        # TODO: add NOT_ prefix to every word up to a punctuation mark
        return lines, labels

    def rem_stopwords(self, lines, labels, args):
        stop_words = set(nltk.corpus.stopwords.words('english'))

        def gen_stopwords(lines):
            for i, line in enumerate(lines):
                new_line = []
                for word in line:
                    if word not in stop_words:
                        new_line.append(word)
                lines[i] = new_line
            return lines

        return gen_stopwords(lines), labels

    def stemming(self, lines, labels, stemmer):
        def gen_stem(lines):
            for i, line in enumerate(lines):
                new_line = []
                for word in line:
                    stemmed = stemmer.stem(word)
                    new_line.append(stemmed)
                lines[i] = new_line
            return lines

        return gen_stem(lines), labels

    def lemmatization(self, lines, labels, lemmatizer):
        def gen_lemma(lines):
            for i, line in enumerate(lines):
                new_line = []
                for word in line:
                    lemma = lemmatizer.lemmatize(word)
                    new_line.append(lemma)
                lines[i] = new_line
            return lines

        return gen_lemma(lines), labels

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

        return missingdict(vocab, default_val=vocab[self.UNK_SYMBOL])

    def vocab(self, lines, vocab_downsize):
        if isinstance(vocab_downsize, int):
            vocab = self._vocab_downsize_tosize(lines, vocab_downsize)
            inv_vocab = {v: k for k, v in vocab.items()}
            return vocab, inv_vocab
        else:
            return self._vocab_downsize_dict(lines, *vocab_downsize)


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

    def __init__(self,
                 train_pos_file,
                 train_neg_file,
                 test_file,
                 eval_size=0.33,
                 random_state=42,
                 preprocessing=Preprocessing(),
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
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train, y_train, test_size=self.eval_size, random_state=self.random_state)

        print("Preprocessing...")
        X_train, y_train = self.preprocessing.transform(X_train, labels=y_train)
        X_eval, y_eval = self.preprocessing.transform(X_eval, labels=y_eval)
        X_test, _ = self.preprocessing.transform(X_test, labels=None)

        print("Generating vocabulary...")
        word_vocab, inv_word_vocab = self.preprocessing.vocab(
            X_train, vocab_downsize=self.vocab_size)
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
            batch = data[train_permutation[i:i + batch_size]]
            if len(batch) == 0:
                raise StopIteration
            else:
                yield batch


if __name__ == "__main__":
    PREFIX = "../data_in/twitter-datasets/"
    EVAL_SIZE = 0.25
    data = Datasets(
        # train_pos_file=PREFIX + "train_pos_full.txt",
        train_pos_file=PREFIX + "train_pos.txt",
        # train_neg_file=PREFIX + "train_neg_full.txt",
        train_neg_file=PREFIX + "train_neg.txt",
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
