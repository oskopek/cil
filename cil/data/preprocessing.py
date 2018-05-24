from collections import Counter
import re

import numpy as np
import nltk

from .utils import UNK_TOKEN, PAD_TOKEN, MissingDict, BASE_VOCAB


class Preprocessing:
    methods = None

    def __init__(
            self,
            standardize=False,
            segment_hashtags=0,
            contractions=False,
            rem_numbers=False,
            punct_squash=False,
            fix_slang=False,
            word_squash=0,
            expl_negations=False,
            rem_stopwords=False,
            stemming=None,
            lemmatization=None,
    ):
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

    def segment_hashtags(self, lines, labels, limit):  # TODO(oskopek): Remove me.
        # https://stackoverflow.com/questions/38125281/split-sentence-without-space-in-python-nltk

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

        # TODO: debug this?

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
                        if len(hashtag) > limit:  # TODO: increase this number?
                            # print("Ignoring segmentation of '{}' (too long)".format(hashtag))
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

    def _vocab_downsize_dict(self, lines, vocab, inv_vocab, padding_size):
        lines = np.asarray(lines)
        data = np.full((len(lines), padding_size), PAD_TOKEN, dtype=object)
        cut_counter = 0
        for i, line in enumerate(lines):
            strs = np.asarray(line).astype(object)
            fin_len = min(padding_size, len(strs))
            data[i, :fin_len] = strs[:fin_len]
            if len(strs) > padding_size:
                cut_counter += 1
        if cut_counter > 0:
            print(f"WARNING: Cut {cut_counter} sentences to length {padding_size}.")

        data = np.vectorize(lambda word: inv_vocab[vocab[word]])(data)
        return data

    def _vocab_downsize_tosize(self, lines, vocab_size):
        counter = Counter()
        for line in lines:
            counter.update(line)

        vocab = dict(BASE_VOCAB)
        for word, _ in counter.most_common(vocab_size - len(BASE_VOCAB)):
            assert word not in vocab
            vocab[word] = len(vocab)
        return MissingDict(vocab, default_val=vocab[UNK_TOKEN])

    def vocab(self, lines, vocab_downsize, padding_size=None):
        if isinstance(vocab_downsize, int):
            vocab = self._vocab_downsize_tosize(lines, vocab_downsize)
            inv_vocab = {v: k for k, v in vocab.items()}
            return vocab, inv_vocab
        else:
            return self._vocab_downsize_dict(lines, *vocab_downsize, padding_size=padding_size)
