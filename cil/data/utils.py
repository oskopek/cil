UNK_TOKEN = '$unk$'
PAD_TOKEN = '$pad$'
BASE_VOCAB = {PAD_TOKEN: 0, UNK_TOKEN: 1}


class MissingDict(dict):
    """Replace missing values with the default value, but do not insert them."""

    def __init__(self, *args, default_val=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_val = default_val

    def __missing__(self, key):
        return self.default_val


def print_outputs(out_file, test_predictions, sentiment_vocab):
    with open(out_file, "w+") as f:
        print("Id,Prediction", file=f)
        for i, prediction in enumerate(test_predictions):
            label = int(sentiment_vocab[prediction]) * 2 - 1
            print(f"{i+1},{label}", file=f)
