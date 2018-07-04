import numpy as np
import os

TOTAL_LEN = 1250000
TRAIN_LEN = 1200000

twitter_dir = 'twitter-datasets'


def generate_permutation(random_seed):
    np.random.seed(random_seed)
    return np.random.permutation(TOTAL_LEN)


def doc_generator(twitter_dir, dataset, include_label=False):
    print("generating {}".format(dataset))
    files = [(os.path.join(twitter_dir, "train_pos_full.txt"), True), (os.path.join(
        twitter_dir, "train_neg_full.txt"), False)]

    for file_path, label in files:
        if label:
            perm = generate_permutation(42)
        else:
            perm = generate_permutation(123)

        with open(file_path, 'r') as f:
            for num, line in enumerate(f):
                if dataset == 'eval' and perm[num] < TRAIN_LEN:
                    continue
                elif dataset == 'train' and perm[num] >= TRAIN_LEN:
                    continue
                if include_label:
                    yield line.rstrip(), label
                else:
                    yield line.rstrip()


def write_to_file(dset, twitter_dir, filename):
    with open(os.path.join(twitter_dir, filename), 'w') as f:
        for line, label in dset:
            print('{},{}'.format(int(label), line), file=f)


train_set = list(doc_generator(twitter_dir, "train", include_label=True))
eval_set = list(doc_generator(twitter_dir, "eval", include_label=True))

write_to_file(train_set, twitter_dir, 'train_data.txt')
write_to_file(eval_set, twitter_dir, 'eval_data.txt')
