# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Twitter Sentiment Classification Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf
import numpy as np

@registry.register_problem
class SentimentTwitter(text_problems.Text2ClassProblem):
  """Twitter sentiment classification."""
  URL = "http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip"

  TOTAL_LEN = 1250000
  TRAIN_LEN = 1200000

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def vocab_filename(self):
    return "sentiment_twitter.vocab.%d" % self.approx_vocab_size

  @property
  def approx_vocab_size(self):
    return 2**13  # 8k vocab suffices for this small dataset.

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    del data_dir
    return ["neg", "pos"]

  def generate_permutation(self, random_seed):
    np.random.seed(random_seed)
    return np.random.permutation(self.TOTAL_LEN)

  def doc_generator(self, twitter_dir, dataset, include_label=False):
      print("generating {}".format(dataset))
      files = [(os.path.join(twitter_dir, "train_pos_full.txt"), True), 
               (os.path.join(twitter_dir, "train_neg_full.txt"), False)]
      
      for file_path, label in files:
        if label:
          perm = self.generate_permutation(42)
        else:
          perm = self.generate_permutation(123)

        with tf.gfile.Open(file_path, 'r') as f: 
          for num, line in enumerate(f):
            if dataset == 'eval' and perm[num] < self.TRAIN_LEN:
              continue
            elif dataset == 'train' and perm[num] >= self.TRAIN_LEN:
              continue
            if include_label:
              yield line.rstrip(), label
            else:
              yield line.rstrip()

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate examples."""
    # Download and extract
    compressed_filename = os.path.basename(self.URL)
    download_path = generator_utils.maybe_download(tmp_dir, compressed_filename,
                                                   self.URL)
    twitter_dir = os.path.join(tmp_dir, "twitter-datasets")
    if not tf.gfile.Exists(twitter_dir):
      import zipfile
      with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    # Generate examples
    print("Datasplit", dataset_split)
    if dataset_split == problem.DatasetSplit.TRAIN:
      dataset = 'train'
      include_label=True
    elif dataset_split == problem.DatasetSplit.EVAL:
      dataset = "eval"
      include_label=True
    for doc, label in self.doc_generator(twitter_dir, dataset, include_label=include_label):
      yield {
          "inputs": doc,
          "label": int(label)
      }


@registry.register_problem
class SentimentTwitterCharacters(SentimentTwitter):
  """IMDB Sentiment classification, character level."""
  vocab_type = text_problems.VocabType.CHARACTER
