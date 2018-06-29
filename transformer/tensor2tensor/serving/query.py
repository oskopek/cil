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
"""Query an exported model. Py2 only. Install tensorflow-serving-api."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from oauth2client.client import GoogleCredentials
from six.moves import input  # pylint: disable=redefined-builtin

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow as tf
from tqdm import tqdm

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("server", None, "Address to Tensorflow Serving server.")
flags.DEFINE_string("servable_name", None, "Name of served model.")
flags.DEFINE_string("problem", None, "Problem name.")
flags.DEFINE_string("data_dir", None, "Data directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", None, "Usr dir for registrations.")
flags.DEFINE_string("test_data", None, "Test data.")
flags.DEFINE_string("output", 'kaggle.csv', "Output file.")
flags.DEFINE_integer("timeout_secs", 10, "Timeout for query.")

# For Cloud ML Engine predictions.
flags.DEFINE_string("cloud_mlengine_model_name", None,
                    "Name of model deployed on Cloud ML Engine.")
flags.DEFINE_string(
    "cloud_mlengine_model_version", None,
    "Version of the model to use. If None, requests will be "
    "sent to the default version.")


def validate_flags():
  """Validates flags are set to acceptable values."""
  if FLAGS.cloud_mlengine_model_name:
    assert not FLAGS.server
    assert not FLAGS.servable_name
  else:
    assert FLAGS.server
    assert FLAGS.servable_name


def make_request_fn():
  """Returns a request function."""
  if FLAGS.cloud_mlengine_model_name:
    request_fn = serving_utils.make_cloud_mlengine_request_fn(
        credentials=GoogleCredentials.get_application_default(),
        model_name=FLAGS.cloud_mlengine_model_name,
        version=FLAGS.cloud_mlengine_model_version)
  else:

    request_fn = serving_utils.make_grpc_request_fn(
        servable_name=FLAGS.servable_name,
        server=FLAGS.server,
        timeout_secs=FLAGS.timeout_secs)
  return request_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  validate_flags()
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  problem = registry.problem(FLAGS.problem)
  hparams = tf.contrib.training.HParams(
      data_dir=os.path.expanduser(FLAGS.data_dir))
  problem.get_hparams(hparams)
  request_fn = make_request_fn()
  if FLAGS.test_data:
    inputs = []
    with open(FLAGS.test_data, 'r') as f:
      with open(FLAGS.output, 'w+') as fout:
        print("Id,Prediction", file=fout)
        for line in tqdm(f):
          num, text = line.rstrip().split(',', 1)
          outputs = serving_utils.predict([text], problem, request_fn)
          print('{},{}'.format(num, "-1" if outputs[0][0] == "neg" else "1"), file=fout)
  else:
    print("Missing test_data nd output file")


if __name__ == "__main__":
  flags.mark_flags_as_required(["problem", "data_dir"])
  tf.app.run()
