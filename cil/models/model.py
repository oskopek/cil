from collections import OrderedDict
from datetime import datetime
import os
from typing import Tuple, Dict, Union, List, Any

from cil.data.datasets import Datasets
from cil.data.twitter_dataset import TwitterDataset
from cil.data.utils import print_outputs

import numpy as np
import tensorflow as tf
import tqdm


class Model:
    EVERY_STEPS = 200
    EVALUATE_EVERY_STEP = 7500  # 10 times per epoch on 32 batch size.

    def _placeholders(self) -> None:
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

        # [batch_size]
        self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
        # [batch_size, max_sentence_len]
        self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")

        # [batch_size, max_sentence_len]
        self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
        # [unique char sequences, max_charseq_len]
        self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
        # [unique char sequences]
        self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")

        # [batch_size]
        self.labels = tf.placeholder(tf.int32, [None], name="labels")
        # [] bool scalar
        self.is_training = tf.placeholder_with_default(False, [], name="is_training")

        if hasattr(self, 'keep_prob'):
            self.keep_prob_cond = tf.cond(self.is_training,
                                          lambda: tf.constant(self.keep_prob, dtype=tf.float32),
                                          lambda: tf.constant(1.0, dtype=tf.float32))

        self.placeholder_tensors = [
            self.sentence_lens, self.word_ids, self.charseq_ids, self.charseqs, self.charseq_lens,
            self.labels, self.is_training
        ]

        # Useful tensors
        self.batch_size = tf.shape(self.word_ids)[0]

    def _summaries_and_init(self) -> None:
        current_accuracy, update_accuracy = tf.metrics.accuracy(self.labels, self.predictions)
        current_loss, update_loss = tf.metrics.mean(self.loss)
        self.reset_metrics = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
        self.current_metrics = [current_accuracy, current_loss]
        self.update_metrics = [update_accuracy, update_loss]

        with self.summary_writer.as_default():
            with tf.contrib.summary.record_summaries_every_n_global_steps(self.EVERY_STEPS):
                self.summaries["train"].extend([
                    tf.contrib.summary.scalar("train/loss", update_loss),
                    tf.contrib.summary.scalar("train/accuracy", update_accuracy)
                ])
            with tf.contrib.summary.always_record_summaries():
                for dataset in ["eval", "test"]:
                    self.summaries[dataset].extend([
                        tf.contrib.summary.scalar(dataset + "/loss", current_loss),
                        tf.contrib.summary.scalar(dataset + "/accuracy", current_accuracy)
                    ])

        # Initialize variables
        self.session.run(tf.global_variables_initializer())
        with self.summary_writer.as_default():
            tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def __init__(self,
                 num_words: int,
                 num_chars: int,
                 *args,
                 padding_size: int = 40,
                 threads: int = 1,
                 seed: int = 42,
                 logdir: str = "logs",
                 expname: str = "exp",
                 **kwargs) -> None:
        super().__init__()
        self.num_words = num_words
        self.num_chars = num_chars
        self.padding_size = padding_size

        self.logdir = logdir
        self.expname = expname
        self.exp_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}-{expname}"
        self.save_dir = os.path.join(logdir, self.exp_id)

        self.summaries: Dict[str, List[tf.Operation]] = dict()
        for key in ["train", "eval", "test"]:
            self.summaries[key] = []

        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        config = {
            "gpu_options": tf.GPUOptions(allow_growth=True),
            "inter_op_parallelism_threads": threads,
            "intra_op_parallelism_threads": threads
        }
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(**config))

        # Construct the graph
        with self.session.graph.as_default():
            self._placeholders()
            self.summary_writer = tf.contrib.summary.create_file_writer(
                self.save_dir, flush_millis=10_000)
            self.predictions, self.loss, self.training_step = self.build_model()
            with tf.name_scope("summaries"):
                self._summaries_and_init()

            print("Variables:", tf.trainable_variables())

    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Operation]:
        """
        Remember to use `with self.session.graph.as_default():`.

        :return: `predictions`, `loss`, and `training_step`.
        """
        raise NotImplementedError("To be overridden.")

    @staticmethod
    def _tqdm_metrics(dataset: str, metrics: List[float], names: List[str]) -> Dict[str, str]:
        d: Dict[str, str] = OrderedDict()
        assert len(metrics) == len(names)
        for metric, name in zip(metrics, names):
            d[f'{dataset}_{name}'] = f"{metric:.4f}"
        return d

    def _train_metrics(self) -> Dict[str, str]:
        train_metrics = self.session.run(self.current_metrics)
        return Model._tqdm_metrics("train", train_metrics, ["acc", "loss"])

    def _eval_metrics(self, data: Datasets, batch_size: int = 1) -> Dict[str, str]:
        dataset = "eval"
        eval_metrics = self.evaluate_epoch(data.eval, dataset, batch_size=batch_size)
        return Model._tqdm_metrics(dataset, eval_metrics, ["acc", "loss"])

    def _build_feed_dict(self,
                         batch: Tuple[Union[np.ndarray, bool], ...],
                         is_training: bool = False,
                         predict: bool = False) -> Dict[tf.Tensor, Union[np.ndarray, bool]]:
        assert is_training == batch[-1]
        d = {tensor: value for tensor, value in zip(self.placeholder_tensors, batch)}
        if predict:
            del d[self.labels]
        return d

    def train_batch(self, batch: Tuple[Union[np.ndarray, bool], ...]) -> List[Any]:
        self.session.run(self.reset_metrics)
        fetches = [self.global_step, self.training_step, self.summaries["train"]]
        return self.session.run(fetches, self._build_feed_dict(batch, is_training=True))[0]

    def train(self, data: Datasets, epochs: int, batch_size: int) -> None:
        def _save(eval_acc: float) -> None:
            test_predictions = self.predict_epoch(data.test, "test", batch_size=1024)
            # Print test predictions
            out_file = f"data_out/pred_{self.exp_id}_epoch_{epoch}_acc{eval_acc}.csv"
            print_outputs(out_file, test_predictions, data.test.vocabulary('labels'))
            print("Exported predictions to", out_file, flush=True)
            print(flush=True)

        def _eval(best_acc: float, cur_step: int) -> float:
            metrics = self._eval_metrics(data, batch_size=batch_size)
            epoch_tqdm.set_postfix(metrics)
            eval_acc = float(metrics["eval_acc"])
            if eval_acc > best_acc:
                best_acc = eval_acc
                _save(eval_acc)
            return best_acc

        best_eval_acc = .0
        step = 0
        with tqdm.tqdm(range(epochs), desc="Epochs") as epoch_tqdm:
            for epoch in epoch_tqdm:
                batch_count, batch_generator = data.train.batches_per_epoch(
                    batch_size, shuffle=True)
                with tqdm.tqdm(range(batch_count), desc=f"Batches [Epoch {epoch}]") as batch_tqdm:
                    for _ in batch_tqdm:
                        batch = next(batch_generator)
                        step = self.train_batch(batch)
                        if step % self.EVALUATE_EVERY_STEP == 0:
                            best_eval_acc = _eval(best_eval_acc, step)
                if step % self.EVALUATE_EVERY_STEP == 0:
                    best_eval_acc = _eval(best_eval_acc, step)

    def evaluate_epoch(self, data: TwitterDataset, dataset: str, batch_size: int) -> List[float]:
        self.session.run(self.reset_metrics)
        for batch in data.batch_per_epoch_generator(1024, shuffle=False):
            self.session.run(self.update_metrics, self._build_feed_dict(batch))
        returns = self.session.run(self.current_metrics + [self.summaries[dataset]])
        return returns[:len(self.current_metrics)]  # return current metrics

    def predict_epoch(self, data: TwitterDataset, dataset: str, batch_size: int) -> List[int]:
        predictions: List[int] = []
        for batch in data.batch_per_epoch_generator(batch_size, shuffle=False):
            batch_predictions = self.session.run(self.predictions,
                                                 self._build_feed_dict(batch, predict=True))
            predictions.extend(batch_predictions)
        self.session.run(self.summaries[dataset])
        return predictions
