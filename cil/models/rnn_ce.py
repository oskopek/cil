from typing import Tuple

import tensorflow as tf

from .rnn import RNN


class RNNCE(RNN):
    def __init__(self, *args, expname: str = "exp", **kwargs) -> None:
        super().__init__(*args, expname=f"charonly_{expname}", **kwargs)

    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Operation]:
        # Construct the graph
        with self.session.graph.as_default():
            with tf.name_scope("char_embeddings"):
                input_chars = self._char_embeddings()

            with tf.name_scope("sentence_rnn"):
                sentence_states = self._sentence_rnn(input_chars)

            with tf.name_scope("fc"):
                output_layer = self._fc(sentence_states)

            with tf.name_scope("predictions"):
                predictions = tf.argmax(output_layer, axis=1)
                predictions = tf.cast(predictions, dtype=tf.int32)

            with tf.name_scope("loss"):
                loss = tf.losses.sparse_softmax_cross_entropy(
                    self.labels, output_layer, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gradients = optimizer.compute_gradients(loss)
                clipped_gradients = [(tf.clip_by_norm(gradient, self.grad_clip), var)
                                     for gradient, var in gradients]
                training_step = optimizer.apply_gradients(
                    clipped_gradients, global_step=self.global_step)

        return predictions, loss, training_step
