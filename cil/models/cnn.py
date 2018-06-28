# arXiv: Yoon Kim 1408.5882
from typing import Tuple

import tensorflow as tf

from .rnn import RNN


class CNN(RNN):
    CLASSES = 2

    def __init__(self,
                 *args,
                 num_channels: int = 512,
                 expname: str = "exp",
                 **kwargs) -> None:
        self.num_channels = num_channels
        super().__init__(*args, expname=f"CNN{num_channels}-{expname}", **kwargs)

    def _sentence_conv(self, inputs):
        def conv1d(inputs,
                   num_channels,
                   kernel_size,
                   stride=1,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=tf.layers.batch_normalization,
                   **kwargs):
            res = tf.layers.conv2d(
                inputs=inputs,
                filters=num_channels,
                kernel_size=kernel_size,
                strides=stride,
                activation=activation_fn,
                padding='valid',
                **kwargs)
            if normalizer_fn is not None:
                res = normalizer_fn(res)
            return res

        assert len(inputs.get_shape()) == 3
        emb_size = inputs.get_shape()[-1]
        seq_len = self.padding_size
        inputs = tf.slice(inputs, begin=[0, 0, 0], size=[-1, seq_len, emb_size])
        inputs = tf.expand_dims(inputs, axis=-1)
        print("input_to_cnn", inputs.get_shape())

        pooled = []
        for kernel_size in [3, 4, 5, 7]:
            name = f"conv_{kernel_size}"
            print(name, end='\t')
            c = conv1d(
                inputs, self.num_channels, kernel_size=[kernel_size, emb_size], name=f"{name}/conv")
            print("conv", c.get_shape(), end='\t')
            mp = tf.layers.max_pooling2d(
                c, pool_size=[seq_len - kernel_size + 1, 1], strides=1, name=f"{name}/pool")
            print("pool", mp.get_shape())
            pooled.append(mp)

        pooled_outputs = tf.concat(pooled, axis=3)
        print("pooled_outputs", pooled_outputs.get_shape())
        flatten = tf.layers.flatten(pooled_outputs)
        print("flatten", flatten.get_shape())
        return flatten

    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Operation]:
        # Construct the graph
        with self.session.graph.as_default():
            with tf.name_scope("char_embeddings"):
                input_chars = self._char_embeddings()

            with tf.name_scope("word_embeddings"):
                input_words = self._word_embeddings()

            with tf.name_scope("conv"):
                inputs = tf.concat([input_chars, input_words], axis=2)
                print("inputs", inputs.get_shape())
                flatten = self._sentence_conv(inputs)

            with tf.name_scope("fc"):
                output_layer = self._fc(flatten)

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

