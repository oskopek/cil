# arXiv: Yoon Kim 1408.5882

# Glove Twitter embeddings (TODO(oskopek): Try different ones)
# !wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
# !unzip glove.twitter.27B.zip
# Produces files: glove.twitter.27B.100d.txt  glove.twitter.27B.200d.txt
#                 glove.twitter.27B.25d.txt   glove.twitter.27B.50d.txt

import tensorflow as tf

from .model import Model


class CNN(Model):
    CLASSES = 2

    def __init__(self,
                 rnn_cell: str,
                 rnn_cell_dim: int,
                 *args,
                 word_embedding: int = 100,
                 char_embedding: int = 100,
                 keep_prob: float = 0.5,
                 learning_rate: float = 1e-4,
                 grad_clip: float = 10.0,
                 num_channels: int = 512,
                 expname: str = "exp",
                 **kwargs) -> None:
        self.rnn_cell = rnn_cell
        self.rnn_cell_dim = rnn_cell_dim
        self.keep_prob = keep_prob
        self.char_embedding = char_embedding
        self.word_embedding = word_embedding
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.num_channels = num_channels

        # Call super last, because our build_model method probably
        # needs above initialization to happen first
        expname = f"{rnn_cell}{rnn_cell_dim}-char{char_embedding}-word{word_embedding}-{expname}"
        super().__init__(*args, expname=expname, **kwargs)

    def _create_cell(self) -> tf.nn.rnn_cell.RNNCell:
        if self.rnn_cell == "LSTM":
            return tf.nn.rnn_cell.LSTMCell(self.rnn_cell_dim)
        elif self.rnn_cell == "GRU":
            return tf.nn.rnn_cell.GRUCell(self.rnn_cell_dim)
        else:
            raise ValueError(f"Unknown rnn_cell {self.rnn_cell}")

    def _char_embeddings(self):
        if self.char_embedding == -1:
            input_chars = tf.one_hot(self.charseqs, self.num_chars)
        else:
            input_chars = tf.nn.embedding_lookup(
                tf.get_variable("char_emb", shape=[self.num_chars, self.char_embedding]),
                self.charseqs)
        print("input_chars", input_chars.get_shape())

        rnn_cell_characters = self._create_cell()
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            rnn_cell_characters,
            rnn_cell_characters,
            input_chars,
            self.charseq_lens,
            dtype=tf.float32,
            scope="rnn_chars")
        if self.rnn_cell == "LSTM":
            state_fw = tf.concat(state_fw, axis=-1)
            state_bw = tf.concat(state_bw, axis=-1)
        input_chars = tf.concat([state_fw, state_bw], axis=1)
        print("input_chars", input_chars.get_shape())

        input_char_words = tf.nn.embedding_lookup(input_chars, self.charseq_ids)
        input_char_words = tf.layers.dropout(
            input_char_words, rate=self.keep_prob, training=self.is_training)
        print("input_char_words", input_char_words.get_shape())
        return input_char_words

    def _word_embeddings(self):
        # TODO: Add GLOVE
        if self.word_embedding == -1:
            input_words = tf.one_hot(self.word_ids, self.num_words)
        else:
            input_words = tf.nn.embedding_lookup(
                tf.get_variable("word_emb", shape=[self.num_words, self.word_embedding]),
                self.word_ids)
            input_words = tf.layers.dropout(
                input_words, rate=self.keep_prob, training=self.is_training)
        print("input_words", input_words.get_shape())
        return input_words

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

    def _fc(self, flatten):
        d0 = tf.layers.dropout(flatten, rate=self.keep_prob, training=self.is_training)
        hidden1 = tf.layers.dense(d0, 256, activation=tf.nn.leaky_relu)
        d1 = tf.layers.dropout(hidden1, rate=self.keep_prob, training=self.is_training)
        # hidden1 = tf.layers.dense(d0, 256, activation=tf.nn.leaky_relu)
        # d1 = tf.layers.dropout(hidden1, rate=self.keep_prob, training=self.is_training)
        output_layer = tf.layers.dense(d1, self.CLASSES, activation=None)
        print("output_layer", output_layer.get_shape())
        return output_layer

    def build_model(self):
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

            predictions = tf.cast(tf.argmax(output_layer, 1), tf.int32, name="predictions")

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
