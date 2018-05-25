# Glove Twitter embeddings (TODO(oskopek): Try different ones)
# !wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
# !unzip glove.twitter.27B.zip
# Produces files: glove.twitter.27B.100d.txt  glove.twitter.27B.200d.txt
#                 glove.twitter.27B.25d.txt   glove.twitter.27B.50d.txt

from typing import Tuple

import tensorflow as tf

from .model import Model


class RNN(Model):
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
                 attention: str = None,
                 attention_size: int = 1,
                 expname: str = "exp",
                 **kwargs) -> None:
        self.rnn_cell = rnn_cell
        self.rnn_cell_dim = rnn_cell_dim
        self.keep_prob = keep_prob
        self.char_embedding = char_embedding
        self.word_embedding = word_embedding
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.attention = attention
        self.attention_size = attention_size

        # Call super last, because our build_model method probably
        # needs above initialization to happen first
        if self.attention:
            expname = f"attn-{expname}"
        expname = f"{rnn_cell}{rnn_cell_dim}-char{char_embedding}-word{word_embedding}-{expname}"
        super().__init__(*args, expname=expname, **kwargs)

    def _create_cell(self) -> tf.nn.rnn_cell.RNNCell:
        if self.rnn_cell == "LSTM":
            return tf.nn.rnn_cell.LSTMCell(self.rnn_cell_dim)
        elif self.rnn_cell == "GRU":
            return tf.nn.rnn_cell.GRUCell(self.rnn_cell_dim)
        else:
            raise ValueError(f"Unknown rnn_cell {self.rnn_cell}. Possible values: LSTM, GRU.")

    def _create_attention(self, encoder_outputs: tf.Tensor,
                          sentence_lens: tf.Tensor) -> tf.contrib.seq2seq.AttentionMechanism:
        if self.attention == "add":
            attention = tf.contrib.seq2seq.BahdanauAttention
        elif self.attention == "mult":
            attention = tf.contrib.seq2seq.LuongAttention
        else:
            raise ValueError(f"Unknown attention {self.attention}. Possible values: add, mult.")
        return attention(
            num_units=self.attention_size,
            memory=encoder_outputs,
            memory_sequence_length=sentence_lens,
            dtype=tf.float32)

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
            inputs=input_chars,
            cell_fw=rnn_cell_characters,
            cell_bw=rnn_cell_characters,
            sequence_length=self.charseq_lens,
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

    @staticmethod
    def _compute_attention(mechanism, cell_output, attention_state):
        # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/seq2seq
        # /python/ops/attention_wrapper.py
        # Alignments shape is [batch_size, 1, memory_time].
        alignments, next_attention_state = mechanism(cell_output, attention_state)
        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = tf.expand_dims(alignments, axis=1)
        # Context is the inner product of alignments and values along the memory time dimension.
        # The attention_mechanism.values shape is [batch_size, memory_time, memory_size].
        # The matmul is over memory_time, so the output shape is [batch_size, 1, memory_size].
        context = tf.matmul(expanded_alignments, mechanism.values)
        # We then squeeze out the singleton dim.
        context = tf.squeeze(context, axis=[1])
        return context, alignments, next_attention_state

    @staticmethod
    def _attention_images_summary(alignments, batch_size, prefix=""):
        # https://github.com/tensorflow/nmt/blob/master/nmt/attention_model.py
        """
        Create attention image and attention summary.
        """
        # Reshape to (batch, tgt_seq_len, src_seq_len, 1)
        seq_len = 40  # TODO(oskopek): Hack
        alignments = tf.slice(alignments, begin=(0, 0), size=(batch_size, seq_len))
        attention_images = tf.reshape(alignments, (batch_size, 40, 1, 1))
        attention_images = tf.transpose(attention_images, perm=(0, 2, 1, 3))  # make img horizontal
        # Scale to range [0, 255]
        attention_images *= 255
        attention_summary = tf.contrib.summary.image(f"{prefix}/attention_images", attention_images)
        return attention_summary

    def _attention_summaries(self, alignments, prefix=""):
        with self.summary_writer.as_default():
            with tf.contrib.summary.record_summaries_every_n_global_steps(self.EVERY_STEPS):
                img = RNN._attention_images_summary(
                    alignments, self.batch_size, prefix=f"{prefix}/train")
                self.summaries["train"].append(img)
            with tf.contrib.summary.always_record_summaries():
                for key in ["eval", "test"]:
                    img = RNN._attention_images_summary(
                        alignments, self.batch_size, prefix=f"{prefix}/{key}")
                    self.summaries[key].append(img)

    def _add_attention(self, outputs, sentence_lens, cell_output, prefix=""):
        attention_mechanism = self._create_attention(outputs, sentence_lens)
        context, alignments, next_attention_state = RNN._compute_attention(
            attention_mechanism,
            cell_output,
            attention_state=attention_mechanism.initial_state(self.batch_size, dtype=tf.float32))
        with tf.name_scope("summaries"):
            self._attention_summaries(alignments, prefix=prefix)
        return context

    def _sentence_rnn(self, inputs):
        rnn_cell_words = self._create_cell()
        (outputs_fw, outputs_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            inputs=inputs,
            cell_bw=rnn_cell_words,
            cell_fw=rnn_cell_words,
            sequence_length=self.sentence_lens,
            dtype=tf.float32,
            scope="rnn_words")

        with tf.variable_scope("attention_fw"):
            c_state_fw, m_state_fw = state_fw
            context_fw = self._add_attention(
                outputs_fw, self.sentence_lens, cell_output=m_state_fw, prefix="attention_fw")
            print("context_fw", context_fw.get_shape())
        with tf.variable_scope("attention_bw"):
            c_state_bw, m_state_bw = state_bw
            context_bw = self._add_attention(
                outputs_bw, self.sentence_lens, cell_output=m_state_bw, prefix="attention_bw")
            print("context_bw", context_bw.get_shape())

        if self.rnn_cell == "LSTM":
            state_fw = tf.concat(state_fw, axis=-1)
            state_bw = tf.concat(state_bw, axis=-1)
        sentence_states = tf.concat([context_fw, context_bw, state_fw, state_bw], axis=1)
        print("sentence_states", sentence_states.get_shape())
        return sentence_states

    def _fc(self, x):
        x = tf.layers.dropout(x, rate=self.keep_prob, training=self.is_training)
        x = tf.layers.dense(x, 256, activation=tf.nn.leaky_relu)
        x = tf.layers.dropout(x, rate=self.keep_prob, training=self.is_training)
        # x = tf.layers.dense(x, 256, activation=tf.nn.leaky_relu)
        # x = tf.layers.dropout(x, rate=self.keep_prob, training=self.is_training)
        output_layer = tf.layers.dense(x, self.CLASSES, activation=None)
        print("output_layer", output_layer.get_shape())
        return output_layer

    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Operation]:
        # Construct the graph
        with self.session.graph.as_default():
            with tf.name_scope("char_embeddings"):
                input_chars = self._char_embeddings()

            with tf.name_scope("word_embeddings"):
                input_words = self._word_embeddings()

            with tf.name_scope("sentence_rnn"):
                inputs = tf.concat([input_chars, input_words], axis=2)
                print("inputs", inputs.get_shape())
                sentence_states = self._sentence_rnn(inputs)

            with tf.name_scope("fc"):
                output_layer = self._fc(sentence_states)

            predictions = tf.cast(tf.argmax(output_layer, 1), tf.int32, name="predictions")

            with tf.name_scope("loss"):
                loss = tf.losses.sparse_softmax_cross_entropy(
                    self.labels, output_layer, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gradients = optimizer.compute_gradients(loss)
                # TODO(oskopek): Only clip RNN gradients?
                clipped_gradients = [(tf.clip_by_norm(gradient, self.grad_clip), var)
                                     for gradient, var in gradients]
                training_step = optimizer.apply_gradients(
                    clipped_gradients, global_step=self.global_step)

        return predictions, loss, training_step
