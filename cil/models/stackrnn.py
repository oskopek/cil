from typing import Tuple

import tensorflow as tf

from .model import Model


class StackRNN(Model):
    CLASSES = 2

    def __init__(self,
                 rnn_cell: str,
                 rnn_cell_dim: int,
                 *args,
                 n_cells: int = 2,
                 word_embedding: int = 100,
                 char_embedding: int = 100,
                 keep_prob: float = 0.5,
                 learning_rate: float = 1e-4,
                 grad_clip: float = 10.0,
                 attention: str = None,
                 attention_size: int = 1,
                 expname: str = "exp",
                 **kwargs) -> None:
        self.n_cells = n_cells
        self.rnn_cell = rnn_cell
        self.rnn_cell_dim = rnn_cell_dim
        self.char_embedding = char_embedding
        self.word_embedding = word_embedding
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

        # Call super last, because our build_model method probably
        # needs above initialization to happen first
        expname = f"char{char_embedding}-word{word_embedding}-{expname}"
        expname = f"{n_cells}xStack{rnn_cell}{rnn_cell_dim}-{expname}"
        super().__init__(*args, expname=expname, **kwargs)

    def _create_cell(self, rnn_cell_dim) -> tf.nn.rnn_cell.RNNCell:
        if self.rnn_cell == "LSTM":
            cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
        elif self.rnn_cell == "GRU":
            cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
        else:
            raise ValueError(f"Unknown rnn_cell {self.rnn_cell}. Possible values: LSTM, GRU.")
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, input_keep_prob=self.keep_prob_cond, output_keep_prob=self.keep_prob_cond)
        return cell

    def _char_embeddings(self):
        if self.char_embedding == -1:
            input_chars = tf.one_hot(self.charseqs, self.num_chars)
        else:
            input_chars = tf.nn.embedding_lookup(
                tf.get_variable("char_emb", shape=[self.num_chars, self.char_embedding]),
                self.charseqs)
        print("input_chars", input_chars.get_shape())

        rnn_cell_characters_fw = self._create_cell(self.char_embedding)
        rnn_cell_characters_bw = self._create_cell(self.char_embedding)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            inputs=input_chars,
            cell_fw=rnn_cell_characters_fw,
            cell_bw=rnn_cell_characters_bw,
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

    def _create_sentence_cell(self):
        coef = 2 if self.rnn_cell == "LSTM" else 1
        cells = [
            self._create_cell(self.word_embedding + 2 * coef * self.char_embedding)
            for _ in range(self.n_cells)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cell

    def _sentence_rnn(self, inputs):
        cell_bw = self._create_sentence_cell()
        cell_fw = self._create_sentence_cell()
        (outputs_fw, outputs_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            inputs=inputs,
            cell_bw=cell_bw,
            cell_fw=cell_fw,
            sequence_length=self.sentence_lens,
            dtype=tf.float32,
            scope="rnn_words")

        sentence_states = []
        for state in (state_fw, state_bw):
            for st in state:
                if self.rnn_cell == "LSTM":
                    st = tf.concat(st, axis=-1)
                sentence_states.append(st)
        sentence_states = tf.concat(sentence_states, axis=-1)
        print("sentence_states", sentence_states.get_shape())
        return sentence_states

    def _fc(self, x):
        x = tf.layers.dropout(x, rate=self.keep_prob, training=self.is_training)
        x = tf.layers.dense(x, 256, activation=tf.nn.relu)
        x = tf.layers.dropout(x, rate=self.keep_prob, training=self.is_training)
        x = tf.layers.dense(x, 256, activation=tf.nn.relu)
        x = tf.layers.dropout(x, rate=self.keep_prob, training=self.is_training)
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
