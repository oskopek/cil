# Glove Twitter embeddings (TODO: Try different ones)
#!wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
#!unzip glove.twitter.27B.zip
# Produces files: glove.twitter.27B.100d.txt  glove.twitter.27B.200d.txt  glove.twitter.27B.25d.txt  glove.twitter.27B.50d.txt

import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.summary as tf_summary

import cil.flags as flags
FLAGS = flags.FLAGS

class Network:
    CLASSES = 2

    def __init__(self,
                 rnn_cell,
                 rnn_cell_dim,
                 num_words,
                 num_chars,
                 logdir,
                 expname,
                 threads=1,
                 seed=42,
                 word_embedding=100,
                 char_embedding=100,
                 keep_prob=0.5,
                 learning_rate=1e-4):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(
                inter_op_parallelism_threads=threads, intra_op_parallelism_threads=threads))

        # Construct the graph
        with self.session.graph.as_default():
            if rnn_cell == "LSTM":
                rnn_cell_co = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell_co = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            self.charseq_ids = tf.placeholder(tf.int32, [None, None])
            self.charseqs = tf.placeholder(tf.int32, [None, None])
            self.charseq_lens = tf.placeholder(tf.int32, [None])
            self.sentiments = tf.placeholder(tf.int32, [None])
            self.is_training = tf.placeholder_with_default(False, [])
            self.keep_prob = tf.placeholder_with_default(1.0, [])

            rnn_cell_co = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_co, self.keep_prob, self.keep_prob)

            if char_embedding == -1:
                input_chars = tf.one_hot(self.charseqs, num_chars)
            else:
                input_chars = tf.nn.embedding_lookup(
                    tf.get_variable("char_emb", shape=[num_chars, char_embedding]), self.charseqs)


#                 input_chars = tf.layers.dropout(input_chars, rate=self.keep_prob, training=self.is_training)
            print("input_chars", input_chars.get_shape())

            if rnn_cell == "LSTM":
                rnn_cell_ce = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell_ce = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                rnn_cell_ce,
                rnn_cell_ce,
                input_chars,
                self.charseq_lens,
                dtype=tf.float32,
                scope="rnn_chars")
            input_chars = tf.concat([state_fw, state_bw], axis=1)
            print("input_chars", input_chars.get_shape())

            input_char_words = tf.nn.embedding_lookup(input_chars, self.charseq_ids)
            input_char_words = tf.layers.dropout(
                input_char_words, rate=self.keep_prob, training=self.is_training)
            print("input_char_words", input_char_words.get_shape())

            if word_embedding == -1:
                input_words = tf.one_hot(self.word_ids, num_words)
            else:
                input_words = tf.nn.embedding_lookup(
                    tf.get_variable("word_emb", shape=[num_words, word_embedding]), self.word_ids)
                input_words = tf.layers.dropout(
                    input_words, rate=self.keep_prob, training=self.is_training)
            print("input_words", input_words.get_shape())

            # TODO: Add GLOVE

            inputs = tf.concat([input_char_words, input_words], axis=2)
            print("inputs", inputs.get_shape())

            (outputs_fw, outputs_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                rnn_cell_co, rnn_cell_co, inputs, self.sentence_lens, dtype=tf.float32)
            states = tf.concat([state_fw, state_bw], axis=1)
            print("states", states.get_shape())

            hidden = tf.layers.dense(states, 64, activation=tf.nn.leaky_relu)
            d1 = tf.layers.dropout(hidden, rate=self.keep_prob, training=self.is_training)
            output_layer = tf.layers.dense(d1, self.CLASSES, activation=None)
            print("output_layer", output_layer.get_shape())

            self.loss = tf.losses.sparse_softmax_cross_entropy(
                self.sentiments, output_layer, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # optimizer = L4.L4Adam()
            gradients = optimizer.compute_gradients(self.loss)
            clipped_gradients = [(tf.clip_by_norm(gradient, 5), var) for gradient, var in gradients]

            self.training = optimizer.apply_gradients(
                clipped_gradients, global_step=self.global_step)

            #self.training = optimizer.apply_gradients(
            #    clipped_gradients, self.loss)

            self.predictions = tf.cast(tf.argmax(output_layer, 1), tf.int32)
            #             self.confusion_matrix = tf.reshape(
            #                 tf.confusion_matrix(self.sentiments, self.predictions,
            #                                     weights=tf.not_equal(self.sentiments, self.predictions),
            #                                     dtype=tf.float32),
            #                 [1, self.CLASSES, self.CLASSES, 1])

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(
                self.sentiments, self.predictions)
            self.current_loss, self.update_loss = tf.metrics.mean(self.loss)
            self.reset_metrics = tf.variables_initializer(
                tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf_summary.create_file_writer(
                "{}/{}-{}".format(logdir, timestamp, expname), flush_millis=5_000)
            self.summaries = {}
            with summary_writer.as_default(), tf_summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [
                    tf_summary.scalar("train/loss", self.update_loss),
                    tf_summary.scalar("train/accuracy", self.update_accuracy)
                ]
            with summary_writer.as_default(), tf_summary.always_record_summaries():
                for dataset in ["eval", "test"]:
                    self.summaries[dataset] = [
                        tf_summary.scalar(dataset + "/loss", self.current_loss),
                        tf_summary.scalar(dataset + "/accuracy", self.current_accuracy),
                        #                                                tf_summary.image(dataset + "/confusion_matrix", self.confusion_matrix)
                    ]

            # Initialize variables
            self.session.run(tf.initialize_all_variables())
            with summary_writer.as_default():
                tf_summary.initialize(session=self.session, graph=self.session.graph)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train_epoch(self, data):
        while not data.epoch_finished():
            self.session.run(self.reset_metrics)
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, sentiments = \
                data.next_batch(FLAGS.batch_size)
            self.session.run(
                [self.training, self.summaries["train"]], {
                    self.sentence_lens: sentence_lens,
                    self.word_ids: word_ids,
                    self.charseq_ids: charseq_ids,
                    self.charseqs: charseqs,
                    self.charseq_lens: charseq_lens,
                    self.sentiments: sentiments,
                    self.is_training: True,
                    self.keep_prob: FLAGS.keep_prob
                })
            print(".", end='')
        print("")

    def evaluate_epoch(self, data, dataset):
        self.session.run(self.reset_metrics)
        while not data.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, sentiments = \
                data.next_batch(FLAGS.batch_size)
            self.session.run(
                [self.update_accuracy, self.update_loss], {
                    self.sentence_lens: sentence_lens,
                    self.word_ids: word_ids,
                    self.charseq_ids: charseq_ids,
                    self.charseqs: charseqs,
                    self.charseq_lens: charseq_lens,
                    self.sentiments: sentiments
                })
        acc, loss, _ = self.session.run(
            [self.current_accuracy, self.current_loss, self.summaries[dataset]])
        return acc, loss

    def predict_epoch(self, data, dataset):
        self.session.run(self.reset_metrics)
        predictions = []
        while not data.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, sentiments = \
                data.next_batch(FLAGS.batch_size)
            batch_predictions = self.session.run(
                self.predictions, {
                    self.sentence_lens: sentence_lens,
                    self.word_ids: word_ids,
                    self.charseq_ids: charseq_ids,
                    self.charseqs: charseqs,
                    self.charseq_lens: charseq_lens
                })
            predictions.extend(batch_predictions)
        self.session.run(self.summaries[dataset])
        return predictions
