import os
import tensorflow as tf

global FLAGS
FLAGS = tf.app.flags.FLAGS


def define_flags():
    def reset_flags():
        import argparse as _argparse
        tf.app.flags._global_parser = _argparse.ArgumentParser()

    reset_flags()
    # Directories
    if os.name == 'nt':
        tf.app.flags.DEFINE_string('checkpoint_dir', 'e:/temp/tensorflow/checkpoints/',
                                   'Directory to save checkpoints in (once per epoch)')
    else:
        tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/tensorflow/checkpoints/',
                                   'Directory to save checkpoints in (once per epoch)')

    # TF parameters
    tf.app.flags.DEFINE_boolean("no_gpu", False, 'Disables GPU usage even if a GPU is available')

    # Optimization parameters
    tf.app.flags.DEFINE_integer('epochs', 15, 'Training epoch count')
    tf.app.flags.DEFINE_integer('batch_size', 32, 'Training batch size')

    # Jupyter notebook params
    # Only to avoid raising UnrecognizedFlagError
    tf.app.flags.DEFINE_string('f', 'kernel', 'Kernel')

    tf.app.flags.DEFINE_string('logdir', 'data_out', 'Logdir name.')
    tf.app.flags.DEFINE_string('rnn_cell', "GRU", 'RNN cell type.')
    tf.app.flags.DEFINE_integer('rnn_cell_dim', 128, 'RNN cell dimension.')
    tf.app.flags.DEFINE_integer('threads', 8, 'Maximum number of threads to use.')
    tf.app.flags.DEFINE_integer('word_embedding', 200, 'word_embedding')
    tf.app.flags.DEFINE_integer('char_embedding', 200, 'char_embedding')
    tf.app.flags.DEFINE_float('keep_prob', 0.5, 'dropout probability')
    tf.app.flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
