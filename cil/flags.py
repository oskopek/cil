import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def define_flags():
    # Directories
    flags.DEFINE_string('logdir', 'data_out', 'Logdir name.')
    flags.DEFINE_string('exp', 'experiment', 'Experiment name.')
    flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                        'Directory in expdir to save checkpoints in (once per epoch)')

    flags.DEFINE_string('checkpoint_path', None, 'Checkpoint to load. If none, ignored.')

    # Data files
    flags.DEFINE_string('train_file', 'data/stories.train.csv', 'Train data file.')
    flags.DEFINE_string('eval_file', 'data/stories.eval.csv', 'Evaluation data file.')
    flags.DEFINE_string('test_file', None, 'Test data file.')

    # Model choice
    flags.DEFINE_string('model', 'RNN', 'Model class name.')

    # TF parameters
    flags.DEFINE_boolean("no_gpu", False, 'Disables GPU usage even if a GPU is available')
    flags.DEFINE_integer('threads', 8, 'Maximum number of threads to use.')
    flags.DEFINE_integer('seed', 42, 'Random seed')

    # Optimization parameters
    flags.DEFINE_integer('epochs', 15, 'Training epoch count')
    flags.DEFINE_integer('batch_size', 32, 'Training batch size')
    flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
    flags.DEFINE_float('grad_clip', 10.0, 'Gradient clipped to L2 norm smaller than or equal to.')

    # Jupyter notebook params
    # Only to avoid raising UnrecognizedFlagError
    flags.DEFINE_string('f', 'kernel', 'Kernel')

    # Other
    flags.DEFINE_string('rnn_cell', "GRU", 'RNN cell type.')
    flags.DEFINE_integer('rnn_cell_dim', 128, 'RNN cell dimension.')
    flags.DEFINE_integer('word_embedding', 200, 'word_embedding')
    flags.DEFINE_integer('char_embedding', 200, 'char_embedding')
    flags.DEFINE_float('keep_prob', 0.5, 'dropout probability')
