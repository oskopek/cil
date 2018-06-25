import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def define_flags():
    # Directories
    flags.DEFINE_string('logdir', 'data_out', 'Logdir name.')
    flags.DEFINE_string('exp', None, 'Experiment name.')
    flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                        'Directory in expdir to save checkpoints in (once per epoch)')
    flags.DEFINE_string('checkpoint_path', None, 'Checkpoint to load. If none, ignored.')

    # Model choice
    flags.DEFINE_string('model', 'RNN', 'Model choice.')

    # TF parameters
    flags.DEFINE_boolean("no_gpu", False, 'Disables GPU usage even if a GPU is available')
    flags.DEFINE_integer('threads', 8, 'Maximum number of threads to use.')
    flags.DEFINE_integer('seed', 42, 'Random seed')

    # Optimization parameters
    flags.DEFINE_integer('epochs', 15, 'Training epoch count')
    flags.DEFINE_integer('batch_size', 128, 'Training batch size')
    flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
    flags.DEFINE_float('grad_clip', 5.0, 'Gradient clipped to L2 norm smaller than or equal to.')

    # Jupyter notebook params
    # Only to avoid raising UnrecognizedFlagError
    flags.DEFINE_string('f', 'kernel', 'Kernel')

    # RNN params
    flags.DEFINE_string('rnn_cell', "LSTM", 'RNN cell type.')
    flags.DEFINE_integer('rnn_cell_dim', 128, 'RNN cell dimension.')
    flags.DEFINE_string('attention', 'add', 'Attention type (add ~ Bahdanau, mult ~ Luong, None).')
    flags.DEFINE_integer('attention_size', 128, 'Attention size.')

    # Embedding params
    flags.DEFINE_integer('word_embedding', 200, 'word_embedding')
    flags.DEFINE_integer('char_embedding', 200, 'char_embedding')

    # General model params
    flags.DEFINE_float('keep_prob', 0.5, 'dropout probability')

    # CNN params
    flags.DEFINE_integer('num_channels', 512, 'Number of CNN channels.')

    # Preprocessing
    flags.DEFINE_bool('enable_preprocessing', False, 'Master kill-switch for preprocessing.')
    flags.DEFINE_bool('standardize', True, 'Standardize text spacing.')
    flags.DEFINE_integer('segment_hashtags', 10, 'Attempt to segment hashtags of length <= value.')
    flags.DEFINE_bool('contractions', True, 'Expand contractions.')
    flags.DEFINE_bool('rem_numbers', True, 'Remove numbers.')
    flags.DEFINE_bool('punct_squash', True, 'Squash repeated punctuation marks.')
    flags.DEFINE_bool('fix_slang', True, 'Fix common slang terms.')
    flags.DEFINE_integer('word_squash', 3, 'Squash N+ occurences of the same word consecutively.')
    flags.DEFINE_bool('rem_stopwords', True, 'Remove stopwords.')
    flags.DEFINE_string('stemmer', 'PorterStemmer', 'Stemmer from nltk.')
    flags.DEFINE_string('lemmatizer', 'WordNetLemmatizer', 'Lemmatizer from nltk.')

    # Necessary preprocessing.
    flags.DEFINE_integer('padding_size', 40, 'Padding size.')
    flags.DEFINE_integer('vocab_size', 20_000, 'Vocabulary size.')
