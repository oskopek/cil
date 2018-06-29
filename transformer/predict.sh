PROBLEM=sentiment_twitter
MODEL=transformer_encoder
HPARAMS=transformer_tiny

CIL_DIR=$HOME
USR_DIR=$CIL_DIR/t2t/usr_dir
DATA_DIR=$CIL_DIR/t2t_data
TMP_DIR=$CIL_DIR/t2t_datagen
TRAIN_DIR=$CIL_DIR/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Predict data if the server is running
python2 -m tensor2tensor.serving.query --server=localhost:9000 --servable_name=$MODEL --problem=$PROBLEM --t2t_usr_dir=$USR_DIR --data_dir=$DATA_DIR --test_data=$TMP_DIR/twitter-datasets/test_data.txt
