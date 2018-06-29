#!/bin/bash
PROBLEM=sentiment_twitter
MODEL=transformer_encoder
HPARAMS=transformer_tiny

# Point to working directory path
CIL_DIR=$HOME
# Point to the current directory where the usr_dir is! 
USR_DIR=$CIL_DIR/transformer/usr_dir
DATA_DIR=$CIL_DIR/t2t_data
TMP_DIR=$CIL_DIR/t2t_datagen
TRAIN_DIR=$CIL_DIR/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
python -m tensor2tensor.bin.t2t_datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USR_DIR

# Train
python -m tensor2tensor.bin.t2t_trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=44000 \
  --t2t_usr_dir=$USR_DIR \
  --eval_throttle_seconds=300 \
  --hparams='batch_size=8000'

python -m tensor2tensor.serving.export \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --output_dir=$TRAIN_DIR

tensorflow_model_server --port=9000 --model_name=$MODEL --model_base_path=$TRAIN_DIR/export/Servo
