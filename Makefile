setup:
	python -m pip install -r requirements.txt --user
	python -c 'import nltk; nltk.download("punkt")'
	cd data_in && ./download_data.sh

clean:
	yapf -ir cil/ glove/ fasttext/

train:
	python -m cil.train

check:
	flake8 cil/ glove/ fasttext/

job:
	bsub -W 24:00 -n 4 -R "rusage[mem=3000, ngpus_excl_p=1]" ./train.sh

status:
	watch -n 1 bbjobs

output:
	bpeek -f

# Experiments

lstm128:
	cp cil/experiments/lstm128.py cil/flags.py

lstm128_ce:
	cp cil/experiments/lstm128_ce.py cil/flags.py

lstm128_we:
	cp cil/experiments/lstm128_we.py cil/flags.py

gru256:
	cp cil/experiments/gru256.py cil/flags.py

stacklstm:
	cp cil/experiments/stacklstm.py cil/flags.py

transformer-train-serve:
	cd transformer && ./train_and_serve.sh

transformer-predict:
	cd transformer && ./predict.sh

glove-setup:
	cd glove && ./setup.sh

glove-run:
	cd glove && ./run.sh

glove-job:
	bsub -W 24:00 -n 4 -R "rusage[mem=13000, ngpus_excl_p=1]" ./train_glove_job.sh

cnn512:
	cp cil/experiments/cnn512.py cil/flags.py

fasttext:
	./train_fasttext.sh

# End experiments

%:
	@:

.PHONY: setup clean train check job status output lstm128 lstm128_ce lstm128_we gru256 stacklstm transformer-train-serve transformer-predict glove-setup glove-run glove-job cnn512 fasttext
