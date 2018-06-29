setup:
	python -m pip install -r requirements.txt --user
	python -c 'import nltk; nltk.download("punkt")'
	cd data && ./download_data.sh

clean:
	yapf -ir cil/ glove/

train:
	python -m cil.train $(filter-out $@,$(MAKECMDGOALS))

check:
	flake8 cil/ glove/

job:
	bsub -W 24:00 -n 4 -R "rusage[mem=2048, ngpus_excl_p=1]" ./train.sh $(filter-out $@,$(MAKECMDGOALS))

status:
	watch -n 1 bbjobs

output:
	bpeek -f

# Experiments

lstm128:
	p

lstm128_ce:
	p

lstm128_we:
	p

gru256:
	p

stacklstm:
	p

transformer:
	p

glove-rf:
	p

glove-lr:
	p

cnn512:
	p

fasttext:
	p

# End experiments

%:
	@:

.PHONY: setup train check job status output
