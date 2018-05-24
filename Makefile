requirements:
	pip install -r requirements.txt

clean:
	yapf -ir cil/

train:
	python -m cil.train

check:
	flake8 cil/

job:
	bsub -W 04:00 -n 4 -R "rusage[mem=2048, ngpus_excl_p=1]" ./train.sh $(filter-out $@,$(MAKECMDGOALS))

status:
	watch -n 1 bbjobs

output:
	bpeek -f

%:
	@:

.PHONY: requirements train check job status output
