# Reproducible entry points. Override variables on the command line, e.g. `make test ENV=myenv`.
ENV      ?= curb-occupancy
# `make run` takes a study config and runs from the study repo's root, e.g.
# make -C <study repo> -f <path to this Makefile> run CONFIG=config/streets/terrebonne_evening.yaml
CONFIG   ?=
RUN      = conda run --no-capture-output -n $(ENV)

.PHONY: env lock test run clean-runs

env:            ## create/update the conda env and install pinned deps + this package
	conda env update -n $(ENV) -f environment.yml
	$(RUN) pip install -r requirements.lock.txt
	$(RUN) pip install --no-deps -e .

lock:           ## re-resolve requirements.in and refresh the lock file
	$(RUN) pip install -r requirements.in
	$(RUN) python tools/lock.py

test:
	$(RUN) pytest -q

run:            ## full pipeline for every pass in the config
	$(RUN) curb-occupancy run $(CONFIG)

clean-runs:
	rm -rf workdir/runs
