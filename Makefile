PYTHON ?= python

.PHONY: check check-static test

check: check-static test

check-static:
	$(PYTHON) -m compileall -q dove_vae_distill scripts tools tests \
		eval_metrics.py eval_temporal_metrics.py inference_script.py
	@for file in scripts/slurm/*.sh scripts/slurm/*.sbatch; do bash -n "$$file"; done

test:
	$(PYTHON) -m pytest -q tests
