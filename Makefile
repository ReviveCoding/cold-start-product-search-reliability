PYTHON ?= python
CONFIG ?= configs/smoke.yaml
RELEASE_ROOT ?= releases
GENERATION ?=

.PHONY: install smoke test integration reproducibility sensitivity lint build verify-build serve serve-release uvicorn-validation publish-release rollback-release release-status force-unlock-release validate-handoff advanced ope clean package docker

install:
	$(PYTHON) -m pip install -c constraints/validated.txt -e .

smoke:
	$(PYTHON) scripts/run_full_pipeline.py --config $(CONFIG)

test:
	$(PYTHON) scripts/run_tests.py

integration:
	$(PYTHON) scripts/integration_validation.py --config $(CONFIG)

reproducibility:
	$(PYTHON) scripts/reproducibility_check.py --config $(CONFIG)

sensitivity:
	$(PYTHON) scripts/run_policy_sensitivity.py --config $(CONFIG)

lint:
	$(PYTHON) -m ruff check src tests scripts

build:
	$(PYTHON) scripts/build_release.py

verify-build:
	$(PYTHON) scripts/verify_build_reproducibility.py

advanced:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src $(PYTHON) scripts/train_dcn.py --train artifacts/smoke/behavior_train_features.csv --validation artifacts/smoke/behavior_validation_features.csv --test artifacts/smoke/behavior_test_features.csv --candidates artifacts/smoke/ranked_test.csv --features artifacts/smoke/behavior_features.json --output-dir artifacts/smoke/dcn_challenger --config $(CONFIG)

serve:
	PRODUCT_SEARCH_ARTIFACT_DIR=artifacts/smoke $(PYTHON) scripts/serve.py

serve-release:
	PRODUCT_SEARCH_RELEASE_ROOT=$(RELEASE_ROOT) $(PYTHON) scripts/serve.py

uvicorn-validation:
	$(PYTHON) scripts/uvicorn_validation.py --artifact-dir artifacts/smoke --requests 16 --concurrency 4

publish-release:
	$(PYTHON) scripts/manage_release.py publish --source artifacts/smoke --release-root $(RELEASE_ROOT) $(if $(GENERATION),--generation $(GENERATION),)

rollback-release:
	$(PYTHON) scripts/manage_release.py rollback --release-root $(RELEASE_ROOT) $(if $(GENERATION),--generation $(GENERATION),)

release-status:
	$(PYTHON) scripts/manage_release.py status --release-root $(RELEASE_ROOT)

force-unlock-release:
	$(PYTHON) scripts/manage_release.py force-unlock --release-root $(RELEASE_ROOT)

validate-handoff:
	$(PYTHON) scripts/validate_candidate_handoff.py --handoff release_candidate_handoff.json

ope:
	$(PYTHON) scripts/run_ope_validation.py --seed 42 --rows 5000

clean:
	$(PYTHON) scripts/clean.py

package:
	$(PYTHON) scripts/package_project.py

docker:
	docker build --tag cold-start-product-search:local .
