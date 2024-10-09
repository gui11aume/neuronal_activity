POETRY := PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry
VENV := .venv
DEV_MARKER := .dev-dependencies-installed
PRECOMMIT_MARKER := .pre-commit-installed
PYTHON_VERSION := $(shell awk -F'"' '/^python = / {print substr($$2, 2)}' pyproject.toml)
PYTHON_INSTALLED_MARKER := .python-$(PYTHON_VERSION)-installed

.PHONY: install install-dev clean train test test-ci

$(PYTHON_INSTALLED_MARKER):
	pyenv install -s $(PYTHON_VERSION)
	touch $(PYTHON_INSTALLED_MARKER)

$(VENV): pyproject.toml $(PYTHON_INSTALLED_MARKER)
	pyenv local $(PYTHON_VERSION)
	$(POETRY) env use $$(pyenv which python)
	$(POETRY) config virtualenvs.in-project true
	touch $(VENV)

install: $(VENV)
	$(POETRY) install --only main
	$(POETRY) run dvc config core.autostage true

$(DEV_MARKER): $(VENV)
	$(POETRY) install --with dev
	$(POETRY) run dvc config core.autostage true
	touch $(DEV_MARKER)

install-dev: $(DEV_MARKER) $(PRECOMMIT_MARKER)

$(PRECOMMIT_MARKER): $(DEV_MARKER)
	pip install pre-commit
	pre-commit install
	touch $(PRECOMMIT_MARKER)

clean:
	rm -rf $(VENV)
	rm -f .python-version $(PRECOMMIT_MARKER) $(PYTHON_INSTALLED_MARKER) $(DEV_MARKER)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Run the training process using DVC, ensuring the setup is completed first
train: install
	$(POETRY) run dvc repro

# Run tests using pytest
test: $(DEV_MARKER)
	$(POETRY) run pytest tests/

# Run tests using pytest and generate reports
test-ci: $(DEV_MARKER)
	$(POETRY) run pytest tests/ $(ARGS) \
		--cov=src \
		--cov-branch \
		--cov-report=term \
		--cov-report=term-missing \
		--cov-report=lcov:coverage.lcov \
		--junitxml=pytest.xml \

# Run a single file (example)
test-one-file: $(DEV_MARKER)
	$(POETRY) run pytest tests/test_pretrain_vector_bert.py

# Run a single case (example)
test-one-case: $(DEV_MARKER)
	$(POETRY) run pytest tests/test_pretrain_vector_bert.py::test_dimensions_forward_pass
