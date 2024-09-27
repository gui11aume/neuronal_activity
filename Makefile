POETRY := PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry
VENV := .venv
PYTHON_VERSION := $(shell awk -F'"' '/^python = / {print substr($$2, 2)}' pyproject.toml)
PYTHON_EXECUTABLE := $(shell pyenv which python)

.PHONY: setup install clean train

setup: $(VENV)

$(VENV): pyproject.toml
	pyenv local $(PYTHON_VERSION)
	$(POETRY) env use $(PYTHON_EXECUTABLE)
	$(POETRY) config virtualenvs.in-project true
	touch $(VENV)

install: $(VENV)
	$(POETRY) install

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Run the training process using DVC, ensuring the setup is completed first
train: setup
	$(POETRY) run CUDA_VISIBLE_DEVICES=0 dvc repro