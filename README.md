# neuronal_activity
Use BERT to model neuronal activity

## Setup Instructions

1. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/your-username/neuronal_activity.git
   cd neuronal_activity
   ```

2. Install pyenv (for Python version management):
   ```bash
   # Install pyenv dependencies
   sudo apt-get update
   sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
   libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
   libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

   # Install pyenv
   curl https://pyenv.run | bash

   # Add pyenv to PATH (add these lines to your ~/.bashrc or ~/.zshrc)
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init --path)"' >> ~/.bashrc

   # Reload your shell
   source ~/.bashrc
   ```

3. Install [Poetry](https://python-poetry.org/) (for Python package and dependency management):
   ```bash
   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -

   # Add Poetry to PATH (add this line to your ~/.bashrc or ~/.zshrc)
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # Reload your shell
   source ~/.bashrc
   ```

4. Optionally, install the required Python version and project dependencies
now (otherwise, this will run automatically the first time you call
`make train`):
   ```bash
   # Install the required Python version
   pyenv install $(cat .python-version)

   # Set up the project environment, install dependencies, and configure DVC
   make install
   ```
## Development Setup

This project uses [ruff](https://github.com/astral-sh/ruff) for linting, [pytest](https://docs.pytest.org/) for testing, [pre-commit hook](https://pre-commit.com/) to enforce code discipline and [DVC](https://dvc.org/) to manage data and models. To install the dependencies for development and the pre-commit hooks, run:

Install the git hook scripts:
   ```bash
   make install-dev
   ```


## Training Configuration

You can modify the training parameters by editing the `config.yaml` file. This file contains various hyperparameters and settings for the model training process.

## Training

To start the training process, simply run:
```
make train
```

This command will execute the DVC pipeline defined in `dvc.yaml`, which includes data preparation and model training stages.

## Additional Information

- The `dvc.yaml` file defines the pipeline stages and their dependencies.
- Training logs and model outputs will be saved in the `logs` and `models` directories respectively.
