# neuronal_activity
Use BERT to model neuronal activity

## Getting Started

### Setup Instructions

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

4. Install the required Python version and project dependencies
   ```bash
   # Install the required Python version
   pyenv install $(cat .python-version)

   # Set up the project environment, install dependencies, and configure DVC
   make install
   ```


### Training Configuration

You can modify the training parameters by editing the `config.yaml` file. This file contains various hyperparameters and settings for the model training process.

### Running the Training Pipeline

To start the training process, simply run:
```
make train
```

This command will execute the DVC pipeline defined in `dvc.yaml`, which includes data preparation and model training stages.

### Additional Information

- The `dvc.yaml` file defines the pipeline stages and their dependencies.
- Training logs and model outputs will be saved in the `logs` and `models` directories respectively.

## Development

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting, [pytest](https://docs.pytest.org/) for testing, [pre-commit hook](https://pre-commit.com/) and [virtualenv](https://virtualenv.pypa.io/) to enforce code discipline and [DVC](https://dvc.org/) to manage data and models. To install the dependencies for development and the pre-commit hooks, run:

Install developer dependencies (including pre-commit hooks):
   ```bash
   make install-dev
   ```

## Editing with VS Code or Cursor

This project uses [VS Code](https://code.visualstudio.com/) or [Cursor](https://cursor.sh/) for development. The `.vscode/settings.json` file is set up with recommended settings for Python development, including Ruff integration. To set up Ruff in VS Code or Cursor:

1. Install the Ruff extension:
   Search for `Ruff` in the Extensions marketplace and install it.

2. The `.vscode/settings.json` file already includes Ruff configuration. It should contain something similar to the following:

   ```json
   {
       "python.testing.pytestArgs": [
           "tests"
       ],
       "python.testing.unittestEnabled": false,
       "python.testing.pytestEnabled": true,
       "editor.formatOnSave": true,
       "[python]": {
           "editor.defaultFormatter": "charliermarsh.ruff",
           "editor.codeActionsOnSave": {
               "source.fixAll": "explicit",
               "source.organizeImports": "explicit"
           }
       },
       "ruff.enable": true,
       "ruff.organizeImports": true,
       "ruff.fixAll": true,
       "ruff.importStrategy": "fromEnvironment",
       "ruff.path": ["${workspaceFolder}/.venv/bin/ruff"],
       "ruff.interpreter": ["${workspaceFolder}/.venv/bin/python"],
       "ruff.nativeServer": "off"
   }
   ```

3. Restart VS Code or Cursor to apply the changes.

Now, Ruff will automatically lint and format your Python code on save, ensuring consistent code style and catching potential issues early in development.

## Testing and Debugging

This project uses pytest for testing. The `.vscode/launch.json` file is set up with configurations to help you run and debug tests easily with [VS Code](https://code.visualstudio.com/) or [Cursor](https://cursor.sh/). This file is located in the `.vscode` directory and contains predefined launch configurations for running tests and debugging.

The `Makefile` is also configured to run tests with `pytest`.

### Running All Tests

To run all tests, you have multiple options:

1. Use the command line:
   ```bash
   make test
   ```
   This command will run all tests in the project.

2. Use VS Code's Test Explorer:
   - Open the Test Explorer view in VS Code (usually found in the sidebar).
   - Click the `Run Tests` button at the top of the Test Explorer (double arrow icon).

3. Use the `Python: Run tests` configuration:
   - Open the Run and Debug view in VS Code (Ctrl+Shift+D or Cmd+Shift+D).
   - Select "Python: Make Test" from the dropdown at the top.
   - Click the green play button or press F5 to run all tests.

### Running a Specific Test

To run a specific test, you can use the provided`Python: Test dimensions_forward_pass` configuration as an example:

1. Open the Run and Debug view in VS Code (Ctrl+Shift+D or Cmd+Shift+D).
2. Select `Python: Test single case (example)` from the dropdown.
3. Click the green play button or press F5 to run this specific test.

This configuration is defined in the `.vscode/launch.json` file and demonstrates how to run a single test function. You can create similar configurations for other specific tests you want to run or debug frequently.

### Debugging with VS Code and Cursor

For more detailed information on how to use the debugger in VS Code and Cursor, you can refer to the following resources:

- [Debugging Python in VS Code](https://code.visualstudio.com/docs/python/debugging)
- [Debugging in Cursor](https://cursor.sh/docs/debugging)
- [Python debug configurations in VS Code](https://code.visualstudio.com/docs/python/debugging#_python-debug-configurations)
- [Advanced Debugging Features in Cursor](https://cursor.sh/docs/advanced-debugging)

These resources provide comprehensive guides on setting up and using the debugger, including how to set breakpoints, step through code, and inspect variables during runtime.

# Code Discipline 💎

## Principles of Code Discipline

### The XYZ problem

A common issue in software development is known as the [XY problem](https://meta.stackexchange.com/q/66377).

> You are trying to solve problem X, and you think solution Y would work, but instead of asking about X when you run into trouble, you ask about Y.

The lesson to learn from the XY problem is not to focus on Y because you think that Y is the right solution. But there is a more subtle lesson here: you should also not focus on X because you think that X is the right problem.

Code discipline is about addressing the XYZ problem: nobody really knows what the real problems are.

### Unexpected Behaviors are Issues

To mitigate the XYZ problem, all unexpected behaviors must be treated as issues. Every exception, odd behavior, unexpected result or running time, edge case, etc. must be documented in a Github issue.

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code discipline. The pre-commit hooks are configured in the `.pre-commit-config.yaml` file. The commit hooks should be running and up-to-date when you commit. If not, reinstall the dependencies with `make install-dev` or follow the instructions below.

Pre-commit hooks are installed automatically when you run `make install-dev`. If commit hooks are not functioning properly, you can manually install the commit hooks with the following commands:

```bash
pre-commit install
```

If at some point the hooks become outdated, you can update them with:

```bash
pre-commit autoupdate
```

In some rare cases (e.g., when hooks are broken or when they get on the way of major code refactoring), it may be necessary to skip the pre-commit hooks for one commit:

```bash
git commit -m "Your commit message with justification" --no-verify
```

**Important Note:**
*If you use `--no-verify`, make sure to document the reason in the commit message. Also note that the pre-commit hooks are also run in the CI/CD pipeline, so they will prevent merging if they fail.*

## Running the Pre-commit Hooks Manually

If you want to run the pre-commit hooks manually before committing, you can use the following command:

```bash
pre-commit run --show-diff-on-failure --color=always --all-files
```

To run a specific hook, you can use the following command:

```bash
pre-commit run <hook-name> --all-files
```

To run a specific hook on a specific file, you can use the following command:

```bash
pre-commit run <hook-name> --files <file-name>
```

## List of Pre-commit Hooks

The hooks include the following:

### General Hooks

- **check-toml**: Validates the syntax of TOML files to ensure they are parseable.
- **check-json**: Validates the syntax of JSON files to ensure they are parseable.
- **check-added-large-files**: Prevents the addition of large files to the repository.
- **check-case-conflict**: Checks for files that may conflict on case-insensitive filesystems.
- **check-merge-conflict**: Detects files containing merge conflict markers.
- **check-yaml**: Validates the syntax of YAML files to ensure they are parseable.
- **detect-private-key**: Scans for the presence of private keys in the codebase.
- **end-of-file-fixer**: Ensures that files end with a newline or are empty.
- **fix-byte-order-marker**: Removes UTF-8 byte order markers from files.
- **mixed-line-ending**: Checks for and replaces mixed line endings in files.
- **trailing-whitespace**: Trims trailing whitespace from lines in files.

### GitHub Actions Hooks

- **check-github-actions**: Validates GitHub Actions workflow files against the official schema.
- **check-github-workflows**: Checks GitHub Workflows for common errors and best practices.

### Poetry Hooks

- **poetry-check**: Validates the structure of the `pyproject.toml` file.
- **poetry-lock**: Checks that the `poetry.lock` file is consistent with the `pyproject.toml`.

### Ruff Hooks

- **ruff**: Runs the Ruff linter with configuration specified in `pyproject.toml`.
- **ruff-format**: Runs the Ruff formatter with configuration specified in `pyproject.toml`.

### MyPy Hook

- **mypy**: Performs static type checking using MyPy, excluding test files.

### Bandit Hook

- **bandit**: Runs the Bandit security linter with default settings.

### Detect Secrets Hook

- **detect-secrets**: Scans for secrets in the codebase using a baseline file.

## Fixing Code Issues with Ruff

Ruff is used to lint and fix code issues. Running the Ruff pre-commit hook will lint and fix the issues automatically when possible. This is done automatically when running `git commit`, but you can also run it manually with the following command:

```bash
pre-commit run ruff --files <file-name>
```

Some issues may persist and are not fixable by Ruff. In that case, you can add to the line that causes the issue a comment of the form `# noqa: <code>` to ignore the issue.

```python
# Ignore "Standard pseudo-random generators are not suitable for cryptographic purposes
random.choice(adjectives)   # noqa: S311
```

You can also ignore issues for a whole file by adding `# ruff: noqa: <code>` to the top of the file. For instance, a test file allowing `assert` statements would start with:

```python
# ruff: noqa: S101

import pytest
# ... rest of imports ...
```

## Using LLMs to enforce code discipline

It is recommended to use LLMs to help with code discipline. Below are some prompts that can be used to help with the most common tasks.

### Commit Messages

First capture the diff of the changes you are about to commit:

```
git diff HEAD > patch.patch
```

Then reference the patch file with @patch.patch in the prompt to write a good commit message:

```
Analyze the output of `git diff HEAD` in @patch.patch and craft a concise, informative commit message for the changes. Ensure the message is clear and informative for future readers who may not be familiar with the current context. Follow these guidelines: 1. Start with a summary line of at most 50 characters, written in the present tense. 2. Do not begin the summary with 'feat:', 'fix:', or any other prefixes. 3. Capitalize the first word of the summary. 4. Do not end the summary line with a period. 5. After the summary, add a blank line followed by a more detailed description if necessary. 6. In the detailed description, explain the 'what' and 'why' of the changes, not the 'how'. 7. Wrap the detailed description at 72 characters. 8. Use bullet points for multiple distinct changes.
Example format:
Summary of changes (50 chars or fewer)

More detailed explanation of the changes, wrapped at
72 characters. Include the motivation for the change
and contrast it with previous behavior.
 - Bullet point for distinct change 1
 - Bullet point for distinct change 2
```


### Docstrings

Use the following prompt to write or update docstrings for all the functions (this addresses most D issues with Ruff):

```
Write or update docstrings for all the functions following the Google format. The docstrings must adhere to PEP 257 conventions and must pass all pydocstyle checks. The docstrings should include a one-line summary, followed by a more detailed description if necessary. Include sections for parameters, return values, and any raised exceptions, but do not add dashed lines under 'Args:', 'Returns:', or 'Raises:' headings. Use triple double quotes for the docstring. Follow the PEP 727 guidelines and do not add type information. Ensure the docstrings are concise yet informative, focusing on the function's purpose and behavior rather than implementation details.
Example format:
"""
Short summary of the function.

More detailed description of the function, including its purpose,
behavior, and any important details.

Args:
    param1: Description of the first parameter.
    param2: Description of the second parameter.

Returns:
    Description of the return value.

Raises:
    KeyError: Description of the exception raised.

"""
```

### Naming Conventions

Use the following prompt to review and update the code to ensure all function and class names adhere to PEP 8 naming conventions (this addresses most N issues with Ruff):

```
Review and update the code to ensure all function and class names adhere to PEP 8 naming conventions. Specifically: 1. Use snake_case for function names (all lowercase with underscores between words). 2. Use CapWords (PascalCase) for class names. 3. Ensure that any acronyms in names follow the appropriate capitalization rules (e.g., HttpResponse rather than HTTPResponse for classes, http_response for functions). 4. Avoid single character names except for counters or iterators. 5. Don't use names that conflict with Python keywords or built-in functions.
```
