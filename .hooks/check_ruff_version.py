#!/usr/bin/env python3  # noqa: D100
import subprocess  # nosec B404
import sys
from pathlib import Path


def get_local_ruff_version():  # noqa: D103
    try:
        ruff_path = Path(".venv/bin/ruff").resolve(strict=True)
        result = subprocess.run(  # nosec: B603
            [str(ruff_path), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split()[-1]
    except FileNotFoundError:
        print("Local Ruff not found in .venv/bin/ruff")
        return None


def get_precommit_ruff_version():  # noqa: D103
    with open(".pre-commit-config.yaml") as f:
        for line in f:
            if "ruff-pre-commit" in line:
                next_line = next(f)
                if "rev:" in next_line:
                    return next_line.split(":")[1].strip().lstrip("v")
    print("Ruff version not found in .pre-commit-config.yaml")
    return None


def compare_versions(v1, v2):  # noqa: D103
    v1_parts = v1.split(".")
    v2_parts = v2.split(".")
    for i in range(max(len(v1_parts), len(v2_parts))):
        v1_part = int(v1_parts[i]) if i < len(v1_parts) else 0
        v2_part = int(v2_parts[i]) if i < len(v2_parts) else 0
        if v1_part > v2_part:
            return 1
        if v1_part < v2_part:
            return -1
    return 0


def main():  # noqa: D103
    local_version = get_local_ruff_version()
    precommit_version = get_precommit_ruff_version()

    if local_version and precommit_version:
        if compare_versions(local_version, precommit_version) == 0:
            print(f"Ruff versions match: {local_version}")
            return 0
        print(
            f"Version mismatch: Local Ruff {local_version} != "
            f"Pre-commit Ruff {precommit_version}",
        )
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
