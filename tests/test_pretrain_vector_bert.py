"""Test cases for the VectorBert model and its components."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import transformers
from lightning.pytorch import Trainer

from neuronal_activity.pretrain_vector_bert import (
    MAX_SLICE_SIZE,
    MIN_SLICE_SIZE,
    TrainHarness,
    TrainingConfig,
    VariableTensorSliceData,
    VectorBert,
    VectorMLMCollator,
)


@pytest.fixture
def harnessed_model() -> TrainHarness:
    """Create and return a TrainHarness instance for testing.

    This fixture creates a TrainHarness instance with a VectorBert model,
    random train and validation data, and default training configuration.

    Returns:
        An instance of TrainHarness.

    """
    model_config = transformers.BertConfig(
        position_embedding_type="relative",
        attn_implementation="eager",
    )
    model = VectorBert(config=model_config, input_dim=256)
    train_data = VariableTensorSliceData(torch.randn(17, 256))
    val_data = VariableTensorSliceData(torch.randn(9, 256))
    train_config = TrainingConfig()
    return TrainHarness(model, train_data, val_data, train_config=train_config)


@pytest.fixture
def examples() -> list[torch.Tensor]:
    """Generate a list of random tensor examples for testing.

    Creates a list of 16 random tensors with varying lengths between
    MIN_SLICE_SIZE and MAX_SLICE_SIZE, and dimension 256.

    Returns:
        A list of torch.Tensor objects representing random examples.

    """
    return [
        torch.randn(torch.randint(MIN_SLICE_SIZE, MAX_SLICE_SIZE + 1, (1,)).item(), 256)
        for _ in range(16)
    ]


def test_variable_tensor_slice_data_slice_sizes():
    """Verify that VariableTensorSliceData produces slices of expected sizes."""
    tensor = VariableTensorSliceData(torch.randn(2048, 256))
    for i in range(len(tensor)):
        assert (
            tensor[i].shape[0] >= MIN_SLICE_SIZE
        ), f"Slice {i} has size {tensor[i].shape[0]}, expected at least {MIN_SLICE_SIZE}"
        assert (
            tensor[i].shape[0] <= MAX_SLICE_SIZE
        ), f"Slice {i} has size {tensor[i].shape[0]}, expected at most {MAX_SLICE_SIZE}"


def test_variable_tensor_slice_data_out_of_bounds():
    """Verify that VariableTensorSliceData raises IndexError for out-of-bounds access."""
    tensor = VariableTensorSliceData(torch.randn(2048, 256))
    with pytest.raises(IndexError):
        tensor[len(tensor)]


def test_vector_mlm_collator():
    """Verify the VectorMLMCollator functionality.

    Checks if the collator correctly processes a batch of examples,
    including proper shaping, padding, masking, and key presence in the output.
    """
    collator = VectorMLMCollator()
    examples = [
        torch.randn(10, 256),  # Example 1: 10 time steps
        torch.randn(15, 256),  # Example 2: 15 time steps
        torch.randn(8, 256),  # Example 3: 8 time steps
    ]

    batch = collator(examples)

    assert set(batch.keys()) == {"inputs", "attention_mask", "targets", "selected"}, (
        f"Expected keys {{'inputs', 'attention_mask', 'targets', 'selected'}}, "
        f"got {batch.keys()}"
    )

    assert batch["inputs"].shape == (3, 15, 256), (
        f"Expected inputs shape (3, 15, 256), " f"got {batch['inputs'].shape}"
    )

    assert batch["attention_mask"].shape == (3, 15), (
        f"Expected attention_mask shape (3, 15), " f"got {batch['attention_mask'].shape}"
    )

    assert batch["targets"].shape == (3, 15, 256), (
        f"Expected targets shape (3, 15, 256), " f"got {batch['targets'].shape}"
    )

    assert batch["selected"].shape == (3, 15), (
        f"Expected selected shape (3, 15), " f"got {batch['selected'].shape}"
    )

    assert torch.all(
        batch["attention_mask"][0, :10],
    ), "Expected all True for first 10 elements of attention_mask[0]"
    assert torch.all(
        ~batch["attention_mask"][0, 10:],
    ), "Expected all False for elements 10-15 of attention_mask[0]"
    assert torch.all(
        batch["attention_mask"][1, :15],
    ), "Expected all True for first 15 elements of attention_mask[1]"
    assert torch.all(
        ~batch["attention_mask"][1, 15:],
    ), "Expected all False for elements after 15 of attention_mask[1]"
    assert torch.all(
        batch["attention_mask"][2, :8],
    ), "Expected all True for first 8 elements of attention_mask[2]"
    assert torch.all(
        ~batch["attention_mask"][2, 8:],
    ), "Expected all False for elements after 8 of attention_mask[2]"

    assert not torch.all(batch["inputs"] == batch["targets"]), "Expected some tokens to be masked"

    assert torch.all(
        batch["selected"] <= batch["attention_mask"],
    ), "Expected selected mask to be within attention mask"

    assert torch.all(
        batch["inputs"][batch["attention_mask"] == 0] == -666.0,  # noqa: PLR2004
    ), "Expected padded elements to be -666.0"


def test_vector_mlm_collator_task_probabilities(examples: list[torch.Tensor]) -> None:
    """Verify the task probabilities of VectorMLMCollator.

    Checks if Task I (standard MLM) occurs with the expected frequency by running
    the collator multiple times and counting the occurrences of Task I.

    Args:
        examples: A list of example tensors for testing.

    """
    collator = VectorMLMCollator()

    task_1_count = 0
    for _ in range(1000):
        batch = collator(examples)

        assert torch.all(
            batch["selected"] <= batch["attention_mask"],
        ), "Expected selected mask to be within attention mask"

        assert torch.all(
            batch["inputs"][batch["attention_mask"] == 0] == -666.0,  # noqa: PLR2004
        ), "Expected padded elements to be -666.0"

        if torch.any(batch["inputs"].sum(dim=-1) == 0):
            task_1_count += 1

    assert 865 <= task_1_count <= 935, f"Task I occurred {task_1_count} times, expected around 900"  # noqa: PLR2004


def test_vector_mlm_collator_masking_rate(examples: list[torch.Tensor]) -> None:
    """Verify the masking rate of VectorMLMCollator.

    Checks if the selection and masking rates are within expected ranges.

    Args:
        examples: A list of example tensors for testing.

    """
    collator = VectorMLMCollator(task_1_prob=1.0)  # Run Task I only
    total_tokens = 0
    selected_tokens = 0
    masked_tokens = 0
    for _ in range(1000):
        batch = collator(examples)
        total_tokens += batch["attention_mask"].sum().item()
        selected_tokens += batch["selected"].sum().item()
        differences = batch["inputs"] != batch["targets"]
        masked_tokens += differences.sum().item() / batch["inputs"].shape[-1]

    selection_rate = selected_tokens / total_tokens
    assert (
        0.115 <= selection_rate <= 0.185  # noqa: PLR2004
    ), f"Selection rate was {selection_rate}, expected around 0.15"

    masking_rate = masked_tokens / total_tokens
    assert (
        0.10 <= masking_rate <= 0.17  # noqa: PLR2004
    ), f"Masking rate was {masking_rate}, expected around 0.135"


def test_dimensions_forward_pass(harnessed_model: TrainHarness) -> None:
    """Verify the dimensions of tensors in a forward pass of the model.

    Ensures that the output dimensions match the input dimensions
    and that all tensor shapes are as expected.

    Args:
        harnessed_model: The harnessed model to test.

    """
    model = harnessed_model.model
    batch = next(iter(harnessed_model.train_dataloader()))
    output = model(batch["inputs"])
    assert output.shape == batch["inputs"].shape
    assert batch["attention_mask"].shape == batch["inputs"].shape[:-1]
    assert batch["targets"].shape == batch["inputs"].shape
    assert batch["selected"].shape == batch["inputs"].shape[:-1]


def test_train_logging(harnessed_model: TrainHarness, tmp_path: Path) -> None:
    """Verify the logging functionality during model training.

    Checks if the trainer correctly logs metrics, creates checkpoints,
    and handles DVC tracking messages.

    Args:
        harnessed_model: The harnessed model to train.
        tmp_path: A temporary directory path for storing logs and checkpoints.

    """
    with patch("logging.warning") as mock_log:
        trainer = Trainer(
            default_root_dir=tmp_path,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            log_every_n_steps=1,
            enable_model_summary=False,
            enable_checkpointing=True,
            accelerator="cpu",
        )
        trainer.fit(harnessed_model)

        log_dir = tmp_path / "lightning_logs" / "version_0"
        metrics_file = log_dir / "metrics.csv"
        checkpoints_dir = log_dir / "checkpoints"

        assert log_dir.exists(), "Log directory was not created"
        assert metrics_file.exists(), "Metrics file was not created"
        assert list(checkpoints_dir.glob("*.ckpt")), "Checkpoint was not created"

        with open(metrics_file) as f:
            content = f.read()
            assert "loss" in content, "Loss was not logged"
            assert "lr" in content, "Learning rate was not logged"
            assert "val_loss" in content, "Validation loss was not logged"

        assert any(
            "Failed to add model summary to DVC." in msg
            for msg in [call.args[0] for call in mock_log.call_args_list]
        ), "Expected 'Failed to add model summary to DVC.' message was not logged"
