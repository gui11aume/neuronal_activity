"""Test cases for the VectorBert model and its components."""

import time
from pathlib import Path
from unittest.mock import patch

import lightning.pytorch as pl
import pytest
import torch
import transformers

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
def examples() -> list[torch.Tensor]:
    """Generate a list of random tensor examples for testing.

    Creates a list of 53 random tensors with varying lengths between
    MIN_SLICE_SIZE and MAX_SLICE_SIZE, and dimension 256.

    Returns:
        A list of torch.Tensor objects representing random examples.

    """
    return [
        torch.randn(torch.randint(MIN_SLICE_SIZE, MAX_SLICE_SIZE + 1, (1,)).item(), 256)
        for _ in range(53)
    ]


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
    # Use a huge learning rate and no warmup
    train_config = TrainingConfig(lr=0.1, lr_warmup_ratio=0.0, lr_decay_ratio=1.0)
    return TrainHarness(model, train_data, val_data, train_config=train_config, dvc_repo=None)


@pytest.fixture
def cpu_trainer() -> pl.Trainer:
    """Create and return a CPU pl.Trainer for testing.

    This fixture creates a pl.Trainer instance configured to run on CPU,
    with a maximum of 1 epoch and 7 steps.

    Returns:
        A pl.Trainer instance.

    """
    csv_logger = pl.loggers.CSVLogger(".", version=f"test_{int(time.time())}")
    return pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        max_steps=7,
        enable_checkpointing=False,
        logger=csv_logger,
    )


@pytest.fixture
def gpu_trainer() -> pl.Trainer:
    """Create and return a GPU pl.Trainer for testing.

    This fixture creates a pl.Trainer instance configured to run on GPU,
    with a maximum of 1 epoch and 7 steps.

    Returns:
        An instance of pl.Trainer.

    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    csv_logger = pl.loggers.CSVLogger(".", version=f"test_{int(time.time())}")
    return pl.Trainer(
        strategy="ddp",
        precision="16-mixed",
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        max_steps=7,
        enable_checkpointing=False,
        logger=csv_logger,
    )


@pytest.mark.dpp
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


@pytest.mark.dpp
def test_variable_tensor_slice_data_out_of_bounds():
    """Verify that VariableTensorSliceData raises IndexError for out-of-bounds access."""
    tensor = VariableTensorSliceData(torch.randn(2048, 256))
    with pytest.raises(IndexError):
        tensor[len(tensor)]


@pytest.mark.dpp
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


@pytest.mark.dpp
@pytest.mark.parametrize("_", range(1000))
def test_vector_mlm_collator_attention_mask(examples: list[torch.Tensor], _: int):  # noqa: PT019
    """Verify that attention mask is applied correctly.

    Checks if the attention mask is applied correctly to the input tensor.

    Args:
        examples: A single example tensor for testing.

    """
    collator = VectorMLMCollator()
    batch = collator(examples)
    assert torch.all(
        batch["inputs"][batch["attention_mask"] == 0] == -666.0,  # noqa: PLR2004
    ), "Expected all padded elements to be -666.0"
    assert torch.all(
        batch["inputs"][batch["attention_mask"] == 1] != -666.0,  # noqa: PLR2004
    ), "Expected only padded elements to be -666.0"


@pytest.mark.dpp
@pytest.mark.benchmark
def test_vector_mlm_collator_masking_rate(
    benchmark,
    examples: list[torch.Tensor],
):
    """Verify the masking rate of VectorMLMCollator.

    Checks if the selection and masking rates are within expected ranges.

    Args:
        benchmark: Pytest benchmark fixture.
        examples: A list of example tensors for testing.

    """
    # Vary the task ratio in `VectorMLMCollator`
    collator = VectorMLMCollator()

    def run_collator(collator, n=1000):
        total_tokens = 0
        selected_tokens = 0
        masked_tokens = 0
        for _ in range(n):
            batch = collator(examples)
            total_tokens += batch["attention_mask"].sum().item()
            selected_tokens += batch["selected"].sum().item()
            differences = batch["inputs"] != batch["targets"]
            masked_tokens += differences.sum().item() / batch["inputs"].shape[-1]
        return total_tokens, selected_tokens, masked_tokens

    # Benchmark while running the test
    total, selected, masked = benchmark(run_collator, collator=collator)

    selection_rate = selected / total
    assert (
        0.115 <= selection_rate <= 0.185  # noqa: PLR2004
    ), f"Selection rate was {selection_rate}, expected around 0.15"

    masking_rate = masked / total
    assert (
        0.10 <= masking_rate <= 0.17  # noqa: PLR2004
    ), f"Masking rate was {masking_rate}, expected around 0.135"


def test_dimensions_forward_pass(harnessed_model: TrainHarness):
    """Verify the dimensions of tensors in a forward pass of the model.

    Ensures that the output dimensions match the input dimensions
    and that all tensor shapes are as expected.

    Args:
        harnessed_model: The harnessed model to test.

    """
    model = harnessed_model.model
    batch = next(iter(harnessed_model.train_dataloader()))
    output = model(inputs=batch["inputs"], attention_mask=batch["attention_mask"])
    assert output.shape == batch["inputs"].shape
    assert batch["attention_mask"].shape == batch["inputs"].shape[:-1]
    assert batch["targets"].shape == batch["inputs"].shape
    assert batch["selected"].shape == batch["inputs"].shape[:-1]


def test_forward_loss_cpu(harnessed_model: TrainHarness):
    """Verify the forward pass of the model on CPU.

    This test ensures that the model can perform a forward pass on CPU.

    Args:
        harnessed_model: The harnessed model to test.

    """
    batch = next(iter(harnessed_model.train_dataloader()))
    loss = harnessed_model.forward_loss(batch)
    assert isinstance(loss.item(), float), "Loss should be a scalar value"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be infinite"


@pytest.mark.gpu
def test_forward_loss_gpu(harnessed_model: TrainHarness):
    """Verify the forward pass of the model on GPU.

    This test ensures that the model can perform a forward pass on GPU,
    and that the resulting loss is correctly computed and on the GPU.

    Args:
        harnessed_model: The harnessed model to test.

    """
    harnessed_model.model.to("cuda")
    batch = next(iter(harnessed_model.train_dataloader()))
    batch = harnessed_model.transfer_batch_to_device(batch, "cuda", dataloader_idx=0)
    loss = harnessed_model.forward_loss(batch)
    assert loss.device.type == "cuda", "Loss should be on GPU"
    assert isinstance(loss.item(), float), "Loss should be a scalar value"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be infinite"


def test_fit_cpu(
    harnessed_model: TrainHarness,
    cpu_trainer: pl.Trainer,
):
    """Verify fit of the model on CPU.

    This test ensures that the model can perform a fit on CPU,
    and that the resulting loss is computed, gradients are set,
    and parameters are updated.

    Args:
        harnessed_model: The harnessed model to test.
        cpu_trainer: The pl.Trainer fixture to use for testing.

    """
    pl.seed_everything(123)
    initial_params = [p.clone().detach() for p in harnessed_model.model.parameters()]
    cpu_trainer.fit(harnessed_model)
    _check_fit_assertions(harnessed_model, cpu_trainer, initial_params)


@pytest.mark.gpu
def test_fit_gpu(
    harnessed_model: TrainHarness,
    gpu_trainer: pl.Trainer,
):
    """Verify fit of the model on GPU.

    This test ensures that the model can perform a fit on GPU,
    and that the resulting loss is computed, gradients are set,
    and parameters are updated.

    Args:
        harnessed_model: The harnessed model to test.
        gpu_trainer: The pl.Trainer fixture to use for testing.

    """
    pl.seed_everything(123)
    initial_params = [p.clone().detach() for p in harnessed_model.model.parameters()]
    gpu_trainer.fit(harnessed_model)
    _check_fit_assertions(harnessed_model, gpu_trainer, initial_params)


def _check_fit_assertions(
    harnessed_model: TrainHarness,
    trainer: pl.Trainer,
    initial_params: list[torch.Tensor],
):
    # Check if training metrics are available
    assert trainer.callback_metrics, "Training metrics should be available"

    # Check if training loss is present and valid
    assert "loss" in trainer.callback_metrics, "Training loss should be present in metrics"
    train_loss = trainer.callback_metrics["loss"]
    assert isinstance(train_loss, torch.Tensor), "Training loss should be a tensor"
    assert train_loss.dim() == 0, "Training loss should be a scalar tensor"
    assert not torch.isnan(train_loss), "Training loss should not be NaN"
    assert not torch.isinf(train_loss), "Training loss should not be infinite"
    assert train_loss >= 0, "Training loss should be non-negative"

    # Check if the model's state dict is not empty after training
    assert harnessed_model.model.state_dict(), "Model state dict should not be empty after training"

    # Check if some parameters were updated
    params_updated = False
    for initial, current in zip(initial_params, harnessed_model.model.parameters(), strict=True):
        if not torch.equal(initial, current):
            params_updated = True
            break
    assert params_updated, "At least some parameters should be updated after training"

    # Check if the training step was called at least once
    assert trainer.num_training_batches > 0, "Training step should have been called"

    # Check if training dataloader was used
    assert trainer.train_dataloader is not None, "Training dataloader should be set"


def test_validate_cpu(
    harnessed_model: TrainHarness,
    cpu_trainer: pl.Trainer,
):
    """Verify validation of the model on CPU.

    This test ensures that the model can perform a validation on CPU,
    and that the resulting loss is computed, gradients are set,
    and parameters are updated.

    Args:
        harnessed_model: The harnessed model to test.
        cpu_trainer: The pl.Trainer fixture to use for testing.

    """
    pl.seed_everything(123)
    initial_params = [p.clone().detach() for p in harnessed_model.model.parameters()]
    cpu_trainer.validate(harnessed_model)
    _check_validate_assertions(harnessed_model, cpu_trainer, initial_params)


@pytest.mark.gpu
def test_validate_gpu(
    harnessed_model: TrainHarness,
    gpu_trainer: pl.Trainer,
):
    """Verify validation of the model on GPU.

    This test ensures that the model can perform a validation on GPU,
    and that the resulting loss is computed, gradients are set,
    and parameters are updated.

    Args:
        harnessed_model: The harnessed model to test.
        gpu_trainer: The pl.Trainer fixture to use for testing.

    """
    pl.seed_everything(123)
    initial_params = [p.clone().detach() for p in harnessed_model.model.parameters()]
    gpu_trainer.validate(harnessed_model)
    _check_validate_assertions(harnessed_model, gpu_trainer, initial_params)


def _check_validate_assertions(
    harnessed_model: TrainHarness,
    trainer: pl.Trainer,
    initial_params: list[torch.Tensor],
):
    # Check if parameters were not updated during validation
    for initial, current in zip(initial_params, harnessed_model.model.parameters(), strict=True):
        assert torch.equal(initial, current), "Parameters should not be updated during validation"

    # Check if validation metrics are available
    assert trainer.callback_metrics, "Validation metrics should be available"

    # Check if validation loss is present and valid
    assert "val_loss" in trainer.callback_metrics, "Validation loss should be present in metrics"
    val_loss = trainer.callback_metrics["val_loss"]
    assert isinstance(val_loss, torch.Tensor), "Validation loss should be a tensor"
    assert val_loss.dim() == 0, "Validation loss should be a scalar tensor"
    assert not torch.isnan(val_loss), "Validation loss should not be NaN"
    assert not torch.isinf(val_loss), "Validation loss should not be infinite"
    assert val_loss >= 0, "Validation loss should be non-negative"
    assert not val_loss.requires_grad, "Loss should not require gradient"
    assert val_loss.grad is None, "Loss should not have a gradient"

    # Check if the model's state dict is not empty after validation
    assert (
        harnessed_model.model.state_dict()
    ), "Model state dict should not be empty after validation"

    # Verify that gradients were not updated during validation
    for param in harnessed_model.model.parameters():
        assert param.grad is None, "Parameters should not have gradients after validation"

    # Check if the validation step was called at least once
    assert len(trainer.num_val_batches) == 1, "There should be one validation data loader"
    assert trainer.num_val_batches[0] > 0, "Validation step should have been called"

    # Check if validation dataloader was used
    assert trainer.val_dataloaders is not None, "Validation dataloader should be set"


def test_train_logging(harnessed_model: TrainHarness, tmp_path: Path):
    """Verify the logging functionality during model training.

    Checks if the pl.Trainer correctly logs metrics, creates checkpoints,
    and handles DVC tracking messages.

    Args:
        harnessed_model: The harnessed model to train.
        tmp_path: A temporary directory path for storing logs and checkpoints.

    """
    with patch("logging.warning"):
        pl.Trainer = pl.Trainer(
            default_root_dir=tmp_path,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            log_every_n_steps=1,
            enable_model_summary=False,
            enable_checkpointing=True,
            accelerator="cpu",
        )
        pl.Trainer.fit(harnessed_model)

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
