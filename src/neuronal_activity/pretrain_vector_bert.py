"""
Implement the VectorBert model and its training harness.

VectorBert is a BERT-based model adapted for processing vector data. The module
includes classes for data handling, model architecture, and training pipeline
using PyTorch Lightning. It also provides utilities for configuration management
and DVC integration for experiment tracking.
"""

import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import dvc.api
import lightning.pytorch as pl
import torch
import transformers
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from safetensors.torch import save_file
from torch import Tensor, jit, optim

from neuronal_activity.utils import generate_codename

DEBUG = False

DOWNSAMPLE_PROB = 0.5
MIN_SLICE_SIZE = 4
MAX_SLICE_SIZE = 48

# Suppress the specific warning about the number of workers
warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers which may be a bottleneck.*",
)


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    batch_size: int = 1536
    max_epochs: int = 20
    lr: float = 1e-4
    lr_warmup_ratio: float = 0.1
    lr_decay_ratio: float = 0.9
    optimizer: str = "AdamW"
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {"betas": (0.9, 0.999)})
    accumulate_grad_batches: int = 1


@dataclass
class VectorBertConfig:
    """Configuration for VectorBert model."""

    bert_config: transformers.BertConfig
    input_dim: int
    dropout_rate: float = 0.1
    additional_kwargs: dict = field(default_factory=dict)


def load_config(config_path: str | None = None) -> TrainingConfig:
    """
    Load training configuration from a YAML file or return default config.

    If no config path is provided, returns the default configuration.
    If a config path is provided, loads the configuration from the YAML file,
    using default values for any missing keys.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Loaded or default training configuration.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.

    """
    if config_path is None:
        return TrainingConfig()

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Use default values for any missing keys
    default_config = asdict(TrainingConfig())
    default_config.update(config_dict)

    return TrainingConfig(**default_config)


class VariableTensorSliceData:
    """
    Represent a dataset of variable-length tensor slices.

    This class provides a way to create a dataset from a tensor, where each item
    is a slice of variable length. It supports random slice sizes and optional
    downsampling.

    Attributes:
        tensor: The input tensor to slice from.

    """

    def __init__(self, tensor: torch.Tensor):
        """
        Initialize the VariableTensorSliceData with a tensor.

        Args:
            tensor: Input tensor to slice from.

        Raises:
            ValueError: If tensor shape[0] is less than 2 * MIN_SLICE_SIZE.

        """
        if tensor.shape[0] < 2 * MIN_SLICE_SIZE:
            raise ValueError(f"Tensor shape[0] too small: {tensor.shape[0]} < {2 * MIN_SLICE_SIZE}")
        self.tensor = tensor

    def __len__(self) -> int:
        """
        Return the number of valid slices in the dataset.

        Returns:
            The number of valid slices.

        """
        return self.tensor.shape[0] - 2 * MIN_SLICE_SIZE + 1

    def __getitem__(self, key: int) -> torch.Tensor:
        """
        Retrieve a tensor slice of random size for the given index.

        Args:
            key: Index of the slice to retrieve.

        Returns:
            A tensor slice of random size.

        Raises:
            IndexError: If the key is out of bounds.

        """
        if key < 0 or key >= len(self):
            raise IndexError(f"Index {key} is out of bounds. Valid range is 0 to {len(self) - 1}.")
        target_size = torch.randint(MIN_SLICE_SIZE, MAX_SLICE_SIZE + 1, (1,)).item()
        if torch.rand(1).item() < DOWNSAMPLE_PROB:
            # Downsample with factor 2
            end = key + 2 * target_size
            return self.tensor[key:end:2].float()
        # No downsampling
        end = key + target_size
        return self.tensor[key:end].float()


@jit.script
class VectorMLMCollator:
    """
    Collator for Vector Masked Language Modeling tasks.

    Handles data preparation for both Task I (standard MLM) and Task II.
    """

    def __init__(
        self,
        task_1_prob: float = 0.9,
        select_prob: float = 0.15,
        mask_prob: float = 0.8,
    ):
        """
        Initialize the VectorMLMCollator with task probabilities.

        Args:
            task_1_prob: Probability of performing Task I. Defaults to 0.9.
            select_prob: Probability of selecting a token for masking. Defaults to 0.15.
            mask_prob: Probability of masking a selected token. Defaults to 0.8.

        Raises:
            ValueError: If any probability is not between 0 and 1.

        """
        if not all(0 <= prob <= 1 for prob in [task_1_prob, select_prob, mask_prob]):
            raise ValueError("All probabilities must be between 0 and 1.")
        self.task_1_prob: float = task_1_prob
        self.select_prob: float = select_prob
        self.mask_prob: float = mask_prob

    def __call__(self, examples: list[Tensor]) -> dict[str, Tensor]:
        """
        Prepare data for Vector Masked Language Modeling tasks.

        Args:
            examples: List of input tensors.

        Returns:
            Dictionary containing prepared data for model input.

        """
        # Create padded tensor and return padding mask.
        padded = torch.nn.utils.rnn.pad_sequence(
            examples,
            batch_first=True,
            padding_value=-666.0,
        )
        padding_mask = padded[:, :, 0] != -666.0  # noqa: PLR2004
        inputs = padded.clone()

        # Run Task I with probability `task_1_prob` and Task II with
        # probability `1 - task_1_prob`.
        if torch.rand(1).item() < self.task_1_prob:
            # Task I: Standard Masked Language Model
            # 1. Select tokens with probability `select_prob`.
            # 2. For selected tokens:
            #    - Replace with 0 (mask) with probability `mask_prob`
            #    - Replace with random token with probability `(1 - mask_prob) / 2`
            #    - Keep unchanged with probability `(1 - mask_prob) / 2`

            # Select tokens (excluding padded ones)
            select_prob = torch.full(inputs.shape[:-1], self.select_prob) * padding_mask
            selected = torch.bernoulli(select_prob).to(torch.bool)

            # Create probability distribution for token replacement
            prob_distribution = torch.tensor(
                [
                    self.mask_prob,  # Mask with 0
                    (1 - self.mask_prob) / 2,  # Replace with random token
                    (1 - self.mask_prob) / 2,  # Keep as is
                ],
                dtype=torch.float32,
            )
            action_prob = prob_distribution.repeat(selected.shape[0], 1)

            # Determine action for each selected token
            action = torch.multinomial(action_prob, num_samples=selected.shape[1], replacement=True)

            # Apply masking (replace with 0)
            inputs[selected & (action == 0)] = 0

            # Apply randomization (replace with random token)
            valid_indices = torch.flatten(torch.nonzero(padding_mask.view(-1)))
            num_replacements = int((selected & (action == 1)).sum())
            random_indices = valid_indices[
                torch.randint(
                    low=0,
                    high=valid_indices.size(0),
                    size=[num_replacements],
                    dtype=torch.int64,
                ),
            ]
            inputs[selected & (action == 1)] = padded.view(-1, padded.shape[-1])[random_indices]

            # Tokens where action == 2 are kept unchanged
        else:
            # Task II: Mask some entries at all time steps
            # 1. Select entire rows (channels) for masking
            # 2. Mask a portion of the selected rows
            # 3. Preserve padding and update selection mask

            # 1. Select entire rows (channels) for masking
            row_selection_prob = torch.full((inputs.shape[0], 1, inputs.shape[2]), 0.5)
            selected_rows = torch.bernoulli(row_selection_prob).to(torch.bool).expand_as(inputs)

            # 2. Mask a portion of the selected rows
            mask_prob = torch.full(inputs.shape, 0.5)
            selected_cells = selected_rows & torch.bernoulli(mask_prob).to(torch.bool)
            inputs[selected_cells] = 0

            # 3. Preserve padding and update selection mask
            inputs[padded == -666.0] = -666.0  # noqa: PLR2004
            selected = padding_mask  # Use padding mask for loss computation

        return {
            "inputs": inputs,
            "attention_mask": padding_mask,
            "targets": padded,
            "selected": selected,
        }


class VectorBert(torch.nn.Module):
    """BERT-based model adapted for processing vector data."""

    def __init__(
        self,
        config: transformers.BertConfig,
        input_dim: int,
        dropout_rate: float = 0.1,
        **kwargs: Any,
    ):
        """
        Initialize the VectorBert model.

        Args:
            config: BERT configuration.
            input_dim: Dimension of input vectors.
            dropout_rate: Dropout rate. Defaults to 0.1.
            **kwargs: Additional keyword arguments for BERT model.

        """
        super().__init__()
        self.config = VectorBertConfig(
            bert_config=config,
            input_dim=input_dim,
            dropout_rate=dropout_rate,
            additional_kwargs=kwargs,
        )
        self.embed_in = torch.nn.Linear(input_dim, config.hidden_size)
        self.bert = transformers.BertModel(config, **kwargs)
        self.embed_out = torch.nn.Linear(config.hidden_size, input_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.embed_in.weight)
        torch.nn.init.xavier_uniform_(self.embed_out.weight)
        torch.nn.init.zeros_(self.embed_in.bias)
        torch.nn.init.zeros_(self.embed_out.bias)

    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Perform forward pass through the VectorBert model.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments for BERT model.

        Returns:
            Output tensor after passing through the model.

        """
        inputs_embeds = self.dropout(self.embed_in(inputs))
        bert_outputs = self.bert(inputs_embeds=inputs_embeds, **kwargs)
        return self.embed_out(self.dropout(bert_outputs.last_hidden_state))


class TrainHarness(pl.LightningModule):
    """
    Lightning module for training the VectorBert model.

    Handles optimization, data loading, and training/validation steps.
    """

    def __init__(
        self,
        model: VectorBert,
        train_data: VariableTensorSliceData,
        val_data: VariableTensorSliceData,
        train_config: TrainingConfig,
    ):
        """
        Initialize the TrainHarness.

        Args:
            model: VectorBert model to train.
            train_data: Training data.
            val_data: Validation data.
            train_config: Training configuration.

        """
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.val_output_list: list[torch.Tensor] = []
        # Save all attributes of `train_config` as hparams
        self.save_hyperparameters(asdict(train_config))

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary containing optimizer and scheduler configurations.

        """
        optimizer_class = getattr(optim, self.hparams.optimizer)
        optimizer = optimizer_class(
            self.model.parameters(),
            lr=float(self.hparams.lr),
            **self.hparams.optimizer_kwargs,
        )

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(self.hparams.lr_warmup_ratio * n_steps)
        n_decay_steps = int(self.hparams.lr_decay_ratio * n_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=n_warmup_steps,
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=n_decay_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create and return the training data loader.

        Returns:
            DataLoader for training data.

        """
        return torch.utils.data.DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=VectorMLMCollator(),
            num_workers=0 if DEBUG else 8,
            persistent_workers=not DEBUG,
            pin_memory=not DEBUG,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create and return the validation data loader.

        Returns:
            DataLoader for validation data.

        """
        return torch.utils.data.DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=VectorMLMCollator(),
            num_workers=0 if DEBUG else 8,
            persistent_workers=not DEBUG,
            pin_memory=not DEBUG,
        )

    def on_fit_start(self):
        """Initialize DVC tracking and log model summary."""
        self.logger.log_hyperparams(self.hparams)
        path_to_model_summary = os.path.join(self.logger.log_dir, "model_summary.txt")
        os.makedirs(os.path.dirname(path_to_model_summary), exist_ok=True)
        with open(path_to_model_summary, "w") as f:
            f.write(str(self.model))
        try:
            repo = dvc.repo.Repo()
        except dvc.exceptions.NotDvcRepoError:
            logging.warning("No DVC tracking")
            return
        try:
            repo.add(path_to_model_summary)
        except dvc.stage.exceptions.StageExternalOutputsError:
            logging.warning("Failed to add model summary to DVC.")

    def forward_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute forward pass and loss.

        Args:
            batch: Dictionary containing input batch data.

        Returns:
            Computed loss tensor.

        """
        targets = batch.pop("targets")
        selected = batch.pop("selected")
        # Remains "inputs" and "attention_mask" in `batch`
        outputs = self.model(**batch)
        return torch.nn.functional.mse_loss(outputs[selected], targets[selected])

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing input batch data.
            batch_idx: Index of the current batch.

        Returns:
            Computed loss tensor.

        """
        loss = self.forward_loss(batch)
        lr = self.lr_schedulers().get_last_lr()[0]
        # Log training loss and learning rate every training step
        self.log_dict(
            dictionary={"loss": loss, "lr": lr},
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):  # noqa: ARG002
        """
        Perform a single validation step.

        Args:
            batch: Dictionary containing input batch data.
            batch_idx: Index of the current batch.

        """
        loss = self.forward_loss(batch)
        # Log validation loss once per validation epoch
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )


if __name__ == "__main__":
    pl.seed_everything(123)
    torch.set_float32_matmul_precision("high")

    repo = dvc.repo.Repo(".")

    train_data_path = sys.argv[1]
    val_data_path = sys.argv[2]
    trained_model_path = sys.argv[3]
    config_path = sys.argv[4] if len(sys.argv) > 4 else None  # noqa: PLR2004

    train_config = load_config(config_path)

    model_config = transformers.BertConfig(
        position_embedding_type="relative",
        attn_implementation="eager",
    )

    train_tensors = torch.load(train_data_path, weights_only=True, map_location="cpu", mmap=True)
    val_tensors = torch.load(val_data_path, weights_only=True, map_location="cpu", mmap=True)

    train_data = VariableTensorSliceData(train_tensors)
    val_data = VariableTensorSliceData(val_tensors)

    model = VectorBert(config=model_config, input_dim=256)
    harnessed_model = TrainHarness(model, train_data, val_data, train_config=train_config)

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    csv_logger = pl.loggers.CSVLogger(".")
    csv_log_dir = csv_logger.log_dir

    checkpoint_callback = ModelCheckpoint(
        dirpath=csv_log_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,  # Save every epoch
    )

    trainer = pl.Trainer(
        default_root_dir=".",
        strategy="ddp_find_unused_parameters_true",
        precision="16-mixed",
        accelerator="gpu",
        devices=1 if DEBUG else -1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        max_epochs=train_config.max_epochs,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        logger=csv_logger,
        log_every_n_steps=1,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Auto-tune the learning rate
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(harnessed_model)
    if lr_finder is not None:
        new_lr = lr_finder.suggestion()
        print(f"Suggested learning rate: {new_lr}")
        harnessed_model.hparams.lr = new_lr
        harnessed_model.save_hyperparameters()
    else:
        print("Learning rate finder failed. Using default learning rate.")

    trainer.fit(harnessed_model)

    trainer.save_checkpoint(f"{trained_model_path}.ckpt")
    save_file(
        {"model_state_dict": model.state_dict(), "config": asdict(model.config)},
        trained_model_path,
    )

    # Commit artifacts
    repo.scm.add([trained_model_path, csv_log_dir])
    repo.scm.commit(f"Train VectorBert model -- {generate_codename()}")

    # Push to remote storage (if configured)
    repo.push()
