import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
import transformers
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from torch import Tensor, jit

DEBUG = False

MIN_SLICE_SIZE = 4
MAX_SLICE_SIZE = 48
MAX_DOWNSAMPLE_SIZE = 96

# Suppress the specific warning about the number of workers
warnings.filterwarnings("ignore", message=".*does not have many workers which may be a bottleneck.*")


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    batch_size: int = 1536
    max_epochs: int = 20
    accumulate_grad_batches: int = 1
    lr_warmup_ratio: float = 0.1
    lr_decay_ratio: float = 0.9


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    if config_path is None:
        return TrainingConfig()
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Use default values for any missing keys
    default_config = asdict(TrainingConfig())
    default_config.update(config_dict)
    
    return TrainingConfig(**default_config)


class VariableTensorSliceData:
    """
    A dataset that returns a slice of random size from MIN_SLICE_SIZE
    to MAX_SLICE_SIZE, downsample with factor of 2 half of the time.
    """
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, key: int) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            # Downsample with factor of 2
            size = min(
                torch.randint(2 * MIN_SLICE_SIZE, 2 * MAX_SLICE_SIZE + 1, (1,)).item(),
                len(self) - 2 * key
            )
            return self.tensor[key : key + size : 2].float()
        else:
            # No downsampling
            size = min(
                torch.randint(MIN_SLICE_SIZE, MAX_SLICE_SIZE + 1, (1,)).item(),
                len(self) - key
            )
            return self.tensor[key : key + size].float()


@jit.script
class VectorMLMCollator:
    def __call__(self, examples: List[Tensor]) -> Dict[str, Tensor]:
        # Create padded tensor and return padding mask.
        padded = torch.nn.utils.rnn.pad_sequence(
            examples, batch_first=True, padding_value=-666.0
        )
        padding_mask = padded[:, :, 0] != -666.0
        inputs = padded.clone()

        # Run Task I with probability TASK_I_PROB (90% by default) and Task II with
        # probability 1 - TASK_I_PROB (10% by default).
        if torch.rand(1).item() < 0.9:
            # Task I: Standard Masked Language Model
            # 1. Select tokens with probability SELECT_PROB (15% by default)
            # 2. For selected tokens:
            #    - Replace with 0 (mask) with probability MASK_PROB (80% by default)
            #    - Replace with random token with probability RAND_PROB (10% by default)
            #    - Keep unchanged with probability KEEP_PROB (10% by default)

            # Select tokens (excluding padded ones)
            select_prob = torch.full(inputs.shape[:-1], 0.15) * padding_mask
            selected = torch.bernoulli(select_prob).to(torch.bool)

            # Create probability distribution for token replacement
            prob_distribution = torch.tensor([0.8, 0.1, 0.1])
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
                    dtype=torch.int64
                )
            ]
            inputs[selected & (action == 1)] = padded.view(-1, padded.shape[-1])[random_indices]

            # Tokens where action == 2 are kept unchanged
        else:
            # Task II: Mask some entries at all time steps
            # 1. Select entire rows (channels) to be masked
            # 2. Mask a portion of the selected rows
            # 3. Preserve padding and update selection mask

            # 1. Select entire rows (channels) to be masked
            row_selection_prob = torch.full((inputs.shape[0], 1, inputs.shape[2]), 1.0)
            selected_rows = torch.bernoulli(row_selection_prob).to(torch.bool)
            selected_rows = selected_rows.expand_as(inputs)

            # 2. Mask a portion of the selected rows
            mask_prob = torch.full(inputs.shape, 0.5)
            selected_cells = selected_rows & torch.bernoulli(mask_prob).to(torch.bool)
            inputs[selected_cells] = 0

            # 3. Preserve padding and update selection mask
            inputs[padded == -666.0] = -666.0  # Preserve padding
            selected = padding_mask  # Use padding mask for loss computation

        return {
            "inputs": inputs,
            "attention_mask": padding_mask,
            "targets": padded,
            "selected": selected,
        }


class VectorBert(torch.nn.Module):
    def __init__(
        self,
        config: transformers.BertConfig,
        input_dim: int,
        dropout_rate: float = 0.1,
        **kwargs: Any
    ):
        super().__init__()
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
        inputs_embeds = self.dropout(self.embed_in(inputs))
        bert_outputs = self.bert(inputs_embeds=inputs_embeds, **kwargs)
        return self.embed_out(self.dropout(bert_outputs.last_hidden_state))


class TrainHarness(pl.LightningModule):
    def __init__(
        self,
        model: VectorBert,
        train_data: VariableTensorSliceData,
        validation_data: VariableTensorSliceData,
        train_config: TrainingConfig
    ):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.val_output_list: List[torch.Tensor] = []
        # Save all attributes of `train_config` as hparams
        self.save_hyperparameters(asdict(train_config))

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.hparams.lr)

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(self.hparams.lr_warmup_ratio * n_steps)
        n_decay_steps = int(self.hparams.lr_decay_ratio * n_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=n_warmup_steps
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=n_decay_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=VectorMLMCollator(),
            num_workers=0 if DEBUG else 8,
            persistent_workers=False if DEBUG else True,
            pin_memory=False if DEBUG else True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.validation_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=VectorMLMCollator(),
            num_workers=0 if DEBUG else 8,
            persistent_workers=False if DEBUG else True,
            pin_memory=False if DEBUG else True,
        )

    def forward_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        targets = batch.pop("targets")
        selected = batch.pop("selected")
        # Remains "inputs" and "attention_mask" in `batch`
        outputs = self.model(**batch)
        return torch.nn.functional.mse_loss(outputs[selected], targets[selected])

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.forward_loss(batch)
        lr = self.lr_schedulers().get_last_lr()[0]
        info = {"loss": loss, "lr": lr}
        self.log_dict(dictionary=info, on_step=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output_list = []

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = self.forward_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_output_list.append(loss)
    
    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.val_output_list).mean()
        self.log('avg_validation_loss', avg_loss, prog_bar=False, logger=True, sync_dist=True)


if __name__ == "__main__":
    pl.seed_everything(123)
    torch.set_float32_matmul_precision("high")

    train_data_path = sys.argv[1]
    validation_data_path = sys.argv[2]
    trained_model_path = sys.argv[3]
    config_path = sys.argv[4] if len(sys.argv) > 4 else None

    train_config = load_config(config_path)

    model_config = transformers.BertConfig(
        position_embedding_type="relative",
        attn_implementation="eager"
    )

    train_tensors = torch.load(
        train_data_path,
        weights_only=True,
        map_location="cpu",
        mmap=True
    )

    validation_tensors = torch.load(
        validation_data_path,
        weights_only=True,
        map_location="cpu",
        mmap=True
    )

    train_data = VariableTensorSliceData(train_tensors)
    validation_data = VariableTensorSliceData(validation_tensors)

    model = VectorBert(config=model_config, input_dim=256)
    harnessed_model = TrainHarness(model, train_data, validation_data, train_config=train_config)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

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
    else:
        print("Learning rate finder failed. Using default learning rate.")

    # Log the training configuration
    csv_logger.log_hyperparams(dict(harnessed_model.hparams))

    trainer.fit(harnessed_model)

    trainer.save_checkpoint(f"{trained_model_path}.ckpt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": dict(model.config)
    }, trained_model_path)
