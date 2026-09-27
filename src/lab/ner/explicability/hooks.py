"""Hugging Face Trainer callback integrating snapshot extraction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from transformers import TrainerCallback

from lab.ner.explicability.config import SnapshotConfig
from lab.ner.explicability.snapshot import capture_snapshot


class ExplicabilityCallback(TrainerCallback):
    def __init__(
        self, trainer, tokenizer, validation_rows: pd.DataFrame, root,
        config: SnapshotConfig, model_name: str | None = None,
        capture_parameters: bool = False,
    ):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.rows = validation_rows
        self.root = root
        self.config = config
        self.model_name = model_name
        self.capture_parameters = capture_parameters
        self.counter = 0
        self.snapshots: list[dict[str, Any]] = []
        self._last_epoch: tuple[float | None, int] | None = None

    def capture_pretrained(self) -> dict[str, Any]:
        snapshot = self._capture("step_000_pretrained", epoch=0.0, step=0)
        self.snapshots.append(snapshot)
        return snapshot

    def capture_final(self, epoch: float | None, step: int) -> dict[str, Any]:
        """Capture the weights Trainer leaves loaded (the best model by default)."""
        snapshot = self._capture("step_final", epoch=epoch, step=step)
        self.snapshots.append(snapshot)
        return snapshot

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics") or {}
        if self.config.trigger == "evaluation" and not any(
            key.startswith("train_") for key in metrics
        ):
            self._capture_next(state)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.config.trigger == "epoch":
            self._capture_next(state)

    def _capture_next(self, state) -> None:
        marker = (state.epoch, state.global_step)
        if marker == self._last_epoch:
            return
        self.counter += 1
        self.snapshots.append(self._capture(f"step_{self.counter:03d}", state.epoch, state.global_step))
        self._last_epoch = marker

    def _capture(self, snapshot_id: str, epoch: float | None, step: int) -> dict[str, Any]:
        model = self.trainer.model_wrapped if self.trainer.model_wrapped is not None else self.trainer.model
        model = model.module if hasattr(model, "module") else model
        return capture_snapshot(
            model=model, tokenizer=self.tokenizer, rows=self.rows, output_root=self.root,
            snapshot_id=snapshot_id, config=self.config, epoch=epoch, step=step,
            checkpoint=self.trainer.state.best_model_checkpoint, model_name=self.model_name,
            capture_parameters=self.capture_parameters,
        )
