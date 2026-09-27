"""CodeCarbon adapter and energy normalization."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schemas import EnergyMetrics


def normalize_energy(
    total_energy_kwh: float | None,
    documents: int,
    tokens: int,
) -> dict[str, float | None]:
    """Normalize measured energy without inventing values for missing observations."""
    if total_energy_kwh is None:
        return {
            "joules_per_1000_documents": None,
            "joules_per_1000_tokens": None,
            "wh_per_1000_documents": None,
        }
    return {
        "joules_per_1000_documents": (
            total_energy_kwh * 3_600_000 * 1000 / documents if documents else None
        ),
        "joules_per_1000_tokens": (
            total_energy_kwh * 3_600_000 * 1000 / tokens if tokens else None
        ),
        "wh_per_1000_documents": (
            total_energy_kwh * 1000 * 1000 / documents if documents else None
        ),
    }


class EnergyTracker:
    """Thin adapter around the same current CodeCarbon API used by NER training."""

    def __init__(self, mode: str, measure_power_secs: int = 1) -> None:
        from codecarbon import EmissionsTracker, OutputMethod
        from codecarbon.output_methods.logger import LoggerOutput
        import logging

        from lab.ner.training.tracking import visible_gpu_ids

        logger = logging.getLogger("lab.ner.quantisation.energy")
        logger.addHandler(logging.NullHandler())
        self.mode = mode
        self.tracker = EmissionsTracker(
            tracking_mode="process",
            gpu_ids=visible_gpu_ids(),
            output_methods=[OutputMethod.LOGGER],
            logging_logger=LoggerOutput(logger=logger),
            allow_multiple_runs=True,
            log_level="error",
            measure_power_secs=measure_power_secs,
        )

    def start(self) -> None:
        self.tracker.start()

    def stop(self, passes: int, documents: int, tokens: int) -> dict[str, Any]:
        self.tracker.stop()
        data = self.tracker.final_emissions_data
        total = data.energy_consumed if data else None
        normalized = normalize_energy(total, documents, tokens)
        metrics = EnergyMetrics(
            mode=self.mode,
            cpu_energy_kwh=getattr(data, "cpu_energy", None),
            gpu_energy_kwh=getattr(data, "gpu_energy", None),
            ram_energy_kwh=getattr(data, "ram_energy", None),
            total_energy_kwh=total,
            duration_s=getattr(data, "duration", None),
            co2_kg=getattr(data, "emissions", None),
            passes=passes,
            documents_processed=documents,
            tokens_processed=tokens,
            **normalized,
        )
        return asdict(metrics)
