"""Configuration loading for manifold experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


@dataclass(frozen=True)
class GraphConfig:
    num_nodes: int = 48
    num_communities: int = 4
    p_in: float = 0.28
    p_out: float = 0.04
    edge_attr_dim: int = 4
    seed: int = 7


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 16
    hidden_dim: int = 64
    control_dim: int = 16
    sheaf_lambda: float = 0.15
    restriction_scale: float = 0.08
    num_classes: int = 2


@dataclass(frozen=True)
class SensingConfig:
    k_hop: int = 2
    budget: int = 8
    noise_std: float = 0.03


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    trajectory_steps: int = 8
    dt: float = 0.05
    controller_lr: float = 0.001
    source_lr: float = 0.001
    source_steps: int = 1
    controller_steps: int = 1
    auc_weight: float = 1.0
    mse_weight: float = 1.0
    final_belief_mse_weight: float = 0.5
    ce_weight: float = 0.05
    control_energy_weight: float = 0.01
    control_target_weight: float = 0.1
    source_scale: float = 0.35
    perturbation_mode: str = "boundary"
    boundary_perturbation_scale: float = 0.5
    boundary_perturbation_fraction: float = 0.35
    boundary_center_swap: bool = True
    boundary_recovery_gain: float = 3.0
    source_freeze_after: int = 3
    source_freeze_for: int = 2
    controller_residual_scale: float = 0.05
    analytic_sheaf_gain: float = 0.0
    analytic_community_gain: float = 0.0
    analytic_topk_multiplier: float = 2.0
    log_every: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    device: str = "auto"
    seed: int = 7
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


T = TypeVar("T")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config at {path} must be a mapping")
    return config_from_mapping(raw)


def config_from_mapping(raw: Mapping[str, Any]) -> ExperimentConfig:
    return _dataclass_from_mapping(ExperimentConfig, raw)


def with_overrides(
    config: ExperimentConfig, *, device: str | None = None
) -> ExperimentConfig:
    """Return a config copy with CLI overrides applied."""
    if device is None:
        return config
    return ExperimentConfig(
        device=device,
        seed=config.seed,
        graph=config.graph,
        model=config.model,
        sensing=config.sensing,
        training=config.training,
    )


def _dataclass_from_mapping(cls: type[T], raw: Mapping[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    valid_names = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - valid_names)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")

    for item in fields(cls):
        if item.name not in raw:
            continue
        value = raw[item.name]
        field_type = item.type
        default_value = item.default
        if is_dataclass(default_value):
            field_type = type(default_value)
        elif item.default_factory is not None:  # type: ignore[attr-defined]
            try:
                default_from_factory = item.default_factory()  # type: ignore[misc]
            except TypeError:
                default_from_factory = None
            if is_dataclass(default_from_factory):
                field_type = type(default_from_factory)
        if (
            isinstance(value, Mapping)
            and isinstance(field_type, type)
            and is_dataclass(field_type)
        ):
            kwargs[item.name] = _dataclass_from_mapping(field_type, value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)
