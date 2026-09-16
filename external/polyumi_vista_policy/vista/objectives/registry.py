"""Objective registry."""

from typing import Any, Dict, Type

from vista.objectives.diffusion import DiffusionObjective
from vista.objectives.flow_matching import FlowMatchingObjective
from vista.objectives.regression import RegressionObjective

OBJECTIVE_REGISTRY: Dict[str, Type] = {
    "diffusion": DiffusionObjective,
    "flow_matching": FlowMatchingObjective,
    "regression": RegressionObjective,
}


def build_objective(name: str, **kwargs: Any):
    if name not in OBJECTIVE_REGISTRY:
        raise KeyError(f"Unknown objective '{name}'. Available: {list(OBJECTIVE_REGISTRY)}")
    if name == "flow_matching":
        if "fm_num_inference_steps" in kwargs and "n_inference_steps" not in kwargs:
            kwargs["n_inference_steps"] = kwargs.pop("fm_num_inference_steps")
        else:
            kwargs.pop("fm_num_inference_steps", None)
        kwargs.pop("num_inference_steps", None)
    return OBJECTIVE_REGISTRY[name](**kwargs)
