# Adapted from: https://github.com/bghira/SimpleTuner/blob/main/helpers/training/quantisation/__init__.py
from typing import Literal

import torch
from optimum.quanto import qtype

from ltx_trainer import logger

QuantizationOptions = Literal[
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
    "fp8-quanto",
    "fp8uz-quanto",
]


def quantize_model(
    model: torch.nn.Module,
    precision: QuantizationOptions,
    quantize_activations: bool = False,
) -> torch.nn.Module:
    """
    Quantize a model using the specified precision settings.
    Args:
        model: The model to quantize.
        precision: The precision level to quantize to (e.g. "int8-quanto", "fp8-quanto").
        quantize_activations: Whether to quantize activations in addition to weights.
    Returns:
        The quantized model, or the original model if no quantization is performed.
    """
    from optimum.quanto import freeze, quantize  # noqa: PLC0415

    weight_quant = _quanto_type_map(precision)
    extra_quanto_args = {
        "exclude": [
            # Input/output projection layers
            "patchify_proj",
            "audio_patchify_proj",
            "proj_out",
            "audio_proj_out",
            # Timestep embedding layers - int4 tinygemm requires strict bfloat16 input
            # and these receive float32 sinusoidal embeddings that are cast to bfloat16
            "*adaln*",
            "time_proj",
            "timestep_embedder*",
            # Caption/text projection layers
            "caption_projection*",
            "audio_caption_projection*",
            # Normalization layers (usually excluded from quantization)
            "*norm*",
        ]
    }
    if quantize_activations:
        logger.debug("Quantizing model weights and activations")
        extra_quanto_args["activations"] = weight_quant
    else:
        logger.debug("Quantizing model weights only")

    quantize(model, weights=weight_quant, **extra_quanto_args)
    freeze(model)
    return model


def _quanto_type_map(precision: QuantizationOptions) -> torch.dtype | qtype | None:
    from optimum.quanto import (  # noqa: PLC0415
        qfloat8,
        qfloat8_e4m3fnuz,
        qint2,
        qint4,
        qint8,
    )

    if precision == "int2-quanto":
        return qint2
    elif precision == "int4-quanto":
        return qint4
    elif precision == "int8-quanto":
        return qint8
    elif precision in ("fp8-quanto", "fp8uz-quanto"):
        if torch.backends.mps.is_available():
            logger.warning(
                "MPS doesn't support dtype float8. "
                "you must select another precision level such as int2, int8, or int8.",
            )
            return None
        if precision == "fp8-quanto":
            return qfloat8
        elif precision == "fp8uz-quanto":
            return qfloat8_e4m3fnuz

    raise ValueError(f"Invalid quantisation level: {precision}")
