"""
Custom CNN Builder
==================
Builds a ``nn.Sequential`` model dynamically from a JSON layer specification.

Supported layer types
---------------------
- Conv2d, BatchNorm2d, ReLU, LeakyReLU, MaxPool2d, AvgPool2d
- Dropout, Flatten, Linear (Dense)

Example layer spec::

    [
        {"type": "Conv2d", "out_channels": 32, "kernel_size": 3, "padding": 1},
        {"type": "BatchNorm2d"},
        {"type": "ReLU"},
        {"type": "MaxPool2d", "kernel_size": 2},
        {"type": "Conv2d", "out_channels": 64, "kernel_size": 3, "padding": 1},
        {"type": "ReLU"},
        {"type": "MaxPool2d", "kernel_size": 2},
        {"type": "Flatten"},
        {"type": "Linear", "out_features": 128},
        {"type": "ReLU"},
        {"type": "Dropout", "p": 0.5},
        {"type": "Linear", "out_features": 10}
    ]
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


def build_custom_cnn(layer_specs: List[Dict[str, Any]],
                     input_channels: int = 3,
                     input_size: tuple = (224, 224),
                     num_classes: int | None = None) -> nn.Sequential:
    """
    Builds a ``nn.Sequential`` model from a list of layer dicts.

    The final ``Linear`` layer's ``out_features`` is overridden by
    *num_classes* if supplied.

    Parameters
    ----------
    layer_specs : list[dict]
        Each dict must have ``"type"`` key.  Other keys are passed as
        keyword arguments to the PyTorch layer constructor.
    input_channels : int
        Number of channels in the input image (default 3 = RGB).
    input_size : tuple
        Spatial size of the input (height, width).
    num_classes : int or None
        If given, override the last ``Linear`` layer's ``out_features``.

    Returns
    -------
    nn.Sequential
    """
    layers: list[nn.Module] = []
    current_channels = input_channels
    current_h, current_w = input_size
    flattened = False

    for i, spec in enumerate(layer_specs):
        spec = dict(spec)  # copy so we don't mutate the original
        layer_type = spec.pop("type")

        if layer_type == "Conv2d":
            out_ch = spec.pop("out_channels", 32)
            ks = spec.pop("kernel_size", 3)
            pad = spec.pop("padding", 0)
            stride = spec.pop("stride", 1)
            layer = nn.Conv2d(current_channels, out_ch, kernel_size=ks,
                              padding=pad, stride=stride, **spec)
            current_channels = out_ch
            current_h = (current_h + 2 * pad - ks) // stride + 1
            current_w = (current_w + 2 * pad - ks) // stride + 1
            layers.append(layer)

        elif layer_type == "BatchNorm2d":
            layers.append(nn.BatchNorm2d(current_channels))

        elif layer_type in ("ReLU", "relu"):
            layers.append(nn.ReLU(inplace=True))

        elif layer_type in ("LeakyReLU", "leakyrelu"):
            layers.append(nn.LeakyReLU(negative_slope=spec.get("negative_slope", 0.01), inplace=True))

        elif layer_type == "MaxPool2d":
            ks = spec.pop("kernel_size", 2)
            stride = spec.pop("stride", ks)
            layer = nn.MaxPool2d(kernel_size=ks, stride=stride, **spec)
            current_h = current_h // stride
            current_w = current_w // stride
            layers.append(layer)

        elif layer_type == "AvgPool2d":
            ks = spec.pop("kernel_size", 2)
            stride = spec.pop("stride", ks)
            layer = nn.AvgPool2d(kernel_size=ks, stride=stride, **spec)
            current_h = current_h // stride
            current_w = current_w // stride
            layers.append(layer)

        elif layer_type == "Dropout":
            layers.append(nn.Dropout(p=spec.get("p", 0.5)))

        elif layer_type == "Flatten":
            layers.append(nn.Flatten())
            flattened = True

        elif layer_type == "Linear":
            if not flattened:
                layers.append(nn.Flatten())
                flattened = True
            in_features = current_channels * current_h * current_w if layers[-1].__class__.__name__ in ("Flatten",) or len([l for l in layers if isinstance(l, nn.Flatten)]) == 1 and not any(isinstance(l, nn.Linear) for l in layers) else spec.get("in_features", 128)
            # For the first Linear after Flatten, compute in_features from spatial dims
            if not any(isinstance(l, nn.Linear) for l in layers):
                in_features = current_channels * current_h * current_w
            else:
                # For subsequent Linear layers, use the previous Linear's out_features
                for l in reversed(layers):
                    if isinstance(l, nn.Linear):
                        in_features = l.out_features
                        break

            out_features = spec.get("out_features", 128)
            # Override last Linear with num_classes if this is the last layer
            if num_classes is not None and i == len(layer_specs) - 1:
                out_features = num_classes

            layers.append(nn.Linear(in_features, out_features))

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    return nn.Sequential(*layers)


def validate_architecture(layer_specs: List[Dict[str, Any]],
                          input_channels: int = 3,
                          input_size: tuple = (224, 224),
                          num_classes: int = 10) -> dict:
    """
    Validates a custom CNN architecture by tracing a dummy tensor through it.

    Returns
    -------
    dict
        ``{"valid": True, "output_shape": [...], "total_params": int}``
        or ``{"valid": False, "error": "..."}``
    """
    try:
        model = build_custom_cnn(layer_specs, input_channels, input_size, num_classes)
        dummy = torch.randn(1, input_channels, *input_size)
        output = model(dummy)
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "valid": True,
            "output_shape": list(output.shape),
            "total_params": total_params,
            "trainable_params": trainable,
            "layer_count": len(list(model.children())),
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
