"""Custom CNN builder: spatial size math and the num_classes override."""

import pytest

torch = pytest.importorskip("torch")

from easyml.cv.custom_cnn import build_custom_cnn, validate_architecture  # noqa: E402


@pytest.mark.parametrize("pool", [
    {"type": "MaxPool2d", "kernel_size": 3, "stride": 2},                  # kernel != stride
    {"type": "MaxPool2d", "kernel_size": 3, "stride": 2, "padding": 1},    # padding
    {"type": "AvgPool2d", "kernel_size": 3, "stride": 2, "padding": 1},
    {"type": "MaxPool2d", "kernel_size": 2, "stride": 2, "ceil_mode": True},
    {"type": "AvgPool2d", "kernel_size": 3, "stride": 1},
])
@pytest.mark.parametrize("size", [(31, 31), (32, 48)])
def test_linear_input_matches_pooled_feature_map(pool, size):
    specs = [
        {"type": "Conv2d", "out_channels": 4, "kernel_size": 3, "padding": 1},
        {"type": "ReLU"},
        pool,
        {"type": "Conv2d", "out_channels": 8, "kernel_size": 5, "stride": 2},
        dict(pool),
        {"type": "Linear", "out_features": 16},
        {"type": "Linear", "out_features": 3},
    ]
    model = build_custom_cnn(specs, input_size=size, num_classes=3)
    assert model(torch.zeros(2, 3, *size)).shape == (2, 3)


def test_num_classes_overrides_last_linear_even_when_layers_follow_it():
    specs = [
        {"type": "Conv2d", "out_channels": 4, "kernel_size": 3},
        {"type": "Flatten"},
        {"type": "Linear", "out_features": 32},
        {"type": "ReLU"},
        {"type": "Linear", "out_features": 10},
        {"type": "Dropout", "p": 0.2},
    ]
    result = validate_architecture(specs, input_size=(16, 16), num_classes=4)
    assert result["valid"], result
    assert result["output_shape"] == [1, 4]
    hidden = build_custom_cnn(specs, input_size=(16, 16), num_classes=4)
    assert [m.out_features for m in hidden if isinstance(m, torch.nn.Linear)] == [32, 4]


def test_feature_map_that_vanishes_is_reported():
    specs = [{"type": "MaxPool2d", "kernel_size": 4}] * 3 + [{"type": "Linear", "out_features": 2}]
    result = validate_architecture(specs, input_size=(32, 32), num_classes=2)
    assert not result["valid"]
    assert "Layer 3 (MaxPool2d)" in result["error"]
