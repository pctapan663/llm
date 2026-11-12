"""Placeholder tests: the packages import and the core dependencies work."""

import importlib

import pytest


@pytest.mark.parametrize("module", ["model", "train", "data"])
def test_package_imports(module):
    assert importlib.import_module(module) is not None


def test_dependencies_available():
    import numpy as np
    import torch

    x = torch.from_numpy(np.arange(4, dtype=np.float32))
    assert x.sum().item() == 6.0
