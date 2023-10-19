import pytest
from mbs import Spectrum
from unittest import TestCase

from operator import add, sub, mul, truediv, pow
import numpy as np

@pytest.fixture
def test_spectrum_int():
    return Spectrum(np.random.poisson(40, size=(101, 93)))

@pytest.fixture
def test_spectrum_float():
    return Spectrum(np.random.random(size=(101, 93)))

@pytest.mark.parametrize("test_op", [add, sub, mul, truediv])
@pytest.mark.parametrize("scalar", [0] + np.random.randint(-10, 10, size=5).tolist() + np.random.random(size=5).tolist())
def test_spectra_rightop_scalar(test_spectrum_int, test_op, scalar):
    test_op(test_spectrum_int, scalar)