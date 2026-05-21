"""
Python tests for uniform_bspline_ceres.

Tests the SplineFitter classes that fit uniform B-splines to (x, y) data.
"""

import math
import pytest

import uniform_bspline_ceres as ubsc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linspace(start, stop, n):
    return [start + (stop - start) * i / (n - 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Smoke tests — import and construct
# ---------------------------------------------------------------------------

def test_import():
    assert ubsc is not None


@pytest.mark.parametrize("cls_name", [
    "SplineFitter1d1d1",
    "SplineFitter1d1d2",
    "SplineFitter1d1d3",
    "SplineFitter1d1d4",
    "SplineFitter1d1d5",
])
def test_construct(cls_name):
    cls = getattr(ubsc, cls_name)
    fitter = cls(num_control_points=10)
    assert fitter is not None


# ---------------------------------------------------------------------------
# Fitting tests
# ---------------------------------------------------------------------------

def test_fit_linear():
    """Fit a linear function y = 2x + 1; a degree-1 spline should recover it."""
    x = linspace(0.0, 1.0, 50)
    y = [2.0 * xi + 1.0 for xi in x]

    fitter = ubsc.SplineFitter1d1d1(num_control_points=5)
    fitter.fit(x, y)

    cp = fitter.get_control_points()
    assert len(cp) == 5
    assert abs(fitter.get_lower_bound() - 0.0) < 1e-10
    assert abs(fitter.get_upper_bound() - 1.0) < 1e-10


def test_fit_exponential_degree3():
    """Fit y = exp(2x) with a cubic spline; check residuals are small."""
    import uniform_bspline as ubs

    x = linspace(0.0, 1.0, 200)
    y = [math.exp(2.0 * xi) for xi in x]

    fitter = ubsc.SplineFitter1d1d3(num_control_points=20)
    fitter.fit(x, y)

    # Reconstruct the spline and evaluate at test points
    cp = fitter.get_control_points()
    spline = ubs.UniformBSpline1d1d3(
        fitter.get_lower_bound(),
        fitter.get_upper_bound(),
        cp,
    )

    for xi in linspace(0.05, 0.95, 20):
        est = spline.evaluate(xi)
        gt = math.exp(2.0 * xi)
        assert abs(est - gt) < 1e-3, f"At x={xi}: est={est}, gt={gt}"


def test_fit_with_smoothness():
    """Fit with smoothness regularization; fitting should still succeed."""
    x = linspace(0.0, 1.0, 100)
    y = [math.sin(2.0 * math.pi * xi) for xi in x]

    fitter = ubsc.SplineFitter1d1d3(num_control_points=15)
    fitter.fit(x, y, smoothness_weight=1e-4)

    cp = fitter.get_control_points()
    assert len(cp) == 15


def test_fit_custom_bounds():
    """Fit on a non-unit interval."""
    x = linspace(-3.0, 3.0, 100)
    y = [xi ** 2 for xi in x]

    fitter = ubsc.SplineFitter1d1d3(num_control_points=12)
    fitter.fit(x, y, lower_bound=-3.0, upper_bound=3.0)

    assert abs(fitter.get_lower_bound() - (-3.0)) < 1e-10
    assert abs(fitter.get_upper_bound() - 3.0) < 1e-10


def test_repr():
    fitter = ubsc.SplineFitter1d1d3(num_control_points=10)
    x = linspace(0.0, 1.0, 20)
    y = [xi for xi in x]
    fitter.fit(x, y)
    r = repr(fitter)
    assert "SplineFitter1d1d3" in r
