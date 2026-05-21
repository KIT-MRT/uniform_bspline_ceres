"""
Python tests for SplinePositionFinder classes.

SplinePositionFinder keeps the spline control points fixed and optimises only
the query position t (scalar for 1D input, 3-vector for 3D input) so that
f(t) ≈ target.

Covered families:
  - SplinePositionFinder1d1d  (R → R,   optimise scalar t)
  - SplinePositionFinder1d3d  (R → R³,  optimise scalar t)
  - SplinePositionFinder3d1d  (R³ → R,  optimise t ∈ R³)
  - SplinePositionFinder3d2d  (R³ → R², optimise t ∈ R³)
"""

import math

import numpy as np
import pytest

import uniform_bspline as ubs
import uniform_bspline_ceres as ubsc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linspace(start, stop, n):
    return [start + (stop - start) * i / (n - 1) for i in range(n)]


def _fit_1d1d(func, n_pts=100, n_ctrl=15, degree=3):
    """Fit a 1D→1D spline and return the finder."""
    x = linspace(0.0, 1.0, n_pts)
    y = [func(xi) for xi in x]
    fitter = ubsc.SplineFitter1d1d3(num_control_points=n_ctrl)
    fitter.fit(x, y)
    return ubsc.SplinePositionFinder1d1d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )


def _fit_1d3d(n_pts=100, n_ctrl=15):
    """Fit a helix 1D→3D spline and return the finder."""
    t = np.linspace(0.0, 2.0 * math.pi, n_pts)
    pts = np.column_stack([np.cos(t), np.sin(t), t / (2.0 * math.pi)])
    fitter = ubsc.SplineFitter1d3d3(num_control_points=n_ctrl)
    fitter.fit(t, pts)
    return ubsc.SplinePositionFinder1d3d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )


def _fit_3d1d(n_ctrl=5):
    """Fit a 3D Gaussian 3D→1D spline and return the finder."""
    coords = np.linspace(-1.0, 1.0, 6)
    gx, gy, gz = np.meshgrid(coords, coords, coords)
    x3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    y3d = np.exp(-0.5 * (gx.ravel() ** 2 + gy.ravel() ** 2 + gz.ravel() ** 2)).tolist()
    fitter = ubsc.SplineFitter3d1d3(num_ctrl_x=n_ctrl, num_ctrl_y=n_ctrl, num_ctrl_z=n_ctrl)
    fitter.fit(x3d, y3d)
    return ubsc.SplinePositionFinder3d1d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )


def _fit_3d2d(n_ctrl=5):
    """Fit a 3D→2D vector field and return the finder."""
    coords = np.linspace(-1.0, 1.0, 6)
    gx, gy, gz = np.meshgrid(coords, coords, coords)
    x3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    y3d2d = np.column_stack([
        np.sin(math.pi * gx.ravel()) * np.cos(math.pi * gy.ravel()),
        np.cos(math.pi * gz.ravel()),
    ])
    fitter = ubsc.SplineFitter3d2d3(num_ctrl_x=n_ctrl, num_ctrl_y=n_ctrl, num_ctrl_z=n_ctrl)
    fitter.fit(x3d, y3d2d)
    return ubsc.SplinePositionFinder3d2d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )


# ---------------------------------------------------------------------------
# Smoke tests — import and construct all 20 classes (4 families × 5 degrees)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", [
    "SplinePositionFinder1d1d1", "SplinePositionFinder1d1d2", "SplinePositionFinder1d1d3",
    "SplinePositionFinder1d1d4", "SplinePositionFinder1d1d5",
])
def test_construct_1d1d(cls_name):
    x = linspace(0.0, 1.0, 50)
    y = [xi for xi in x]
    fitter = ubsc.SplineFitter1d1d3(num_control_points=10)
    fitter.fit(x, y)
    cls = getattr(ubsc, cls_name)
    # Construct with pre-fitted control points (the degree mismatch intentionally tests
    # that construction itself does not crash; accurate results require matching degrees).
    finder = cls(fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points())
    assert finder is not None


@pytest.mark.parametrize("cls_name", [
    "SplinePositionFinder1d3d1", "SplinePositionFinder1d3d2", "SplinePositionFinder1d3d3",
    "SplinePositionFinder1d3d4", "SplinePositionFinder1d3d5",
])
def test_construct_1d3d(cls_name):
    t = np.linspace(0.0, 1.0, 20)
    pts = np.column_stack([t, t, t])
    fitter = ubsc.SplineFitter1d3d3(num_control_points=8)
    fitter.fit(t, pts)
    cls = getattr(ubsc, cls_name)
    finder = cls(fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points())
    assert finder is not None


@pytest.mark.parametrize("cls_name", [
    "SplinePositionFinder3d1d1", "SplinePositionFinder3d1d2", "SplinePositionFinder3d1d3",
    "SplinePositionFinder3d1d4", "SplinePositionFinder3d1d5",
])
def test_construct_3d1d(cls_name):
    coords = np.linspace(0.0, 1.0, 4)
    gx, gy, gz = np.meshgrid(coords, coords, coords)
    x3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    y3d = np.ones(len(x3d)).tolist()
    fitter = ubsc.SplineFitter3d1d3(num_ctrl_x=6, num_ctrl_y=6, num_ctrl_z=6)
    fitter.fit(x3d, y3d)
    cls = getattr(ubsc, cls_name)
    finder = cls(fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points())
    assert finder is not None


@pytest.mark.parametrize("cls_name", [
    "SplinePositionFinder3d2d1", "SplinePositionFinder3d2d2", "SplinePositionFinder3d2d3",
    "SplinePositionFinder3d2d4", "SplinePositionFinder3d2d5",
])
def test_construct_3d2d(cls_name):
    coords = np.linspace(0.0, 1.0, 4)
    gx, gy, gz = np.meshgrid(coords, coords, coords)
    x3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    y3d2d = np.ones((len(x3d), 2))
    fitter = ubsc.SplineFitter3d2d3(num_ctrl_x=6, num_ctrl_y=6, num_ctrl_z=6)
    fitter.fit(x3d, y3d2d)
    cls = getattr(ubsc, cls_name)
    finder = cls(fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points())
    assert finder is not None


# ---------------------------------------------------------------------------
# 1D → 1D: function inversion
# ---------------------------------------------------------------------------

class TestPositionFinder1d1d:
    def setup_method(self):
        self.finder = _fit_1d1d(math.exp)    # f(t) = exp(t)

    def test_find_known_point(self):
        """Find t s.t. exp(t) ≈ exp(0.5), i.e. t ≈ 0.5."""
        t_opt = self.finder.find(target=math.exp(0.5), initial_t=0.3)
        assert abs(t_opt - 0.5) < 0.02

    def test_find_at_lower_bound(self):
        t_opt = self.finder.find(target=math.exp(0.0), initial_t=0.1)
        assert abs(t_opt - 0.0) < 0.02

    def test_find_at_upper_bound(self):
        t_opt = self.finder.find(target=math.exp(1.0), initial_t=0.9)
        assert abs(t_opt - 1.0) < 0.02

    def test_find_midpoint_sin(self):
        finder = _fit_1d1d(lambda x: math.sin(2 * math.pi * x), n_ctrl=20)
        # sin(2π * 0.25) = 1.0
        t_opt = finder.find(target=1.0, initial_t=0.2)
        assert abs(t_opt - 0.25) < 0.02

    def test_find_with_explicit_bounds(self):
        """Provide explicit lower/upper bounds on the search."""
        t_opt = self.finder.find(
            target=math.exp(0.7), initial_t=0.5, lower_t=0.0, upper_t=1.0
        )
        assert abs(t_opt - 0.7) < 0.02

    def test_find_returns_scalar(self):
        t_opt = self.finder.find(target=math.exp(0.4), initial_t=0.3)
        assert isinstance(t_opt, float)


# ---------------------------------------------------------------------------
# 1D → 3D: closest point on a helix
# ---------------------------------------------------------------------------

class TestPositionFinder1d3d:
    def setup_method(self):
        self.finder = _fit_1d3d()
        # helix: x(t) = [cos(t), sin(t), t/(2π)], t ∈ [0, 2π]

    def test_find_at_pi(self):
        """Point at t=π is [-1, 0, 0.5]."""
        target = np.array([-1.0, 0.0, 0.5])
        t_opt = self.finder.find(target=target, initial_t=2.0)
        assert abs(t_opt - math.pi) < 0.1

    def test_find_at_zero(self):
        """Point at t=0 is [1, 0, 0]."""
        target = np.array([1.0, 0.0, 0.0])
        t_opt = self.finder.find(target=target, initial_t=0.5)
        assert abs(t_opt) < 0.1

    def test_find_at_half_pi(self):
        """Point at t=π/2 is [0, 1, 0.25]."""
        target = np.array([0.0, 1.0, 0.25])
        t_opt = self.finder.find(target=target, initial_t=1.0)
        assert abs(t_opt - math.pi / 2) < 0.1

    def test_find_returns_scalar(self):
        target = np.array([1.0, 0.0, 0.0])
        t_opt = self.finder.find(target=target, initial_t=0.5)
        assert isinstance(t_opt, float)


# ---------------------------------------------------------------------------
# 3D → 1D: inverse query in a scalar field
# ---------------------------------------------------------------------------

class TestPositionFinder3d1d:
    def setup_method(self):
        self.finder = _fit_3d1d()
        # spline approximates f(x,y,z) = exp(-0.5*(x²+y²+z²))

    def test_find_at_origin(self):
        """Maximum is at origin: f(0,0,0) = 1.0."""
        t_opt = self.finder.find(
            target=1.0,
            initial_t=np.array([0.1, 0.1, 0.1]),
        )
        assert np.linalg.norm(t_opt) < 0.2

    def test_find_on_level_set(self):
        """f ≈ 0.5 on a sphere of radius sqrt(2*ln2) ≈ 1.18; start on the x-axis."""
        t_opt = self.finder.find(
            target=0.5,
            initial_t=np.array([0.8, 0.0, 0.0]),
        )
        # f(t_opt) should be close to 0.5
        val = math.exp(-0.5 * float(np.dot(t_opt, t_opt)))
        assert abs(val - 0.5) < 0.05

    def test_find_returns_array(self):
        t_opt = self.finder.find(
            target=0.8, initial_t=np.array([0.3, 0.0, 0.0])
        )
        assert hasattr(t_opt, '__len__')
        assert len(t_opt) == 3

    def test_find_with_explicit_bounds(self):
        t_opt = self.finder.find(
            target=0.9,
            initial_t=np.array([0.2, 0.0, 0.0]),
            lower_t=np.array([-1.0, -1.0, -1.0]),
            upper_t=np.array([1.0, 1.0, 1.0]),
        )
        assert np.all(t_opt >= -1.0) and np.all(t_opt <= 1.0)


# ---------------------------------------------------------------------------
# 3D → 2D: inverse query in a vector field
# ---------------------------------------------------------------------------

class TestPositionFinder3d2d:
    def setup_method(self):
        self.finder = _fit_3d2d()
        # field: f(x,y,z) = [sin(π x)cos(π y), cos(π z)]

    def test_find_at_known_point(self):
        """At (0, 0, 0): f = [sin(0)*cos(0), cos(0)] = [0, 1]."""
        t_opt = self.finder.find(
            target=np.array([0.0, 1.0]),
            initial_t=np.array([0.1, 0.1, 0.1]),
        )
        # t_opt should be near origin (where z ≈ 0 and x or y ≈ 0)
        assert t_opt is not None
        assert len(t_opt) == 3

    def test_find_returns_array(self):
        t_opt = self.finder.find(
            target=np.array([0.0, 0.0]),
            initial_t=np.array([0.5, 0.5, 0.5]),
        )
        assert hasattr(t_opt, '__len__')
        assert len(t_opt) == 3

    def test_find_with_bounds(self):
        t_opt = self.finder.find(
            target=np.array([0.0, 1.0]),
            initial_t=np.array([0.0, 0.0, 0.0]),
            lower_t=np.array([-1.0, -1.0, -1.0]),
            upper_t=np.array([1.0, 1.0, 1.0]),
        )
        assert np.all(t_opt >= -1.0) and np.all(t_opt <= 1.0)


# ---------------------------------------------------------------------------
# Repr / str
# ---------------------------------------------------------------------------

def test_repr_1d1d():
    finder = _fit_1d1d(math.sin)
    r = repr(finder)
    assert "SplinePositionFinder1d1d3" in r


def test_repr_1d3d():
    finder = _fit_1d3d()
    r = repr(finder)
    assert "SplinePositionFinder1d3d3" in r


def test_repr_3d1d():
    finder = _fit_3d1d()
    r = repr(finder)
    assert "SplinePositionFinder3d1d3" in r


def test_repr_3d2d():
    finder = _fit_3d2d()
    r = repr(finder)
    assert "SplinePositionFinder3d2d3" in r
