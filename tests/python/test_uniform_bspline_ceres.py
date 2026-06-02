"""
Python binding tests for uniform_bspline_ceres.

These tests mirror the C++ evaluator/generator tests so that the Python API
is verified to be consistent with the C++ implementation.

Run with:
    pytest tests/python/
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


def _fit_exp(n_cp=20, smw=0.0):
    x = linspace(0.0, 1.0, 100)
    y = [math.exp(2.0 * xi) for xi in x]
    fitter = ubsc.SplineFitter1d1d3(num_control_points=n_cp)
    fitter.fit(x, y, smoothness_weight=smw)
    return fitter


def _fit_helix(n_cp=20):
    t = np.linspace(0.0, 2.0 * math.pi, 200)
    pts = np.column_stack([np.cos(t), np.sin(t), t / (2.0 * math.pi)])
    fitter = ubsc.SplineFitter1d3d3(num_control_points=n_cp)
    fitter.fit(t, pts)
    return fitter


def _fit_3d1d(n_ctrl=5):
    coords = np.linspace(-1.0, 1.0, 6)
    gx, gy, gz = np.meshgrid(coords, coords, coords)
    x3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    y3d = np.exp(-0.5 * (gx.ravel() ** 2 + gy.ravel() ** 2 + gz.ravel() ** 2)).tolist()
    fitter = ubsc.SplineFitter3d1d3(num_ctrl_x=n_ctrl, num_ctrl_y=n_ctrl, num_ctrl_z=n_ctrl)
    fitter.fit(x3d, y3d)
    return fitter


def _fit_3d2d(n_ctrl=5):
    coords = np.linspace(-1.0, 1.0, 6)
    gx, gy, gz = np.meshgrid(coords, coords, coords)
    x3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    y3d2d = np.column_stack([
        np.sin(math.pi * gx.ravel()) * np.cos(math.pi * gy.ravel()),
        np.cos(math.pi * gz.ravel()),
    ])
    fitter = ubsc.SplineFitter3d2d3(num_ctrl_x=n_ctrl, num_ctrl_y=n_ctrl, num_ctrl_z=n_ctrl)
    fitter.fit(x3d, y3d2d)
    return fitter


# ===========================================================================
# SplineFitter 1D -> 1D
# ===========================================================================

class TestSplineFitter1d1d:

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_construct_all_degrees(self, degree):
        cls = getattr(ubsc, f"SplineFitter1d1d{degree}")
        assert cls(num_control_points=10) is not None

    def test_fit_linear(self):
        x = linspace(0.0, 1.0, 50)
        y = [2.0 * xi + 1.0 for xi in x]
        fitter = ubsc.SplineFitter1d1d1(num_control_points=5)
        fitter.fit(x, y)
        assert len(fitter.get_control_points()) == 5
        assert abs(fitter.get_lower_bound() - 0.0) < 1e-10
        assert abs(fitter.get_upper_bound() - 1.0) < 1e-10

    def test_fit_exponential(self):
        fitter = _fit_exp(n_cp=20)
        spline = ubs.UniformBSpline1d1d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        for xi in linspace(0.05, 0.95, 20):
            assert abs(spline.evaluate(xi) - math.exp(2.0 * xi)) < 1e-3

    def test_fit_with_smoothness(self):
        x = linspace(0.0, 1.0, 100)
        y = [math.sin(2.0 * math.pi * xi) for xi in x]
        fitter = ubsc.SplineFitter1d1d3(num_control_points=15)
        fitter.fit(x, y, smoothness_weight=1e-4)
        assert len(fitter.get_control_points()) == 15

    def test_fit_custom_bounds(self):
        x = linspace(-3.0, 3.0, 100)
        y = [xi ** 2 for xi in x]
        fitter = ubsc.SplineFitter1d1d3(num_control_points=12)
        fitter.fit(x, y, lower_bound=-3.0, upper_bound=3.0)
        assert abs(fitter.get_lower_bound() - (-3.0)) < 1e-10
        assert abs(fitter.get_upper_bound() - 3.0) < 1e-10

    def test_repr(self):
        fitter = ubsc.SplineFitter1d1d3(num_control_points=10)
        fitter.fit(linspace(0.0, 1.0, 20), linspace(0.0, 1.0, 20))
        assert "SplineFitter1d1d3" in repr(fitter)

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_all_degrees_recover_linear(self, degree):
        x = linspace(0.0, 1.0, 50)
        y = [2.0 * xi + 1.0 for xi in x]
        fitter = getattr(ubsc, f"SplineFitter1d1d{degree}")(num_control_points=10)
        fitter.fit(x, y)
        spline = getattr(ubs, f"UniformBSpline1d1d{degree}")(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        for xi in np.linspace(0.05, 0.95, 10):
            assert abs(spline.evaluate(xi) - (2.0 * xi + 1.0)) < 1e-6


# ===========================================================================
# SplineFitter 1D -> 3D
# ===========================================================================

class TestSplineFitter1d3d:

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_construct_all_degrees(self, degree):
        cls = getattr(ubsc, f"SplineFitter1d3d{degree}")
        assert cls(num_control_points=10) is not None

    def test_fit_helix(self):
        fitter = _fit_helix()
        spline = ubs.UniformBSpline1d3d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        np.testing.assert_allclose(spline.evaluate(math.pi), [-1.0, 0.0, 0.5], atol=0.05)

    def test_control_points_shape(self):
        fitter = _fit_helix(n_cp=15)
        assert fitter.get_control_points().shape == (15, 3)


# ===========================================================================
# SplineFitter 3D -> 1D
# ===========================================================================

class TestSplineFitter3d1d:

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_construct_all_degrees(self, degree):
        cls = getattr(ubsc, f"SplineFitter3d1d{degree}")
        n = max(4, degree + 1)
        assert cls(num_ctrl_x=n, num_ctrl_y=n, num_ctrl_z=n) is not None

    def test_control_points_shape(self):
        fitter = _fit_3d1d(n_ctrl=5)
        assert fitter.get_control_points().shape == (5, 5, 5)


# ===========================================================================
# SplineFitter 3D -> 2D
# ===========================================================================

class TestSplineFitter3d2d:

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_construct_all_degrees(self, degree):
        cls = getattr(ubsc, f"SplineFitter3d2d{degree}")
        n = max(4, degree + 1)
        assert cls(num_ctrl_x=n, num_ctrl_y=n, num_ctrl_z=n) is not None

    def test_control_points_shape(self):
        fitter = _fit_3d2d(n_ctrl=5)
        assert fitter.get_control_points().shape == (5, 5, 5, 2)


# ===========================================================================
# SplinePositionFinder 1D -> 1D
# ===========================================================================

class TestSplinePositionFinder1d1d:

    def setup_method(self):
        fitter = _fit_exp(n_cp=20)
        self.finder = ubsc.SplinePositionFinder1d1d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_construct_all_degrees(self, degree):
        fitter = _fit_exp()
        cls = getattr(ubsc, f"SplinePositionFinder1d1d{degree}")
        assert cls(fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()) is not None

    def test_find_at_lower_bound(self):
        t_opt = self.finder.find(target=math.exp(0.0), initial_t=0.1)
        assert abs(t_opt - 0.0) < 0.02

    def test_find_at_upper_bound(self):
        t_opt = self.finder.find(target=math.exp(2.0), initial_t=0.9)
        assert abs(t_opt - 1.0) < 0.02

    def test_find_with_explicit_bounds(self):
        t_opt = self.finder.find(target=math.exp(1.4), initial_t=0.5, lower_t=0.0, upper_t=1.0)
        assert abs(t_opt - 0.7) < 0.02

    def test_find_returns_scalar(self):
        assert isinstance(self.finder.find(target=math.exp(0.8), initial_t=0.3), float)

    def test_repr(self):
        assert "SplinePositionFinder1d1d3" in repr(self.finder)


# ===========================================================================
# SplinePositionFinder 1D -> 3D
# ===========================================================================

class TestSplinePositionFinder1d3d:

    def setup_method(self):
        fitter = _fit_helix()
        self.finder = ubsc.SplinePositionFinder1d3d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_construct_all_degrees(self, degree):
        fitter = _fit_helix()
        cls = getattr(ubsc, f"SplinePositionFinder1d3d{degree}")
        assert cls(fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()) is not None

    def test_find_at_zero(self):
        t_opt = self.finder.find(target=np.array([1.0, 0.0, 0.0]), initial_t=0.5)
        assert abs(t_opt) < 0.1

    def test_find_at_half_pi(self):
        t_opt = self.finder.find(target=np.array([0.0, 1.0, 0.25]), initial_t=1.0)
        assert abs(t_opt - math.pi / 2) < 0.1

    def test_find_returns_scalar(self):
        t_opt = self.finder.find(target=np.array([1.0, 0.0, 0.0]), initial_t=0.5)
        assert isinstance(t_opt, float)

    def test_repr(self):
        assert "SplinePositionFinder1d3d3" in repr(self.finder)


# ===========================================================================
# SplinePositionFinder 3D -> 1D
# ===========================================================================

class TestSplinePositionFinder3d1d:

    def setup_method(self):
        fitter = _fit_3d1d()
        self.finder = ubsc.SplinePositionFinder3d1d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_construct_all_degrees(self, degree):
        fitter = _fit_3d1d(n_ctrl=max(5, degree + 1))
        cls = getattr(ubsc, f"SplinePositionFinder3d1d{degree}")
        assert cls(fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()) is not None

    def test_find_near_maximum(self):
        t_opt = self.finder.find(target=1.0, initial_t=np.array([0.1, 0.1, 0.1]))
        assert np.linalg.norm(t_opt) < 0.2

    def test_find_returns_array_of_length_3(self):
        t_opt = self.finder.find(target=0.8, initial_t=np.array([0.3, 0.0, 0.0]))
        assert len(t_opt) == 3

    def test_find_with_explicit_bounds(self):
        t_opt = self.finder.find(
            target=0.9, initial_t=np.array([0.2, 0.0, 0.0]),
            lower_t=np.array([-1.0, -1.0, -1.0]), upper_t=np.array([1.0, 1.0, 1.0]),
        )
        assert np.all(t_opt >= -1.0) and np.all(t_opt <= 1.0)

    def test_repr(self):
        assert "SplinePositionFinder3d1d3" in repr(self.finder)


# ===========================================================================
# SplinePositionFinder 3D -> 2D
# ===========================================================================

class TestSplinePositionFinder3d2d:

    def setup_method(self):
        fitter = _fit_3d2d()
        self.finder = ubsc.SplinePositionFinder3d2d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_construct_all_degrees(self, degree):
        fitter = _fit_3d2d(n_ctrl=max(5, degree + 1))
        cls = getattr(ubsc, f"SplinePositionFinder3d2d{degree}")
        assert cls(fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()) is not None

    def test_find_returns_array_of_length_3(self):
        t_opt = self.finder.find(target=np.array([0.0, 1.0]), initial_t=np.array([0.1, 0.1, 0.1]))
        assert len(t_opt) == 3

    def test_find_with_bounds(self):
        t_opt = self.finder.find(
            target=np.array([0.0, 1.0]), initial_t=np.array([0.0, 0.0, 0.0]),
            lower_t=np.array([-1.0, -1.0, -1.0]), upper_t=np.array([1.0, 1.0, 1.0]),
        )
        assert np.all(t_opt >= -1.0) and np.all(t_opt <= 1.0)

    def test_repr(self):
        assert "SplinePositionFinder3d2d3" in repr(self.finder)


# ===========================================================================
# Analytic accuracy tests
# ===========================================================================

class TestFitterAccuracy:

    def test_1d1d_exponential_residuals(self):
        x = linspace(0.0, 1.0, 200)
        y = [math.exp(2.0 * xi) for xi in x]
        fitter = ubsc.SplineFitter1d1d3(num_control_points=30)
        fitter.fit(x, y, smoothness_weight=1e-6, smoothness_order=2)
        spline = ubs.UniformBSpline1d1d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        for xi in np.linspace(0.05, 0.95, 20):
            assert abs(spline.evaluate(xi) - math.exp(2.0 * xi)) < 1e-3

    def test_1d1d_sine_residuals(self):
        x = linspace(0.0, 1.0, 200)
        y = [math.sin(2.0 * math.pi * xi) for xi in x]
        fitter = ubsc.SplineFitter1d1d3(num_control_points=30)
        fitter.fit(x, y, smoothness_weight=1e-6, smoothness_order=2)
        spline = ubs.UniformBSpline1d1d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        for xi in np.linspace(0.05, 0.95, 20):
            assert abs(spline.evaluate(xi) - math.sin(2.0 * math.pi * xi)) < 1e-3

    def test_1d1d_non_unit_interval(self):
        x = np.linspace(-3.0, 3.0, 100).tolist()
        y = [xi ** 2 for xi in x]
        fitter = ubsc.SplineFitter1d1d3(num_control_points=20)
        fitter.fit(x, y, lower_bound=-3.0, upper_bound=3.0, smoothness_weight=1e-6)
        spline = ubs.UniformBSpline1d1d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        for xi in np.linspace(-2.5, 2.5, 20):
            assert abs(spline.evaluate(xi) - xi ** 2) < 0.01

    def test_1d3d_helix(self):
        fitter = _fit_helix(n_cp=30)
        spline = ubs.UniformBSpline1d3d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        for ti in np.linspace(0.2, 2.0 * math.pi - 0.2, 20):
            pos = spline.evaluate(ti)
            expected = np.array([math.cos(ti), math.sin(ti), ti / (2.0 * math.pi)])
            assert np.linalg.norm(pos - expected) < 0.05

    def test_3d1d_gaussian_center(self):
        fitter = _fit_3d1d(n_ctrl=6)
        spline = ubs.UniformBSpline3d1d3()
        spline.set_bounds(fitter.get_lower_bound(), fitter.get_upper_bound())
        spline.set_control_points(fitter.get_control_points())
        assert abs(spline.evaluate(np.array([0.0, 0.0, 0.0])) - 1.0) < 0.02

    def test_3d2d_vector_field_at_origin(self):
        fitter = _fit_3d2d(n_ctrl=6)
        spline = ubs.UniformBSpline3d2d3()
        spline.set_bounds(fitter.get_lower_bound(), fitter.get_upper_bound())
        spline.set_control_points(fitter.get_control_points())
        np.testing.assert_allclose(spline.evaluate(np.array([0.0, 0.0, 0.0])), [0.0, 1.0], atol=0.05)


class TestPositionFinderAccuracy:

    def test_1d1d_multiple_targets(self):
        fitter = _fit_exp(n_cp=30, smw=1e-8)
        finder = ubsc.SplinePositionFinder1d1d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        for t_true in [0.1, 0.3, 0.5, 0.7, 0.9]:
            t_opt = finder.find(target=math.exp(2.0 * t_true), initial_t=t_true + 0.05)
            assert abs(t_opt - t_true) < 0.02, f"at t={t_true}: got {t_opt:.4f}"

    def test_1d3d_helix_closest_point(self):
        fitter = _fit_helix(n_cp=30)
        finder = ubsc.SplinePositionFinder1d3d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        t_opt = finder.find(target=np.array([-1.0, 0.0, 0.5]), initial_t=2.5)
        assert abs(t_opt - math.pi) < 0.1

    def test_3d1d_gaussian_level_set(self):
        fitter = _fit_3d1d(n_ctrl=6)
        finder = ubsc.SplinePositionFinder3d1d3(
            fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
        )
        t_opt = finder.find(target=0.5, initial_t=np.array([0.8, 0.0, 0.0]))
        r_expected = math.sqrt(2.0 * math.log(2.0))
        assert abs(float(np.linalg.norm(t_opt)) - r_expected) < 0.1


# ===========================================================================
# Example snippet tests
# ===========================================================================

def test_example_fitter1d1d():
    x = [i / 100.0 for i in range(101)]
    y = [math.exp(2.0 * xi) for xi in x]
    fitter = ubsc.SplineFitter1d1d3(num_control_points=20)
    fitter.fit(x, y, smoothness_weight=1e-4, smoothness_order=2)
    spline = ubs.UniformBSpline1d1d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )
    assert abs(spline.evaluate(0.5) - math.exp(1.0)) < 0.01


def test_example_fitter1d3d():
    t = np.linspace(0.0, 2.0 * math.pi, 200)
    pts = np.column_stack([np.cos(t), np.sin(t), t / (2.0 * math.pi)])
    fitter = ubsc.SplineFitter1d3d3(num_control_points=20)
    fitter.fit(t, pts, smoothness_weight=1e-4, smoothness_order=2)
    spline = ubs.UniformBSpline1d3d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )
    np.testing.assert_allclose(spline.evaluate(math.pi), [-1.0, 0.0, 0.5], atol=0.05)


def test_example_fitter3d1d():
    fitter = _fit_3d1d()
    spline = ubs.UniformBSpline3d1d3()
    spline.set_bounds(fitter.get_lower_bound(), fitter.get_upper_bound())
    spline.set_control_points(fitter.get_control_points())
    assert abs(spline.evaluate(np.array([0.0, 0.0, 0.0])) - 1.0) < 0.05


def test_example_fitter3d2d():
    fitter = _fit_3d2d()
    spline = ubs.UniformBSpline3d2d3()
    spline.set_bounds(fitter.get_lower_bound(), fitter.get_upper_bound())
    spline.set_control_points(fitter.get_control_points())
    np.testing.assert_allclose(spline.evaluate(np.array([0.0, 0.0, 0.0])), [0.0, 1.0], atol=0.05)


def test_example_position_finder1d1d():
    fitter = ubsc.SplineFitter1d1d3(num_control_points=20)
    fitter.fit(
        [i / 100.0 for i in range(101)],
        [math.exp(2.0 * i / 100.0) for i in range(101)],
        smoothness_weight=1e-4, smoothness_order=2,
    )
    finder = ubsc.SplinePositionFinder1d1d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )
    assert abs(finder.find(target=math.exp(1.0), initial_t=0.3) - 0.5) < 0.02


def test_example_position_finder1d3d():
    fitter = _fit_helix(n_cp=20)
    finder = ubsc.SplinePositionFinder1d3d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )
    assert abs(finder.find(target=np.array([-1.0, 0.0, 0.5]), initial_t=2.0) - math.pi) < 0.1


def test_example_position_finder3d1d():
    fitter = _fit_3d1d()
    finder = ubsc.SplinePositionFinder3d1d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )
    t_opt = finder.find(target=0.5, initial_t=np.array([0.8, 0.0, 0.0]))
    r_expected = math.sqrt(2.0 * math.log(2.0))
    assert abs(float(np.linalg.norm(t_opt)) - r_expected) < 0.1


def test_example_position_finder3d2d():
    fitter = _fit_3d2d()
    finder = ubsc.SplinePositionFinder3d2d3(
        fitter.get_lower_bound(), fitter.get_upper_bound(), fitter.get_control_points()
    )
    t_opt = finder.find(target=np.array([0.0, 1.0]), initial_t=np.array([0.0, 0.0, 0.0]))
    spline = ubs.UniformBSpline3d2d3()
    spline.set_bounds(fitter.get_lower_bound(), fitter.get_upper_bound())
    spline.set_control_points(fitter.get_control_points())
    np.testing.assert_allclose(spline.evaluate(t_opt), [0.0, 1.0], atol=0.1)
