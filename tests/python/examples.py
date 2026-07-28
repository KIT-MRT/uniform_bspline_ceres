"""
Runnable Python examples used as Doxygen snippets in docs/mainpage.md.

Each named region between  ## [Tag]  /  ## [Tag]  markers is embedded
verbatim into the documentation via  \snippet examples.py Tag.
"""

import math

import numpy as np
import uniform_bspline as ubs
import uniform_bspline_ceres as ubsc

# ---------------------------------------------------------------------------
# See the Doxygen documentation for a more detailed explanation of the following example.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1D -> 1D spline fitting with SplineFitter1d1d
# ---------------------------------------------------------------------------

## [Fitter1d1d_Python]
x = [i / 100.0 for i in range(101)]
y = [math.exp(2.0 * xi) for xi in x]

fitter = ubsc.SplineFitter1d1d3(num_control_points=20)
fitter.fit(x, y, smoothness_weight=1e-4, smoothness_order=2)  # penalise 2nd derivative

# Reconstruct via uniform_bspline for evaluation
spline = ubs.UniformBSpline1d1d3(
    fitter.get_lower_bound(),
    fitter.get_upper_bound(),
    fitter.get_control_points(),
)
y_est = spline.evaluate(0.5)   # ≈ exp(1.0) ≈ 2.718
## [Fitter1d1d_Python]

# ---------------------------------------------------------------------------
# 1D -> 3D trajectory fitting with SplineFitter1d3d
# ---------------------------------------------------------------------------

## [Fitter1d3d_Python]
# Helix: x(t) = [cos(t), sin(t), t / (2*pi)]
t = np.linspace(0.0, 2.0 * math.pi, 200)
pts = np.column_stack([np.cos(t), np.sin(t), t / (2.0 * math.pi)])  # shape (200, 3)

fitter_3d = ubsc.SplineFitter1d3d3(num_control_points=20)
fitter_3d.fit(t, pts, smoothness_weight=1e-4, smoothness_order=2)

# Reconstruct via uniform_bspline for evaluation
spline_3d = ubs.UniformBSpline1d3d3(
    fitter_3d.get_lower_bound(),
    fitter_3d.get_upper_bound(),
    fitter_3d.get_control_points(),  # shape (20, 3)
)
pos = spline_3d.evaluate(math.pi)   # ≈ [cos(π), sin(π), 0.5] = [-1, 0, 0.5]
## [Fitter1d3d_Python]

# ---------------------------------------------------------------------------
# 3D -> 1D cost volume fitting with SplineFitter3d1d
# ---------------------------------------------------------------------------

## [Fitter3d1d_Python]
# Sample a 3D Gaussian on a 6×6×6 grid
coords = np.linspace(-1.0, 1.0, 6)
gx, gy, gz = np.meshgrid(coords, coords, coords)
x3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])   # shape (216, 3)
y3d = np.exp(-0.5 * (gx.ravel() ** 2 + gy.ravel() ** 2 + gz.ravel() ** 2)).tolist()

fitter_cv = ubsc.SplineFitter3d1d3(num_ctrl_x=5, num_ctrl_y=5, num_ctrl_z=5)
fitter_cv.fit(x3d, y3d)

# Reconstruct via uniform_bspline for evaluation
spline_cv = ubs.UniformBSpline3d1d3(
    fitter_cv.get_lower_bound(),
    fitter_cv.get_upper_bound(),
    fitter_cv.get_control_points(),  # shape (5, 5, 5)
)
cost = spline_cv.evaluate(np.array([0.0, 0.0, 0.0]))  # ≈ exp(0) = 1.0
## [Fitter3d1d_Python]

# ---------------------------------------------------------------------------
# 3D -> 2D deformation field fitting with SplineFitter3d2d
# ---------------------------------------------------------------------------

## [Fitter3d2d_Python]
# Sample a smooth 3D->2D vector field on a 6×6×6 grid
# field: f(x,y,z) = [sin(pi*x)*cos(pi*y), cos(pi*z)]
coords = np.linspace(-1.0, 1.0, 6)
gx, gy, gz = np.meshgrid(coords, coords, coords)
x3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])   # shape (216, 3)
y3d2d = np.column_stack([
    np.sin(np.pi * gx.ravel()) * np.cos(np.pi * gy.ravel()),
    np.cos(np.pi * gz.ravel()),
])  # shape (216, 2)

fitter_df = ubsc.SplineFitter3d2d3(num_ctrl_x=5, num_ctrl_y=5, num_ctrl_z=5)
fitter_df.fit(x3d, y3d2d)

# Reconstruct via uniform_bspline for evaluation
spline_df = ubs.UniformBSpline3d2d3(
    fitter_df.get_lower_bound(),
    fitter_df.get_upper_bound(),
    fitter_df.get_control_points(),  # shape (5, 5, 5, 2)
)
field = spline_df.evaluate(np.array([0.0, 0.0, 0.0]))  # ≈ [sin(0)*cos(0), cos(0)] = [0, 1]
## [Fitter3d2d_Python]

# ---------------------------------------------------------------------------
# 1D -> 1D spline position finding with SplinePositionFinder1d1d
# ---------------------------------------------------------------------------

## [PositionFinder1d1d_Python]
finder_1d1d = ubsc.SplinePositionFinder1d1d3(
    fitter.get_lower_bound(),
    fitter.get_upper_bound(),
    fitter.get_control_points(),
)
# Find t* such that spline(t*) ≈ math.exp(1.0) ≈ 2.718 (i.e. invert f(t) = exp(2t) at t=0.5)
t_opt_1d1d = finder_1d1d.find(target=math.exp(1.0), initial_t=0.3)  # ≈ 0.5
## [PositionFinder1d1d_Python]

# ---------------------------------------------------------------------------
# 1D -> 3D closest point on curve with SplinePositionFinder1d3d
# ---------------------------------------------------------------------------

## [PositionFinder1d3d_Python]
finder_1d3d = ubsc.SplinePositionFinder1d3d3(
    fitter_3d.get_lower_bound(),
    fitter_3d.get_upper_bound(),
    fitter_3d.get_control_points(),  # shape (20, 3)
)
# Find t* closest to [-1, 0, 0.5] on the helix  (should be ≈ π)
t_opt_1d3d = finder_1d3d.find(
    target=np.array([-1.0, 0.0, 0.5]),
    initial_t=2.0,
)  # ≈ π ≈ 3.14
## [PositionFinder1d3d_Python]

# ---------------------------------------------------------------------------
# 3D -> 1D inverse query in a cost volume with SplinePositionFinder3d1d
# ---------------------------------------------------------------------------

## [PositionFinder3d1d_Python]
finder_3d1d = ubsc.SplinePositionFinder3d1d3(
    fitter_cv.get_lower_bound(),
    fitter_cv.get_upper_bound(),
    fitter_cv.get_control_points(),  # shape (5, 5, 5)
)
# Find t* in R^3 where the Gaussian ≈ 0.5  (should lie on the unit sphere shell)
t_opt_3d1d = finder_3d1d.find(
    target=0.5,
    initial_t=np.array([0.8, 0.0, 0.0]),
)  # ≈ (1/sqrt(2), 0, 0) · sqrt(2*ln2)
## [PositionFinder3d1d_Python]

# ---------------------------------------------------------------------------
# 3D -> 2D inverse query in a vector field with SplinePositionFinder3d2d
# ---------------------------------------------------------------------------

## [PositionFinder3d2d_Python]
finder_3d2d = ubsc.SplinePositionFinder3d2d3(
    fitter_df.get_lower_bound(),
    fitter_df.get_upper_bound(),
    fitter_df.get_control_points(),  # shape (5, 5, 5, 2)
)
# Find t* in R^3 where the vector field ≈ [0, 1]  (should be near origin where z≈0)
t_opt_3d2d = finder_3d2d.find(
    target=np.array([0.0, 1.0]),
    initial_t=np.array([0.1, 0.1, 0.1]),
)
## [PositionFinder3d2d_Python]
