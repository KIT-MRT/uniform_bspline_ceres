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
fitter_cv.fit(x3d, y3d, smoothness_weight=1e-4, smoothness_order=2)

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
fitter_df.fit(x3d, y3d2d, smoothness_weight=1e-4, smoothness_order=2)

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

# ---------------------------------------------------------------------------
# 1D -> 1D joint optimisation: fit control points AND unknown positions t_i
# ---------------------------------------------------------------------------

## [JointOptimizer1d1d_Python]
# Ground-truth: spline is y = sin(2π t), t ∈ [0, 1].
# Simulate 30 measurements with noisy initial t guesses.
n_obs = 30
t_true = np.linspace(0.05, 0.95, n_obs)
y_joint = np.sin(2 * math.pi * t_true)
# Perturb initial positions by ±0.05
rng = np.random.default_rng(0)
t_init = np.clip(t_true + rng.uniform(-0.05, 0.05, n_obs), 0.0, 1.0).tolist()

opt_1d1d = ubsc.SplineJointOptimizer1d1d3(num_control_points=12)
opt_1d1d.fit(
    y=y_joint.tolist(),
    initial_t=t_init,
    lower_bound=0.0,
    upper_bound=1.0,
    smoothness_weight=1e-4,
    smoothness_order=2,
)
ctrl_joint = opt_1d1d.get_control_points()   # list of 12 doubles
t_opt_1d1d = opt_1d1d.get_positions()        # list of 30 optimised t_i
## [JointOptimizer1d1d_Python]

# ---------------------------------------------------------------------------
# 1D -> 3D joint optimisation: fit 3D trajectory AND arc-length parameters
# ---------------------------------------------------------------------------

## [JointOptimizer1d3d_Python]
# Ground-truth 3D helix: x(t) = [cos(2πt), sin(2πt), t], t ∈ [0, 1].
t_true_3d = np.linspace(0.05, 0.95, 20)
y_helix = np.column_stack([
    np.cos(2 * math.pi * t_true_3d),
    np.sin(2 * math.pi * t_true_3d),
    t_true_3d,
])
t_init_3d = np.clip(t_true_3d + rng.uniform(-0.05, 0.05, 20), 0.0, 1.0).tolist()

opt_1d3d = ubsc.SplineJointOptimizer1d3d3(num_control_points=12)
opt_1d3d.fit(
    y=y_helix,
    initial_t=t_init_3d,
    lower_bound=0.0,
    upper_bound=1.0,
    smoothness_weight=1e-4,
    smoothness_order=2,
)
ctrl_joint_3d = opt_1d3d.get_control_points()  # shape (12, 3)
t_opt_1d3d  = opt_1d3d.get_positions()         # list of 20 optimised scalars
## [JointOptimizer1d3d_Python]

# ---------------------------------------------------------------------------
# 3D -> 1D joint optimisation: fit scalar field AND 3D query positions
# ---------------------------------------------------------------------------

## [JointOptimizer3d1d_Python]
# Ground-truth: f(t) = t_x + t_y + t_z, domain [0,1]^3.
rng2 = np.random.default_rng(1)
t_true_grid = rng2.uniform(0.1, 0.9, (15, 3))
y_grid = t_true_grid.sum(axis=1).tolist()
t_init_grid = np.clip(t_true_grid + rng2.uniform(-0.05, 0.05, t_true_grid.shape), 0.0, 1.0)

opt_3d1d = ubsc.SplineJointOptimizer3d1d3(num_ctrl_x=4, num_ctrl_y=4, num_ctrl_z=4)
opt_3d1d.fit(
    y=y_grid,
    initial_t=t_init_grid,
    lower_bound=np.array([0.0, 0.0, 0.0]),
    upper_bound=np.array([1.0, 1.0, 1.0]),
    smoothness_weight=1e-4,
    smoothness_order=2,
)
ctrl_joint_vol  = opt_3d1d.get_control_points()   # shape (4, 4, 4)
t_opt_3d1d      = opt_3d1d.get_positions()        # shape (15, 3)
## [JointOptimizer3d1d_Python]

# ---------------------------------------------------------------------------
# 3D -> 2D joint optimisation: fit vector field AND 3D query positions
# ---------------------------------------------------------------------------

## [JointOptimizer3d2d_Python]
# Ground-truth: g(t) = [t_x, t_y], domain [0,1]^3.
t_true_vf = rng2.uniform(0.1, 0.9, (15, 3))
y_vf = t_true_vf[:, :2]
t_init_vf = np.clip(t_true_vf + rng2.uniform(-0.05, 0.05, t_true_vf.shape), 0.0, 1.0)

opt_3d2d = ubsc.SplineJointOptimizer3d2d3(num_ctrl_x=4, num_ctrl_y=4, num_ctrl_z=4)
opt_3d2d.fit(
    y=y_vf,
    initial_t=t_init_vf,
    lower_bound=np.array([0.0, 0.0, 0.0]),
    upper_bound=np.array([1.0, 1.0, 1.0]),
    smoothness_weight=1e-4,
    smoothness_order=2,
)
ctrl_joint_vf  = opt_3d2d.get_control_points()    # shape (4, 4, 4, 2)
t_opt_3d2d     = opt_3d2d.get_positions()         # shape (15, 3)
## [JointOptimizer3d2d_Python]

