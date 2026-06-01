# uniform_bspline_ceres

[![CI](https://github.com/KIT-MRT/uniform_bspline_ceres/actions/workflows/ci.yml/badge.svg)](https://github.com/KIT-MRT/uniform_bspline_ceres/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/KIT-MRT/uniform_bspline_ceres)](https://github.com/KIT-MRT/uniform_bspline_ceres/releases)
[![PyPI](https://img.shields.io/pypi/v/uniform_bspline_ceres)](https://pypi.org/project/uniform_bspline_ceres/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://kit-mrt.github.io/uniform_bspline_ceres)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: BSL-1.0](https://img.shields.io/badge/License-BSL_1.0-lightblue.svg)](https://www.boost.org/LICENSE_1_0.txt)

A header-only C++ library for fitting and optimizing uniform B-splines using the [Ceres Solver](http://ceres-solver.org/). It builds on top of [uniform_bspline](https://github.com/KIT-MRT/uniform_bspline) and extends it with Ceres-compatible cost functors, enabling spline parameters — control points, evaluation positions, or both — to be optimized jointly with other variables in a non-linear least-squares problem.

**Features:**

- **Evaluator** — Evaluate a spline at a *fixed* parameter position inside a Ceres cost function (sparse, autodiff-friendly).
- **Generator** — Generate a fully auto-differentiable spline object when the parameter position is itself an optimization variable.
- **Smoothness priors** — Exact 1-D integral-based smoothness residuals and grid-based approximations for N-D splines.
- **Header-only** — just add to your include path, no compilation required.
- **Python bindings** — easily install via pybind11.

## Quick Start

### C++

#### Fitting a 1-D spline with the Evaluator API

```cpp
#include <uniform_bspline_ceres/uniform_bspline_ceres.hpp>

using Spline = ubs::UniformBSpline<double, 3, double, double, std::vector<double>>;

class ExponentialResidual {
public:
    ExponentialResidual(const ubs::UniformBSplineCeresEvaluator<Spline>& eval, double meas)
        : eval_(eval), meas_(meas) {}

    template <typename T>
    bool operator()(const T* c0, const T* c1, const T* c2, const T* c3, T* residual) const {
        eval_.evaluate(c0, c1, c2, c3, residual);
        *residual -= T(meas_);
        return true;
    }
private:
    ubs::UniformBSplineCeresEvaluator<Spline> eval_;
    double meas_;
};

// Build and solve
std::vector<double> controlPoints(20, 0.0);
Spline spline(controlPoints);
ubs::UniformBSplineCeres<Spline> splineCeres(spline);

ceres::Problem problem;
std::vector<double*> params(splineCeres.getNumPointParameterPointers());
for (int i = 0; i < numMeas; ++i) {
    double x = double(i) / numMeas;
    const auto data = splineCeres.getPointData(x);
    splineCeres.fillParameterPointers(data, params.begin(), params.end());
    auto* cost = new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1, 1, 1>(
        new ExponentialResidual(splineCeres.getEvaluator(data), std::exp(2.0 * x)));
    problem.AddResidualBlock(cost, nullptr, params);
}
ceres::Solver::Options opts;
ceres::Solver::Summary summary;
ceres::Solve(opts, &problem, &summary);
```

#### Optimizing the evaluation position with the Generator API

Use the Generator API when the control points are fixed and only the query position `t`
needs to be optimised — for example to invert a spline or find the closest point on a curve:

```cpp
template <typename T>
using Spline = ubs::UniformBSpline<T, 3, T, T, std::vector<T>>;

// Suppose spline is already fitted; we want t* such that spline(t*) ≈ target.
double target = 3.5;
double t = 0.5;  // initial guess

struct PositionResidual {
    ubs::UniformBSplineCeresGenerator<Spline> gen;
    double target;
    bool operator()(double const* const* params, double* residual) const {
        // params[0..order-1] are control points, params[order] is t
        auto s = gen(params, params + gen.numControlPointBlocks());
        residual[0] = s.evaluate(*params[gen.numControlPointBlocks()]) - target;
        return true;
    }
};

ubs::UniformBSplineCeres<Spline<double>> splineCeres(fittedSpline);
const auto data = splineCeres.getRangeData(0.0, 1.0);
auto gen = splineCeres.getGenerator<Spline>(data);

ceres::Problem problem;
std::vector<double*> params;
splineCeres.fillParameterPointers(data, std::back_inserter(params));
params.push_back(&t);

auto* cost = new ceres::DynamicAutoDiffCostFunction<PositionResidual>(
    new PositionResidual{gen, target});
for (int i = 0; i < static_cast<int>(params.size()); ++i)
    cost->AddParameterBlock(1);
cost->SetNumResiduals(1);
problem.AddResidualBlock(cost, nullptr, params);

// Fix control points, only optimise t
for (std::size_t i = 0; i + 1 < params.size(); ++i)
    problem.SetParameterBlockConstant(params[i]);
problem.SetParameterLowerBound(&t, 0, 0.0);
problem.SetParameterUpperBound(&t, 0, 1.0);

ceres::Solver::Options opts;
ceres::Solver::Summary summary;
ceres::Solve(opts, &problem, &summary);
// t now holds t*
```

#### Smoothness / Regularization

Add an integral-based smoothness prior on the first derivative:

```cpp
const double weight = 1e-5;
splineCeres.addSmoothnessResiduals<1>(problem, weight);
```

For N-D splines (grid of control points), use the grid-based approximation:

```cpp
splineCeres.addSmoothnessResidualsGrid<1>(problem, weight);
```

### Python

```python
import uniform_bspline_ceres as ubsc

# Fit y = exp(2x) with a cubic (degree-3) spline
fitter = ubsc.SplineFitter1d1d3(num_control_points=20)
x = [i / 199 for i in range(200)]
y = [2.718 ** (2 * xi) for xi in x]
fitter.fit(x, y)

ctrl = fitter.get_control_points()   # list of fitted control points
lb   = fitter.get_lower_bound()      # lower bound of the fitted range
ub   = fitter.get_upper_bound()      # upper bound of the fitted range

# Add smoothness regularization
fitter.fit(x, y, smoothness_weight=1e-4)

# Fit on a non-unit interval
x2 = [-3 + 6 * i / 99 for i in range(100)]
y2 = [xi ** 2 for xi in x2]
fitter.fit(x2, y2, lower_bound=-3.0, upper_bound=3.0)
```

#### Finding the query position on a fitted spline

After fitting, use `SplinePositionFinder` to invert the spline — i.e. find the parameter
`t*` that maps to a given target value:

```python
import uniform_bspline_ceres as ubsc

# --- 1D → 1D: find t* such that spline(t*) ≈ target ---
fitter = ubsc.SplineFitter1d1d3(num_control_points=20)
x = [i / 199 for i in range(200)]
y = [2.718 ** (2 * xi) for xi in x]   # y = exp(2x)
fitter.fit(x, y)

finder = ubsc.SplinePositionFinder1d1d3(
    lower_bound=fitter.get_lower_bound(),
    upper_bound=fitter.get_upper_bound(),
    control_points=fitter.get_control_points(),
)
# Find t* such that spline(t*) ≈ exp(2 * 0.3)
t_star = finder.find(target=2.718 ** 0.6, initial_t=0.5)
print(t_star)   # ≈ 0.3

# --- 1D → 3D: find t* on a curve closest to a 3D point ---
finder_3d = ubsc.SplinePositionFinder1d3d3(
    lower_bound=0.0,
    upper_bound=1.0,
    control_points=ctrl_matrix,   # (num_control_points, 3) numpy array
)
t_star = finder_3d.find(target=[0.5, 0.5, 0.5], initial_t=0.5)
```

The naming convention is `SplinePositionFinderNdMd{Degree}`, mirroring the fitter classes.
Available variants: `1d1d`, `1d3d`, `3d1d`, `3d2d`, each with degrees 1–5.

For the full usage guide see the **[documentation](https://kit-mrt.github.io/uniform_bspline_ceres)**.

## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| CMake | >= 3.16 | required |
| Eigen3 | >= 3.3 | required |
| Ceres Solver | >= 2.0 | required |
| uniform_bspline | >= 1.0.0 | required |
| GTest | >= 1.10 | optional — C++ tests only |
| Google Benchmark | >= 1.5 | optional — C++ tests only |
| pybind11 | >= 2.11 | optional — Python bindings only |
| Python | >= 3.8 | optional — Python bindings only |
| numpy | >= 1.21 | optional — Python bindings only |

### Option A — shell script (Ubuntu/Debian)

```bash
./install_dependencies.sh                  # core only (includes uniform_bspline)
./install_dependencies.sh --tests          # core + C++ test dependencies
./install_dependencies.sh --python         # core + Python bindings
./install_dependencies.sh --tests --python # all of the above
```

### Option B — vcpkg (cross-platform: Linux / macOS / Windows)

```bash
vcpkg install                                                # core only (includes uniform_bspline)
vcpkg install --x-feature=tests                              # core + C++ test dependencies
vcpkg install --x-feature=python-bindings                    # core + Python bindings
vcpkg install --x-feature=tests --x-feature=python-bindings  # all of the above
```

Then configure CMake with:
```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

### Option C — Dev Container (zero-setup)

Open in VS Code → **Reopen in Container**.
All dependencies (core, GTest, and Python bindings) are installed automatically via `.devcontainer/devcontainer.json`.

## Installation

### C++ -- CMake

```bash
git clone https://github.com/KIT-MRT/uniform_bspline_ceres.git  # clone the repository
cd uniform_bspline_ceres
cmake -S . -B build                                              # configure
cmake --build build --parallel $(nproc)                          # compile
sudo cmake --install build                                       # install system-wide
```

To also build the Doxygen HTML documentation:
```bash
cmake -S . -B build -DBUILD_DOCUMENTATION=ON  # configure with docs enabled
cmake --build build --target docs             # generate documentation
```

CMake options:

| Option | Default | Description |
|---|---|---|
| `BUILD_TESTS` | `OFF` | Build C++ GTest tests |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Build Python (pybind11) bindings |
| `BUILD_DOCUMENTATION` | `OFF` | Build Doxygen documentation |

Then in your own project:
```cmake
find_package(uniform_bspline_ceres REQUIRED)
target_link_libraries(my_target PRIVATE uniform_bspline_ceres::uniform_bspline_ceres)
```

### C++ -- FetchContent (no install needed)

```cmake
include(FetchContent)
FetchContent_Declare(
    uniform_bspline_ceres
    GIT_REPOSITORY https://github.com/KIT-MRT/uniform_bspline_ceres.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(uniform_bspline_ceres)
target_link_libraries(my_target PRIVATE uniform_bspline_ceres::uniform_bspline_ceres)
```

### Python -- pip

#### 1. Install from PyPI

```bash
pip install uniform_bspline_ceres
```

#### 2. Install from GitHub

```bash
pip install git+https://github.com/KIT-MRT/uniform_bspline_ceres.git
```

#### 3. Install from a local clone

```bash
git clone https://github.com/KIT-MRT/uniform_bspline_ceres.git
cd uniform_bspline_ceres
pip install .
```

## Testing

### C++

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure
```

### Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install .[test]
pytest tests/python/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSL-1.0 — see [LICENSE](LICENSE).

## Citation

If you use this library in academic work, please cite:

```bibtex
@article{Beck2021_1000131090,
    author       = {Beck, Johannes},
    year         = {2021},
    title        = {Camera Calibration with Non-Central Local Camera Models},
    doi          = {10.5445/IR/1000131090},
    publisher    = {{Karlsruher Institut für Technologie (KIT)}},
    school       = {Karlsruher Institut für Technologie (KIT)}
}
```
