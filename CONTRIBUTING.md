# Contributing

Thank you for your interest in contributing to `uniform_bspline_ceres`!

## Reporting Issues

Please use the [GitHub issue tracker](https://github.com/KIT-MRT/uniform_bspline_ceres/issues) to report bugs or request features.
Include a minimal reproducible example and your environment details (OS, compiler, CMake version, Ceres version).

## Development Setup

```bash
git clone https://github.com/KIT-MRT/uniform_bspline_ceres.git
cd uniform_bspline_ceres
./install_dependencies.sh --tests   # install C++ and test deps
```

Or open in VS Code and select **Reopen in Container** for a zero-setup environment.

## Building and Testing

**C++:**

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure
```

**Python:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install .[test]         # installs the package + pytest
pytest tests/python/ -v
```

## Code Style

- C++17, header-only. Follow the style of the surrounding code.
- Python bindings in `bindings/python/uniform_bspline_ceres_py.cpp` — keep one `bind_*` call per type combination.
- No trailing whitespace, Unix line endings.

## Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Make sure all CI checks pass (C++ tests, Doxygen build).
3. Keep commits focused; use descriptive commit messages.
4. Open a pull request against `main` and fill in the description.
