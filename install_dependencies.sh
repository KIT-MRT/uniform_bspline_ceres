#!/usr/bin/env bash
# Install dependencies for uniform_bspline_ceres
# Usage:
#   ./install_dependencies.sh                     # core C++ dependencies
#   ./install_dependencies.sh --tests             # core + C++ test dependencies (libgtest-dev, libbenchmark-dev)
#   ./install_dependencies.sh --python            # core + Python bindings dependencies
#   ./install_dependencies.sh --tests --python    # all of the above

set -euo pipefail

INSTALL_PYTHON=false
INSTALL_TESTS=false
for arg in "$@"; do
    case "$arg" in
        --python) INSTALL_PYTHON=true ;;
        --tests)  INSTALL_TESTS=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "Installing core dependencies..."
sudo apt-get update -qq
sudo apt-get install -y cmake libeigen3-dev libboost-dev libceres-dev

# uniform_bspline is header-only and not available via apt — clone and install it
echo "Installing uniform_bspline..."
tmp=$(mktemp -d)
git clone --depth 1 --branch remove_catkin_dep https://github.com/KIT-MRT/uniform_bspline.git "$tmp"
cmake -S "$tmp" -B "$tmp/build" -DCMAKE_BUILD_TYPE=Release
sudo cmake --install "$tmp/build"
rm -rf "$tmp"

if [ "$INSTALL_TESTS" = true ]; then
    echo "Installing C++ test dependencies..."
    sudo apt-get install -y libgtest-dev libbenchmark-dev
fi

if [ "$INSTALL_PYTHON" = true ]; then
    echo "Installing Python bindings dependencies..."
    sudo apt-get install -y pybind11-dev python3-dev python3-pip
fi

echo "Done."
