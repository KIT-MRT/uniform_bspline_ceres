cmake_minimum_required(VERSION 3.16)

find_package(pybind11 REQUIRED)
find_package(Ceres REQUIRED)

pybind11_add_module(uniform_bspline_ceres_pyext
    uniform_bspline_ceres_py.cpp
)

set_target_properties(uniform_bspline_ceres_pyext PROPERTIES OUTPUT_NAME uniform_bspline_ceres)


target_link_libraries(uniform_bspline_ceres_pyext PRIVATE
    uniform_bspline_ceres::uniform_bspline_ceres
    Ceres::ceres
)

install(TARGETS uniform_bspline_ceres_pyext
    LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
)
