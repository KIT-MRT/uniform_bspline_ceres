cmake_minimum_required(VERSION 3.16)
project(uniform_bspline_ceres VERSION 1.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(BUILD_PYTHON_BINDINGS  "Build Python bindings via pybind11" OFF)
option(BUILD_TESTS            "Build C++ unit tests"               OFF)
option(BUILD_DOCUMENTATION    "Build Doxygen HTML documentation"   OFF)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

if(NOT TARGET uniform_bspline::uniform_bspline)
    find_package(uniform_bspline 1.0.0 QUIET)
    if(NOT uniform_bspline_FOUND)
        message(STATUS "uniform_bspline not found — fetching via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            uniform_bspline
            GIT_REPOSITORY https://github.com/KIT-MRT/uniform_bspline.git
            GIT_TAG        remove_catkin_dep
        )
        # Disable uniform_bspline's own tests when used as a sub-dependency
        set(BUILD_TESTS_SAVED ${BUILD_TESTS})
        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(uniform_bspline)
        # Restore for this project's own BUILD_TESTS option
        set(BUILD_TESTS ${BUILD_TESTS_SAVED} CACHE BOOL "" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# Library target (header-only interface)
# ---------------------------------------------------------------------------
add_library(uniform_bspline_ceres INTERFACE)
add_library(uniform_bspline_ceres::uniform_bspline_ceres ALIAS uniform_bspline_ceres)

target_include_directories(uniform_bspline_ceres
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/uniform_bspline_ceres>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/uniform_bspline_ceres>
)

target_link_libraries(uniform_bspline_ceres
    INTERFACE
        uniform_bspline::uniform_bspline
        Eigen3::Eigen
        Ceres::ceres
)

target_compile_features(uniform_bspline_ceres INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# Python bindings
# ---------------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings/python)
endif()

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------
if(BUILD_DOCUMENTATION)
    find_package(Doxygen REQUIRED)
    add_custom_target(docs
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/build/docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating Doxygen HTML documentation"
        VERBATIM
    )
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS uniform_bspline_ceres
    EXPORT uniform_bspline_ceresTargets
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT uniform_bspline_ceresTargets
    FILE uniform_bspline_ceresTargets.cmake
    NAMESPACE uniform_bspline_ceres::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/uniform_bspline_ceres
)

configure_package_config_file(
    cmake/uniform_bspline_ceresConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/uniform_bspline_ceresConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/uniform_bspline_ceres
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/uniform_bspline_ceresConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/uniform_bspline_ceresConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/uniform_bspline_ceresConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/uniform_bspline_ceres
)
