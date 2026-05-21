find_package(GTest 1.10 REQUIRED)
find_package(benchmark REQUIRED)
find_package(Ceres REQUIRED)

set(TEST_SOURCES
    evaluator.cpp
    evaluator_example.cpp
    evaluator_optimization.cpp
    generator.cpp
    generator_example.cpp
    generator_spline_autodiff.cpp
    smoothness.cpp
    smoothness_prior.cpp
)

add_executable(uniform_bspline_ceres_tests ${TEST_SOURCES})

target_link_libraries(uniform_bspline_ceres_tests
    PRIVATE
        uniform_bspline_ceres::uniform_bspline_ceres
        GTest::GTest
        GTest::Main
        Ceres::ceres
)

target_include_directories(uniform_bspline_ceres_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(uniform_bspline_ceres_tests)

add_executable(uniform_bspline_ceres_benchmarks benchmark.cpp)

target_link_libraries(uniform_bspline_ceres_benchmarks
    PRIVATE
        uniform_bspline_ceres::uniform_bspline_ceres
        GTest::GTest
        benchmark::benchmark
        Ceres::ceres
)

target_include_directories(uniform_bspline_ceres_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}
)
