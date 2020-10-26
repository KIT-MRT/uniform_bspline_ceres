#include "uniform_bspline_ceres.hpp"

#include <benchmark/benchmark.h>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace {
template <typename Spline_>
class Residual {
public:
    Residual(const ubs::UniformBSplineCeresEvaluator<Spline_>& evaluator, double measurement)
            : evaluator_(evaluator), measurement_{measurement} {
    }
    template <typename T>
    inline bool operator()(const T* c0, const T* c1, const T* c2, const T* c3, T* residual) const {
        evaluator_.evaluate(c0, c1, c2, c3, residual);
        *residual -= static_cast<T>(measurement_);
        return true;
    }

private:
    ubs::UniformBSplineCeresEvaluator<Spline_> evaluator_;
    double measurement_;
};


template <typename Spline>
void addToProblem(const Eigen::Ref<const Eigen::Matrix3Xd>& measurements,
                  ubs::UniformBSplineCeres<Spline>& spline,
                  ceres::Problem& problem) {
    std::vector<double*> parameters(spline.ControlPointsSupport);
    for (int c = 0; c < int(measurements.cols()); c++) {
        const Eigen::Vector3d& measurement{measurements.col(c)};

        const auto data = spline.getPointData(measurement.head<2>());
        spline.fillParameterPointers(data, std::begin(parameters), std::end(parameters));

        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Residual<Spline>, 1, 1, 1, 1, 1>(
                                     new Residual<Spline>(spline.getEvaluator(data), measurement.z())),
                                 new ceres::TrivialLoss,
                                 parameters);
    }
}

struct BenchmarkParameters {
    int64_t numControlPoints{10};
    int64_t numMeasurements{10000};
};

using namespace benchmark;
using Spline = ubs::UniformBSpline<double, 1, Eigen::Vector2d, double, Eigen::MatrixXd>;

void constructProblemOnNumMeasurements(State& state) {

    BenchmarkParameters p;
    p.numMeasurements = state.range();

    const Eigen::Matrix3Xd measurements{(1 + Eigen::Array3Xd::Random(3, state.range())) / 2};
    Spline spline{Eigen::MatrixXd::Zero(p.numControlPoints, p.numControlPoints)};
    ubs::UniformBSplineCeres<Spline> splineCeres{spline};

    for ([[maybe_unused]] auto _ : state) {
        ceres::Problem problem;
        addToProblem<Spline>(measurements, splineCeres, problem);
    }
    state.SetComplexityN(state.range());
}

void solveProblemOnNumMeasurements(State& state) {
    BenchmarkParameters p;
    p.numMeasurements = state.range();

    const Eigen::Matrix3Xd measurements{(1 + Eigen::Array3Xd::Random(3, p.numMeasurements)) / 2};
    Spline spline{Eigen::MatrixXd::Zero(p.numControlPoints, p.numControlPoints)};
    ubs::UniformBSplineCeres<Spline> splineCeres{spline};
    ceres::Problem problem;
    addToProblem<Spline>(measurements, splineCeres, problem);

    for ([[maybe_unused]] auto _ : state) {
        ceres::Solver::Summary summary;
        Solve(ceres::Solver::Options(), &problem, &summary);
    }
    state.SetComplexityN(state.range());
}

void solveProblemOnNumControlPoints(State& state) {
    BenchmarkParameters p;
    p.numControlPoints = state.range();

    Spline spline{Eigen::MatrixXd::Zero(p.numControlPoints, p.numControlPoints)};
    const Eigen::Matrix3Xd measurements{(1 + Eigen::Array3Xd::Random(3, p.numMeasurements)) / 2};
    ubs::UniformBSplineCeres<Spline> splineCeres{spline};
    ceres::Problem problem;
    addToProblem<Spline>(measurements, splineCeres, problem);

    for (auto _ : state) { // NOLINT
        ceres::Solver::Summary summary;
        Solve(ceres::Solver::Options(), &problem, &summary);
    }
    state.SetComplexityN(state.range());
}

} // anonymous namespace

BENCHMARK(constructProblemOnNumMeasurements)->Unit(kMillisecond)->RangeMultiplier(10)->Range(1e2, 1e5)->Complexity();
BENCHMARK(solveProblemOnNumMeasurements)
    ->Unit(kMillisecond)
    ->RangeMultiplier(10)
    ->Range(1e2, 1e5)
    ->Complexity()
    ->Iterations(1);
BENCHMARK(solveProblemOnNumControlPoints)
    ->Unit(kMillisecond)
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Complexity()
    ->Iterations(1);


int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return RUN_ALL_TESTS();
}