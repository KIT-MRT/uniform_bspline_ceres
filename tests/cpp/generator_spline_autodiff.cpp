#include <random>

#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>
#include <gtest/gtest.h>
#include <uniform_bspline/uniform_bspline.hpp>

#include "uniform_bspline_ceres.hpp"

namespace {

using Spline = ubs::UniformBSpline11d<3>;

template <typename Spline>
typename Spline::ValueType evaluateTestFunction(const Spline& spline) {
    using T = typename Spline::ValueType;

    T res{};
    constexpr int NumTestPoints = 50;
    for (int i = 0; i < NumTestPoints; ++i) {
        T testPoint = T(double(i) / double(NumTestPoints - 1));

        res += spline.evaluate(testPoint);
        res += spline.derivative(testPoint, 1);
    }

    res += spline.template smoothness<3>();

    return res;
}

class CostFunctor1DSpline {
public:
    explicit CostFunctor1DSpline(int numControlPoints) : numControlPoints_{numControlPoints} {
    }

    template <typename T>
    bool operator()(const T* const* controlPointsRaw, T* residual) const {
        using OptSpline = ubs::UniformBSpline11<T, Spline::Degree>;
        typename OptSpline::ControlPointsType controlPoints(numControlPoints_);

        std::copy(*controlPointsRaw, *controlPointsRaw + numControlPoints_, controlPoints.begin());
        OptSpline spline(controlPoints);

        *residual = evaluateTestFunction(spline);
        return true;
    }

private:
    int numControlPoints_;
};

} // namespace

TEST(CeresBSplineOptimization, Opt1d) {
    std::vector<double> controlPoints(15);
    std::mt19937 rng;
    std::uniform_real_distribution<> dist(0, 1.0);
    std::generate(controlPoints.begin(), controlPoints.end(), [&]() { return dist(rng); });

    ceres::Problem::Options problemOptions;
    problemOptions.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problemOptions);
    auto costFunction = std::make_unique<ceres::DynamicAutoDiffCostFunction<CostFunctor1DSpline>>(
        (new CostFunctor1DSpline(controlPoints.size())));
    costFunction->AddParameterBlock(controlPoints.size());
    costFunction->SetNumResiduals(1);
    problem.AddResidualBlock(costFunction.get(), nullptr, controlPoints.data());

    // Check value, derivative and smoothness computation.
    std::vector<double> residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &residuals, nullptr, nullptr);
    ASSERT_EQ((int)residuals.size(), 1);
    {
        Spline spline(controlPoints);
        EXPECT_EQ(evaluateTestFunction(spline), residuals[0]);
    }

    // Check gradient.
    ceres::GradientChecker gradientChecker(costFunction.get(), nullptr, ceres::NumericDiffOptions());

    std::vector<double*> parameterBlocks(1);
    parameterBlocks[0] = controlPoints.data();

    ceres::GradientChecker::ProbeResults results;
    EXPECT_TRUE(gradientChecker.Probe(parameterBlocks.data(), 1e-6, &results)) << results.error_log;
}
