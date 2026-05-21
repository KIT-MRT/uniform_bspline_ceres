#include "uniform_bspline_ceres.hpp"

#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/problem.h>
#include <gtest/gtest.h>

template <typename T>
using Spline = ubs::UniformBSpline11<T, 5>;
using SplineGenerator = ubs::UniformBSplineCeresGenerator<Spline>;

namespace {
class SplineCostFunction {
public:
    SplineCostFunction(int numControlPoints, const SplineGenerator& generator)
            : numControlPoints_{numControlPoints}, generator_{generator} {
    }

    template <typename T>
    bool operator()(const T* const* parameterPointers, T* residual) {
        auto spline = generator_.generate(parameterPointers);

        T val = parameterPointers[numControlPoints_][0];
        residual[0] = spline.evaluate(val);
        residual[1] = spline.derivative(val, 1);
        return true;
    }

private:
    int numControlPoints_;
    SplineGenerator generator_;
};
} // namespace

TEST(UniformBSplineCeres, SplineGenerator) {
    double lowerBound = 1.0;
    double upperBound = 7.0;

    std::vector<double> controlPoints{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    Spline<double> spline(lowerBound, upperBound, controlPoints);
    ubs::UniformBSplineCeres<Spline<double>> splineCeres(spline);

    const auto data = splineCeres.getRangeData(1.0, 6.9999999);

    std::vector<double*> parameterPointers(splineCeres.getNumRangeParameterPointers(data));
    splineCeres.fillParameterPointers(data, parameterPointers.begin(), parameterPointers.end());

    auto gen = splineCeres.getGenerator<Spline>(data);
    auto partialBSpline = gen.generate(parameterPointers.data());
    EXPECT_EQ(partialBSpline.getLowerBound(), 1.0);
    EXPECT_EQ(partialBSpline.getUpperBound(), 7.0);
    EXPECT_NEAR(partialBSpline.evaluate(2.101), spline.evaluate(2.101), 1e-15);

    auto costFunction = std::make_unique<ceres::DynamicAutoDiffCostFunction<SplineCostFunction>>(
        new SplineCostFunction(parameterPointers.size(), gen));

    const double evalPoint = 3.143;
    double modEvalPoint = evalPoint;
    parameterPointers.push_back(&modEvalPoint);

    for (int i = 0; i < int(parameterPointers.size()); ++i) {
        costFunction->AddParameterBlock(1);
    }
    costFunction->SetNumResiduals(2);

    ceres::Problem problem;
    problem.AddResidualBlock(costFunction.release(), nullptr, parameterPointers);

    std::vector<double> residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &residuals, nullptr, nullptr);

    ASSERT_EQ(residuals.size(), 2U);
    EXPECT_NEAR(residuals[0], spline.evaluate(evalPoint), 1e-15);
    EXPECT_NEAR(residuals[1], spline.derivative(evalPoint, 1), 1e-15);
}

TEST(UniformBSplineCeres, GetRangeData) {
    const double lowerBound = -7.45;
    const double upperBound = 23.456;

    std::vector<double> controlPoints{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    Spline<double> spline(lowerBound, upperBound, controlPoints);
    ubs::UniformBSplineCeres<Spline<double>> splineCeres(spline);

    const int numTestPoints = 100;
    for (int i = 0; i < numTestPoints; ++i) {
        const double p = (lowerBound + 0.1) + (upperBound - lowerBound - 0.2) * double(i) / (numTestPoints - 1);

        const auto data = splineCeres.getRangeData(p, 0.1, true);
        std::vector<double*> parameterPointers(splineCeres.getNumRangeParameterPointers(data));
        splineCeres.fillParameterPointers(data, parameterPointers.begin(), parameterPointers.end());

        auto gen = splineCeres.getGenerator<Spline>(data);
        auto partialBSpline = gen.generate(parameterPointers.data());

        EXPECT_LE((int)parameterPointers.size(), Spline<double>::Order + 1);
        EXPECT_NEAR(partialBSpline.evaluate(p - 0.1), spline.evaluate(p - 0.1), 1e-14);
        EXPECT_NEAR(partialBSpline.evaluate(p), spline.evaluate(p), 1e-14);
        EXPECT_NEAR(partialBSpline.evaluate(p + 0.1), spline.evaluate(p + 0.1), 1e-14);
    }
}
