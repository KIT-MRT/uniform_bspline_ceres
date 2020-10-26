#include "uniform_bspline_ceres.hpp"

#include <ceres/ceres.h>
#include <gtest/gtest.h>

// See README.md for a more detailed explanation of the following example.

//! [Spline]
template <typename T>
using Spline = ubs::UniformBSpline<T, 3, T, T, std::vector<T>>;
//! [Spline]

namespace {
//! [Residual]
class SplineMinimumResidual {
public:
    explicit SplineMinimumResidual(const ubs::UniformBSplineCeresGenerator<Spline>& generator, int numControlPoints)
            : generator_(generator), numControlPoints_(numControlPoints) {
    }

    template <typename T>
    bool operator()(const T* const* paramPointers, T* residual) const {
        const T* const* controlPoints = paramPointers;
        const T& pos = *(paramPointers[numControlPoints_]);

        auto spline = generator_.generate(controlPoints);
        residual[0] = spline.evaluate(pos);
        residual[1] = spline.derivative(pos, 1);
        return true;
    }

private:
    ubs::UniformBSplineCeresGenerator<Spline> generator_;
    int numControlPoints_;
};
//! [Residual]

} // namespace

TEST(UniformBSplineCeres, ExampleEvaluator1D) { // NOLINT(readability-function-size)
    //! [Init]
    std::vector<double> controlPoints{6.0, 1.0, 0.0, 1.0, 2.0, 3.0, 6.0};
    Spline<double> spline(controlPoints);
    ubs::UniformBSplineCeres<Spline<double>> splineCeres(spline);

    double t = 0.8;
    //! [Init]

    //! [Range_Data]
    const auto data = splineCeres.getRangeData(0.0, 1.0);
    //! [Range_Data]

    //! [Parameter_Pointers]
    const int numControlPoints = splineCeres.getNumRangeParameterPointers(data);
    std::vector<double*> parameterPointers(numControlPoints + 1);

    splineCeres.fillParameterPointers(data, parameterPointers.begin(), parameterPointers.begin() + numControlPoints);
    parameterPointers.back() = &t;
    //! [Parameter_Pointers]

    //! [Cost_Function]
    ubs::UniformBSplineCeresGenerator<Spline> generator = splineCeres.getGenerator<Spline>(data);

    auto costFunction = std::make_unique<ceres::DynamicAutoDiffCostFunction<SplineMinimumResidual>>(
        new SplineMinimumResidual(generator, numControlPoints));

    for (int i = 0; i < numControlPoints; ++i) {
        costFunction->AddParameterBlock(1);
    }
    costFunction->AddParameterBlock(1);
    costFunction->SetNumResiduals(2);
    //! [Cost_Function]

    //! [Add_Residual]
    ceres::Problem problem;
    problem.AddResidualBlock(costFunction.release(), nullptr, parameterPointers);
    //! [Add_Residual]

    //! [Set_Bounds]
    problem.SetParameterLowerBound(&t, 0, 0.0);
    problem.SetParameterUpperBound(&t, 0, 1.0);
    //! [Set_Bounds]

    //! [Fix_Control_Points]
    spline.getControlPointsContainer().forEach([&](double& c) { problem.SetParameterBlockConstant(&c); });
    //! [Fix_Control_Points]

    //! [Solve]
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.parameter_tolerance = 1e-15;
    options.gradient_tolerance = 1e-15;
    options.function_tolerance = 1e-15;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    //! [Solve]

    EXPECT_NEAR(t, 0.25, 1e-6);
}
