#include "uniform_bspline_ceres.hpp"

#include <ceres/ceres.h>
#include <gtest/gtest.h>

// See README.md for a more detailed explanation of the following example.

//! [Spline]
using Spline = ubs::UniformBSpline<double, 3, double, double, std::vector<double>>;
//! [Spline]

namespace {
//! [Residual]
class ExponentialResidual {
public:
    explicit ExponentialResidual(const ubs::UniformBSplineCeresEvaluator<Spline>& splineEvaluator, double measurement)
            : splineEvaluator_(splineEvaluator), measurement_{measurement} {
    }

    template <typename T>
    bool operator()(const T* c0, const T* c1, const T* c2, const T* c3, T* residual) const {
        splineEvaluator_.evaluate(c0, c1, c2, c3, residual);
        *residual -= T(measurement_);
        return true;
    }

private:
    ubs::UniformBSplineCeresEvaluator<Spline> splineEvaluator_;
    double measurement_;
};
//! [Residual]

} // namespace

TEST(UniformBSplineCeres, ExampleEvaluator1D) { // NOLINT(readability-function-size)
    //! [Init]
    std::vector<double> controlPoints(20, 0.0);
    Spline spline(controlPoints);
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);
    //! [Init]

    const int numMeasurements = 1000;

    //! [Problem_Setup]
    std::vector<double*> parameterPointers(splineCeres.getNumPointParameterPointers());
    ceres::Problem problem;

    for (int i = 0; i < numMeasurements; ++i) {
        const double posX = double(i) / double(numMeasurements);
        const double posY = std::exp(posX * 2.0);

        const auto data = splineCeres.getPointData(posX);
        splineCeres.fillParameterPointers(data, parameterPointers.begin(), parameterPointers.end());
        ubs::UniformBSplineCeresEvaluator<Spline> evaluator = splineCeres.getEvaluator(data);

        auto* costFunctor = new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1, 1, 1>(
            new ExponentialResidual(evaluator, posY));

        problem.AddResidualBlock(costFunctor, nullptr, parameterPointers);
    }
    //! [Problem_Setup]

    //! [Solve]
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    //! [Solve]

    const int numTestPoints = 2000;
    for (int i = 0; i < numTestPoints; ++i) {
        const double posX = double(i) / double(numTestPoints);

        const double estY = spline.evaluate(posX);
        const double gtY = std::exp(posX * 2.0);
        EXPECT_NEAR(gtY, estY, 1e-5);
    }

    //! [Smoothing]
    const double weight = 1e-5;
    splineCeres.addSmoothnessResiduals<1>(problem, weight);
    //! [Smoothing]

    //! [Smoothing_Grid]
    splineCeres.addSmoothnessResidualsGrid<1>(problem, weight);
    //! [Smoothing_Grid]

    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
}
