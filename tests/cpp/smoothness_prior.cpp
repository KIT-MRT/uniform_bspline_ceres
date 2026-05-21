#include "uniform_bspline_ceres.hpp"

#include <random>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include "test_helper.hpp"
#include "uniform_bspline_ceres.hpp"

namespace {
template <typename Spline_>
class FitCostFunctor {
public:
    static_assert(Spline_::OutputDims == 1, "Invalid number of output dims specified.");

    explicit FitCostFunctor(const ubs::UniformBSplineCeresEvaluator<Spline_>& splineEvaluator, double val)
            : splineEvaluator_(splineEvaluator), val_(val) {
    }

    template <typename T>
    bool operator()(T const* const* paramPointers, T* residual) const {
        splineEvaluator_.evaluate(paramPointers, residual);
        *residual -= T(val_);
        return true;
    }

private:
    ubs::UniformBSplineCeresEvaluator<Spline_> splineEvaluator_;
    double val_;
};

} // namespace

TEST(UniformBSplineCeres, PlaneFitPrior) { // NOLINT(readability-function-size)
    using Spline = ubs::UniformBSpline<double, 3, Eigen::Vector2d, double, Eigen::MatrixXd>;

    Eigen::MatrixXd controlPoints = Eigen::MatrixXd::Random(20, 20);
    Spline spline(controlPoints);

    // Generate measurements.
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);

    std::vector<double*> parameterPointers(splineCeres.ControlPointsSupport);
    ceres::Problem problem;
    const int numMeasurements = 10;

    const Eigen::Vector3d supportPoint(1.0, 2.0, 3.0);
    const Eigen::Vector3d dir1 = Eigen::Vector3d(2.0, 0.12, 0.32).normalized();
    const Eigen::Vector3d dir2 = Eigen::Vector3d(0.1, 1.5, 1.74).normalized();

    Eigen::Matrix2d a = Eigen::Matrix2d::Zero();
    a << dir1.head<2>(), dir2.head<2>();

    const Eigen::Matrix2d aInv = a.inverse();

    // Create ceres problem.
    for (int i = 0; i < numMeasurements; ++i) {
        for (int j = 0; j < numMeasurements; ++j) {
            Eigen::Vector2d planePos{};
            planePos << i, j;
            planePos = planePos / (numMeasurements - 1) * 0.2 - supportPoint.head<2>();

            const Eigen::Vector2d rs = aInv * planePos;
            const Eigen::Vector3d pos = supportPoint + dir1 * rs[0] + dir2 * rs[1];

            const auto data = splineCeres.getPointData(pos.head<2>());
            splineCeres.fillParameterPointers(data, parameterPointers.begin(), parameterPointers.end());
            ubs::UniformBSplineCeresEvaluator<Spline> evaluator = splineCeres.getEvaluator(data);

            auto* costFunctor = new ceres::DynamicAutoDiffCostFunction<FitCostFunctor<Spline>>(
                new FitCostFunctor<Spline>(evaluator, pos[2]));
            for (int k = 0; k < splineCeres.ControlPointsSupport; ++k) {
                costFunctor->AddParameterBlock(1);
            }
            costFunctor->SetNumResiduals(1);

            problem.AddResidualBlock(costFunctor, nullptr, parameterPointers);
        }
    }

    splineCeres.addSmoothnessResiduals<2>(problem);

    // Solve problem.
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.initial_trust_region_radius = 1e16;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Check result.
    test_util::linspace(std::array<int, 2>{200, 200}, [&](const Eigen::Vector2d& pos) {
        const Eigen::Vector3d splinePos(pos[0], pos[1], spline.evaluate(pos));

        Eigen::Vector2d planePos = pos - supportPoint.head<2>();
        Eigen::Vector2d rs = aInv * planePos;
        const Eigen::Vector3d gtPos = supportPoint + dir1 * rs[0] + dir2 * rs[1];

        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(gtPos[i], splinePos[i], 1e-8) << pos;
        }
    });
}
