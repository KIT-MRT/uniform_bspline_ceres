#include "uniform_bspline_ceres.hpp"

#include <random>

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include "test_helper.hpp"

namespace {

template <typename Spline_>
class StaticCostFunctor {
public:
    explicit StaticCostFunctor(const ubs::UniformBSplineCeresEvaluator<Spline_>& splineEvaluator, double gtVal)
            : splineEvaluator_(splineEvaluator), gtVal_(gtVal) {
    }

    template <typename T, typename... Ts>
    bool operator()(const T* p1, Ts*... ps) const {
        splineEvaluator_.evaluate(p1, ps...);

        // The last element of the control points is the place to store the output.
        T* residual = std::get<sizeof...(Ts)>(std::make_tuple(p1, ps...));

        for (int i = 0; i < Spline_::OutputDims; ++i) {
            residual[i] -= T(gtVal_);
        }
        return true;
    }

private:
    ubs::UniformBSplineCeresEvaluator<Spline_> splineEvaluator_;
    double gtVal_;
};

template <typename Spline_>
class DynamicCostFunctor {
public:
    explicit DynamicCostFunctor(const ubs::UniformBSplineCeresEvaluator<Spline_>& splineEvaluator, double gtVal)
            : splineEvaluator_(splineEvaluator), gtVal_(gtVal) {
    }

    template <typename T>
    bool operator()(T const* const* paramPointers, T* residual) const {
        splineEvaluator_.evaluate(paramPointers, residual);

        for (int i = 0; i < Spline_::OutputDims; ++i) {
            residual[i] -= T(gtVal_);
        }
        return true;
    }

private:
    ubs::UniformBSplineCeresEvaluator<Spline_> splineEvaluator_;
    double gtVal_;
};

} // namespace

TEST(UniformBSplineCeres, Optimization1D) {
    using TestSpline = ubs::UniformBSpline<double, 3, double, double, Eigen::VectorXd>;
    const double gtVal = 10.0;

    Eigen::VectorXd points = Eigen::VectorXd::Constant(20, gtVal * 2.0);
    points += Eigen::VectorXd::Random(points.size());

    TestSpline spline(points);

    ubs::UniformBSplineCeres<TestSpline> ceresSpline(spline);

    ceres::Problem problem;

    const int numPoints = 100;
    for (int i = 0; i < numPoints; ++i) {
        double pos = double(i) / numPoints;

        std::vector<double*> paramPointers(ceresSpline.getNumPointParameterPointers());
        const auto data = ceresSpline.getPointData(pos);
        auto splineEvaluator = ceresSpline.getEvaluator(data);
        ceresSpline.fillParameterPointers(data, paramPointers.begin(), paramPointers.end());

        auto* costFunctor = new ceres::AutoDiffCostFunction<StaticCostFunctor<TestSpline>, 1, 1, 1, 1, 1>(
            new StaticCostFunctor<TestSpline>(splineEvaluator, gtVal));

        problem.AddResidualBlock(costFunctor, nullptr, paramPointers);
    }

    ceres::Solver::Options options;
    options.initial_trust_region_radius = 1e14; // This is a linear problem.
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ASSERT_TRUE(summary.IsSolutionUsable());

    const auto& controlPoints = spline.getControlPoints();
    for (int i = 0; i < int(controlPoints.size()); ++i) {
        EXPECT_NEAR(controlPoints[i], gtVal, 1e-10);
    }
}

namespace {
template <int Degree, int InputDims, int OutputDims>
void testUniformBSplineCeresOptNd() { // NOLINT(readability-function-size)
    // Spline used to optimize.
    using Spline = ubs::EigenUniformBSpline<double, Degree, InputDims, OutputDims>;

    // The ground truth value (the optimization fits the B-spline to this value).
    std::mt19937 rd;
    std::uniform_real_distribution<> dist(-100.0, 100.0);
    const double gtVal = dist(rd);

    // Initialize the number of control points.
    std::array<int, Spline::InputDims> numMeasurements{};
    std::array<int, Spline::InputDims> controlPointDims{};
    for (int i = 0; i < Spline::InputDims; ++i) {
        controlPointDims[i] = Spline::Order + 2 + i;
        numMeasurements[i] = controlPointDims[i] * 5;
    }

    // Initialize the control points randomly.
    typename Spline::ControlPointsType points(controlPointDims);
    std::for_each(points.data(), points.data() + points.num_elements(), [&](typename Spline::OutputType& val) {
        val = Spline::OutputType::Random();
    });

    // Create a spline.
    Spline spline(points);
    ubs::UniformBSplineCeres<Spline> ceresSpline(spline);

    // Build the ceres problem.
    ceres::Problem problem;

    test_util::linspace(numMeasurements, [&](const Eigen::Matrix<double, Spline::InputDims, 1>& pos) {
        std::vector<double*> paramPointers(ceresSpline.ControlPointsSupport);
        const auto data = ceresSpline.getPointData(pos);
        auto splineEvaluator = ceresSpline.getEvaluator(data);
        ceresSpline.fillParameterPointers(data, paramPointers.begin(), paramPointers.end());

        auto* costFunctor = new ceres::DynamicAutoDiffCostFunction<DynamicCostFunctor<Spline>>(
            new DynamicCostFunctor<Spline>(splineEvaluator, gtVal));

        for (int n = 0; n < ceresSpline.ControlPointsSupport; ++n) {
            costFunctor->AddParameterBlock(OutputDims);
        }

        costFunctor->SetNumResiduals(OutputDims);

        problem.AddResidualBlock(costFunctor, nullptr, paramPointers);
    });

    // Solve the problem.
    ceres::Solver::Options options;
    options.initial_trust_region_radius = 1e14; // This is a linear problem.
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ASSERT_TRUE(summary.IsSolutionUsable());

    // Check result (each control point must now be equal to the ground truth value).
    const auto& controlPoints = spline.getControlPoints();
    for (int i = 0; i < int(controlPoints.num_elements()); ++i) {
        for (int j = 0; j < Spline::OutputDims; ++j) {
            EXPECT_NEAR(controlPoints.data()[i][j], gtVal, 1e-5) << "index (" << i << ", " << j << ")";
        }
    }
} // namespace

TEST(UniformBSplineCeres, Optimization11) {
    testUniformBSplineCeresOptNd<2, 1, 1>();
    testUniformBSplineCeresOptNd<3, 1, 1>();
    testUniformBSplineCeresOptNd<4, 1, 1>();
}

TEST(UniformBSplineCeres, Optimization21) {
    testUniformBSplineCeresOptNd<2, 2, 1>();
    testUniformBSplineCeresOptNd<3, 2, 1>();
    testUniformBSplineCeresOptNd<4, 2, 1>();
}

TEST(UniformBSplineCeres, Optimization12) {
    testUniformBSplineCeresOptNd<2, 1, 2>();
    testUniformBSplineCeresOptNd<3, 1, 2>();
    testUniformBSplineCeresOptNd<4, 1, 2>();
}

TEST(UniformBSplineCeres, Optimization22) {
    testUniformBSplineCeresOptNd<2, 2, 2>();
    testUniformBSplineCeresOptNd<3, 2, 2>();
    testUniformBSplineCeresOptNd<4, 2, 2>();
}

TEST(UniformBSplineCeres, Optimization32) {
    testUniformBSplineCeresOptNd<2, 3, 2>();
}

TEST(UniformBSplineCeres, Optimization23) {
    testUniformBSplineCeresOptNd<2, 2, 3>();
    testUniformBSplineCeresOptNd<3, 2, 3>();
}

TEST(UniformBSplineCeres, OptimizationDeriv1D) { // NOLINT(readability-function-size)
    using TestSpline = ubs::UniformBSpline<double, 3, double, double, Eigen::VectorXd>;

    const double gtV = 2.0;
    const double gtDeriv = 10.0;

    Eigen::VectorXd points = Eigen::VectorXd::Constant(20, gtDeriv * 2.0);
    points += Eigen::VectorXd::Random(points.size());

    TestSpline spline(points);

    ubs::UniformBSplineCeres<TestSpline> ceresSpline(spline);

    ceres::Problem problem;

    std::vector<double*> paramPointers(ceresSpline.getNumPointParameterPointers());
    {
        const auto data = ceresSpline.getPointData(0.0);
        auto splineEvaluator = ceresSpline.getEvaluator(data);
        ceresSpline.fillParameterPointers(data, paramPointers.begin(), paramPointers.end());

        auto* costFunctor = new ceres::AutoDiffCostFunction<StaticCostFunctor<TestSpline>, 1, 1, 1, 1, 1>(
            new StaticCostFunctor<TestSpline>(splineEvaluator, gtV));

        problem.AddResidualBlock(costFunctor, nullptr, paramPointers);
    }

    const int numPoints = 100;
    for (int i = 0; i < numPoints; ++i) {
        double pos = double(i) / numPoints;

        const auto data = ceresSpline.getPointData(pos);
        auto splineEvaluator = ceresSpline.getEvaluator(data, {1});
        ceresSpline.fillParameterPointers(data, paramPointers.begin(), paramPointers.end());

        auto* costFunctor = new ceres::AutoDiffCostFunction<StaticCostFunctor<TestSpline>, 1, 1, 1, 1, 1>(
            new StaticCostFunctor<TestSpline>(splineEvaluator, gtDeriv));

        problem.AddResidualBlock(costFunctor, nullptr, paramPointers);
    }

    ceres::Solver::Options options;
    options.initial_trust_region_radius = 1e14; // This is a linear problem.
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ASSERT_TRUE(summary.IsSolutionUsable());

    const int numTestPoints = 1000;
    for (int i = 0; i < numTestPoints; ++i) {
        const double pos = double(i) / double(numTestPoints);
        const double gtVal = gtDeriv * pos + gtV;
        const double val = spline.evaluate(pos);

        EXPECT_NEAR(gtVal, val, 1e-7);
    }
}

} // namespace
