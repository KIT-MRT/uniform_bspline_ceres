#include "uniform_bspline_ceres.hpp"

#include <random>

#include <gtest/gtest.h>
#include <uniform_bspline/uniform_bspline.hpp>

namespace {
template <typename T>
using EigenAlignedVec = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename Spline, int Derivative>
void testSmoothness1D(const typename Spline::ControlPointsType& controlPoints) {
    Spline spline(-2.0, 6.0, controlPoints);
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);

    ceres::Problem problem;
    splineCeres.template addSmoothnessResiduals<Derivative>(problem);

    double cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    EXPECT_NEAR(spline.template smoothness<Derivative>(), cost, 1e-8);
}

template <int OutputDim, int Degree, int Derivative>
void testSmoothness1DEigen(int numControlPoints) {
    using Vectord = Eigen::Matrix<double, OutputDim, 1>;
    using ControlPoints = EigenAlignedVec<Vectord>;
    using Spline = ubs::UniformBSpline<double, Degree, double, Vectord, ControlPoints>;

    ControlPoints controlPoints(numControlPoints);
    for (auto& p : controlPoints) {
        p.setRandom();
    }

    Spline spline(-3.0, 7.0, controlPoints);
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);

    ceres::Problem problem;
    splineCeres.template addSmoothnessResiduals<Derivative>(problem);

    double cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    EXPECT_NEAR(spline.template smoothness<Derivative>().sum(), cost, 1e-8);
}

template <typename Spline, int Derivative>
void testSmoothness1DGrid(const typename Spline::ControlPointsType& controlPoints) {
    Spline spline(3.0, 8.0, controlPoints);
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);

    ceres::Problem problem;
    splineCeres.template addSmoothnessResidualsGrid<Derivative>(problem);

    double cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    EXPECT_NEAR(spline.template smoothness<Derivative>(), cost, 1e-8);
}
} // anonymous namespace

TEST(UniformBSplineCeres, Smoothness1D1D) {
    std::mt19937 rng;
    std::uniform_real_distribution<> dist(0.0, 0.3);

    std::vector<double> controlPoints(20, 0.0);
    for (auto& c : controlPoints) {
        c = dist(rng);
    }

    testSmoothness1D<ubs::UniformBSpline11<double, 1>, 0>(controlPoints);
    testSmoothness1D<ubs::UniformBSpline11<double, 2>, 0>(controlPoints);

    testSmoothness1D<ubs::UniformBSpline11<double, 1>, 1>(controlPoints);
    testSmoothness1D<ubs::UniformBSpline11<double, 2>, 1>(controlPoints);
    testSmoothness1D<ubs::UniformBSpline11<double, 3>, 1>(controlPoints);

    testSmoothness1D<ubs::UniformBSpline11<double, 2>, 2>(controlPoints);
    testSmoothness1D<ubs::UniformBSpline11<double, 3>, 2>(controlPoints);

    testSmoothness1DGrid<ubs::UniformBSpline11<double, 1>, 0>(controlPoints);

    testSmoothness1DGrid<ubs::UniformBSpline11<double, 1>, 1>(controlPoints);
    testSmoothness1DGrid<ubs::UniformBSpline11<double, 2>, 1>(controlPoints);
    testSmoothness1DGrid<ubs::UniformBSpline11<double, 3>, 1>(controlPoints);

    testSmoothness1DGrid<ubs::UniformBSpline11<double, 2>, 2>(controlPoints);
    testSmoothness1DGrid<ubs::UniformBSpline11<double, 3>, 2>(controlPoints);
}

TEST(UniformBSplineCeres, Smoothness1DND) {
    testSmoothness1DEigen<1, 1, 0>(10);
    testSmoothness1DEigen<1, 1, 1>(10);
    testSmoothness1DEigen<1, 2, 1>(11);
    testSmoothness1DEigen<1, 3, 1>(12);
    testSmoothness1DEigen<1, 4, 1>(13);
    testSmoothness1DEigen<1, 4, 2>(13);

    testSmoothness1DEigen<6, 1, 0>(10);
    testSmoothness1DEigen<6, 1, 1>(10);
    testSmoothness1DEigen<6, 2, 1>(11);
    testSmoothness1DEigen<6, 3, 1>(12);
    testSmoothness1DEigen<6, 4, 1>(13);
    testSmoothness1DEigen<6, 4, 2>(13);

    testSmoothness1DEigen<20, 4, 1>(8);
}

TEST(UniformBSplineCeres, Smoothness2D1D) {
    Eigen::MatrixXd controlPoints = Eigen::MatrixXd::Random(10, 17);

    using Spline = ubs::UniformBSpline<double, 3, Eigen::Vector2d, double, Eigen::MatrixXd>;
    Spline spline({1.0, 2.0}, {7.0, 6.0}, controlPoints);
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);

    {
        ceres::Problem problem;
        splineCeres.addSmoothnessResiduals<0>(problem);

        double cost = 0.0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
        EXPECT_NEAR(spline.smoothness<0>(), cost, 1e-6);
    }

    {
        ceres::Problem problem;
        splineCeres.addSmoothnessResiduals<1>(problem);

        double cost = 0.0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
        EXPECT_NEAR(spline.smoothness<1>(), cost, 1e-6);
    }
}

TEST(UniformBSplineCeres, Smoothness2D3D) {
    using Spline = ubs::UniformBSpline<double, 3, Eigen::Vector2d, Eigen::Vector3d>;

    Spline::ControlPointsType controlPoints(boost::extents[10][15]);
    std::for_each(controlPoints.data(), controlPoints.data() + controlPoints.num_elements(), [](Eigen::Vector3d& v) {
        v.setRandom();
    });

    Spline spline({7.0, 1.0}, {56.0, 7.0}, controlPoints);
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);

    {
        ceres::Problem problem;
        splineCeres.addSmoothnessResiduals<0>(problem);

        double cost = 0.0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
        EXPECT_NEAR(spline.smoothness<0>().sum(), cost, 1e-6);
    }

    {
        ceres::Problem problem;
        splineCeres.addSmoothnessResiduals<1>(problem);

        double cost = 0.0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
        EXPECT_NEAR(spline.smoothness<1>().sum(), cost, 1e-6);
    }
}

TEST(UniformBSplineCeres, SmoothnessGrid1D2D) {
    std::mt19937 rng;
    std::uniform_real_distribution<> dist(0.0, 0.3);

    EigenAlignedVec<Eigen::Vector2d> controlPoints(20);
    for (auto& c : controlPoints) {
        c << dist(rng), dist(rng);
    }

    using Spline = ubs::UniformBSpline<double, 3, double, Eigen::Vector2d, EigenAlignedVec<Eigen::Vector2d>>;
    Spline spline(1.0, 7.0, controlPoints);
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);

    ceres::Problem problem;
    splineCeres.addSmoothnessResidualsGrid<1>(problem);

    double cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    EXPECT_NEAR(spline.smoothness<1>().sum(), cost, 1e-8);
}

TEST(UniformBSplineCeres, SmoothnessGrid2D1D) {
    Eigen::MatrixXd controlPoints = Eigen::MatrixXd::Random(10, 20);

    using Spline1D = ubs::UniformBSpline<double, 3, double, double, Eigen::VectorXd>;

    using Spline = ubs::UniformBSpline<double, 3, Eigen::Vector2d, double, Eigen::MatrixXd>;
    Spline spline({1.0, 2.0}, {7.0, 6.0}, controlPoints);
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);

    ceres::Problem problem;
    splineCeres.addSmoothnessResidualsGrid<1>(problem);

    // Compute ground truth.
    double gtSmoothness = 0.0;
    for (int r = 0; r < int(controlPoints.rows()); ++r) {
        gtSmoothness += Spline1D(1.0, 7.0, controlPoints.row(r)).smoothness<1>();
    }
    for (int c = 0; c < int(controlPoints.cols()); ++c) {
        gtSmoothness += Spline1D(2.0, 6.0, controlPoints.col(c)).smoothness<1>();
    }

    double cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    EXPECT_NEAR(gtSmoothness, cost, 1e-6);
}
