#include "uniform_bspline_ceres.hpp"

#include <random>

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include "test_helper.hpp"

namespace {

template <typename Spline_>
class TestSplineEvaluator {
public:
    explicit TestSplineEvaluator(ubs::UniformBSplineCeresEvaluator<Spline_>& spline) : spline_(spline) {
    }

    template <typename... Ts>
    void operator()(Ts&&... ts) {
        spline_.evaluate(std::forward<Ts>(ts)...);
    }

private:
    ubs::UniformBSplineCeresEvaluator<Spline_>& spline_;
};

} // namespace

TEST(UniformBSplineCeres, Evaluator1D) {
    using Spline = ubs::UniformBSpline<double, 3, double, double, Eigen::VectorXd>;

    Eigen::VectorXd points = Eigen::VectorXd::Random(20);

    Spline spline(points);
    ubs::UniformBSplineCeres<Spline> ceresSpline(spline);

    const int numPoints = 1000;
    for (int i = 0; i < numPoints; ++i) {
        double pos = double(i) / numPoints;

        std::array<double*, 4> pts{};

        const auto data = ceresSpline.getPointData(pos);
        ceresSpline.fillParameterPointers(data, pts.begin(), pts.end());
        ubs::UniformBSplineCeresEvaluator<Spline> splineEvaluator = ceresSpline.getEvaluator(data);

        double val = std::numeric_limits<double>::quiet_NaN();
        splineEvaluator.evaluate(pts[0], pts[1], pts[2], pts[3], &val);

        EXPECT_DOUBLE_EQ(spline.evaluate(pos), val);
    }
}

namespace {

template <int Degree, int InputDims, int OutputDims>
void testEvaluateND(const Eigen::Matrix<double, InputDims, 1>& lowerBound,
                    const Eigen::Matrix<double, InputDims, 1>& upperBound) {
    using Spline = ubs::EigenUniformBSpline<double, Degree, InputDims, OutputDims>;

    std::array<int, Spline::InputDims> controlPointDims{};
    for (int i = 0; i < Spline::InputDims; ++i) {
        controlPointDims[i] = Spline::Order + 2 + i;
    }

    typename Spline::ControlPointsType points(controlPointDims);
    std::for_each(points.data(), points.data() + points.num_elements(), [&](typename Spline::OutputType& val) {
        val = Spline::OutputType::Random();
    });

    Spline spline(lowerBound, upperBound, points);
    ubs::UniformBSplineCeres<Spline> ceresSpline(spline);

    std::mt19937 rng;

    const int numPoints = 10000;
    for (int pointIdx = 0; pointIdx < numPoints; ++pointIdx) {
        typename Spline::InputType pos{};
        for (int i = 0; i < InputDims; ++i) {
            std::uniform_real_distribution<> dist(lowerBound[i], upperBound[i]);
            pos[i] = dist(rng);
        }

        std::array<double*, ubs::UniformBSplineCeres<Spline>::ControlPointsSupport + 1> ptrs{};
        const auto data = ceresSpline.getPointData(pos);
        ceresSpline.fillParameterPointers(data, ptrs.begin(), ptrs.end());
        ubs::UniformBSplineCeresEvaluator<Spline> splineEvaluator = ceresSpline.getEvaluator(data);

        typename Spline::OutputType result = Spline::OutputType::Random();
        ptrs.back() = result.data();

        test_util::apply(TestSplineEvaluator<Spline>(splineEvaluator), test_util::asTuple(ptrs));

        const auto gtVal = spline.evaluate(pos);
        EXPECT_TRUE(gtVal.isApprox(result));
    }
}

} // namespace

TEST(UniformBSplineCeres, EvaluatorND) {
    using Vec1d = Eigen::Matrix<double, 1, 1>;

    testEvaluateND<2, 1, 1>(Vec1d(0.0), Vec1d(1.0));
    testEvaluateND<2, 1, 1>(Vec1d(-5.0), Vec1d(8.0));
    testEvaluateND<3, 1, 1>(Vec1d(-7.0), Vec1d(10.0));
    testEvaluateND<4, 1, 1>(Vec1d(-9.0), Vec1d(12.0));

    testEvaluateND<2, 1, 2>(Vec1d(-1.0), Vec1d(12.0));
    testEvaluateND<3, 1, 2>(Vec1d(-2.0), Vec1d(17.0));
    testEvaluateND<4, 1, 2>(Vec1d(-3.0), Vec1d(19.0));

    testEvaluateND<2, 2, 1>({1.0, 2.0}, {3.0, 4.0});
    testEvaluateND<3, 2, 1>({2.0, 7.0}, {5.0, 40.0});
    testEvaluateND<4, 2, 1>({3.0, 6.0}, {4.0, 39.0});

    testEvaluateND<2, 2, 3>({4.0, 5.0}, {100.0, 45.0});
    testEvaluateND<3, 2, 3>({5.0, 4.0}, {101.0, 48.0});
    testEvaluateND<4, 2, 3>({6.0, 3.0}, {102.0, 50.0});
}

TEST(UniformBSplineCeres, EvaluatorDerivative1D) {
    using Spline = ubs::UniformBSpline<double, 3, double, double, Eigen::VectorXd>;

    Eigen::VectorXd points = Eigen::VectorXd::Random(20);

    Spline spline(points);
    ubs::UniformBSplineCeres<Spline> ceresSpline(spline);

    const int numPoints = 1000;
    for (int i = 0; i < numPoints; ++i) {
        double pos = double(i) / numPoints;

        std::array<double*, 4> pts{};

        const auto data = ceresSpline.getPointData(pos);
        ceresSpline.fillParameterPointers(data, pts.begin(), pts.end());
        ubs::UniformBSplineCeresEvaluator<Spline> splineEvaluator = ceresSpline.getEvaluator(data, {1});

        double val = std::numeric_limits<double>::quiet_NaN();
        splineEvaluator.evaluate(pts[0], pts[1], pts[2], pts[3], &val);

        EXPECT_DOUBLE_EQ(spline.derivative(pos, 1), val);
    }
}

TEST(UniformBSplineCeres, ParameterPointerValidity) {
    const int numRPoints = 20 * 10;
    const int numCPoints = 10 * 10;

    Eigen::MatrixXd controlPoints(Eigen::MatrixXd::Zero(numRPoints / 10, numCPoints / 10));

    using Spline = ubs::UniformBSpline<double, 1, Eigen::Vector2d, double, Eigen::MatrixXd>;
    Spline spline(controlPoints);
    ubs::UniformBSplineCeres<Spline> splineCeres(spline);

    std::vector<double*> params(splineCeres.ControlPointsSupport);

    const auto& internalC = spline.getControlPoints();
    const double* const startPtr = internalC.data();
    const double* const endPtr = internalC.data() + internalC.size();

    for (int r = 0; r < numRPoints; ++r) {
        for (int c = 0; c < numCPoints; ++c) {
            const auto data = splineCeres.getPointData({double(r) / numRPoints, double(c) / numCPoints});
            splineCeres.fillParameterPointers(data, params.begin(), params.end());
            for (double* p : params) {
                EXPECT_GE(p, startPtr);
                EXPECT_LT(p, endPtr);
            }
        }
    }
}
