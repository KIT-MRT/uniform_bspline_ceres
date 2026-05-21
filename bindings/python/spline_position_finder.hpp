#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/multi_array.hpp>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <uniform_bspline/multi_array.hpp>
#include <uniform_bspline/uniform_bspline.hpp>

namespace ubs {

// ---------------------------------------------------------------------------
// SplinePositionFinder1d1d — given a fixed 1D→1D spline, find t* s.t.
//   spline(t*) ≈ target  (minimises (spline(t) - target)^2).
// ---------------------------------------------------------------------------

/**
 * @brief Finds the query position on a fixed 1D→1D spline closest to a target value.
 *
 * The control points are held constant; Ceres optimises the scalar position @f$t@f$.
 *
 * @tparam Degree  Polynomial degree of the B-spline (≥ 1).
 */
template <int Degree>
class SplinePositionFinder1d1d {
public:
    using Spline = ubs::UniformBSpline<double, Degree, double, double, std::vector<double>>;

    /**
     * @brief Construct from spline parameters.
     * @param lowerBound  Lower bound of the spline domain.
     * @param upperBound  Upper bound of the spline domain.
     * @param controlPoints  Fitted control points (e.g. from SplineFitter1d1d).
     */
    SplinePositionFinder1d1d(double lowerBound, double upperBound,
                              std::vector<double> controlPoints)
        : spline_(lowerBound, upperBound, std::move(controlPoints)) {}

    /**
     * @brief Find @f$t^*@f$ that minimises @f$(f(t) - \text{target})^2@f$.
     *
     * @param target  Desired output value.
     * @param initialT  Initial guess for the query position.
     * @param lowerT  Lower bound on @f$t@f$ (defaults to spline domain lower bound).
     * @param upperT  Upper bound on @f$t@f$ (defaults to spline domain upper bound).
     * @param maxIterations  Maximum Ceres solver iterations.
     * @return Optimal query position @f$t^*@f$.
     */
    double find(double target,
                double initialT,
                double lowerT = std::numeric_limits<double>::quiet_NaN(),
                double upperT = std::numeric_limits<double>::quiet_NaN(),
                int maxIterations = 200) const {
        double t = initialT;

        struct Residual {
            Spline spline;
            double target;
            bool operator()(const double* t_ptr, double* r) const {
                *r = spline.evaluate(*t_ptr) - target;
                return true;
            }
        };

        ceres::Problem problem;
        problem.AddResidualBlock(
            new ceres::NumericDiffCostFunction<Residual, ceres::CENTRAL, 1, 1>(
                new Residual{spline_, target}),
            nullptr, &t);

        if (!std::isnan(lowerT)) { problem.SetParameterLowerBound(&t, 0, lowerT); }
        if (!std::isnan(upperT)) { problem.SetParameterUpperBound(&t, 0, upperT); }

        ceres::Solver::Options opts;
        opts.max_num_iterations = maxIterations;
        opts.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        return t;
    }

private:
    Spline spline_;
};

// ---------------------------------------------------------------------------
// SplinePositionFinder1d3d — given a fixed 1D→3D spline, find t* s.t.
//   spline(t*) ≈ target  (closest point on curve to a 3D target).
// ---------------------------------------------------------------------------

/**
 * @brief Finds the scalar parameter @f$t^*@f$ on a 1D→3D curve closest to a 3D target.
 *
 * Minimises @f$\|f(t) - \text{target}\|^2@f$ over the scalar @f$t@f$.
 *
 * @tparam Degree  Polynomial degree of the B-spline (≥ 1).
 */
template <int Degree>
class SplinePositionFinder1d3d {
public:
    using CtrlContainer = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
    using Spline = ubs::UniformBSpline<double, Degree, double, Eigen::Vector3d, CtrlContainer>;

    template <typename T>
    using CtrlContainerT = std::vector<Eigen::Matrix<T, 3, 1>,
                                       Eigen::aligned_allocator<Eigen::Matrix<T, 3, 1>>>;
    template <typename T>
    using SplineT = ubs::UniformBSpline<T, Degree, T, Eigen::Matrix<T, 3, 1>, CtrlContainerT<T>>;

    /**
     * @brief Construct from spline parameters.
     * @param lowerBound  Lower bound of the spline domain.
     * @param upperBound  Upper bound of the spline domain.
     * @param controlPoints  Fitted control points (e.g. from SplineFitter1d3d).
     */
    SplinePositionFinder1d3d(double lowerBound, double upperBound, CtrlContainer controlPoints)
        : spline_(lowerBound, upperBound, std::move(controlPoints)) {}

    /**
     * @brief Find @f$t^*@f$ minimising @f$\|f(t) - \text{target}\|^2@f$.
     *
     * @param target  Target 3D point (closest point query).
     * @param initialT  Initial guess for the query position.
     * @param lowerT  Lower bound on @f$t@f$.
     * @param upperT  Upper bound on @f$t@f$.
     * @param maxIterations  Maximum Ceres solver iterations.
     * @return Optimal parameter @f$t^*@f$.
     */
    double find(const Eigen::Vector3d& target,
                double initialT,
                double lowerT = std::numeric_limits<double>::quiet_NaN(),
                double upperT = std::numeric_limits<double>::quiet_NaN(),
                int maxIterations = 200) const {
        double t = initialT;

        ceres::Problem problem;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Residual, 3, 1>(new Residual{spline_, target}),
            nullptr, &t);

        if (!std::isnan(lowerT)) { problem.SetParameterLowerBound(&t, 0, lowerT); }
        if (!std::isnan(upperT)) { problem.SetParameterUpperBound(&t, 0, upperT); }

        ceres::Solver::Options opts;
        opts.max_num_iterations = maxIterations;
        opts.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        return t;
    }

private:
    struct Residual {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Spline spline;
        Eigen::Vector3d target;

        template <typename T>
        bool operator()(const T* t_ptr, T* r) const {
            Eigen::Matrix<T, 3, 1> v;
            if constexpr (std::is_same_v<T, double>) {
                v = spline.evaluate(t_ptr[0]).template cast<T>() - target.template cast<T>();
            } else {
                v = spline.template cast<SplineT<T>>().evaluate(t_ptr[0]) - target.template cast<T>();
            }
            r[0] = v[0]; r[1] = v[1]; r[2] = v[2];
            return true;
        }
    };

    Spline spline_;
};

// ---------------------------------------------------------------------------
// SplinePositionFinder3d1d — given a fixed 3D→1D spline, find t* ∈ ℝ³ s.t.
//   spline(t*) ≈ target  (inverse query in a 3D scalar field).
// ---------------------------------------------------------------------------

/**
 * @brief Finds the 3D query position @f$\mathbf{t}^*@f$ on a 3D→1D spline closest
 *        to a scalar target value.
 *
 * Minimises @f$(f(\mathbf{t}) - \text{target})^2@f$ over @f$\mathbf{t} \in \mathbb{R}^3@f$.
 *
 * @tparam Degree  Polynomial degree of the B-spline (≥ 1).
 */
template <int Degree>
class SplinePositionFinder3d1d {
public:
    using CtrlArray = ubs::EigenAlignedMultiArray<double, 3>;
    using Spline = ubs::UniformBSpline<double, Degree, Eigen::Vector3d, double, CtrlArray>;

    /**
     * @brief Construct from spline parameters.
     * @param lowerBound  Lower bound in each axis.
     * @param upperBound  Upper bound in each axis.
     * @param controlPoints  Fitted control points (e.g. from SplineFitter3d1d).
     */
    SplinePositionFinder3d1d(Eigen::Vector3d lowerBound, Eigen::Vector3d upperBound,
                              CtrlArray controlPoints)
        : spline_(lowerBound, upperBound, std::move(controlPoints)) {}

    /**
     * @brief Find @f$\mathbf{t}^*@f$ minimising @f$(f(\mathbf{t}) - \text{target})^2@f$.
     *
     * @param target  Desired scalar output value.
     * @param initialT  Initial guess (3D).
     * @param lowerT  Per-axis lower bounds (NaN = unconstrained).
     * @param upperT  Per-axis upper bounds (NaN = unconstrained).
     * @param maxIterations  Maximum Ceres solver iterations.
     * @return Optimal 3D query position @f$\mathbf{t}^*@f$.
     */
    Eigen::Vector3d find(
        double target,
        Eigen::Vector3d initialT,
        Eigen::Vector3d lowerT = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN()),
        Eigen::Vector3d upperT = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN()),
        int maxIterations = 200) const {
        double t[3] = {initialT[0], initialT[1], initialT[2]};

        struct Residual {
            Spline spline;
            double target;
            bool operator()(const double* t_raw, double* r) const {
                *r = spline.evaluate(Eigen::Vector3d(t_raw[0], t_raw[1], t_raw[2])) - target;
                return true;
            }
        };

        ceres::Problem problem;
        problem.AddResidualBlock(
            new ceres::NumericDiffCostFunction<Residual, ceres::CENTRAL, 1, 3>(
                new Residual{spline_, target}),
            nullptr, t);

        for (int i = 0; i < 3; ++i) {
            if (!std::isnan(lowerT[i])) { problem.SetParameterLowerBound(t, i, lowerT[i]); }
            if (!std::isnan(upperT[i])) { problem.SetParameterUpperBound(t, i, upperT[i]); }
        }

        ceres::Solver::Options opts;
        opts.max_num_iterations = maxIterations;
        opts.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        return Eigen::Vector3d(t[0], t[1], t[2]);
    }

private:
    Spline spline_;
};

// ---------------------------------------------------------------------------
// SplinePositionFinder3d2d — given a fixed 3D→2D spline, find t* ∈ ℝ³ s.t.
//   spline(t*) ≈ target  (inverse query in a 3D vector field).
// ---------------------------------------------------------------------------

/**
 * @brief Finds the 3D query position @f$\mathbf{t}^*@f$ on a 3D→2D spline closest
 *        to a 2D target value.
 *
 * Minimises @f$\|f(\mathbf{t}) - \text{target}\|^2@f$ over
 * @f$\mathbf{t} \in \mathbb{R}^3@f$.
 *
 * @tparam Degree  Polynomial degree of the B-spline (≥ 1).
 */
template <int Degree>
class SplinePositionFinder3d2d {
public:
    using CtrlArray = ubs::EigenAlignedMultiArray<Eigen::Vector2d, 3>;
    using Spline =
        ubs::UniformBSpline<double, Degree, Eigen::Vector3d, Eigen::Vector2d, CtrlArray>;

    /**
     * @brief Construct from spline parameters.
     * @param lowerBound  Lower bound in each axis.
     * @param upperBound  Upper bound in each axis.
     * @param controlPoints  Fitted control points (e.g. from SplineFitter3d2d).
     */
    SplinePositionFinder3d2d(Eigen::Vector3d lowerBound, Eigen::Vector3d upperBound,
                              CtrlArray controlPoints)
        : spline_(lowerBound, upperBound, std::move(controlPoints)) {}

    /**
     * @brief Find @f$\mathbf{t}^*@f$ minimising @f$\|f(\mathbf{t}) - \text{target}\|^2@f$.
     *
     * @param target  Desired 2D output value.
     * @param initialT  Initial guess (3D).
     * @param lowerT  Per-axis lower bounds (NaN = unconstrained).
     * @param upperT  Per-axis upper bounds (NaN = unconstrained).
     * @param maxIterations  Maximum Ceres solver iterations.
     * @return Optimal 3D query position @f$\mathbf{t}^*@f$.
     */
    Eigen::Vector3d find(
        const Eigen::Vector2d& target,
        Eigen::Vector3d initialT,
        Eigen::Vector3d lowerT = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN()),
        Eigen::Vector3d upperT = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN()),
        int maxIterations = 200) const {
        double t[3] = {initialT[0], initialT[1], initialT[2]};

        struct Residual {
            Spline spline;
            Eigen::Vector2d target;
            bool operator()(const double* t_raw, double* r) const {
                const auto v =
                    spline.evaluate(Eigen::Vector3d(t_raw[0], t_raw[1], t_raw[2])) - target;
                r[0] = v[0]; r[1] = v[1];
                return true;
            }
        };

        ceres::Problem problem;
        problem.AddResidualBlock(
            new ceres::NumericDiffCostFunction<Residual, ceres::CENTRAL, 2, 3>(
                new Residual{spline_, target}),
            nullptr, t);

        for (int i = 0; i < 3; ++i) {
            if (!std::isnan(lowerT[i])) { problem.SetParameterLowerBound(t, i, lowerT[i]); }
            if (!std::isnan(upperT[i])) { problem.SetParameterUpperBound(t, i, upperT[i]); }
        }

        ceres::Solver::Options opts;
        opts.max_num_iterations = maxIterations;
        opts.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        return Eigen::Vector3d(t[0], t[1], t[2]);
    }

private:
    Spline spline_;
};

} // namespace ubs
