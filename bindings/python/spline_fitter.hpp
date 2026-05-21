#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/multi_array.hpp>
#include <ceres/ceres.h>
#include <uniform_bspline/multi_array.hpp>
#include <uniform_bspline/uniform_bspline.hpp>

#include "uniform_bspline_ceres.hpp"
#include "uniform_bspline_ceres_evaluator.hpp"

namespace ubs {

// ---------------------------------------------------------------------------
// Internal residual functors
// ---------------------------------------------------------------------------
namespace internal {

/// Residual for scalar output (1D output): residual = spline(x) - measurement
template <typename Spline>
class ScalarResidual {
public:
    ScalarResidual(const UniformBSplineCeresEvaluator<Spline>& evaluator, double measurement)
        : evaluator_(evaluator), measurement_(measurement) {}

    template <typename T>
    bool operator()(const T* const* params, T* residual) const {
        evaluator_.evaluate(params, residual);
        *residual -= static_cast<T>(measurement_);
        return true;
    }

private:
    UniformBSplineCeresEvaluator<Spline> evaluator_;
    double measurement_;
};

/// Residual for fixed-size Eigen vector output (OutDim-D output).
template <typename Spline, int OutDim>
class VectorNdResidual {
public:
    using MeasType = Eigen::Matrix<double, OutDim, 1>;

    VectorNdResidual(const UniformBSplineCeresEvaluator<Spline>& evaluator, const MeasType& measurement)
        : evaluator_(evaluator), measurement_(measurement) {}

    template <typename T>
    bool operator()(const T* const* params, T* residual) const {
        evaluator_.evaluate(params, residual);
        for (int i = 0; i < OutDim; ++i) {
            residual[i] -= static_cast<T>(measurement_[i]);
        }
        return true;
    }

private:
    UniformBSplineCeresEvaluator<Spline> evaluator_;
    MeasType measurement_;
};

} // namespace internal

// ---------------------------------------------------------------------------
// SplineFitter1d1d — fit a 1D→1D uniform B-spline to (x, y) data using Ceres.
// ---------------------------------------------------------------------------

/**
 * @brief Fits a 1D→1D uniform B-spline to scattered (x, y) data.
 *
 * Internally sets up and solves a Ceres NLS problem. The spline bounds are
 * determined automatically from the x data range unless explicitly set.
 *
 * @tparam Degree  Polynomial degree of the B-spline (≥ 1).
 */
template <int Degree>
class SplineFitter1d1d {
public:
    using Spline = ubs::UniformBSpline<double, Degree, double, double, std::vector<double>>;

    /**
     * @brief Construct with a fixed number of control points.
     * @param numControlPoints  Number of control points (>= Order = Degree + 1).
     */
    explicit SplineFitter1d1d(int numControlPoints)
        : numControlPoints_(numControlPoints) {
        if (numControlPoints < Degree + 1) {
            throw std::invalid_argument("numControlPoints must be >= Degree + 1");
        }
    }

    /**
     * @brief Fit the spline to the provided data.
     *
     * @param x  Input positions (must be sorted ascending, within [lower, upper]).
     * @param y  Corresponding output values. Must have the same length as x.
     * @param lowerBound  Lower bound of the spline domain. Defaults to min(x).
     * @param upperBound  Upper bound of the spline domain. Defaults to max(x).
     * @param smoothnessWeight  Regularization weight on the derivative (0 = disabled).
     * @param smoothnessOrder  Derivative order used for regularization (1, 2, or 3).
     * @param maxIterations  Maximum Ceres solver iterations.
     */
    void fit(const std::vector<double>& x,
             const std::vector<double>& y,
             double lowerBound = std::numeric_limits<double>::quiet_NaN(),
             double upperBound = std::numeric_limits<double>::quiet_NaN(),
             double smoothnessWeight = 0.0,
             int smoothnessOrder = 1,
             int maxIterations = 200) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("x and y must have the same length");
        }
        if (x.empty()) {
            throw std::invalid_argument("x must not be empty");
        }

        if (std::isnan(lowerBound)) { lowerBound = *std::min_element(x.begin(), x.end()); }
        if (std::isnan(upperBound)) { upperBound = *std::max_element(x.begin(), x.end()); }

        controlPoints_.assign(numControlPoints_, 0.0);
        Spline spline(lowerBound, upperBound, controlPoints_);
        UniformBSplineCeres<Spline> splineCeres(spline);

        std::vector<double*> paramPtrs(splineCeres.getNumPointParameterPointers());
        ceres::Problem problem;

        for (std::size_t i = 0; i < x.size(); ++i) {
            const auto data = splineCeres.getPointData(x[i]);
            splineCeres.fillParameterPointers(data, paramPtrs.begin(), paramPtrs.end());
            auto evaluator = splineCeres.getEvaluator(data);

            auto* costFn = new ceres::DynamicAutoDiffCostFunction<internal::ScalarResidual<Spline>>(
                new internal::ScalarResidual<Spline>(evaluator, y[i]));
            costFn->AddParameterBlock(1);  // repeated Order times
            for (int j = 1; j < Spline::Order; ++j) {
                costFn->AddParameterBlock(1);
            }
            costFn->SetNumResiduals(1);
            problem.AddResidualBlock(costFn, nullptr, paramPtrs);
        }

        if (smoothnessWeight > 0.0) {
            const int order = std::max(1, std::min(smoothnessOrder, Degree));
            if constexpr (Degree >= 3) {
                if (order == 3) { splineCeres.template addSmoothnessResiduals<3>(problem, smoothnessWeight); }
                else if (order == 2) { splineCeres.template addSmoothnessResiduals<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResiduals<1>(problem, smoothnessWeight); }
            } else if constexpr (Degree >= 2) {
                if (order >= 2) { splineCeres.template addSmoothnessResiduals<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResiduals<1>(problem, smoothnessWeight); }
            } else {
                splineCeres.template addSmoothnessResiduals<1>(problem, smoothnessWeight);
            }
        }

        ceres::Solver::Options options;
        options.max_num_iterations = maxIterations;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        controlPoints_ = spline.getControlPoints();
        lowerBound_ = lowerBound;
        upperBound_ = upperBound;
    }

    /** @brief Return the fitted control points (valid after fit()). */
    const std::vector<double>& getControlPoints() const { return controlPoints_; }

    /** @brief Return the lower bound of the fitted spline. */
    double getLowerBound() const { return lowerBound_; }

    /** @brief Return the upper bound of the fitted spline. */
    double getUpperBound() const { return upperBound_; }

    /** @brief Build and return the fitted spline object. */
    Spline getSpline() const {
        return Spline(lowerBound_, upperBound_, controlPoints_);
    }

private:
    int numControlPoints_;
    double lowerBound_{0.0};
    double upperBound_{1.0};
    std::vector<double> controlPoints_;
};

// ---------------------------------------------------------------------------
// SplineFitter1d3d — fit a 1D→3D uniform B-spline (e.g. 3D trajectory).
// ---------------------------------------------------------------------------

/**
 * @brief Fits a 1D→3D uniform B-spline to scattered (x, y) data.
 *
 * Useful for fitting 3D trajectories parameterized by a scalar (time, arc-length).
 *
 * @tparam Degree  Polynomial degree of the B-spline (≥ 1).
 */
template <int Degree>
class SplineFitter1d3d {
public:
    using CtrlContainer = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
    using Spline = ubs::UniformBSpline<double, Degree, double, Eigen::Vector3d, CtrlContainer>;

    explicit SplineFitter1d3d(int numControlPoints) : numControlPoints_(numControlPoints) {
        if (numControlPoints < Degree + 1) {
            throw std::invalid_argument("numControlPoints must be >= Degree + 1");
        }
    }

    /**
     * @brief Fit the spline to the provided data.
     *
     * @param x  Scalar input positions. Length must match rows of y.
     * @param y  Output values, shape (N, 3). Each row is one 3D measurement.
     * @param lowerBound  Lower bound. Defaults to min(x).
     * @param upperBound  Upper bound. Defaults to max(x).
     * @param smoothnessWeight  Regularization weight (0 = disabled).
     * @param smoothnessOrder  Derivative order used for regularization (1, 2, or 3).
     * @param maxIterations  Maximum Ceres solver iterations.
     */
    void fit(const std::vector<double>& x,
             const Eigen::MatrixXd& y,
             double lowerBound = std::numeric_limits<double>::quiet_NaN(),
             double upperBound = std::numeric_limits<double>::quiet_NaN(),
             double smoothnessWeight = 0.0,
             int smoothnessOrder = 1,
             int maxIterations = 200) {
        if (y.cols() != 3) {
            throw std::invalid_argument("y must have 3 columns");
        }
        if (static_cast<Eigen::Index>(x.size()) != y.rows()) {
            throw std::invalid_argument("x and y must have the same number of rows");
        }
        if (x.empty()) {
            throw std::invalid_argument("x must not be empty");
        }

        if (std::isnan(lowerBound)) { lowerBound = *std::min_element(x.begin(), x.end()); }
        if (std::isnan(upperBound)) { upperBound = *std::max_element(x.begin(), x.end()); }

        CtrlContainer ctrlPts(numControlPoints_, Eigen::Vector3d::Zero());
        Spline spline(lowerBound, upperBound, ctrlPts);
        UniformBSplineCeres<Spline> splineCeres(spline);

        const int numBlocks = splineCeres.getNumPointParameterPointers();
        std::vector<double*> paramPtrs(numBlocks);
        ceres::Problem problem;

        for (std::size_t i = 0; i < x.size(); ++i) {
            const auto data = splineCeres.getPointData(x[i]);
            splineCeres.fillParameterPointers(data, paramPtrs.begin(), paramPtrs.end());
            auto evaluator = splineCeres.getEvaluator(data);
            const Eigen::Vector3d meas = y.row(static_cast<Eigen::Index>(i)).transpose();

            auto* costFn =
                new ceres::DynamicAutoDiffCostFunction<internal::VectorNdResidual<Spline, 3>>(
                    new internal::VectorNdResidual<Spline, 3>(evaluator, meas));
            for (int j = 0; j < numBlocks; ++j) {
                costFn->AddParameterBlock(3); // each control point is Eigen::Vector3d
            }
            costFn->SetNumResiduals(3);
            problem.AddResidualBlock(costFn, nullptr, paramPtrs);
        }

        if (smoothnessWeight > 0.0) {
            const int order = std::max(1, std::min(smoothnessOrder, Degree));
            if constexpr (Degree >= 3) {
                if (order == 3) { splineCeres.template addSmoothnessResiduals<3>(problem, smoothnessWeight); }
                else if (order == 2) { splineCeres.template addSmoothnessResiduals<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResiduals<1>(problem, smoothnessWeight); }
            } else if constexpr (Degree >= 2) {
                if (order >= 2) { splineCeres.template addSmoothnessResiduals<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResiduals<1>(problem, smoothnessWeight); }
            } else {
                splineCeres.template addSmoothnessResiduals<1>(problem, smoothnessWeight);
            }
        }

        ceres::Solver::Options options;
        options.max_num_iterations = maxIterations;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        const auto& fitted = spline.getControlPoints();
        controlPoints_.resize(numControlPoints_);
        for (int i = 0; i < numControlPoints_; ++i) {
            controlPoints_[i] = fitted[i];
        }
        lowerBound_ = lowerBound;
        upperBound_ = upperBound;
    }

    /** @brief Return control points as (numControlPoints, 3) matrix. */
    Eigen::MatrixXd getControlPoints() const {
        Eigen::MatrixXd result(numControlPoints_, 3);
        for (int i = 0; i < numControlPoints_; ++i) {
            result.row(i) = controlPoints_[i].transpose();
        }
        return result;
    }

    double getLowerBound() const { return lowerBound_; }
    double getUpperBound() const { return upperBound_; }

private:
    int numControlPoints_;
    double lowerBound_{0.0};
    double upperBound_{1.0};
    CtrlContainer controlPoints_;
};

// ---------------------------------------------------------------------------
// SplineFitter2d1d — fit a 2D→1D uniform B-spline (e.g. height map, cost field).
// ---------------------------------------------------------------------------

/**
 * @brief Fits a 2D→1D uniform B-spline to scattered (x, y) data.
 *
 * Useful for fitting height maps or 2D scalar cost fields.
 *
 * @tparam Degree  Polynomial degree of the B-spline (≥ 1).
 */
template <int Degree>
class SplineFitter2d1d {
public:
    using Spline = ubs::UniformBSpline<double, Degree, Eigen::Vector2d, double, Eigen::MatrixXd>;

    /**
     * @param numCtrlX  Number of control points along the first axis.
     * @param numCtrlY  Number of control points along the second axis.
     */
    SplineFitter2d1d(int numCtrlX, int numCtrlY) : numCtrlX_(numCtrlX), numCtrlY_(numCtrlY) {
        if (numCtrlX < Degree + 1 || numCtrlY < Degree + 1) {
            throw std::invalid_argument("numCtrlX and numCtrlY must be >= Degree + 1");
        }
    }

    /**
     * @brief Fit the spline to the provided data.
     *
     * @param x  2D input positions, shape (N, 2).
     * @param y  Scalar output values, length N.
     * @param lowerBound  Lower bound in each axis. Defaults to min(x) per axis.
     * @param upperBound  Upper bound in each axis. Defaults to max(x) per axis.
     * @param smoothnessWeight  Regularization weight (0 = disabled).
     * @param smoothnessOrder  Derivative order used for regularization (1, 2, or 3).
     * @param maxIterations  Maximum Ceres solver iterations.
     */
    void fit(const Eigen::MatrixXd& x,
             const std::vector<double>& y,
             Eigen::Vector2d lowerBound =
                 Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN()),
             Eigen::Vector2d upperBound =
                 Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN()),
             double smoothnessWeight = 0.0,
             int smoothnessOrder = 1,
             int maxIterations = 200) {
        if (x.cols() != 2) {
            throw std::invalid_argument("x must have 2 columns");
        }
        if (static_cast<Eigen::Index>(y.size()) != x.rows()) {
            throw std::invalid_argument("x and y must have the same number of rows");
        }
        if (y.empty()) {
            throw std::invalid_argument("x must not be empty");
        }

        for (int d = 0; d < 2; ++d) {
            if (std::isnan(lowerBound[d])) { lowerBound[d] = x.col(d).minCoeff(); }
            if (std::isnan(upperBound[d])) { upperBound[d] = x.col(d).maxCoeff(); }
        }

        Eigen::MatrixXd ctrlPts = Eigen::MatrixXd::Zero(numCtrlX_, numCtrlY_);
        Spline spline(lowerBound, upperBound, ctrlPts);
        UniformBSplineCeres<Spline> splineCeres(spline);

        const int numBlocks = splineCeres.getNumPointParameterPointers();
        std::vector<double*> paramPtrs(numBlocks);
        ceres::Problem problem;

        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            const Eigen::Vector2d xi = x.row(i).transpose();
            const auto data = splineCeres.getPointData(xi);
            splineCeres.fillParameterPointers(data, paramPtrs.begin(), paramPtrs.end());
            auto evaluator = splineCeres.getEvaluator(data);

            auto* costFn = new ceres::DynamicAutoDiffCostFunction<internal::ScalarResidual<Spline>>(
                new internal::ScalarResidual<Spline>(evaluator, y[static_cast<std::size_t>(i)]));
            for (int j = 0; j < numBlocks; ++j) {
                costFn->AddParameterBlock(1);
            }
            costFn->SetNumResiduals(1);
            problem.AddResidualBlock(costFn, nullptr, paramPtrs);
        }

        if (smoothnessWeight > 0.0) {
            const int order = std::max(1, std::min(smoothnessOrder, Degree));
            if constexpr (Degree >= 3) {
                if (order == 3) { splineCeres.template addSmoothnessResidualsGrid<3>(problem, smoothnessWeight); }
                else if (order == 2) { splineCeres.template addSmoothnessResidualsGrid<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResidualsGrid<1>(problem, smoothnessWeight); }
            } else if constexpr (Degree >= 2) {
                if (order >= 2) { splineCeres.template addSmoothnessResidualsGrid<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResidualsGrid<1>(problem, smoothnessWeight); }
            } else {
                splineCeres.template addSmoothnessResidualsGrid<1>(problem, smoothnessWeight);
            }
        }

        ceres::Solver::Options options;
        options.max_num_iterations = maxIterations;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        controlPoints_ = spline.getControlPoints();
        lowerBound_ = lowerBound;
        upperBound_ = upperBound;
    }

    /** @brief Return control points as (numCtrlX, numCtrlY) matrix. */
    const Eigen::MatrixXd& getControlPoints() const { return controlPoints_; }

    Eigen::Vector2d getLowerBound() const { return lowerBound_; }
    Eigen::Vector2d getUpperBound() const { return upperBound_; }

private:
    int numCtrlX_;
    int numCtrlY_;
    Eigen::Vector2d lowerBound_{Eigen::Vector2d::Zero()};
    Eigen::Vector2d upperBound_{Eigen::Vector2d::Ones()};
    Eigen::MatrixXd controlPoints_;
};

// ---------------------------------------------------------------------------
// SplineFitter3d1d — fit a 3D→1D uniform B-spline (e.g. 3D cost volume).
// ---------------------------------------------------------------------------

/**
 * @brief Fits a 3D→1D uniform B-spline to scattered (x, y) data.
 *
 * Useful for fitting 3D scalar cost volumes or occupancy fields.
 *
 * @tparam Degree  Polynomial degree of the B-spline (≥ 1).
 */
template <int Degree>
class SplineFitter3d1d {
public:
    using CtrlArray = ubs::EigenAlignedMultiArray<double, 3>;
    using Spline = ubs::UniformBSpline<double, Degree, Eigen::Vector3d, double, CtrlArray>;

    /**
     * @param numCtrlX  Number of control points along the first axis.
     * @param numCtrlY  Number of control points along the second axis.
     * @param numCtrlZ  Number of control points along the third axis.
     */
    SplineFitter3d1d(int numCtrlX, int numCtrlY, int numCtrlZ)
        : numCtrlX_(numCtrlX), numCtrlY_(numCtrlY), numCtrlZ_(numCtrlZ),
          controlPoints_(boost::extents[1][1][1]) {
        if (numCtrlX < Degree + 1 || numCtrlY < Degree + 1 || numCtrlZ < Degree + 1) {
            throw std::invalid_argument("all num_ctrl_* must be >= Degree + 1");
        }
    }

    /**
     * @brief Fit the spline to the provided data.
     *
     * @param x  3D input positions, shape (N, 3).
     * @param y  Scalar output values, length N.
     * @param lowerBound  Lower bound in each axis. Defaults to min(x) per axis.
     * @param upperBound  Upper bound in each axis. Defaults to max(x) per axis.
     * @param smoothnessWeight  Regularization weight (0 = disabled).
     * @param smoothnessOrder  Derivative order used for regularization (1, 2, or 3).
     * @param maxIterations  Maximum Ceres solver iterations.
     */
    void fit(const Eigen::MatrixXd& x,
             const std::vector<double>& y,
             Eigen::Vector3d lowerBound =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             Eigen::Vector3d upperBound =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             double smoothnessWeight = 0.0,
             int smoothnessOrder = 1,
             int maxIterations = 200) {
        if (x.cols() != 3) {
            throw std::invalid_argument("x must have 3 columns");
        }
        if (static_cast<Eigen::Index>(y.size()) != x.rows()) {
            throw std::invalid_argument("x and y must have the same number of rows");
        }
        if (y.empty()) {
            throw std::invalid_argument("x must not be empty");
        }

        for (int d = 0; d < 3; ++d) {
            if (std::isnan(lowerBound[d])) { lowerBound[d] = x.col(d).minCoeff(); }
            if (std::isnan(upperBound[d])) { upperBound[d] = x.col(d).maxCoeff(); }
        }

        CtrlArray ctrlPts(boost::extents[numCtrlX_][numCtrlY_][numCtrlZ_]);
        std::fill(ctrlPts.data(), ctrlPts.data() + ctrlPts.num_elements(), 0.0);
        Spline spline(lowerBound, upperBound, ctrlPts);
        UniformBSplineCeres<Spline> splineCeres(spline);

        const int numBlocks = splineCeres.getNumPointParameterPointers();
        std::vector<double*> paramPtrs(numBlocks);
        ceres::Problem problem;

        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            const Eigen::Vector3d xi = x.row(i).transpose();
            const auto data = splineCeres.getPointData(xi);
            splineCeres.fillParameterPointers(data, paramPtrs.begin(), paramPtrs.end());
            auto evaluator = splineCeres.getEvaluator(data);

            auto* costFn = new ceres::DynamicAutoDiffCostFunction<internal::ScalarResidual<Spline>>(
                new internal::ScalarResidual<Spline>(evaluator, y[static_cast<std::size_t>(i)]));
            for (int j = 0; j < numBlocks; ++j) {
                costFn->AddParameterBlock(1);
            }
            costFn->SetNumResiduals(1);
            problem.AddResidualBlock(costFn, nullptr, paramPtrs);
        }

        if (smoothnessWeight > 0.0) {
            const int order = std::max(1, std::min(smoothnessOrder, Degree));
            if constexpr (Degree >= 3) {
                if (order == 3) { splineCeres.template addSmoothnessResidualsGrid<3>(problem, smoothnessWeight); }
                else if (order == 2) { splineCeres.template addSmoothnessResidualsGrid<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResidualsGrid<1>(problem, smoothnessWeight); }
            } else if constexpr (Degree >= 2) {
                if (order >= 2) { splineCeres.template addSmoothnessResidualsGrid<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResidualsGrid<1>(problem, smoothnessWeight); }
            } else {
                splineCeres.template addSmoothnessResidualsGrid<1>(problem, smoothnessWeight);
            }
        }

        ceres::Solver::Options options;
        options.max_num_iterations = maxIterations;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        controlPoints_ = spline.getControlPoints();
        lowerBound_ = lowerBound;
        upperBound_ = upperBound;
    }

    /** @brief Return control points as (numCtrlX, numCtrlY, numCtrlZ) 3D array. */
    const CtrlArray& getControlPointsArray() const { return controlPoints_; }

    int getNumCtrlX() const { return numCtrlX_; }
    int getNumCtrlY() const { return numCtrlY_; }
    int getNumCtrlZ() const { return numCtrlZ_; }

    Eigen::Vector3d getLowerBound() const { return lowerBound_; }
    Eigen::Vector3d getUpperBound() const { return upperBound_; }

private:
    int numCtrlX_;
    int numCtrlY_;
    int numCtrlZ_;
    Eigen::Vector3d lowerBound_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d upperBound_{Eigen::Vector3d::Ones()};
    CtrlArray controlPoints_;
};

// ---------------------------------------------------------------------------
// SplineFitter3d2d — fit a 3D→2D uniform B-spline (e.g. 3D deformation field).
// ---------------------------------------------------------------------------

/**
 * @brief Fits a 3D→2D uniform B-spline to scattered (x, y) data.
 *
 * Useful for fitting 2D vector fields defined over a 3D domain (e.g. a planar
 * displacement field parameterized in 3D space).
 *
 * @tparam Degree  Polynomial degree of the B-spline (≥ 1).
 */
template <int Degree>
class SplineFitter3d2d {
public:
    using CtrlArray = ubs::EigenAlignedMultiArray<Eigen::Vector2d, 3>;
    using Spline = ubs::UniformBSpline<double, Degree, Eigen::Vector3d, Eigen::Vector2d, CtrlArray>;

    /**
     * @param numCtrlX  Number of control points along the first axis.
     * @param numCtrlY  Number of control points along the second axis.
     * @param numCtrlZ  Number of control points along the third axis.
     */
    SplineFitter3d2d(int numCtrlX, int numCtrlY, int numCtrlZ)
        : numCtrlX_(numCtrlX), numCtrlY_(numCtrlY), numCtrlZ_(numCtrlZ),
          controlPoints_(boost::extents[1][1][1]) {
        if (numCtrlX < Degree + 1 || numCtrlY < Degree + 1 || numCtrlZ < Degree + 1) {
            throw std::invalid_argument("all num_ctrl_* must be >= Degree + 1");
        }
    }

    /**
     * @brief Fit the spline to the provided data.
     *
     * @param x  3D input positions, shape (N, 3).
     * @param y  2D output values, shape (N, 2).
     * @param lowerBound  Lower bound in each axis. Defaults to min(x) per axis.
     * @param upperBound  Upper bound in each axis. Defaults to max(x) per axis.
     * @param smoothnessWeight  Regularization weight (0 = disabled).
     * @param smoothnessOrder  Derivative order used for regularization (1, 2, or 3).
     * @param maxIterations  Maximum Ceres solver iterations.
     */
    void fit(const Eigen::MatrixXd& x,
             const Eigen::MatrixXd& y,
             Eigen::Vector3d lowerBound =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             Eigen::Vector3d upperBound =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             double smoothnessWeight = 0.0,
             int smoothnessOrder = 1,
             int maxIterations = 200) {
        if (x.cols() != 3) {
            throw std::invalid_argument("x must have 3 columns");
        }
        if (y.cols() != 2) {
            throw std::invalid_argument("y must have 2 columns");
        }
        if (x.rows() != y.rows()) {
            throw std::invalid_argument("x and y must have the same number of rows");
        }
        if (x.rows() == 0) {
            throw std::invalid_argument("x must not be empty");
        }

        for (int d = 0; d < 3; ++d) {
            if (std::isnan(lowerBound[d])) { lowerBound[d] = x.col(d).minCoeff(); }
            if (std::isnan(upperBound[d])) { upperBound[d] = x.col(d).maxCoeff(); }
        }

        CtrlArray ctrlPts(boost::extents[numCtrlX_][numCtrlY_][numCtrlZ_]);
        std::fill(ctrlPts.data(), ctrlPts.data() + ctrlPts.num_elements(),
                  Eigen::Vector2d::Zero());
        Spline spline(lowerBound, upperBound, ctrlPts);
        UniformBSplineCeres<Spline> splineCeres(spline);

        const int numBlocks = splineCeres.getNumPointParameterPointers();
        std::vector<double*> paramPtrs(numBlocks);
        ceres::Problem problem;

        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            const Eigen::Vector3d xi = x.row(i).transpose();
            const auto data = splineCeres.getPointData(xi);
            splineCeres.fillParameterPointers(data, paramPtrs.begin(), paramPtrs.end());
            auto evaluator = splineCeres.getEvaluator(data);
            const Eigen::Vector2d meas = y.row(i).transpose();

            auto* costFn =
                new ceres::DynamicAutoDiffCostFunction<internal::VectorNdResidual<Spline, 2>>(
                    new internal::VectorNdResidual<Spline, 2>(evaluator, meas));
            for (int j = 0; j < numBlocks; ++j) {
                costFn->AddParameterBlock(2); // each control point is Eigen::Vector2d
            }
            costFn->SetNumResiduals(2);
            problem.AddResidualBlock(costFn, nullptr, paramPtrs);
        }

        if (smoothnessWeight > 0.0) {
            const int order = std::max(1, std::min(smoothnessOrder, Degree));
            if constexpr (Degree >= 3) {
                if (order == 3) { splineCeres.template addSmoothnessResidualsGrid<3>(problem, smoothnessWeight); }
                else if (order == 2) { splineCeres.template addSmoothnessResidualsGrid<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResidualsGrid<1>(problem, smoothnessWeight); }
            } else if constexpr (Degree >= 2) {
                if (order >= 2) { splineCeres.template addSmoothnessResidualsGrid<2>(problem, smoothnessWeight); }
                else { splineCeres.template addSmoothnessResidualsGrid<1>(problem, smoothnessWeight); }
            } else {
                splineCeres.template addSmoothnessResidualsGrid<1>(problem, smoothnessWeight);
            }
        }

        ceres::Solver::Options options;
        options.max_num_iterations = maxIterations;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        controlPoints_ = spline.getControlPoints();
        lowerBound_ = lowerBound;
        upperBound_ = upperBound;
    }

    /** @brief Return control points as (numCtrlX, numCtrlY, numCtrlZ) 3D array of Vector2d. */
    const CtrlArray& getControlPointsArray() const { return controlPoints_; }

    int getNumCtrlX() const { return numCtrlX_; }
    int getNumCtrlY() const { return numCtrlY_; }
    int getNumCtrlZ() const { return numCtrlZ_; }

    Eigen::Vector3d getLowerBound() const { return lowerBound_; }
    Eigen::Vector3d getUpperBound() const { return upperBound_; }

private:
    int numCtrlX_;
    int numCtrlY_;
    int numCtrlZ_;
    Eigen::Vector3d lowerBound_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d upperBound_{Eigen::Vector3d::Ones()};
    CtrlArray controlPoints_;
};

} // namespace ubs


