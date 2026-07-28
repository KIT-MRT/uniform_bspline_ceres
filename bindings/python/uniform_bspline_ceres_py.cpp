#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/multi_array.hpp>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <uniform_bspline/multi_array.hpp>
#include <uniform_bspline/uniform_bspline.hpp>

#include "uniform_bspline_ceres.hpp"
#include "uniform_bspline_ceres_evaluator.hpp"

namespace py = pybind11;

// ===========================================================================
// SplineFitter classes
// ===========================================================================

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

// ===========================================================================
// SplinePositionFinder classes
// ===========================================================================

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

// ---------------------------------------------------------------------------
// Bind SplineFitter1d1d<Degree>.
// ---------------------------------------------------------------------------
template <int Degree>
void bind_spline_fitter_1d1d(py::module& m, const char* name) {
    using Fitter = ubs::SplineFitter1d1d<Degree>;

    py::class_<Fitter>(m, name)
        .def(py::init<int>(), py::arg("num_control_points"),
             "Create a 1D->1D B-spline fitter.")

        .def("fit",
             [](Fitter& self,
                const std::vector<double>& x,
                const std::vector<double>& y,
                double lower_bound,
                double upper_bound,
                double smoothness_weight,
                int smoothness_order,
                int max_iterations) {
                 self.fit(x, y, lower_bound, upper_bound, smoothness_weight, smoothness_order, max_iterations);
             },
             py::arg("x"),
             py::arg("y"),
             py::arg("lower_bound") = std::numeric_limits<double>::quiet_NaN(),
             py::arg("upper_bound") = std::numeric_limits<double>::quiet_NaN(),
             py::arg("smoothness_weight") = 0.0,
             py::arg("smoothness_order") = 1,
             py::arg("max_iterations") = 200,
             "Fit the spline to (x, y) data. lower_bound/upper_bound default to "
             "min(x)/max(x). smoothness_weight > 0 adds a regularizer on the "
             "smoothness_order-th derivative.")

        .def("get_control_points",
             [](const Fitter& self) -> std::vector<double> {
                 return self.getControlPoints();
             },
             "Return the fitted control points.")

        .def("get_lower_bound", &Fitter::getLowerBound,
             "Return the lower bound of the fitted spline.")

        .def("get_upper_bound", &Fitter::getUpperBound,
             "Return the upper bound of the fitted spline.")

        .def("__repr__", [name](const Fitter& self) {
            return std::string(name) +
                   "(lower=" + std::to_string(self.getLowerBound()) +
                   ", upper=" + std::to_string(self.getUpperBound()) +
                   ", n_ctrl=" + std::to_string(self.getControlPoints().size()) + ")";
        });
}

// ---------------------------------------------------------------------------
// Bind SplineFitter1d3d<Degree>.
// ---------------------------------------------------------------------------
//! [CustomBinding_Example]
template <int Degree>
void bind_spline_fitter_1d3d(py::module& m, const char* name) {
    using Fitter = ubs::SplineFitter1d3d<Degree>;

    py::class_<Fitter>(m, name)
        .def(py::init<int>(), py::arg("num_control_points"),
             "Create a 1D->3D B-spline fitter (e.g. 3D trajectory).")

        .def("fit",
             [](Fitter& self,
                const std::vector<double>& x,
                const Eigen::MatrixXd& y,
                double lower_bound,
                double upper_bound,
                double smoothness_weight,
                int smoothness_order,
                int max_iterations) {
                 self.fit(x, y, lower_bound, upper_bound, smoothness_weight, smoothness_order, max_iterations);
             },
             py::arg("x"),
             py::arg("y"),
             py::arg("lower_bound") = std::numeric_limits<double>::quiet_NaN(),
             py::arg("upper_bound") = std::numeric_limits<double>::quiet_NaN(),
             py::arg("smoothness_weight") = 0.0,
             py::arg("smoothness_order") = 1,
             py::arg("max_iterations") = 200,
             "Fit the spline to (x, y) data. y must be shape (N, 3).")

        .def("get_control_points", &Fitter::getControlPoints,
             "Return the fitted control points as numpy array of shape (M, 3).")

        .def("get_lower_bound", &Fitter::getLowerBound,
             "Return the lower bound of the fitted spline.")

        .def("get_upper_bound", &Fitter::getUpperBound,
             "Return the upper bound of the fitted spline.")

        .def("__repr__", [name](const Fitter& self) {
            return std::string(name) +
                   "(lower=" + std::to_string(self.getLowerBound()) +
                   ", upper=" + std::to_string(self.getUpperBound()) +
                   ", n_ctrl=" + std::to_string(self.getControlPoints().rows()) + ")";
        });
}
//! [CustomBinding_Example]

// ---------------------------------------------------------------------------
// Bind SplineFitter2d1d<Degree>.
// ---------------------------------------------------------------------------
template <int Degree>
void bind_spline_fitter_2d1d(py::module& m, const char* name) {
    using Fitter = ubs::SplineFitter2d1d<Degree>;

    py::class_<Fitter>(m, name)
        .def(py::init<int, int>(), py::arg("num_ctrl_x"), py::arg("num_ctrl_y"),
             "Create a 2D->1D B-spline fitter (e.g. height map, cost field).")

        .def("fit",
             [](Fitter& self,
                const Eigen::MatrixXd& x,
                const std::vector<double>& y,
                Eigen::Vector2d lower_bound,
                Eigen::Vector2d upper_bound,
                double smoothness_weight,
                int smoothness_order,
                int max_iterations) {
                 self.fit(x, y, lower_bound, upper_bound, smoothness_weight, smoothness_order, max_iterations);
             },
             py::arg("x"),
             py::arg("y"),
             py::arg("lower_bound") =
                 Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("upper_bound") =
                 Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("smoothness_weight") = 0.0,
             py::arg("smoothness_order") = 1,
             py::arg("max_iterations") = 200,
             "Fit the spline to (x, y) data. x must be shape (N, 2).")

        .def("get_control_points", &Fitter::getControlPoints,
             "Return the fitted control points as numpy array of shape (num_ctrl_x, num_ctrl_y).")

        .def("get_lower_bound", &Fitter::getLowerBound,
             "Return the lower bound (Vector2d) of the fitted spline.")

        .def("get_upper_bound", &Fitter::getUpperBound,
             "Return the upper bound (Vector2d) of the fitted spline.")

        .def("__repr__", [name](const Fitter& self) {
            const auto& ctrl = self.getControlPoints();
            return std::string(name) + "(n_ctrl=[" + std::to_string(ctrl.rows()) + "," +
                   std::to_string(ctrl.cols()) + "])";
        });
}

// ---------------------------------------------------------------------------
// Bind SplineFitter3d1d<Degree>.
// ---------------------------------------------------------------------------
template <int Degree>
void bind_spline_fitter_3d1d(py::module& m, const char* name) {
    using Fitter = ubs::SplineFitter3d1d<Degree>;

    py::class_<Fitter>(m, name)
        .def(py::init<int, int, int>(),
             py::arg("num_ctrl_x"), py::arg("num_ctrl_y"), py::arg("num_ctrl_z"),
             "Create a 3D->1D B-spline fitter (e.g. 3D cost volume).")

        .def("fit",
             [](Fitter& self,
                const Eigen::MatrixXd& x,
                const std::vector<double>& y,
                Eigen::Vector3d lower_bound,
                Eigen::Vector3d upper_bound,
                double smoothness_weight,
                int smoothness_order,
                int max_iterations) {
                 self.fit(x, y, lower_bound, upper_bound, smoothness_weight, smoothness_order, max_iterations);
             },
             py::arg("x"),
             py::arg("y"),
             py::arg("lower_bound") =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("upper_bound") =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("smoothness_weight") = 0.0,
             py::arg("smoothness_order") = 1,
             py::arg("max_iterations") = 200,
             "Fit the spline to (x, y) data. x must be shape (N, 3).")

        .def("get_control_points",
             [](const Fitter& self) -> py::array_t<double> {
                 const auto& arr = self.getControlPointsArray();
                 const int nx = self.getNumCtrlX();
                 const int ny = self.getNumCtrlY();
                 const int nz = self.getNumCtrlZ();
                 py::array_t<double> result({nx, ny, nz});
                 std::copy(arr.data(), arr.data() + arr.num_elements(),
                           result.mutable_data());
                 return result;
             },
             "Return the fitted control points as numpy array of shape (nx, ny, nz).")

        .def("get_lower_bound", &Fitter::getLowerBound,
             "Return the lower bound (Vector3d) of the fitted spline.")

        .def("get_upper_bound", &Fitter::getUpperBound,
             "Return the upper bound (Vector3d) of the fitted spline.")

        .def("__repr__", [name](const Fitter& self) {
            return std::string(name) + "(n_ctrl=[" + std::to_string(self.getNumCtrlX()) + "," +
                   std::to_string(self.getNumCtrlY()) + "," +
                   std::to_string(self.getNumCtrlZ()) + "])";
        });
}

// ---------------------------------------------------------------------------
// Bind SplineFitter3d2d<Degree>.
// ---------------------------------------------------------------------------
template <int Degree>
void bind_spline_fitter_3d2d(py::module& m, const char* name) {
    using Fitter = ubs::SplineFitter3d2d<Degree>;

    py::class_<Fitter>(m, name)
        .def(py::init<int, int, int>(),
             py::arg("num_ctrl_x"), py::arg("num_ctrl_y"), py::arg("num_ctrl_z"),
             "Create a 3D->2D B-spline fitter (e.g. 3D deformation field).")

        .def("fit",
             [](Fitter& self,
                const Eigen::MatrixXd& x,
                const Eigen::MatrixXd& y,
                Eigen::Vector3d lower_bound,
                Eigen::Vector3d upper_bound,
                double smoothness_weight,
                int smoothness_order,
                int max_iterations) {
                 self.fit(x, y, lower_bound, upper_bound, smoothness_weight, smoothness_order, max_iterations);
             },
             py::arg("x"),
             py::arg("y"),
             py::arg("lower_bound") =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("upper_bound") =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("smoothness_weight") = 0.0,
             py::arg("smoothness_order") = 1,
             py::arg("max_iterations") = 200,
             "Fit the spline to (x, y) data. x must be shape (N, 3), y shape (N, 2).")

        .def("get_control_points",
             [](const Fitter& self) -> py::array_t<double> {
                 const auto& arr = self.getControlPointsArray();
                 const int nx = self.getNumCtrlX();
                 const int ny = self.getNumCtrlY();
                 const int nz = self.getNumCtrlZ();
                 // shape: (nx, ny, nz, 2) — last dim is the 2D output
                 py::array_t<double> result({nx, ny, nz, 2});
                 auto buf = result.mutable_unchecked<4>();
                 for (int ix = 0; ix < nx; ++ix)
                     for (int iy = 0; iy < ny; ++iy)
                         for (int iz = 0; iz < nz; ++iz) {
                             buf(ix, iy, iz, 0) = arr[ix][iy][iz][0];
                             buf(ix, iy, iz, 1) = arr[ix][iy][iz][1];
                         }
                 return result;
             },
             "Return the fitted control points as numpy array of shape (nx, ny, nz, 2).")

        .def("get_lower_bound", &Fitter::getLowerBound,
             "Return the lower bound (Vector3d) of the fitted spline.")

        .def("get_upper_bound", &Fitter::getUpperBound,
             "Return the upper bound (Vector3d) of the fitted spline.")

        .def("__repr__", [name](const Fitter& self) {
            return std::string(name) + "(n_ctrl=[" + std::to_string(self.getNumCtrlX()) + "," +
                   std::to_string(self.getNumCtrlY()) + "," +
                   std::to_string(self.getNumCtrlZ()) + "])"; 
        });
}

// ---------------------------------------------------------------------------
// Bind SplinePositionFinder1d1d<Degree>.
// ---------------------------------------------------------------------------
template <int Degree>
void bind_spline_position_finder_1d1d(py::module& m, const char* name) {
    using Finder = ubs::SplinePositionFinder1d1d<Degree>;

    py::class_<Finder>(m, name)
        .def(py::init<double, double, std::vector<double>>(),
             py::arg("lower_bound"), py::arg("upper_bound"), py::arg("control_points"),
             "Create a 1D->1D position finder from a fitted spline.")

        .def("find",
             [](const Finder& self, double target, double initial_t,
                double lower_t, double upper_t, int max_iterations) {
                 return self.find(target, initial_t, lower_t, upper_t, max_iterations);
             },
             py::arg("target"),
             py::arg("initial_t"),
             py::arg("lower_t") = std::numeric_limits<double>::quiet_NaN(),
             py::arg("upper_t") = std::numeric_limits<double>::quiet_NaN(),
             py::arg("max_iterations") = 200,
             "Find t* minimising (spline(t) - target)^2. Returns optimal scalar t.");
}

// ---------------------------------------------------------------------------
// Bind SplinePositionFinder1d3d<Degree>.
// ---------------------------------------------------------------------------
//! [CustomBinding_Example_Finder]
template <int Degree>
void bind_spline_position_finder_1d3d(py::module& m, const char* name) {
    using Finder = ubs::SplinePositionFinder1d3d<Degree>;
    using CtrlContainer =
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

    py::class_<Finder>(m, name)
        .def(py::init([](double lb, double ub, const Eigen::MatrixXd& cp) {
                 if (cp.cols() != 3) {
                     throw std::invalid_argument("control_points must have 3 columns");
                 }
                 CtrlContainer ctrl(cp.rows());
                 for (Eigen::Index i = 0; i < cp.rows(); ++i) {
                     ctrl[i] = cp.row(i).transpose();
                 }
                 return Finder(lb, ub, std::move(ctrl));
             }),
             py::arg("lower_bound"), py::arg("upper_bound"), py::arg("control_points"),
             "Create a 1D->3D position finder. control_points is shape (M, 3).")

        .def("find",
             [](const Finder& self, const Eigen::Vector3d& target, double initial_t,
                double lower_t, double upper_t, int max_iterations) {
                 return self.find(target, initial_t, lower_t, upper_t, max_iterations);
             },
             py::arg("target"),
             py::arg("initial_t"),
             py::arg("lower_t") = std::numeric_limits<double>::quiet_NaN(),
             py::arg("upper_t") = std::numeric_limits<double>::quiet_NaN(),
             py::arg("max_iterations") = 200,
             "Find t* minimising ||spline(t) - target||^2 (closest point on curve). "
             "Returns optimal scalar t.");
}
//! [CustomBinding_Example_Finder]

// ---------------------------------------------------------------------------
// Bind SplinePositionFinder3d1d<Degree>.
// ---------------------------------------------------------------------------
template <int Degree>
void bind_spline_position_finder_3d1d(py::module& m, const char* name) {
    using Finder = ubs::SplinePositionFinder3d1d<Degree>;
    using CtrlArray = ubs::EigenAlignedMultiArray<double, 3>;

    py::class_<Finder>(m, name)
        .def(py::init([](Eigen::Vector3d lb, Eigen::Vector3d ub,
                         const py::array_t<double>& cp) {
                 if (cp.ndim() != 3) {
                     throw std::invalid_argument("control_points must be 3-dimensional");
                 }
                 const int nx = static_cast<int>(cp.shape(0));
                 const int ny = static_cast<int>(cp.shape(1));
                 const int nz = static_cast<int>(cp.shape(2));
                 CtrlArray arr(boost::extents[nx][ny][nz]);
                 auto r = cp.unchecked<3>();
                 for (int ix = 0; ix < nx; ++ix)
                     for (int iy = 0; iy < ny; ++iy)
                         for (int iz = 0; iz < nz; ++iz)
                             arr[ix][iy][iz] = r(ix, iy, iz);
                 return Finder(lb, ub, std::move(arr));
             }),
             py::arg("lower_bound"), py::arg("upper_bound"), py::arg("control_points"),
             "Create a 3D->1D position finder. control_points is shape (nx, ny, nz).")

        .def("find",
             [](const Finder& self, double target, Eigen::Vector3d initial_t,
                Eigen::Vector3d lower_t, Eigen::Vector3d upper_t, int max_iterations) {
                 return self.find(target, initial_t, lower_t, upper_t, max_iterations);
             },
             py::arg("target"),
             py::arg("initial_t"),
             py::arg("lower_t") =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("upper_t") =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("max_iterations") = 200,
             "Find t* in R^3 minimising (spline(t) - target)^2. Returns optimal Vector3d.");
}

// ---------------------------------------------------------------------------
// Bind SplinePositionFinder3d2d<Degree>.
// ---------------------------------------------------------------------------
template <int Degree>
void bind_spline_position_finder_3d2d(py::module& m, const char* name) {
    using Finder = ubs::SplinePositionFinder3d2d<Degree>;
    using CtrlArray = ubs::EigenAlignedMultiArray<Eigen::Vector2d, 3>;

    py::class_<Finder>(m, name)
        .def(py::init([](Eigen::Vector3d lb, Eigen::Vector3d ub,
                         const py::array_t<double>& cp) {
                 if (cp.ndim() != 4 || cp.shape(3) != 2) {
                     throw std::invalid_argument(
                         "control_points must be shape (nx, ny, nz, 2)");
                 }
                 const int nx = static_cast<int>(cp.shape(0));
                 const int ny = static_cast<int>(cp.shape(1));
                 const int nz = static_cast<int>(cp.shape(2));
                 CtrlArray arr(boost::extents[nx][ny][nz]);
                 auto r = cp.unchecked<4>();
                 for (int ix = 0; ix < nx; ++ix)
                     for (int iy = 0; iy < ny; ++iy)
                         for (int iz = 0; iz < nz; ++iz)
                             arr[ix][iy][iz] = Eigen::Vector2d(r(ix, iy, iz, 0),
                                                               r(ix, iy, iz, 1));
                 return Finder(lb, ub, std::move(arr));
             }),
             py::arg("lower_bound"), py::arg("upper_bound"), py::arg("control_points"),
             "Create a 3D->2D position finder. control_points is shape (nx, ny, nz, 2).")

        .def("find",
             [](const Finder& self, const Eigen::Vector2d& target, Eigen::Vector3d initial_t,
                Eigen::Vector3d lower_t, Eigen::Vector3d upper_t, int max_iterations) {
                 return self.find(target, initial_t, lower_t, upper_t, max_iterations);
             },
             py::arg("target"),
             py::arg("initial_t"),
             py::arg("lower_t") =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("upper_t") =
                 Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
             py::arg("max_iterations") = 200,
             "Find t* in R^3 minimising ||spline(t) - target||^2. Returns optimal Vector3d.");
}

PYBIND11_MODULE(uniform_bspline_ceres, m)
{
    m.doc() = "Python bindings for uniform_bspline_ceres. Provides SplineFitter "
              "classes that fit uniform B-splines to (x, y) data using Ceres.";

    bind_spline_fitter_1d1d<1>(m, "SplineFitter1d1d1");
    bind_spline_fitter_1d1d<2>(m, "SplineFitter1d1d2");
    bind_spline_fitter_1d1d<3>(m, "SplineFitter1d1d3");
    bind_spline_fitter_1d1d<4>(m, "SplineFitter1d1d4");
    bind_spline_fitter_1d1d<5>(m, "SplineFitter1d1d5");

    bind_spline_fitter_1d3d<1>(m, "SplineFitter1d3d1");
    bind_spline_fitter_1d3d<2>(m, "SplineFitter1d3d2");
    bind_spline_fitter_1d3d<3>(m, "SplineFitter1d3d3");
    bind_spline_fitter_1d3d<4>(m, "SplineFitter1d3d4");
    bind_spline_fitter_1d3d<5>(m, "SplineFitter1d3d5");

    bind_spline_fitter_2d1d<1>(m, "SplineFitter2d1d1");
    bind_spline_fitter_2d1d<2>(m, "SplineFitter2d1d2");
    bind_spline_fitter_2d1d<3>(m, "SplineFitter2d1d3");
    bind_spline_fitter_2d1d<4>(m, "SplineFitter2d1d4");
    bind_spline_fitter_2d1d<5>(m, "SplineFitter2d1d5");

    bind_spline_fitter_3d1d<1>(m, "SplineFitter3d1d1");
    bind_spline_fitter_3d1d<2>(m, "SplineFitter3d1d2");
    bind_spline_fitter_3d1d<3>(m, "SplineFitter3d1d3");
    bind_spline_fitter_3d1d<4>(m, "SplineFitter3d1d4");
    bind_spline_fitter_3d1d<5>(m, "SplineFitter3d1d5");

    bind_spline_fitter_3d2d<1>(m, "SplineFitter3d2d1");
    bind_spline_fitter_3d2d<2>(m, "SplineFitter3d2d2");
    bind_spline_fitter_3d2d<3>(m, "SplineFitter3d2d3");
    bind_spline_fitter_3d2d<4>(m, "SplineFitter3d2d4");
    bind_spline_fitter_3d2d<5>(m, "SplineFitter3d2d5");

    bind_spline_position_finder_1d1d<1>(m, "SplinePositionFinder1d1d1");
    bind_spline_position_finder_1d1d<2>(m, "SplinePositionFinder1d1d2");
    bind_spline_position_finder_1d1d<3>(m, "SplinePositionFinder1d1d3");
    bind_spline_position_finder_1d1d<4>(m, "SplinePositionFinder1d1d4");
    bind_spline_position_finder_1d1d<5>(m, "SplinePositionFinder1d1d5");

    bind_spline_position_finder_1d3d<1>(m, "SplinePositionFinder1d3d1");
    bind_spline_position_finder_1d3d<2>(m, "SplinePositionFinder1d3d2");
    bind_spline_position_finder_1d3d<3>(m, "SplinePositionFinder1d3d3");
    bind_spline_position_finder_1d3d<4>(m, "SplinePositionFinder1d3d4");
    bind_spline_position_finder_1d3d<5>(m, "SplinePositionFinder1d3d5");

    bind_spline_position_finder_3d1d<1>(m, "SplinePositionFinder3d1d1");
    bind_spline_position_finder_3d1d<2>(m, "SplinePositionFinder3d1d2");
    bind_spline_position_finder_3d1d<3>(m, "SplinePositionFinder3d1d3");
    bind_spline_position_finder_3d1d<4>(m, "SplinePositionFinder3d1d4");
    bind_spline_position_finder_3d1d<5>(m, "SplinePositionFinder3d1d5");

    bind_spline_position_finder_3d2d<1>(m, "SplinePositionFinder3d2d1");
    bind_spline_position_finder_3d2d<2>(m, "SplinePositionFinder3d2d2");
    bind_spline_position_finder_3d2d<3>(m, "SplinePositionFinder3d2d3");
    bind_spline_position_finder_3d2d<4>(m, "SplinePositionFinder3d2d4");
    bind_spline_position_finder_3d2d<5>(m, "SplinePositionFinder3d2d5");

}

