#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "spline_fitter.hpp"
#include "spline_position_finder.hpp"

namespace py = pybind11;

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

