/*
 * Scion C++ backend — element-level kernels via pybind11.
 *
 * Build with:
 *   cd src/scion/_backends/cpp
 *   pip install .
 *
 * This file provides a starting point. Performance-critical routines
 * (element stiffness, mass matrix, etc.) live here and are called from
 * the Python assembler when the C++ backend is active.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

// Example: 2-D Quad4 element stiffness (same logic as the Rust kernel)
MatrixXd assemble_quad4_stiffness(
    const Eigen::Ref<const MatrixXd>& coords,   // (4, 2)
    const Eigen::Ref<const MatrixXd>& C          // (3, 3)
) {
    MatrixXd ke = MatrixXd::Zero(8, 8);

    // 2x2 Gauss points
    double gp = 0.5773502691896257;
    double gauss[4][2] = {{-gp, -gp}, {gp, -gp}, {gp, gp}, {-gp, gp}};

    for (auto& qp : gauss) {
        double xi = qp[0], eta = qp[1];

        // dN/dxi (4x2)
        Eigen::Matrix<double, 4, 2> dN_dxi;
        dN_dxi << -(1 - eta) / 4, -(1 - xi) / 4,
                   (1 - eta) / 4, -(1 + xi) / 4,
                   (1 + eta) / 4,  (1 + xi) / 4,
                  -(1 + eta) / 4,  (1 - xi) / 4;

        // Jacobian
        Eigen::Matrix2d J = dN_dxi.transpose() * coords;
        double detJ = J.determinant();
        Eigen::Matrix<double, 4, 2> dN_dx = dN_dxi * J.inverse();

        // B-matrix (3x8)
        Eigen::Matrix<double, 3, 8> B = Eigen::Matrix<double, 3, 8>::Zero();
        for (int a = 0; a < 4; ++a) {
            B(0, 2 * a)     = dN_dx(a, 0);
            B(1, 2 * a + 1) = dN_dx(a, 1);
            B(2, 2 * a)     = dN_dx(a, 1);
            B(2, 2 * a + 1) = dN_dx(a, 0);
        }

        ke += B.transpose() * C * B * detJ;
    }

    return ke;
}

PYBIND11_MODULE(_cpp_core, m) {
    m.doc() = "Scion C++ backend";
    m.def("assemble_quad4_stiffness", &assemble_quad4_stiffness,
          "Quad4 element stiffness matrix",
          py::arg("coords"), py::arg("C"));
}
