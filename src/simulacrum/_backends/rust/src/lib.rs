///! Simulacrum Rust backend — performance-critical kernels exposed to Python via PyO3.
///!
///! Build with: `maturin develop --release`

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute element stiffness matrix for a Quad4 element (example kernel).
///
/// Parameters
/// ----------
/// coords : (4, 2) node coordinates
/// C      : (3, 3) constitutive matrix (Voigt, plane stress/strain)
///
/// Returns
/// -------
/// ke : (8, 8) element stiffness matrix
#[pyfunction]
fn assemble_quad4_stiffness<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'_, f64>,
    c_mat: PyReadonlyArray2<'_, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let coords = coords.as_array();
    let c_mat = c_mat.as_array();

    // 2x2 Gauss quadrature points and weights
    let gauss_pts: [(f64, f64); 4] = [
        (-0.577350269, -0.577350269),
        ( 0.577350269, -0.577350269),
        ( 0.577350269,  0.577350269),
        (-0.577350269,  0.577350269),
    ];

    let mut ke = Array2::<f64>::zeros((8, 8));

    for &(xi, eta) in &gauss_pts {
        // Shape function derivatives in parametric space
        let dn_dxi = [
            [-(1.0 - eta) / 4.0, -(1.0 - xi) / 4.0],
            [ (1.0 - eta) / 4.0, -(1.0 + xi) / 4.0],
            [ (1.0 + eta) / 4.0,  (1.0 + xi) / 4.0],
            [-(1.0 + eta) / 4.0,  (1.0 - xi) / 4.0],
        ];

        // Jacobian
        let mut j = [[0.0f64; 2]; 2];
        for a in 0..4 {
            for i in 0..2 {
                for k in 0..2 {
                    j[i][k] += dn_dxi[a][i] * coords[[a, k]];
                }
            }
        }
        let det_j = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        let inv_j = [
            [ j[1][1] / det_j, -j[0][1] / det_j],
            [-j[1][0] / det_j,  j[0][0] / det_j],
        ];

        // Shape function derivatives in physical space
        let mut dn_dx = [[0.0f64; 2]; 4];
        for a in 0..4 {
            for i in 0..2 {
                for k in 0..2 {
                    dn_dx[a][i] += inv_j[i][k] * dn_dxi[a][k];
                }
            }
        }

        // B-matrix (3 x 8)
        let mut b = [[0.0f64; 8]; 3];
        for a in 0..4 {
            b[0][2 * a]     = dn_dx[a][0];
            b[1][2 * a + 1] = dn_dx[a][1];
            b[2][2 * a]     = dn_dx[a][1];
            b[2][2 * a + 1] = dn_dx[a][0];
        }

        // ke += B^T C B * det_J * w  (w = 1 for standard 2x2 Gauss)
        // tmp = C @ B  (3x8)
        let mut cb = [[0.0f64; 8]; 3];
        for i in 0..3 {
            for j_col in 0..8 {
                for k in 0..3 {
                    cb[i][j_col] += c_mat[[i, k]] * b[k][j_col];
                }
            }
        }

        // ke += B^T @ (C @ B) * det_J
        for i in 0..8 {
            for j_col in 0..8 {
                let mut val = 0.0;
                for k in 0..3 {
                    val += b[k][i] * cb[k][j_col];
                }
                ke[[i, j_col]] += val * det_j;
            }
        }
    }

    ke.into_pyarray(py)
}

/// Python module definition
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(assemble_quad4_stiffness, m)?)?;
    Ok(())
}
