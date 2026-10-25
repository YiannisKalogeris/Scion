//! PyO3 bindings for Scion's native MPM kernels.
//!
//! Thin by design: every decision that matters lives in `scion-mpm`, which
//! has no Python dependency. This file only checks shapes, pins the arrays
//! down as contiguous buffers and releases the GIL.

use numpy::{PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scion_mpm::Scatter;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// Reusable particle-to-grid scatter for one stencil.
///
/// Build once per stencil rebuild, call `add` as often as the assembly needs.
#[pyclass(name = "Scatter", module = "scion_rust")]
pub struct PyScatter {
    inner: Scatter,
}

#[pymethods]
impl PyScatter {
    /// `node_id` is the `(n_particles, npe)` stencil; `n_nodes` the grid size.
    #[new]
    fn new(node_id: PyReadonlyArray2<i64>, n_nodes: usize) -> PyResult<Self> {
        // as_slice refuses a non-contiguous array rather than copying it
        // silently -- a hidden copy of a million-particle stencil would eat
        // the speed-up this exists to provide.
        let ids = node_id.as_slice().map_err(|_| {
            err("node_id must be C-contiguous int64".to_string())
        })?;
        Ok(Self { inner: Scatter::new(ids, n_nodes).map_err(err)? })
    }

    #[getter]
    fn n_nodes(&self) -> usize { self.inner.n_nodes() }

    #[getter]
    fn n_entries(&self) -> usize { self.inner.n_entries() }

    /// Accumulate `vals` into `out[:, col : col + vals.shape[-1]]`.
    ///
    /// `vals` arrives already flattened to `(n_entries, k)`; the Python side
    /// owns that reshape, which is free for a contiguous array.
    #[pyo3(signature = (out, vals, col = 0))]
    fn add(&self, py: Python<'_>, mut out: PyReadwriteArray2<f64>,
           vals: PyReadonlyArray2<f64>, col: usize) -> PyResult<()> {
        let ncomp = out.shape()[1];
        let k = vals.shape()[1];
        let v = vals.as_slice().map_err(|_| {
            err("vals must be C-contiguous float64".to_string())
        })?;
        let o = out.as_slice_mut().map_err(|_| {
            err("out must be C-contiguous float64".to_string())
        })?;
        // The readwrite guard holds the borrow, so no Python code can touch
        // these buffers while the GIL is released.
        py.allow_threads(|| self.inner.add(o, ncomp, col, v, k)).map_err(err)
    }
}

/// One-shot form: build the transpose, scatter, discard it.
///
/// A drop-in for `np.add.at`. Prefer `Scatter` wherever the same stencil is
/// scattered more than once, which is every assembly in the library.
#[pyfunction]
#[pyo3(signature = (out, node_id, vals, col = 0))]
fn scatter_add(py: Python<'_>, out: PyReadwriteArray2<f64>,
               node_id: PyReadonlyArray2<i64>, vals: PyReadonlyArray2<f64>,
               col: usize) -> PyResult<()> {
    let n_nodes = out.shape()[0];
    let sc = PyScatter::new(node_id, n_nodes)?;
    sc.add(py, out, vals, col)
}

/// Number of threads rayon will use, for the record in a benchmark.
#[pyfunction]
fn num_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

#[pymodule]
fn scion_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScatter>()?;
    m.add_function(wrap_pyfunction!(scatter_add, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}

