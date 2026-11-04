//! PyO3 bindings for Scion's native MPM kernels.
//!
//! Thin by design: every decision that matters lives in `scion-mpm`, which
//! has no Python dependency. This file only checks shapes, pins the arrays
//! down as contiguous buffers and releases the GIL.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
            PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1};
use scion_mpm::{coo_to_csr, BlockPattern, Scatter};

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

/// Reusable CSR pattern for a block tangent on one stencil.
///
/// Build once per stencil rebuild -- once a load step -- and accumulate into
/// it on every Newton iteration. See `scion_mpm::block_csr` for why this is
/// bit-identical to the COO path it replaces and independent of the chunking.
#[pyclass(name = "BlockPattern", module = "scion_rust")]
pub struct PyBlockPattern {
    inner: BlockPattern,
}

#[pymethods]
impl PyBlockPattern {
    /// `node_id` is the `(n_particles, npe)` stencil, `nc` the dofs per node.
    #[new]
    fn new(node_id: PyReadonlyArray2<i64>, n_nodes: usize, nc: usize)
           -> PyResult<Self> {
        let npe = node_id.shape()[1];
        let ids = node_id.as_slice().map_err(|_| {
            err("node_id must be C-contiguous int64".to_string())
        })?;
        Ok(Self {
            inner: BlockPattern::new(ids, n_nodes, npe, nc).map_err(err)?,
        })
    }

    #[getter]
    fn nnz(&self) -> usize { self.inner.nnz() }

    #[getter]
    fn n_dof(&self) -> usize { self.inner.n_dof() }

    #[getter]
    fn n_particles(&self) -> usize { self.inner.n_particles() }

    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.indptr().to_vec().into_pyarray_bound(py)
    }

    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        py.allow_threads(|| self.inner.indices()).into_pyarray_bound(py)
    }

    /// Add the blocks of particles `lo .. hi` into `data`.
    ///
    /// Both arrays arrive flattened; the Python side owns those reshapes,
    /// which are free for a contiguous array and would otherwise pin this
    /// binding to one dimensionality.
    fn accumulate(&self, py: Python<'_>, mut data: PyReadwriteArray1<f64>,
                  blocks: PyReadonlyArray1<f64>, lo: usize, hi: usize)
                  -> PyResult<()> {
        let b = blocks.as_slice().map_err(|_| {
            err("blocks must be C-contiguous float64".to_string())
        })?;
        let d = data.as_slice_mut().map_err(|_| {
            err("data must be C-contiguous float64".to_string())
        })?;
        py.allow_threads(|| self.inner.accumulate(d, b, lo, hi)).map_err(err)
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

/// COO -> CSR, summing duplicates: a faster `coo_matrix(...).tocsr()`.
///
/// Returns `(indptr, indices, data)` for the caller to hand to
/// `scipy.sparse.csr_matrix`. Not bit-identical to SciPy -- see the note on
/// `scion_mpm::csr` -- but deterministic and thread-count independent.
#[pyfunction]
fn coo_to_csr_py<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<i64>,
    cols: PyReadonlyArray1<i64>,
    vals: PyReadonlyArray1<f64>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i32>>,
               Bound<'py, PyArray1<f64>>)> {
    let r = rows.as_slice().map_err(|_| err("rows must be C-contiguous int64".into()))?;
    let c = cols.as_slice().map_err(|_| err("cols must be C-contiguous int64".into()))?;
    let v = vals.as_slice().map_err(|_| err("vals must be C-contiguous float64".into()))?;
    let (ip, ix, d) = py
        .allow_threads(|| coo_to_csr(r, c, v, n_rows, n_cols))
        .map_err(err)?;
    Ok((ip.into_pyarray_bound(py), ix.into_pyarray_bound(py), d.into_pyarray_bound(py)))
}

/// Number of threads rayon will use, for the record in a benchmark.
#[pyfunction]
fn num_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

#[pymodule]
fn scion_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScatter>()?;
    m.add_class::<PyBlockPattern>()?;
    m.add_function(wrap_pyfunction!(scatter_add, m)?)?;
    m.add_function(wrap_pyfunction!(coo_to_csr_py, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}

