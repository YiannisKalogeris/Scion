//! Smoothed particle hydrodynamics in two dimensions: kernels, neighbour
//! search, corrected operators and a diffusion solver, with no Python in the
//! loop so `cargo test` exercises it with no interpreter.
//!
//! **Two dimensions only, deliberately.** The Python package
//! (`scion.discretization.sph`) is dimension-general and is the reference
//! this crate is tested against; the example it exists for is 2-D, and a
//! half-general 3-D path that is never run is worse than an honest 2-D one.
//! Generalising means the correction-matrix inverse and the cell stencil,
//! not the structure.
//!
//! The reference is the NumPy path, never the other way round: a
//! disagreement is this crate's bug until proved otherwise.

pub mod diffusion;
pub mod elastic;
pub mod fluid;
pub mod kernel;
pub mod levelset;
pub mod neighbors;
pub mod operators;
pub mod vtk;

pub use diffusion::{lattice, DiffusionSolver, DirectionalDiffusivity};
pub use elastic::{ElasticSolver, SaintVenantKirchhoffSolid};
pub use fluid::{FluidSolver, WeaklyCompressibleFluid};
pub use kernel::Kernel;
pub use levelset::LevelSet;
pub use neighbors::NeighborList;
pub use vtk::{Args, Field, Series};
