//! Where the time goes in the 2-D diffusion case, in Rust.
//!
//!     cargo run -p scion-sph --release --example bench -- [refine] [steps]
//!
//! Prints the two halves separately -- building the neighbour list, the
//! kernel values and the correction matrices, against the stepping loop --
//! because they scale differently and the reference case is so short that
//! setup dominates it. The matching NumPy run is
//! `python -m examples.sph.benchmark`, which calls this and tabulates both.

use scion_sph::{lattice, DiffusionSolver, DirectionalDiffusivity, Kernel};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let refine: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(1);
    let steps: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(50);

    let (length, height) = (2.0_f64, 0.4_f64);
    let dp = height / (40.0 * refine as f64);
    let h = 1.3 * dp;
    let (positions, volumes) = lattice([0.0, 0.0], [length, height], dp);
    let n = positions.len();

    let mut phi: Vec<f64> = positions
        .iter()
        .map(|p| {
            if p[0] >= 1.0 {
                (-2500.0 * (p[0] - 1.5) * (p[0] - 1.5)).exp()
            } else if p[0] >= 0.45 && p[0] <= 0.55 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let t0 = Instant::now();
    let mut solver = DiffusionSolver::new(
        positions,
        volumes,
        DirectionalDiffusivity::isotropic(1.0e-4),
        Kernel::WendlandC2,
        h,
        [0.0, 0.0],
        [length, height],
        [true, false],
        true,
        f64::EPSILON.sqrt(),
    );
    let setup = t0.elapsed().as_secs_f64();

    let dt = solver.time_step();
    let t1 = Instant::now();
    for _ in 0..steps {
        solver.step(&mut phi, dt);
    }
    let stepping = t1.elapsed().as_secs_f64();

    // printed as a machine-readable line so the Python side can parse it
    println!(
        "rust refine={refine} particles={n} pairs={} setup={setup:.6} \
         steps={steps} stepping={stepping:.6} per_step={:.6e}",
        solver.pairs.len(),
        stepping / steps as f64
    );
}
