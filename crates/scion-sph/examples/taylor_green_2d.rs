//! SPHinXsys example 5, the 2-D Taylor-Green vortex, in Rust.
//!
//!     cargo run -p scion-sph --release --example taylor_green_2d -- [end] [n] [out.csv]
//!     cargo run -p scion-sph --release --example taylor_green_2d -- 1.0 100 \
//!         --vtk ../../examples/sph/results/taylor_green_2d_vtk
//!
//! `--vtk <dir>` writes the same `particles.pvd` series the NumPy run writes,
//! with the same array names, and the periodic cell beside it as `cell.vtu`.
//!
//! The unit square, periodic both ways, Re = 100, Ma = 0.1. The velocity
//! field is an exact solution of the incompressible Navier-Stokes equations
//! and decays as `exp(-8 pi^2 nu t)`, which is what the printed ratio
//! compares against. The matching NumPy run is
//! `python -m examples.sph.taylor_green_2d`, and that is the reference.

use scion_sph::fluid::{FluidSolver, WeaklyCompressibleFluid};
use scion_sph::operators::{gradient, linear_gradient_correction};
use scion_sph::vtk::{self, Field, Series};
use scion_sph::{lattice, Args, Kernel};
use std::io::Write;

/// `dv/dx - du/dy` from the *corrected* SPH gradient.
///
/// The correction is not optional even though this domain is periodic both
/// ways and no support is ever truncated: measured against the closed form
/// `4 pi cos(2 pi x) cos(2 pi y)` at n = 50 / 100 / 200 the uncorrected
/// gradient plateaus at 3.3e-2 / 2.8e-2 / 2.7e-2 while the corrected one
/// gives 7.4e-3 / 1.9e-3 / 4.7e-4, ratios 3.99 and 4.00. Refining holds
/// `h/dp` fixed, so the kernel's smoothing error is a constant fraction
/// rather than something refinement removes. The solver keeps no `B` of its
/// own -- its momentum and continuity terms are the antisymmetric pair
/// forms, picked for conservation -- so one is built here for the diagnostic.
fn vorticity(solver: &FluidSolver) -> Vec<f64> {
    let b = linear_gradient_correction(&solver.pairs, &solver.volume, &solver.dw, f64::EPSILON);
    let u: Vec<f64> = solver.velocity.iter().map(|v| v[0]).collect();
    let v: Vec<f64> = solver.velocity.iter().map(|v| v[1]).collect();
    let grad_u = gradient(&solver.pairs, &u, &solver.volume, &solver.dw, Some(&b));
    let grad_v = gradient(&solver.pairs, &v, &solver.volume, &solver.dw, Some(&b));
    (0..solver.len()).map(|p| grad_v[p][0] - grad_u[p][1]).collect()
}

fn main() {
    let args = Args::parse();
    let end_time: f64 = args.parsed(0, 1.0);
    let n: usize = args.parsed(1, 100);
    let path = args.positional(2).map(str::to_string);

    let side = 1.0_f64;
    let (rho0, u_ref, reynolds, mach) = (1.0_f64, 1.0_f64, 100.0_f64, 0.1_f64);
    let spacing = side / n as f64;
    let fluid = WeaklyCompressibleFluid::new(rho0, u_ref / mach, rho0 * u_ref * side / reynolds);
    let (x, _) = lattice([0.0, 0.0], [side, side], spacing);
    let mut solver = FluidSolver::new(
        x.clone(),
        spacing,
        fluid,
        Kernel::WendlandC2,
        1.3,
        [0.0, 0.0],
        [side, side],
        [true, true],
    );
    let two_pi = 2.0 * std::f64::consts::PI;
    for (p, pos) in x.iter().enumerate() {
        solver.velocity[p] = [
            -(two_pi * pos[0]).cos() * (two_pi * pos[1]).sin(),
            (two_pi * pos[0]).sin() * (two_pi * pos[1]).cos(),
        ];
    }

    let rate = 8.0 * std::f64::consts::PI.powi(2) * fluid.viscosity / fluid.rho0;
    let energy0 = solver.kinetic_energy();
    println!(
        "{} particles, {} pairs, velocity decays at exp(-{rate:.4} t)",
        solver.len(),
        solver.pairs.len()
    );

    let interval = 0.1;
    let steps = (end_time / interval).round() as usize;
    let mut rows = vec![(0.0, energy0, solver.maximum_speed())];

    let mut series = args.flag("vtk").map(|dir| {
        // no boundary to draw -- the domain has no edge -- but without the
        // square there is nothing to say the four vortices tile, or to
        // notice a particle leaving one side and arriving at the other
        let mut points = Vec::new();
        let cell = [[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]];
        let segments = vtk::closed_loop(&mut points, &cell);
        let cell_path = std::path::Path::new(dir).join("cell.vtu");
        vtk::write_lines(&cell_path, &points, &segments).expect("cannot write the cell");
        println!("wrote {} (the periodic cell; not a wall)", cell_path.display());
        Series::new(dir, "particles").expect("cannot open the series directory")
    });
    let write_frame = |series: &mut Series, s: &FluidSolver| {
        let speed: Vec<f64> = s
            .velocity
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .collect();
        let omega = vorticity(s);
        series
            .add(
                s.time,
                &s.positions,
                &[
                    ("velocity", Field::Vector(&s.velocity)),
                    ("speed", Field::Scalar(&speed)),
                    ("pressure", Field::Scalar(&s.pressure)),
                    ("vorticity", Field::Scalar(&omega)),
                ],
            )
            .expect("cannot write a frame");
    };
    if let Some(series) = series.as_mut() {
        write_frame(series, &solver);
    }

    let start = std::time::Instant::now();
    for k in 0..steps {
        solver.advance((k + 1) as f64 * interval, u_ref);
        rows.push((solver.time, solver.kinetic_energy(), solver.maximum_speed()));
        if let Some(series) = series.as_mut() {
            write_frame(series, &solver);
        }
    }
    if let Some(series) = series.as_ref() {
        println!(
            "wrote {} frames to {}",
            series.len(),
            series.collection_path().display()
        );
    }
    println!("{:.1} s of wall time", start.elapsed().as_secs_f64());
    println!("{:>6} {:>15} {:>12} {:>7} {:>10}", "t", "kinetic energy", "exact", "ratio", "max speed");
    for (t, energy, speed) in &rows {
        let exact = energy0 * (-2.0 * rate * t).exp();
        println!("{t:6.2} {energy:15.6e} {exact:12.4e} {:7.3} {speed:10.5}", energy / exact);
    }

    if let Some(path) = path {
        let mut file = std::fs::File::create(&path).expect("cannot open the output file");
        writeln!(file, "time,kinetic_energy,max_speed").unwrap();
        for (t, energy, speed) in &rows {
            writeln!(file, "{t:.17e},{energy:.17e},{speed:.17e}").unwrap();
        }
        println!("wrote {path}");
    }
}
