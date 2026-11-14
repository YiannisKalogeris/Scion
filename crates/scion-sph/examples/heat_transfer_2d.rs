//! SPHinXsys's example 6, the 2-D heat-transfer (anisotropic-diffusion) case,
//! in Rust.
//!
//!     cargo run -p scion-sph --release --example heat_transfer_2d -- [out.csv]
//!     cargo run -p scion-sph --release --example heat_transfer_2d -- \
//!         --vtk ../../examples/sph/results/heat_transfer_2d_vtk
//!
//! `--vtk <dir>` writes the same `particles.pvd` series the NumPy run writes,
//! with the same array names. It raises `--substeps` to 24 if left at 1, for
//! the reason the NumPy example gives: the stable step takes this case to
//! `t = 1` in **three** steps, so a series written at the natural step is
//! three frames of an animation nobody can read. Sub-stepping changes
//! nothing but the sampling -- there is no boundary here, the particles
//! never move, and only `phi` evolves.
//!
//! A 2 x 0.4 block at `dp = H/40`, periodic along `x` and free along `y`,
//! Wendland C2 at `h = 1.3 dp`, corrected kernel gradients, Heun in time, and
//! eleven observers along the mid-line. The same case runs in
//! `examples/sph/heat_transfer_2d.py`; the CSV written here is what
//! `tests/sph/test_rust_agreement.py` compares against the NumPy path, and
//! the NumPy path is the reference.

use scion_sph::vtk::{Field, Series};
use scion_sph::{lattice, Args, DiffusionSolver, DirectionalDiffusivity, Kernel};
use std::io::Write;

fn main() {
    let args = Args::parse();
    let length = 2.0_f64;
    let height = 0.4_f64;
    let dp = height / 40.0;
    let h = 1.3 * dp;
    let d_coeff = 1.0e-4;
    let bias_coeff = 0.0;
    let alpha = std::f64::consts::PI / 6.0;
    let end_time = 1.0_f64;

    let (positions, volumes) = lattice([0.0, 0.0], [length, height], dp);
    let diffusivity = DirectionalDiffusivity {
        d_coeff,
        bias_coeff,
        bias_direction: [alpha.cos(), alpha.sin()],
    };
    let mut solver = DiffusionSolver::new(
        positions.clone(),
        volumes,
        diffusivity,
        Kernel::WendlandC2,
        h,
        [0.0, 0.0],
        [length, height],
        [true, false],
        true,
        f64::EPSILON.sqrt(),
    );

    // the initial condition, exactly as the reference case states it: a unit
    // block on [0.45, 0.55] and a narrow Gaussian centred at x = 1.5
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

    let observers: Vec<[f64; 2]> = (0..11)
        .map(|i| [0.05 * length + 0.9 * length * i as f64 / 10.0, 0.5 * height])
        .collect();

    let substeps: usize = args
        .flag("substeps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(if args.flag("vtk").is_some() { 24 } else { 1 })
        .max(1);
    let dt = solver.time_step() / substeps as f64;
    let steps = (end_time / dt).ceil() as usize;
    println!(
        "{} particles, {} pairs, dt = {:.6}, {} steps to t = {:.3}",
        positions.len(),
        solver.pairs.len(),
        dt,
        steps,
        steps as f64 * dt
    );

    let mut series = args.flag("vtk").map(|dir| {
        // no outline file: periodic in x, free surfaces in y, so there is no
        // solid anywhere, and the fixed lattice fills its rectangle exactly
        // -- the particles *are* the outline
        Series::new(dir, "particles").expect("cannot open the series directory")
    });
    // taken once rather than read through `solver` each frame: the particle
    // set is fixed, so the volumes never change, and holding a borrow of the
    // solver inside the closure would block its own `step`
    let volumes = solver.volumes.clone();
    let write_frame = |series: &mut Series, t: f64, phi: &[f64]| {
        series
            .add(
                t,
                &positions,
                &[
                    ("phi", Field::Scalar(phi)),
                    ("volume", Field::Scalar(&volumes)),
                ],
            )
            .expect("cannot write a frame");
    };
    if let Some(series) = series.as_mut() {
        write_frame(series, 0.0, &phi);
    }

    let mut rows: Vec<(f64, Vec<f64>)> = vec![(0.0, solver.observe(&observers, &phi))];
    let mut t = 0.0;
    for _ in 0..steps {
        solver.step(&mut phi, dt);
        t += dt;
        rows.push((t, solver.observe(&observers, &phi)));
        if let Some(series) = series.as_mut() {
            write_frame(series, t, &phi);
        }
    }

    let total: f64 = phi.iter().zip(solver.volumes.iter()).map(|(p, v)| p * v).sum();
    println!("species integral at the end: {total:.17e}");
    for (t, row) in &rows {
        print!("t = {t:7.4}");
        for v in row {
            print!(" {v:12.6e}");
        }
        println!();
    }

    let path = std::env::args().nth(1).unwrap_or_else(|| "heat_transfer_2d.csv".to_string());
    let mut file = std::fs::File::create(&path).expect("cannot open the output file");
    writeln!(file, "time,{}", (0..11).map(|i| format!("phi_{i}")).collect::<Vec<_>>().join(","))
        .unwrap();
    for (t, row) in &rows {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:.17e}")).collect();
        writeln!(file, "{t:.17e},{}", cells.join(",")).unwrap();
    }
    // the final field too, so the comparison is not only at eleven points
    let field = path.replace(".csv", "_field.csv");
    let mut file = std::fs::File::create(&field).expect("cannot open the field file");
    writeln!(file, "x,y,phi").unwrap();
    for (p, v) in positions.iter().zip(phi.iter()) {
        writeln!(file, "{:.17e},{:.17e},{:.17e}", p[0], p[1], v).unwrap();
    }
    println!("wrote {path} and {field}");
}
