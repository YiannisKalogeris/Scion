//! SPHinXsys example 9, the 2-D dam break with static confinement, in Rust.
//!
//!     cargo run -p scion-sph --release --example dam_break_confinement -- [end] [out.csv]
//!     cargo run -p scion-sph --release --example dam_break_confinement -- 20.0 \
//!         --vtk ../../examples/sph/results/dam_break_confinement_vtk
//!
//! `--vtk <dir>` writes the same `particles.pvd` series the NumPy run writes,
//! with the same array names and the tank and obstacle beside it as
//! `solid.vtu`, so ParaView opens either code's output the same way.
//!
//! A 2 x 1 column of water collapses in a 5.366 x 5.366 tank holding a
//! triangular obstacle. Neither the tank nor the obstacle is made of
//! particles: both are signed distance fields, carried as one level set
//! because the two solids are disjoint and further apart than a kernel
//! support. The matching NumPy run is
//! `python -m examples.sph.dam_break_confinement`, and that is the reference.

use scion_sph::fluid::{FluidSolver, WeaklyCompressibleFluid};
use scion_sph::levelset::{box_interior_distance, polygon_distance, LevelSet};
use scion_sph::vtk::{self, Field, Series};
use scion_sph::{lattice, Args, Kernel};
use std::io::Write;

const TANK: f64 = 5.366;
const COLUMN_LENGTH: f64 = 2.0;
const COLUMN_HEIGHT: f64 = 1.0;
const SPACING: f64 = 0.025;
const RHO0: f64 = 1.0;
const GRAVITY: f64 = 1.0;

fn main() {
    let args = Args::parse();
    let end_time: f64 = args.parsed(0, 2.0);
    let path = args.positional(1).map(str::to_string);

    let u_ref = 2.0 * (GRAVITY * COLUMN_HEIGHT).sqrt();
    let c0 = 10.0 * u_ref;
    let margin = 4.0 * SPACING;
    let lower = [-margin, -margin];
    let upper = [TANK + margin, TANK + margin];

    let triangle = [
        [0.5 * TANK, 0.05 * TANK],
        [0.5 * TANK + 0.5 * COLUMN_LENGTH, 0.05 * TANK + 0.5 * COLUMN_HEIGHT],
        [0.5 * TANK + 0.5 * COLUMN_LENGTH, 0.05 * TANK],
    ];
    // phi > 0 inside the solid: outside the tank, or inside the triangle
    let solid = |p: [f64; 2]| {
        box_interior_distance(p, [0.0, 0.0], [TANK, TANK]).max(polygon_distance(p, &triangle))
    };

    let (x, _) = lattice([0.0, 0.0], [COLUMN_LENGTH, COLUMN_HEIGHT], SPACING);
    let fluid = WeaklyCompressibleFluid::new(RHO0, c0, 0.0);
    let mut solver = FluidSolver::new(
        x,
        SPACING,
        fluid,
        Kernel::WendlandC2,
        1.3,
        lower,
        upper,
        [false, false],
    );
    let start_levelset = std::time::Instant::now();
    solver.confinement = vec![LevelSet::new(
        solid,
        lower,
        upper,
        SPACING,
        Kernel::WendlandC2,
        1.3 * SPACING,
    )];
    solver.free_surface = true;
    solver.riemann_pressure = true;
    solver.transport_coefficient = 0.0;
    solver.gravity = [0.0, -GRAVITY];
    println!(
        "{} particles, level set {:?} cells in {:.2} s",
        solver.len(),
        solver.confinement[0].counts,
        start_levelset.elapsed().as_secs_f64()
    );

    let energy = |s: &FluidSolver| -> f64 {
        (0..s.len())
            .map(|p| {
                s.mass[p]
                    * (0.5 * (s.velocity[p][0].powi(2) + s.velocity[p][1].powi(2))
                        + GRAVITY * s.positions[p][1])
            })
            .sum()
    };
    let escaped = |s: &FluidSolver| -> usize {
        s.positions
            .iter()
            .filter(|p| s.confinement[0].signed_distance(**p) > 0.0)
            .count()
    };

    let interval = 0.1;
    let steps = (end_time / interval).round() as usize;
    let mut rows = vec![(0.0, energy(&solver), solver.maximum_speed(), escaped(&solver) as f64)];

    let mut series = args.flag("vtk").map(|dir| {
        // the tank and the triangle are not made of particles, so nothing in
        // the frames can draw them: write them once, from the same constants
        // the level set was built from
        let mut points = Vec::new();
        let tank = [[0.0, 0.0], [TANK, 0.0], [TANK, TANK], [0.0, TANK]];
        let mut segments = vtk::closed_loop(&mut points, &tank);
        segments.extend(vtk::closed_loop(&mut points, &triangle));
        let solid_path = std::path::Path::new(dir).join("solid.vtu");
        vtk::write_lines(&solid_path, &points, &segments).expect("cannot write the outline");
        println!("wrote {} (the tank and the obstacle)", solid_path.display());
        Series::new(dir, "particles").expect("cannot open the series directory")
    });
    let write_frame = |series: &mut Series, s: &FluidSolver| {
        let speed: Vec<f64> = s
            .velocity
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .collect();
        series
            .add(
                s.time,
                &s.positions,
                &[
                    ("velocity", Field::Vector(&s.velocity)),
                    ("speed", Field::Scalar(&speed)),
                    ("pressure", Field::Scalar(&s.pressure)),
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
        rows.push((
            solver.time,
            energy(&solver),
            solver.maximum_speed(),
            escaped(&solver) as f64,
        ));
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
    println!("{:>6} {:>12} {:>8} {:>10} {:>8}", "t", "energy", "E/E0", "max speed", "escaped");
    let e0 = rows[0].1;
    for (t, e, speed, out) in rows.iter().step_by((rows.len() / 12).max(1)) {
        println!("{t:6.2} {e:12.6} {:8.4} {speed:10.4} {out:8.0}", e / e0);
    }

    if let Some(path) = path {
        let mut file = std::fs::File::create(&path).expect("cannot open the output file");
        writeln!(file, "time,mechanical_energy,max_speed,escaped").unwrap();
        for (t, e, speed, out) in &rows {
            writeln!(file, "{t:.17e},{e:.17e},{speed:.17e},{out:.17e}").unwrap();
        }
        println!("wrote {path}");
    }
}
