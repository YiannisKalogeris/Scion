//! SPHinXsys example 13, the 2-D oscillating beam, in Rust.
//!
//!     cargo run -p scion-sph --release --example oscillating_beam -- [end] [out.csv]
//!     cargo run -p scion-sph --release --example oscillating_beam -- 1.0 \
//!         --vtk ../../examples/sph/results/oscillating_beam_vtk
//!
//! `--vtk <dir>` writes the same `particles.pvd` series the NumPy run writes,
//! with the same array names -- including `clamped`, since the boundary
//! condition is invisible in the positions -- and the undeformed outline
//! beside it as `reference.vtu`, which a total-Lagrangian run otherwise never
//! shows: every frame carries the *deformed* configuration.
//!
//! A cantilever clamped into a thicker base, released from the first
//! Euler-Bernoulli mode. Total Lagrangian, so the neighbour list and the
//! correction matrix are built once and never touched. The matching NumPy
//! run is `python -m examples.sph.oscillating_beam`, and that is the
//! reference.

use scion_sph::elastic::{ElasticSolver, SaintVenantKirchhoffSolid};
use scion_sph::vtk::{self, Field, Series};
use scion_sph::{Args, Kernel};
use std::io::Write;

const LENGTH: f64 = 0.2;
const THICKNESS: f64 = 0.02;
const INSERT: f64 = 0.06;
const RHO0: f64 = 1.0e3;
const YOUNGS: f64 = 2.0e6;
const POISSON: f64 = 0.3975;
const VELOCITY_FRACTION: f64 = 0.05;
const KL: f64 = 1.875;
/// the observer body carries its own smoothing length in the reference case
/// (`defineAdaptationRatios(1.15, 2.0)`), and the tip reading depends on it
const OBSERVER_H_RATIO: f64 = 1.15;

fn mode_shape(x_hat: f64) -> f64 {
    let m = KL.sin() + KL.sinh();
    let n = KL.cos() + KL.cosh();
    let q = 2.0 * (KL.cos() * KL.sinh() - KL.sin() * KL.cosh());
    (m * ((KL * x_hat).cos() - (KL * x_hat).cosh())
        - n * ((KL * x_hat).sin() - (KL * x_hat).sinh()))
        / q
}

fn in_box(p: [f64; 2], lower: [f64; 2], upper: [f64; 2]) -> bool {
    p[0] >= lower[0] && p[0] <= upper[0] && p[1] >= lower[1] && p[1] <= upper[1]
}

fn beam_region(p: [f64; 2]) -> bool {
    in_box(p, [-INSERT, -0.5 * THICKNESS], [LENGTH, 0.5 * THICKNESS])
}

fn base_region(p: [f64; 2], base_width: f64) -> bool {
    in_box(
        p,
        [-INSERT - base_width, -0.5 * THICKNESS - base_width],
        [0.0, 0.5 * THICKNESS + base_width],
    )
}

fn main() {
    let args = Args::parse();
    let end_time: f64 = args.parsed(0, 1.0);
    let path = args.positional(1).map(str::to_string);

    let spacing = THICKNESS / 10.0;
    let base_width = 4.0 * spacing;
    let lower = [-INSERT - base_width, -0.5 * LENGTH];
    let upper = [LENGTH + 3.0 * base_width, 0.5 * LENGTH];

    let nx = ((upper[0] - lower[0]) / spacing).round() as usize;
    let ny = ((upper[1] - lower[1]) / spacing).round() as usize;
    let mut positions = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            let p = [
                lower[0] + (i as f64 + 0.5) * spacing,
                lower[1] + (j as f64 + 0.5) * spacing,
            ];
            if beam_region(p) || base_region(p, base_width) {
                positions.push(p);
            }
        }
    }
    let volumes = vec![spacing * spacing; positions.len()];
    let fixed: Vec<bool> = positions
        .iter()
        .map(|&p| base_region(p, base_width) && !beam_region(p))
        .collect();

    let solid = SaintVenantKirchhoffSolid::new(RHO0, YOUNGS, POISSON);
    let mut solver = ElasticSolver::new(
        positions.clone(),
        volumes,
        solid,
        Kernel::WendlandC2,
        1.3,
        spacing,
    );
    for (p, pos) in positions.iter().enumerate() {
        if pos[0] > 0.0 {
            solver.velocity[p][1] = VELOCITY_FRACTION * solid.c0() * mode_shape(pos[0] / LENGTH);
        }
    }
    println!(
        "{} particles, {} pairs, {} clamped, c0 = {:.4}",
        solver.len(),
        solver.pairs.len(),
        fixed.iter().filter(|f| **f).count(),
        solid.c0()
    );

    let tip = [LENGTH, 0.0];
    let interval = 0.005;
    let mut rows = vec![(0.0, 0.0, solver.kinetic_energy(), solver.strain_energy())];

    // the clamp is the collar of the base block minus the beam's own
    // cross-section, so the beam flexes inside the insert -- which is why the
    // measured period sits about a percent above Euler-Bernoulli. Nothing in
    // the positions shows that, so it is written as a field.
    let clamped: Vec<f64> = fixed.iter().map(|f| if *f { 1.0 } else { 0.0 }).collect();
    let reference = solver.reference.clone();
    let mut series = args.flag("vtk").map(|dir| {
        let base = [
            [-INSERT - base_width, -0.5 * THICKNESS - base_width],
            [0.0, -0.5 * THICKNESS - base_width],
            [0.0, 0.5 * THICKNESS + base_width],
            [-INSERT - base_width, 0.5 * THICKNESS + base_width],
        ];
        let beam = [
            [-INSERT, -0.5 * THICKNESS],
            [LENGTH, -0.5 * THICKNESS],
            [LENGTH, 0.5 * THICKNESS],
            [-INSERT, 0.5 * THICKNESS],
        ];
        let mut points = Vec::new();
        let mut segments = vtk::closed_loop(&mut points, &base);
        segments.extend(vtk::closed_loop(&mut points, &beam));
        let outline = std::path::Path::new(dir).join("reference.vtu");
        vtk::write_lines(&outline, &points, &segments).expect("cannot write the outline");
        println!("wrote {} (the undeformed base and beam)", outline.display());
        Series::new(dir, "particles").expect("cannot open the series directory")
    });
    let write_frame = |series: &mut Series, time: f64, x: &[[f64; 2]], v: &[[f64; 2]]| {
        let u: Vec<[f64; 2]> = (0..x.len())
            .map(|p| [x[p][0] - reference[p][0], x[p][1] - reference[p][1]])
            .collect();
        let magnitude: Vec<f64> = u.iter().map(|d| (d[0] * d[0] + d[1] * d[1]).sqrt()).collect();
        series
            .add(
                time,
                x,
                &[
                    ("displacement", Field::Vector(&u)),
                    ("magnitude", Field::Scalar(&magnitude)),
                    ("velocity", Field::Vector(v)),
                    ("clamped", Field::Scalar(&clamped)),
                ],
            )
            .expect("cannot write a frame");
    };
    if let Some(series) = series.as_mut() {
        write_frame(series, 0.0, &solver.positions, &solver.velocity);
    }

    let start = std::time::Instant::now();
    let mut next_sample = interval;
    let mut steps = 0usize;
    while solver.time < end_time {
        let dt = solver.time_step(0.6);
        solver.step(dt, &fixed);
        steps += 1;
        if solver.time >= next_sample {
            let dy: Vec<f64> = (0..solver.len())
                .map(|p| solver.positions[p][1] - solver.reference[p][1])
                .collect();
            rows.push((
                solver.time,
                solver.probe_with(tip, &dy, OBSERVER_H_RATIO * spacing),
                solver.kinetic_energy(),
                solver.strain_energy(),
            ));
            if let Some(series) = series.as_mut() {
                write_frame(series, solver.time, &solver.positions, &solver.velocity);
            }
            next_sample += interval;
        }
    }
    if let Some(series) = series.as_ref() {
        println!(
            "wrote {} frames to {}",
            series.len(),
            series.collection_path().display()
        );
    }
    println!("{steps} steps, {:.1} s of wall time", start.elapsed().as_secs_f64());

    // period from the zero crossings of the tip displacement
    let mut crossings = Vec::new();
    for k in 0..rows.len() - 1 {
        let (t0, y0) = (rows[k].0, rows[k].1);
        let (t1, y1) = (rows[k + 1].0, rows[k + 1].1);
        if (y0 < 0.0) != (y1 < 0.0) {
            crossings.push(t0 + (t1 - t0) * (-y0 / (y1 - y0)));
        }
    }
    if crossings.len() >= 3 {
        let gaps: f64 = crossings.windows(2).map(|w| w[1] - w[0]).sum();
        let period = 2.0 * gaps / (crossings.len() - 1) as f64;
        println!("measured period {period:.5} s (frequency {:.4} Hz)", 1.0 / period);
    }
    let peak = rows.iter().map(|r| r.1.abs()).fold(0.0f64, f64::max);
    println!("peak tip displacement {peak:.5} m");

    if let Some(path) = path {
        let mut file = std::fs::File::create(&path).expect("cannot open the output file");
        writeln!(file, "time,tip_dy,kinetic_energy,strain_energy").unwrap();
        for (t, dy, ke, se) in &rows {
            writeln!(file, "{t:.17e},{dy:.17e},{ke:.17e},{se:.17e}").unwrap();
        }
        println!("wrote {path}");
    }
}
