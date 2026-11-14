//! Total-Lagrangian SPH for an elastic solid.
//!
//! Mirrors `scion.discretization.sph.elastic`, which is the reference.
//!
//! The neighbour list, kernel values, volumes and correction matrix are all
//! built once in the **reference** configuration and never touched again.
//! Nothing is searched and nothing is rebuilt, which is also why the tensile
//! instability of an updated-Lagrangian SPH solid cannot arise here: the
//! kernel never sees the deformed configuration.
//!
//! `F` is *integrated* from `dF/dt` rather than recomputed from positions --
//! that is the quantity the Verlet splitting needs at the half step.
//!
//! `stress_PK1_B = P(F) B^T`: the transpose is load-bearing at a free
//! surface and invisible on a uniform lattice, which is most of a beam.

use crate::kernel::Kernel;
use crate::neighbors::{self, NeighborList};
use crate::operators;

#[derive(Clone, Copy, Debug)]
pub struct SaintVenantKirchhoffSolid {
    pub rho0: f64,
    pub youngs_modulus: f64,
    pub poisson: f64,
}

impl SaintVenantKirchhoffSolid {
    pub fn new(rho0: f64, youngs_modulus: f64, poisson: f64) -> Self {
        Self { rho0, youngs_modulus, poisson }
    }

    pub fn shear_modulus(&self) -> f64 {
        0.5 * self.youngs_modulus / (1.0 + self.poisson)
    }

    pub fn lame_lambda(&self) -> f64 {
        self.youngs_modulus * self.poisson
            / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
    }

    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / 3.0 / (1.0 - 2.0 * self.poisson)
    }

    /// Built from the **bulk** modulus; it sets the time step.
    pub fn c0(&self) -> f64 {
        (self.bulk_modulus() / self.rho0).sqrt()
    }

    /// `S = lambda tr(E) I + 2 G E` on the Green strain.
    pub fn stress_pk2(&self, f: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
        let ftf = [
            [f[0][0] * f[0][0] + f[1][0] * f[1][0], f[0][0] * f[0][1] + f[1][0] * f[1][1]],
            [f[0][1] * f[0][0] + f[1][1] * f[1][0], f[0][1] * f[0][1] + f[1][1] * f[1][1]],
        ];
        let e = [
            [0.5 * (ftf[0][0] - 1.0), 0.5 * ftf[0][1]],
            [0.5 * ftf[1][0], 0.5 * (ftf[1][1] - 1.0)],
        ];
        let trace = e[0][0] + e[1][1];
        let (lam, two_g) = (self.lame_lambda(), 2.0 * self.shear_modulus());
        [
            [lam * trace + two_g * e[0][0], two_g * e[0][1]],
            [two_g * e[1][0], lam * trace + two_g * e[1][1]],
        ]
    }

    pub fn stress_pk1(&self, f: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
        matmul(f, self.stress_pk2(f))
    }

    pub fn pair_damping(&self, strain_rate: f64, smoothing_length: f64) -> f64 {
        0.5 * self.rho0 * self.c0() * strain_rate * smoothing_length
    }
}

fn matmul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

fn transpose(a: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}

pub struct ElasticSolver {
    pub reference: Vec<[f64; 2]>,
    pub positions: Vec<[f64; 2]>,
    pub velocity: Vec<[f64; 2]>,
    pub force: Vec<[f64; 2]>,
    pub force_prior: Vec<[f64; 2]>,
    pub volumes: Vec<f64>,
    pub mass: Vec<f64>,
    pub deformation: Vec<[[f64; 2]; 2]>,
    pub deformation_rate: Vec<[[f64; 2]; 2]>,
    pub b: Vec<[[f64; 2]; 2]>,
    pub pairs: NeighborList,
    pub w: Vec<f64>,
    pub dw: Vec<f64>,
    pub weight: Vec<f64>,
    pub solid: SaintVenantKirchhoffSolid,
    pub kernel: Kernel,
    pub h: f64,
    pub spacing: f64,
    pub dissipation: f64,
    pub time: f64,
}

impl ElasticSolver {
    pub fn new(
        positions: Vec<[f64; 2]>,
        volumes: Vec<f64>,
        solid: SaintVenantKirchhoffSolid,
        kernel: Kernel,
        h_ratio: f64,
        spacing: f64,
    ) -> Self {
        let n = positions.len();
        let h = h_ratio * spacing;
        // the reference configuration is unbounded and non-periodic
        let (lower, upper) = extent(&positions, kernel.support(h));
        let pairs = neighbors::build(&positions, kernel.support(h), lower, upper, [false, false]);
        let w: Vec<f64> = pairs.r.iter().map(|&r| kernel.w(r, h)).collect();
        let dw: Vec<f64> = pairs.r.iter().map(|&r| kernel.dw_dr(r, h)).collect();
        let b = operators::linear_gradient_correction(&pairs, &volumes, &dw, f64::EPSILON.sqrt());
        let w0 = kernel.w(0.0, h);
        let weight = w.iter().map(|v| v / w0).collect();
        let mass = volumes.iter().map(|v| solid.rho0 * v).collect();
        let eye = [[1.0, 0.0], [0.0, 1.0]];
        Self {
            reference: positions.clone(),
            positions,
            velocity: vec![[0.0; 2]; n],
            force: vec![[0.0; 2]; n],
            force_prior: vec![[0.0; 2]; n],
            volumes,
            mass,
            deformation: vec![eye; n],
            deformation_rate: vec![[[0.0; 2]; 2]; n],
            b,
            pairs,
            w,
            dw,
            weight,
            solid,
            kernel,
            h,
            spacing,
            dissipation: 0.25,
            time: 0.0,
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn first_half(&mut self, dt: f64) {
        let mut stress = vec![[[0.0f64; 2]; 2]; self.len()];
        for p in 0..self.len() {
            for d in 0..2 {
                self.positions[p][d] += 0.5 * dt * self.velocity[p][d];
                for c in 0..2 {
                    self.deformation[p][d][c] += 0.5 * dt * self.deformation_rate[p][d][c];
                }
            }
            stress[p] = matmul(self.solid.stress_pk1(self.deformation[p]), transpose(self.b[p]));
        }

        self.force.iter_mut().for_each(|f| *f = [0.0, 0.0]);
        for k in 0..self.pairs.len() {
            let (i, j) = (self.pairs.i[k], self.pairs.j[k]);
            let mut jump_x = [0.0f64; 2];
            let mut jump_v = [0.0f64; 2];
            for d in 0..2 {
                jump_x[d] = self.positions[i][d] - self.positions[j][d];
                jump_v[d] = self.velocity[i][d] - self.velocity[j][d];
            }
            let scale = 2.0 / self.pairs.r[k];
            let strain_rate = scale * scale * (jump_x[0] * jump_v[0] + jump_x[1] * jump_v[1]);
            let damping = self.dissipation
                * self.weight[k]
                * self.solid.pair_damping(strain_rate, self.h);
            let mut total = [[0.0f64; 2]; 2];
            for a in 0..2 {
                for c in 0..2 {
                    total[a][c] = stress[i][a][c]
                        + stress[j][a][c]
                        + 0.5 * (self.deformation[i][a][c] + self.deformation[j][a][c]) * damping;
                }
            }
            let e = self.pairs.e[k];
            let traction = [
                (total[0][0] * e[0] + total[0][1] * e[1]) * self.dw[k],
                (total[1][0] * e[0] + total[1][1] * e[1]) * self.dw[k],
            ];
            let scale_i = self.mass[i] / self.solid.rho0 * self.volumes[j];
            let scale_j = self.mass[j] / self.solid.rho0 * self.volumes[i];
            for d in 0..2 {
                self.force[i][d] += scale_i * traction[d];
                self.force[j][d] -= scale_j * traction[d];
            }
        }
        for p in 0..self.len() {
            for d in 0..2 {
                self.velocity[p][d] +=
                    dt * (self.force_prior[p][d] + self.force[p][d]) / self.mass[p];
            }
        }
    }

    pub fn second_half(&mut self, dt: f64) {
        for p in 0..self.len() {
            for d in 0..2 {
                self.positions[p][d] += 0.5 * dt * self.velocity[p][d];
            }
        }
        let mut rate = vec![[[0.0f64; 2]; 2]; self.len()];
        for k in 0..self.pairs.len() {
            let (i, j) = (self.pairs.i[k], self.pairs.j[k]);
            let e = self.pairs.e[k];
            for a in 0..2 {
                let jump = self.velocity[i][a] - self.velocity[j][a];
                for c in 0..2 {
                    let term = -jump * self.dw[k] * e[c];
                    rate[i][a][c] += self.volumes[j] * term;
                    rate[j][a][c] += self.volumes[i] * term;
                }
            }
        }
        for p in 0..self.len() {
            self.deformation_rate[p] = matmul(rate[p], self.b[p]);
            for a in 0..2 {
                for c in 0..2 {
                    self.deformation[p][a][c] += 0.5 * dt * self.deformation_rate[p][a][c];
                }
            }
        }
    }

    /// Hold the listed particles at their reference position and at rest.
    pub fn constrain(&mut self, fixed: &[bool]) {
        for p in 0..self.len() {
            if fixed[p] {
                self.positions[p] = self.reference[p];
                self.velocity[p] = [0.0, 0.0];
            }
        }
    }

    pub fn time_step(&self, cfl: f64) -> f64 {
        let mut smallest = f64::INFINITY;
        for p in 0..self.len() {
            let ax = (self.force[p][0] + self.force_prior[p][0]) / self.mass[p];
            let ay = (self.force[p][1] + self.force_prior[p][1]) / self.mass[p];
            let acceleration = (ax * ax + ay * ay).sqrt();
            let speed = (self.velocity[p][0].powi(2) + self.velocity[p][1].powi(2)).sqrt();
            let by_force = (self.h / (acceleration + 1e-300)).sqrt();
            let by_sound = self.h / (self.solid.c0() + speed);
            smallest = smallest.min(by_force.min(by_sound));
        }
        cfl * smallest
    }

    pub fn step(&mut self, dt: f64, fixed: &[bool]) {
        self.first_half(dt);
        self.constrain(fixed);
        self.second_half(dt);
        self.time += dt;
    }

    pub fn kinetic_energy(&self) -> f64 {
        (0..self.len())
            .map(|p| 0.5 * self.mass[p] * (self.velocity[p][0].powi(2) + self.velocity[p][1].powi(2)))
            .sum()
    }

    pub fn strain_energy(&self) -> f64 {
        (0..self.len())
            .map(|p| {
                let f = self.deformation[p];
                let ftf = [
                    [f[0][0] * f[0][0] + f[1][0] * f[1][0], f[0][0] * f[0][1] + f[1][0] * f[1][1]],
                    [f[0][1] * f[0][0] + f[1][1] * f[1][0], f[0][1] * f[0][1] + f[1][1] * f[1][1]],
                ];
                let e = [
                    [0.5 * (ftf[0][0] - 1.0), 0.5 * ftf[0][1]],
                    [0.5 * ftf[1][0], 0.5 * (ftf[1][1] - 1.0)],
                ];
                let s = self.solid.stress_pk2(f);
                let density = 0.5
                    * (s[0][0] * e[0][0] + s[0][1] * e[0][1] + s[1][0] * e[1][0] + s[1][1] * e[1][1]);
                self.volumes[p] * density
            })
            .sum()
    }

    /// Shepard interpolation at a point of the **reference** configuration,
    /// which is what makes the observer follow the material.
    ///
    /// `h` is the *observer's* smoothing length, which an observer body
    /// carries separately from the solid's. The beam case uses 1.15 dp
    /// against the body's 1.3 dp, and the tip reading depends on it: 1.15
    /// reproduces SPHinXsys's recorded 0.198877, 1.3 gives 0.198772.
    pub fn probe_with(&self, point: [f64; 2], field: &[f64], h: f64) -> f64 {
        let support = self.kernel.support(h);
        let (mut num, mut den) = (0.0, 0.0);
        for p in 0..self.len() {
            let r = ((self.reference[p][0] - point[0]).powi(2)
                + (self.reference[p][1] - point[1]).powi(2))
            .sqrt();
            if r < support {
                let weight = self.kernel.w(r, h) * self.volumes[p];
                num += weight * field[p];
                den += weight;
            }
        }
        if den > 0.0 {
            num / den
        } else {
            f64::NAN
        }
    }

    /// The same probe at the body's own smoothing length.
    pub fn probe(&self, point: [f64; 2], field: &[f64]) -> f64 {
        self.probe_with(point, field, self.h)
    }
}

fn extent(points: &[[f64; 2]], pad: f64) -> ([f64; 2], [f64; 2]) {
    let mut lower = [f64::INFINITY; 2];
    let mut upper = [f64::NEG_INFINITY; 2];
    for p in points {
        for d in 0..2 {
            lower[d] = lower[d].min(p[d]);
            upper[d] = upper[d].max(p[d]);
        }
    }
    for d in 0..2 {
        lower[d] -= pad;
        upper[d] += pad;
    }
    (lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(nx: usize, ny: usize, dp: f64) -> (Vec<[f64; 2]>, Vec<f64>) {
        let mut x = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                x.push([(i as f64 + 0.5) * dp, (j as f64 + 0.5) * dp]);
            }
        }
        let n = x.len();
        (x, vec![dp * dp; n])
    }

    fn solver() -> ElasticSolver {
        let dp = 0.002;
        let (x, v) = block(40, 10, dp);
        ElasticSolver::new(
            x,
            v,
            SaintVenantKirchhoffSolid::new(1.0e3, 2.0e6, 0.3975),
            Kernel::WendlandC2,
            1.3,
            dp,
        )
    }

    /// At small strain the Saint-Venant-Kirchhoff law must be Hooke's law.
    /// Catches a factor on either Lame constant, which no qualitative check
    /// would.
    #[test]
    fn reduces_to_linear_elasticity_at_small_strain() {
        let solid = SaintVenantKirchhoffSolid::new(1.0e3, 2.0e6, 0.3975);
        let f = [[1.0 + 1e-6, 0.0], [0.0, 1.0]];
        let p = solid.stress_pk1(f);
        let expected_xx = (solid.lame_lambda() + 2.0 * solid.shear_modulus()) * 1e-6;
        let expected_yy = solid.lame_lambda() * 1e-6;
        assert!((p[0][0] / expected_xx - 1.0).abs() < 1e-5);
        assert!((p[1][1] / expected_yy - 1.0).abs() < 1e-5);
    }

    #[test]
    fn undeformed_solid_is_stress_free() {
        let solid = SaintVenantKirchhoffSolid::new(1.0e3, 2.0e6, 0.3975);
        let p = solid.stress_pk1([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(p, [[0.0, 0.0], [0.0, 0.0]]);
    }

    /// A body at rest in its reference configuration must stay there. The
    /// correction matrix is what makes this exact at a free surface.
    #[test]
    fn an_undeformed_body_at_rest_does_not_move() {
        let mut s = solver();
        let fixed = vec![false; s.len()];
        for _ in 0..20 {
            s.step(1e-6, &fixed);
        }
        let worst = s
            .velocity
            .iter()
            .map(|v| v[0].abs().max(v[1].abs()))
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-12, "velocity {worst}");
    }

    /// A uniform velocity is a rigid translation: no strain may develop.
    #[test]
    fn rigid_translation_builds_no_strain() {
        let mut s = solver();
        let fixed = vec![false; s.len()];
        s.velocity.iter_mut().for_each(|v| *v = [1.0, -0.5]);
        for _ in 0..20 {
            s.step(1e-6, &fixed);
        }
        let worst = s
            .deformation
            .iter()
            .map(|f| {
                (f[0][0] - 1.0)
                    .abs()
                    .max(f[1][1] - 1.0)
                    .max(f[0][1].abs())
                    .max(f[1][0].abs())
            })
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-12, "deformation gradient drifted by {worst}");
    }
}
