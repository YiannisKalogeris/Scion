//! Diffusion of one species on a fixed particle set.
//!
//! The anisotropic inter-particle coefficient is the directional projection
//! `D_ij = 1 / (e . D^-1 e)`, which SPHinXsys computes as `1/|T e|^2` with
//! `T` the inverse Cholesky factor of `D`. The same number, and this form
//! says what it is: the harmonic projection of the tensor on the line joining
//! the two particles, equal to `d` exactly when `D = d I`.
//!
//! `step` is Heun's method, written the way SPHinXsys writes it: a forward
//! Euler stage, a second rate from the intermediate state, and the average of
//! the two states.

use crate::kernel::Kernel;
use crate::neighbors::{self, NeighborList};
use crate::operators;

/// `D = d I + d_bias (b (x) b)`.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalDiffusivity {
    pub d_coeff: f64,
    pub bias_coeff: f64,
    pub bias_direction: [f64; 2],
}

impl DirectionalDiffusivity {
    pub fn isotropic(d_coeff: f64) -> Self {
        Self { d_coeff, bias_coeff: 0.0, bias_direction: [1.0, 0.0] }
    }

    pub fn tensor(&self) -> [[f64; 2]; 2] {
        let b = self.bias_direction;
        let norm = (b[0] * b[0] + b[1] * b[1]).sqrt();
        let b = if norm > 0.0 { [b[0] / norm, b[1] / norm] } else { b };
        [
            [self.d_coeff + self.bias_coeff * b[0] * b[0], self.bias_coeff * b[0] * b[1]],
            [self.bias_coeff * b[0] * b[1], self.d_coeff + self.bias_coeff * b[1] * b[1]],
        ]
    }

    pub fn reference(&self) -> f64 {
        self.d_coeff.max(self.d_coeff + self.bias_coeff)
    }

    /// `1 / (e . D^-1 e)`.
    pub fn inter_particle(&self, e: [f64; 2]) -> f64 {
        let m = self.tensor();
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let inv = [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]];
        let q = e[0] * (inv[0][0] * e[0] + inv[0][1] * e[1])
            + e[1] * (inv[1][0] * e[0] + inv[1][1] * e[1]);
        1.0 / q
    }

    /// SPHinXsys's rule, `0.5 h^2 / (D_ref * dim)`, with `dim = 2` here.
    pub fn time_step(&self, h: f64) -> f64 {
        0.5 * h * h / self.reference() / 2.0
    }
}

pub struct DiffusionSolver {
    pub positions: Vec<[f64; 2]>,
    pub volumes: Vec<f64>,
    pub pairs: NeighborList,
    pub w: Vec<f64>,
    pub dw: Vec<f64>,
    pub b: Option<Vec<[[f64; 2]; 2]>>,
    pub d_ij: Vec<f64>,
    /// the edge weights, built once from the positions
    pub conductance: Vec<f64>,
    pub kernel: Kernel,
    pub h: f64,
    pub lower: [f64; 2],
    pub upper: [f64; 2],
    pub periodic: [bool; 2],
    pub diffusivity: DirectionalDiffusivity,
    scratch: Vec<f64>,
}

impl DiffusionSolver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        positions: Vec<[f64; 2]>,
        volumes: Vec<f64>,
        diffusivity: DirectionalDiffusivity,
        kernel: Kernel,
        h: f64,
        lower: [f64; 2],
        upper: [f64; 2],
        periodic: [bool; 2],
        correction: bool,
        correction_eps: f64,
    ) -> Self {
        let pairs = neighbors::build(&positions, kernel.support(h), lower, upper, periodic);
        let w: Vec<f64> = pairs.r.iter().map(|&r| kernel.w(r, h)).collect();
        let dw: Vec<f64> = pairs.r.iter().map(|&r| kernel.dw_dr(r, h)).collect();
        let b = if correction {
            Some(operators::linear_gradient_correction(&pairs, &volumes, &dw, correction_eps))
        } else {
            None
        };
        let d_ij: Vec<f64> = pairs.e.iter().map(|&e| diffusivity.inter_particle(e)).collect();
        let conductance = operators::pair_conductance(&pairs, &dw, &d_ij, b.as_deref());
        let n = positions.len();
        Self {
            positions,
            volumes,
            pairs,
            w,
            dw,
            b,
            d_ij,
            conductance,
            kernel,
            h,
            lower,
            upper,
            periodic,
            diffusivity,
            scratch: vec![0.0; n],
        }
    }

    pub fn rate(&self, phi: &[f64], out: &mut [f64]) {
        operators::laplacian_rate(&self.pairs, phi, &self.volumes, &self.conductance, out);
    }

    /// One Heun (RK2) step, in place.
    pub fn step(&mut self, phi: &mut [f64], dt: f64) {
        let n = phi.len();
        let mut rate = std::mem::take(&mut self.scratch);
        self.rate(phi, &mut rate);
        let mut first: Vec<f64> = (0..n).map(|k| phi[k] + dt * rate[k]).collect();
        self.rate(&first, &mut rate);
        for k in 0..n {
            first[k] += dt * rate[k];
            phi[k] = 0.5 * (phi[k] + first[k]);
        }
        self.scratch = rate;
    }

    pub fn time_step(&self) -> f64 {
        self.diffusivity.time_step(self.h)
    }

    /// Shepard-normalised interpolation at arbitrary points; brute force,
    /// because the observer sets this exists for are a handful of points.
    pub fn observe(&self, points: &[[f64; 2]], phi: &[f64]) -> Vec<f64> {
        let support = self.kernel.support(self.h);
        let length = [self.upper[0] - self.lower[0], self.upper[1] - self.lower[1]];
        points
            .iter()
            .map(|p| {
                let (mut num, mut den) = (0.0, 0.0);
                for (k, x) in self.positions.iter().enumerate() {
                    let mut d = [x[0] - p[0], x[1] - p[1]];
                    for c in 0..2 {
                        if self.periodic[c] {
                            d[c] -= length[c] * (d[c] / length[c]).round();
                        }
                    }
                    let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
                    if r < support {
                        let weight = self.kernel.w(r, self.h) * self.volumes[k];
                        num += weight * phi[k];
                        den += weight;
                    }
                }
                if den > 0.0 {
                    num / den
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
}

/// Cell-centred particles filling a box, as SPHinXsys's lattice generator
/// places them: `lower + (i + 1/2) dp`, never on the face.
pub fn lattice(lower: [f64; 2], upper: [f64; 2], dp: f64) -> (Vec<[f64; 2]>, Vec<f64>) {
    let nx = ((upper[0] - lower[0]) / dp).round() as usize;
    let ny = ((upper[1] - lower[1]) / dp).round() as usize;
    let mut x = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            x.push([
                lower[0] + (i as f64 + 0.5) * dp,
                lower[1] + (j as f64 + 0.5) * dp,
            ]);
        }
    }
    let n = x.len();
    (x, vec![dp * dp; n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_projection_is_the_coefficient_itself() {
        let d = DirectionalDiffusivity::isotropic(1e-4);
        for k in 0..16 {
            let a = k as f64 * 0.3;
            let e = [a.cos(), a.sin()];
            assert!((d.inter_particle(e) - 1e-4).abs() < 1e-19);
        }
    }

    /// A biased tensor must read its two principal values along its own axes,
    /// which is the check that catches an un-inverted or transposed tensor.
    #[test]
    fn directional_projection_reads_the_principal_values() {
        let d = DirectionalDiffusivity {
            d_coeff: 1e-4,
            bias_coeff: 3e-4,
            bias_direction: [(0.5f64).cos(), (0.5f64).sin()],
        };
        let along = [(0.5f64).cos(), (0.5f64).sin()];
        let across = [-(0.5f64).sin(), (0.5f64).cos()];
        assert!((d.inter_particle(along) - 4e-4).abs() < 1e-18);
        assert!((d.inter_particle(across) - 1e-4).abs() < 1e-18);
    }

    #[test]
    fn a_uniform_field_does_not_move() {
        let dp = 0.02;
        let (x, vol) = lattice([0.0, 0.0], [2.0, 0.4], dp);
        let mut s = DiffusionSolver::new(
            x.clone(),
            vol,
            DirectionalDiffusivity::isotropic(1e-4),
            Kernel::WendlandC2,
            1.3 * dp,
            [0.0, 0.0],
            [2.0, 0.4],
            [true, false],
            true,
            0.0,
        );
        let mut phi = vec![0.7; x.len()];
        let dt = s.time_step();
        for _ in 0..5 {
            s.step(&mut phi, dt);
        }
        let worst = phi.iter().map(|v| (v - 0.7).abs()).fold(0.0f64, f64::max);
        assert!(worst < 1e-15, "uniform field moved by {worst}");
    }
}
