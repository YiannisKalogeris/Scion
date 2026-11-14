//! Weakly compressible SPH: the dual-criteria scheme, in Rust.
//!
//! Mirrors `scion.discretization.sph.fluid`, which is the reference. The
//! structure is the same and so are the conventions:
//!
//! * an advection step carries the neighbour list, the density summation,
//!   the viscous force and the transport-velocity shift; several acoustic
//!   steps inside it carry pressure and continuity on a **frozen** neighbour
//!   list, which is SPHinXsys's scheme and not an approximation introduced
//!   here;
//! * each acoustic step is two halves straddling the position update, which
//!   is what makes it second order;
//! * the Riemann density dissipation is on in the first half and the
//!   pressure dissipation off in the second, as the Taylor-Green case runs.
//!
//! Two dimensions only, as the rest of this crate.

use crate::kernel::Kernel;
use crate::levelset::LevelSet;
use crate::neighbors::{self, NeighborList};

#[derive(Clone, Copy, Debug)]
pub struct WeaklyCompressibleFluid {
    pub rho0: f64,
    pub c0: f64,
    pub viscosity: f64,
}

impl WeaklyCompressibleFluid {
    pub fn new(rho0: f64, c0: f64, viscosity: f64) -> Self {
        Self { rho0, c0, viscosity }
    }

    /// `rho0 c0^2`, the reference pressure of the linear equation of state.
    pub fn p0(&self) -> f64 {
        self.rho0 * self.c0 * self.c0
    }

    /// `p = c0^2 (rho - rho0)`. The linear form SPHinXsys uses, not Tait.
    pub fn pressure(&self, rho: f64) -> f64 {
        self.p0() * (rho / self.rho0 - 1.0)
    }
}

/// The kernel sum over an infinite regular lattice.
///
/// The density summation divides by this so an undisturbed lattice reads
/// exactly `rho0`. Computed from the lattice rather than from a continuum
/// integral, which would give `1/dp^2` and be wrong by the kernel's own
/// discretisation -- about 1% at `h = 1.3 dp`.
pub fn lattice_number_density(kernel: Kernel, h: f64, spacing: f64) -> f64 {
    let depth = (kernel.support(h) / spacing) as i32 + 1;
    let mut sigma = 0.0;
    for j in -depth..=depth {
        for i in -depth..=depth {
            let r = ((i as f64 * spacing).powi(2) + (j as f64 * spacing).powi(2)).sqrt();
            if r < kernel.support(h) {
                sigma += kernel.w(r, h);
            }
        }
    }
    sigma
}

pub struct FluidSolver {
    pub positions: Vec<[f64; 2]>,
    pub velocity: Vec<[f64; 2]>,
    pub density: Vec<f64>,
    pub pressure: Vec<f64>,
    pub mass: Vec<f64>,
    pub volume: Vec<f64>,
    pub drho_dt: Vec<f64>,
    pub force: Vec<[f64; 2]>,
    pub force_prior: Vec<[f64; 2]>,
    pub gravity: [f64; 2],
    pub fluid: WeaklyCompressibleFluid,
    pub kernel: Kernel,
    pub h: f64,
    pub spacing: f64,
    pub sigma0: f64,
    pub lower: [f64; 2],
    pub upper: [f64; 2],
    pub periodic: [bool; 2],
    pub riemann_density: bool,
    pub riemann_pressure: bool,
    pub riemann_limiter: f64,
    pub transport_coefficient: f64,
    pub transport_limiter: f64,
    /// static confinement: level sets whose interior is solid, replacing
    /// wall particles entirely
    pub confinement: Vec<LevelSet>,
    /// a free surface cannot hold a density below the reference one, and a
    /// kernel truncated by it reads exactly that
    pub free_surface: bool,
    pub time: f64,
    pub pairs: NeighborList,
    pub w: Vec<f64>,
    pub dw: Vec<f64>,
}

impl FluidSolver {
    pub fn new(
        positions: Vec<[f64; 2]>,
        spacing: f64,
        fluid: WeaklyCompressibleFluid,
        kernel: Kernel,
        h_ratio: f64,
        lower: [f64; 2],
        upper: [f64; 2],
        periodic: [bool; 2],
    ) -> Self {
        let n = positions.len();
        let h = h_ratio * spacing;
        let sigma0 = lattice_number_density(kernel, h, spacing);
        let mass = vec![fluid.rho0 * spacing * spacing; n];
        let mut solver = Self {
            positions,
            velocity: vec![[0.0; 2]; n],
            density: vec![fluid.rho0; n],
            pressure: vec![0.0; n],
            volume: mass.iter().map(|m| m / fluid.rho0).collect(),
            mass,
            drho_dt: vec![0.0; n],
            force: vec![[0.0; 2]; n],
            force_prior: vec![[0.0; 2]; n],
            gravity: [0.0; 2],
            fluid,
            kernel,
            h,
            spacing,
            sigma0,
            lower,
            upper,
            periodic,
            riemann_density: true,
            riemann_pressure: false,
            riemann_limiter: 3.0,
            transport_coefficient: 0.2,
            transport_limiter: 100.0,
            confinement: Vec::new(),
            free_surface: false,
            time: 0.0,
            pairs: NeighborList {
                i: Vec::new(),
                j: Vec::new(),
                dx: Vec::new(),
                r: Vec::new(),
                e: Vec::new(),
                n_particles: n,
            },
            w: Vec::new(),
            dw: Vec::new(),
        };
        solver.update_configuration();
        solver
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn update_configuration(&mut self) {
        neighbors::wrap(&mut self.positions, self.lower, self.upper, self.periodic);
        self.pairs = neighbors::build(
            &self.positions,
            self.kernel.support(self.h),
            self.lower,
            self.upper,
            self.periodic,
        );
        self.w = self.pairs.r.iter().map(|&r| self.kernel.w(r, self.h)).collect();
        self.dw = self.pairs.r.iter().map(|&r| self.kernel.dw_dr(r, self.h)).collect();
    }

    pub fn density_summation(&mut self) {
        let w0 = self.kernel.w(0.0, self.h);
        let mut sigma = vec![w0; self.len()];
        for k in 0..self.pairs.len() {
            sigma[self.pairs.i[k]] += self.w[k];
            sigma[self.pairs.j[k]] += self.w[k];
        }
        for level_set in &self.confinement {
            for p in 0..self.positions.len() {
                let reference_volume = self.mass[p] / self.fluid.rho0;
                sigma[p] += level_set.kernel_integral_at(self.positions[p]) / reference_volume;
            }
        }
        for p in 0..self.len() {
            self.density[p] = sigma[p] * self.fluid.rho0 / self.sigma0;
            if self.free_surface {
                self.density[p] = self.density[p].max(self.fluid.rho0);
            }
            self.volume[p] = self.mass[p] / self.density[p];
        }
    }

    /// Morris-type viscous force. The `0.01 h` regularises the `1/r`.
    pub fn viscous_force(&self) -> Vec<[f64; 2]> {
        let mut force = vec![[0.0f64; 2]; self.len()];
        if self.fluid.viscosity == 0.0 {
            return force;
        }
        for k in 0..self.pairs.len() {
            let (i, j) = (self.pairs.i[k], self.pairs.j[k]);
            let weight = 2.0 * self.fluid.viscosity * self.dw[k]
                / (self.pairs.r[k] + 0.01 * self.h)
                * self.volume[i]
                * self.volume[j];
            for d in 0..2 {
                let term = weight * (self.velocity[i][d] - self.velocity[j][d]);
                force[i][d] += term;
                force[j][d] -= term;
            }
        }
        force
    }

    pub fn transport_velocity_shift(&mut self) {
        let mut inconsistency = vec![[0.0f64; 2]; self.len()];
        for k in 0..self.pairs.len() {
            let (i, j) = (self.pairs.i[k], self.pairs.j[k]);
            for d in 0..2 {
                let gw = 2.0 * self.dw[k] * self.pairs.e[k][d];
                inconsistency[i][d] -= gw * self.volume[j];
                inconsistency[j][d] += gw * self.volume[i];
            }
        }
        for p in 0..self.len() {
            let squared = inconsistency[p][0].powi(2) + inconsistency[p][1].powi(2);
            let limiter = (self.transport_limiter * self.h * self.h * squared).min(1.0);
            for d in 0..2 {
                self.positions[p][d] +=
                    self.transport_coefficient * self.h * self.h * limiter * inconsistency[p][d];
            }
        }
    }

    pub fn pressure_step(&mut self, dt: f64) {
        for p in 0..self.len() {
            self.density[p] += 0.5 * dt * self.drho_dt[p];
            self.pressure[p] = self.fluid.pressure(self.density[p]);
            for d in 0..2 {
                self.positions[p][d] += 0.5 * dt * self.velocity[p][d];
            }
        }
        let mut rate = vec![0.0; self.len()];
        for k in 0..self.pairs.len() {
            let (i, j) = (self.pairs.i[k], self.pairs.j[k]);
            let p_sum = self.pressure[i] + self.pressure[j];
            let scale = -p_sum * self.dw[k] * self.volume[i] * self.volume[j];
            for d in 0..2 {
                let term = scale * self.pairs.e[k][d];
                self.force[i][d] += term;
                self.force[j][d] -= term;
            }
            if self.riemann_density {
                let jump = (self.pressure[i] - self.pressure[j])
                    / (self.fluid.rho0 * self.fluid.c0)
                    * self.dw[k];
                rate[i] += jump * self.volume[j];
                rate[j] -= jump * self.volume[i];
            }
        }
        for level_set in &self.confinement {
            for p in 0..self.positions.len() {
                // the solid as a mirror at equal pressure
                let g = level_set.kernel_gradient_at(self.positions[p]);
                let scale = 2.0 * self.mass[p] * self.pressure[p] / self.density[p];
                self.force[p][0] -= scale * g[0];
                self.force[p][1] -= scale * g[1];
            }
        }
        for p in 0..self.len() {
            self.drho_dt[p] = if self.riemann_density {
                rate[p] * self.density[p]
            } else {
                0.0
            };
            for d in 0..2 {
                self.velocity[p][d] +=
                    dt * (self.force_prior[p][d] + self.force[p][d]) / self.mass[p];
            }
        }
    }

    pub fn density_step(&mut self, dt: f64) {
        for p in 0..self.len() {
            for d in 0..2 {
                self.positions[p][d] += 0.5 * dt * self.velocity[p][d];
            }
        }
        let mut rate = vec![0.0; self.len()];
        let mut dissipation = vec![[0.0f64; 2]; self.len()];
        let rho0c0 = self.fluid.rho0 * self.fluid.c0;
        for k in 0..self.pairs.len() {
            let (i, j) = (self.pairs.i[k], self.pairs.j[k]);
            let mut u_jump = 0.0;
            for d in 0..2 {
                u_jump += (self.velocity[i][d] - self.velocity[j][d]) * self.pairs.e[k][d];
            }
            let term = u_jump * self.dw[k];
            rate[i] += term * self.volume[j];
            rate[j] += term * self.volume[i];
            if self.riemann_pressure {
                let limited = (self.riemann_limiter * u_jump.max(0.0) / self.fluid.c0).min(1.0);
                let scale =
                    rho0c0 * u_jump * limited * self.dw[k] * self.volume[i] * self.volume[j];
                for d in 0..2 {
                    let f = scale * self.pairs.e[k][d];
                    dissipation[i][d] += f;
                    dissipation[j][d] -= f;
                }
            }
        }
        for p in 0..self.len() {
            self.drho_dt[p] += rate[p] * self.density[p];
        }
        for level_set in &self.confinement {
            for p in 0..self.positions.len() {
                // the same mirror with the velocity reversed, which is no slip
                let g = level_set.kernel_gradient_at(self.positions[p]);
                self.drho_dt[p] += self.density[p] * 2.0
                    * (self.velocity[p][0] * g[0] + self.velocity[p][1] * g[1]);
            }
        }
        for p in 0..self.len() {
            self.density[p] += 0.5 * dt * self.drho_dt[p];
            self.force[p] = dissipation[p];
        }
        self.bound_to_surface();
    }

    /// Push back any particle that has entered the solid.
    ///
    /// The confinement force is strong but not infinite, so a particle can
    /// cross during a step; left there it reads a kernel integral above one
    /// half, is pushed harder, and the error feeds itself.
    pub fn bound_to_surface(&mut self) {
        let margin = 0.5 * self.spacing;
        for level_set in &self.confinement {
            for p in 0..self.positions.len() {
                let phi = level_set.signed_distance(self.positions[p]);
                if phi > -margin {
                    let n = level_set.normal(self.positions[p]);
                    self.positions[p][0] -= (phi + margin) * n[0];
                    self.positions[p][1] -= (phi + margin) * n[1];
                }
            }
        }
    }

    pub fn acoustic_time_step(&self, cfl: f64) -> f64 {
        let fastest = self
            .velocity
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0f64, f64::max);
        cfl * self.h / (self.fluid.c0 + fastest)
    }

    pub fn advection_time_step(&self, reference_speed: f64, cfl: f64) -> f64 {
        let mut worst: f64 = 0.0;
        for p in 0..self.len() {
            let speed_sq = self.velocity[p][0].powi(2) + self.velocity[p][1].powi(2);
            let fx = self.force_prior[p][0] + self.force[p][0];
            let fy = self.force_prior[p][1] + self.force[p][1];
            let scale = 4.0 * self.h * (fx * fx + fy * fy).sqrt() / self.mass[p];
            worst = worst.max(speed_sq.max(scale));
        }
        let viscous = self.fluid.viscosity / self.fluid.rho0 / self.h;
        cfl * self.h / worst.sqrt().max(reference_speed).max(viscous)
    }

    pub fn advance(&mut self, end_time: f64, reference_speed: f64) {
        while self.time < end_time {
            self.force_prior = self.combined_prior();
            let advection_dt = self.advection_time_step(reference_speed, 0.25);
            self.density_summation();
            self.force_prior = self.combined_prior();
            self.transport_velocity_shift();
            let mut elapsed = 0.0;
            while elapsed < advection_dt {
                let dt = self.acoustic_time_step(0.6).min(advection_dt - elapsed);
                self.pressure_step(dt);
                self.density_step(dt);
                elapsed += dt;
                self.time += dt;
            }
            self.update_configuration();
        }
    }

    fn combined_prior(&self) -> Vec<[f64; 2]> {
        let viscous = self.viscous_force();
        (0..self.len())
            .map(|p| {
                [
                    viscous[p][0] + self.gravity[0] * self.mass[p],
                    viscous[p][1] + self.gravity[1] * self.mass[p],
                ]
            })
            .collect()
    }

    pub fn kinetic_energy(&self) -> f64 {
        (0..self.len())
            .map(|p| {
                0.5 * self.mass[p]
                    * (self.velocity[p][0].powi(2) + self.velocity[p][1].powi(2))
            })
            .sum()
    }

    pub fn maximum_speed(&self) -> f64 {
        self.velocity
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0f64, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion::lattice;

    fn taylor_green(n: usize) -> FluidSolver {
        let dp = 1.0 / n as f64;
        let (x, _) = lattice([0.0, 0.0], [1.0, 1.0], dp);
        let fluid = WeaklyCompressibleFluid::new(1.0, 10.0, 0.01);
        let mut s = FluidSolver::new(
            x.clone(),
            dp,
            fluid,
            Kernel::WendlandC2,
            1.3,
            [0.0, 0.0],
            [1.0, 1.0],
            [true, true],
        );
        for (p, pos) in x.iter().enumerate() {
            s.velocity[p] = [
                -(2.0 * std::f64::consts::PI * pos[0]).cos()
                    * (2.0 * std::f64::consts::PI * pos[1]).sin(),
                (2.0 * std::f64::consts::PI * pos[0]).sin()
                    * (2.0 * std::f64::consts::PI * pos[1]).cos(),
            ];
        }
        s
    }

    /// An undisturbed periodic lattice must read exactly the reference
    /// density -- that is what `sigma0` is for, and it is exact rather than
    /// approximate because both sides are the same lattice sum.
    #[test]
    fn density_summation_reproduces_the_reference_density() {
        let mut s = taylor_green(40);
        s.density_summation();
        let worst = s
            .density
            .iter()
            .map(|r| (r - 1.0).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-12, "density off by {worst}");
    }

    /// Nothing drives a fluid at rest and at uniform density, so nothing may
    /// move. Catches a stray self-term or an asymmetric pair sum.
    #[test]
    fn a_fluid_at_rest_stays_at_rest() {
        let mut s = taylor_green(40);
        s.velocity.iter_mut().for_each(|v| *v = [0.0, 0.0]);
        s.density_summation();
        for _ in 0..20 {
            s.pressure_step(1e-3);
            s.density_step(1e-3);
        }
        assert!(s.maximum_speed() < 1e-13, "speed {}", s.maximum_speed());
    }

    /// A uniform translation is a rigid body motion: the pressure force must
    /// cancel exactly and the density must not change.
    #[test]
    fn uniform_translation_is_rigid() {
        let mut s = taylor_green(40);
        s.velocity.iter_mut().for_each(|v| *v = [1.0, 0.0]);
        s.density_summation();
        for _ in 0..20 {
            s.pressure_step(1e-3);
            s.density_step(1e-3);
        }
        let spread = s
            .velocity
            .iter()
            .map(|v| (v[0] - 1.0).abs().max(v[1].abs()))
            .fold(0.0f64, f64::max);
        assert!(spread < 1e-12, "velocity spread {spread}");
    }
}
