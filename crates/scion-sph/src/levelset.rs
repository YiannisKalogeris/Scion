//! Level-set geometry: a solid boundary with no solid particles.
//!
//! Mirrors `scion.discretization.sph.levelset`, which is the reference.
//!
//! `phi > 0` is **inside the solid**. The normal `grad phi / |grad phi|`
//! therefore points into the solid, which is what the surface bounding
//! relies on.
//!
//! Two integrals carry the whole boundary condition:
//!
//! ```text
//! K(x) = INT_solid W(|x - y|) dy      G(x) = INT_solid grad W(x - y) dy
//! ```
//!
//! They depend on the geometry alone, so they are built once on a grid and
//! interpolated. **That grid's spacing must be the particle spacing and its
//! origin must be an integer number of spacings from the wall** -- a finer
//! grid converges to the continuum integral, and the continuum integral is
//! not what the density summation wants. See the note in the Python module.

use crate::kernel::Kernel;

/// Signed distance to a box, **positive outside** it.
pub fn box_interior_distance(p: [f64; 2], lower: [f64; 2], upper: [f64; 2]) -> f64 {
    let mut excess = [0.0f64; 2];
    for d in 0..2 {
        excess[d] = (lower[d] - p[d]).max(p[d] - upper[d]);
    }
    let outside = (excess[0].max(0.0).powi(2) + excess[1].max(0.0).powi(2)).sqrt();
    if outside > 0.0 {
        outside
    } else {
        excess[0].max(excess[1])
    }
}

/// Signed distance to a closed polygon, **positive inside** it.
pub fn polygon_distance(p: [f64; 2], polygon: &[[f64; 2]]) -> f64 {
    let n = polygon.len();
    let mut best = f64::INFINITY;
    let mut inside = false;
    for k in 0..n {
        let a = polygon[k];
        let b = polygon[(k + 1) % n];
        let edge = [b[0] - a[0], b[1] - a[1]];
        let length_sq = edge[0] * edge[0] + edge[1] * edge[1];
        let t = (((p[0] - a[0]) * edge[0] + (p[1] - a[1]) * edge[1]) / length_sq).clamp(0.0, 1.0);
        let closest = [a[0] + t * edge[0], a[1] + t * edge[1]];
        let d = ((p[0] - closest[0]).powi(2) + (p[1] - closest[1]).powi(2)).sqrt();
        best = best.min(d);
        // crossing number along +x
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x_cross = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * edge[0];
            if p[0] < x_cross {
                inside = !inside;
            }
        }
    }
    if inside {
        best
    } else {
        -best
    }
}

/// The solid fraction of a cell, from `phi` and `grad phi`.
pub fn cut_cell_fraction(phi: f64, gradient: [f64; 2], spacing: f64) -> f64 {
    let squared = gradient[0] * gradient[0] + gradient[1] * gradient[1];
    let inverse = 1.0 / (squared + 1e-300);
    let half_width = 0.5 * spacing;
    let mut fraction = 0.0;
    for d in 0..2 {
        let measure = phi / (gradient[d].abs() + 1e-300);
        let heaviside = (0.5 + 0.5 * measure / half_width).clamp(0.0, 1.0);
        fraction += gradient[d] * gradient[d] * inverse * heaviside;
    }
    fraction
}

pub struct LevelSet {
    pub lower: [f64; 2],
    pub spacing: f64,
    pub counts: [usize; 2],
    pub phi: Vec<f64>,
    pub gradient: Vec<[f64; 2]>,
    pub fraction: Vec<f64>,
    pub kernel_integral: Vec<f64>,
    pub kernel_gradient_integral: Vec<[f64; 2]>,
}

impl LevelSet {
    fn at(&self, i: usize, j: usize) -> usize {
        i * self.counts[1] + j
    }

    pub fn new<F>(phi_of: F, lower: [f64; 2], upper: [f64; 2], spacing: f64, kernel: Kernel, h: f64) -> Self
    where
        F: Fn([f64; 2]) -> f64,
    {
        let counts = [
            (((upper[0] - lower[0]) / spacing).round() as usize).max(1),
            (((upper[1] - lower[1]) / spacing).round() as usize).max(1),
        ];
        let n = counts[0] * counts[1];
        let mut phi = vec![0.0; n];
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                let p = [
                    lower[0] + (i as f64 + 0.5) * spacing,
                    lower[1] + (j as f64 + 0.5) * spacing,
                ];
                phi[i * counts[1] + j] = phi_of(p);
            }
        }
        let mut set = Self {
            lower,
            spacing,
            counts,
            phi,
            gradient: vec![[0.0; 2]; n],
            fraction: vec![0.0; n],
            kernel_integral: vec![0.0; n],
            kernel_gradient_integral: vec![[0.0; 2]; n],
        };
        set.build_gradient();
        set.build_fraction();
        set.build_integrals(kernel, h);
        set
    }

    /// Central differences, one-sided at the edge -- `np.gradient`'s rule.
    fn build_gradient(&mut self) {
        let (nx, ny) = (self.counts[0], self.counts[1]);
        for i in 0..nx {
            for j in 0..ny {
                let gx = if nx == 1 {
                    0.0
                } else if i == 0 {
                    (self.phi[self.at(1, j)] - self.phi[self.at(0, j)]) / self.spacing
                } else if i == nx - 1 {
                    (self.phi[self.at(nx - 1, j)] - self.phi[self.at(nx - 2, j)]) / self.spacing
                } else {
                    (self.phi[self.at(i + 1, j)] - self.phi[self.at(i - 1, j)]) / (2.0 * self.spacing)
                };
                let gy = if ny == 1 {
                    0.0
                } else if j == 0 {
                    (self.phi[self.at(i, 1)] - self.phi[self.at(i, 0)]) / self.spacing
                } else if j == ny - 1 {
                    (self.phi[self.at(i, ny - 1)] - self.phi[self.at(i, ny - 2)]) / self.spacing
                } else {
                    (self.phi[self.at(i, j + 1)] - self.phi[self.at(i, j - 1)]) / (2.0 * self.spacing)
                };
                let k = self.at(i, j);
                self.gradient[k] = [gx, gy];
            }
        }
    }

    fn build_fraction(&mut self) {
        for k in 0..self.phi.len() {
            self.fraction[k] = cut_cell_fraction(self.phi[k], self.gradient[k], self.spacing);
        }
    }

    /// `sum_d alpha(x + d) w(d)` -- a correlation, not a convolution. The
    /// flip is invisible for `W`, which is even, and silently negates `G`.
    fn build_integrals(&mut self, kernel: Kernel, h: f64) {
        let depth = (kernel.support(h) / self.spacing).ceil() as isize;
        let cell_volume = self.spacing * self.spacing;
        let (nx, ny) = (self.counts[0] as isize, self.counts[1] as isize);
        for i in 0..nx {
            for j in 0..ny {
                let (mut integral, mut gx, mut gy) = (0.0, 0.0, 0.0);
                for di in -depth..=depth {
                    for dj in -depth..=depth {
                        let dx = di as f64 * self.spacing;
                        let dy = dj as f64 * self.spacing;
                        let r = (dx * dx + dy * dy).sqrt();
                        if r >= kernel.support(h) {
                            continue;
                        }
                        // edge replication outside the grid
                        let ii = (i + di).clamp(0, nx - 1) as usize;
                        let jj = (j + dj).clamp(0, ny - 1) as usize;
                        let alpha = self.fraction[self.at(ii, jj)];
                        integral += kernel.w(r, h) * alpha * cell_volume;
                        if r > 0.0 {
                            let dw = kernel.dw_dr(r, h) * alpha * cell_volume;
                            gx += -dw * dx / r;
                            gy += -dw * dy / r;
                        }
                    }
                }
                let k = self.at(i as usize, j as usize);
                self.kernel_integral[k] = integral;
                self.kernel_gradient_integral[k] = [gx, gy];
            }
        }
    }

    /// Bilinear interpolation with the edge value outside the grid.
    fn interpolate<T, F>(&self, point: [f64; 2], get: F) -> T
    where
        T: Copy,
        F: Fn(usize, usize) -> T,
        T: std::ops::Mul<f64, Output = T> + std::ops::Add<Output = T>,
    {
        let (nx, ny) = (self.counts[0], self.counts[1]);
        let fx = ((point[0] - self.lower[0]) / self.spacing - 0.5).clamp(0.0, (nx - 1) as f64);
        let fy = ((point[1] - self.lower[1]) / self.spacing - 0.5).clamp(0.0, (ny - 1) as f64);
        let i0 = fx.floor() as usize;
        let j0 = fy.floor() as usize;
        let i1 = (i0 + 1).min(nx - 1);
        let j1 = (j0 + 1).min(ny - 1);
        let tx = fx - i0 as f64;
        let ty = fy - j0 as f64;
        get(i0, j0) * ((1.0 - tx) * (1.0 - ty))
            + get(i1, j0) * (tx * (1.0 - ty))
            + get(i0, j1) * ((1.0 - tx) * ty)
            + get(i1, j1) * (tx * ty)
    }

    pub fn signed_distance(&self, point: [f64; 2]) -> f64 {
        self.interpolate(point, |i, j| self.phi[self.at(i, j)])
    }

    pub fn kernel_integral_at(&self, point: [f64; 2]) -> f64 {
        self.interpolate(point, |i, j| self.kernel_integral[self.at(i, j)])
    }

    pub fn kernel_gradient_at(&self, point: [f64; 2]) -> [f64; 2] {
        let gx = self.interpolate(point, |i, j| self.kernel_gradient_integral[self.at(i, j)][0]);
        let gy = self.interpolate(point, |i, j| self.kernel_gradient_integral[self.at(i, j)][1]);
        [gx, gy]
    }

    /// The unit normal, pointing into the solid.
    pub fn normal(&self, point: [f64; 2]) -> [f64; 2] {
        let gx = self.interpolate(point, |i, j| self.gradient[self.at(i, j)][0]);
        let gy = self.interpolate(point, |i, j| self.gradient[self.at(i, j)][1]);
        let norm = (gx * gx + gy * gy).sqrt() + 1e-300;
        [gx / norm, gy / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half_space(spacing: f64) -> LevelSet {
        let h = 1.3 * 0.025;
        LevelSet::new(|p| -p[1], [-0.3, -0.3], [0.3, 0.3], spacing, Kernel::WendlandC2, h)
    }

    /// `K(d) + K(-d) = 1` for a half-space: what falls inside the solid plus
    /// what falls outside is the whole kernel. Exact, and it fails for any
    /// mistake in the cut-cell fraction.
    #[test]
    fn half_space_kernel_integral_is_antisymmetric() {
        let ls = half_space(0.025 / 4.0);
        for k in 0..12 {
            let d = (k as f64 - 6.0) * 0.008;
            let up = ls.kernel_integral_at([0.0, d]);
            let down = ls.kernel_integral_at([0.0, -d]);
            assert!((up + down - 1.0).abs() < 2e-3, "K({d}) + K(-{d}) = {}", up + down);
        }
        assert!((ls.kernel_integral_at([0.0, 0.0]) - 0.5).abs() < 2e-3);
    }

    /// The gradient integral is normal to a flat wall and points into it.
    #[test]
    fn half_space_kernel_gradient_is_normal_and_inward() {
        let ls = half_space(0.025 / 4.0);
        for k in 0..8 {
            let d = (k as f64 - 4.0) * 0.008;
            let g = ls.kernel_gradient_at([0.0, d]);
            assert!(g[0].abs() < 1e-12, "tangential component {}", g[0]);
            assert!(g[1] <= 0.0, "gradient points out of the solid: {}", g[1]);
        }
    }

    #[test]
    fn box_distance_is_positive_outside() {
        assert!((box_interior_distance([0.5, 0.5], [0.0, 0.0], [1.0, 1.0]) + 0.5).abs() < 1e-15);
        assert!((box_interior_distance([-0.1, 0.5], [0.0, 0.0], [1.0, 1.0]) - 0.1).abs() < 1e-15);
    }

    #[test]
    fn polygon_distance_is_positive_inside() {
        let triangle = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert!(polygon_distance([0.1, 0.1], &triangle) > 0.0);
        assert!(polygon_distance([1.0, 1.0], &triangle) < 0.0);
    }
}
