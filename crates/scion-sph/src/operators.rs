//! SPH operators over a [`NeighborList`].
//!
//! The correction matrix is the inverse of the renormalisation tensor
//! `A_i = sum_j V_j (-dW/dr) r_ij (e (x) e)`, which makes the corrected
//! gradient exact for a linear field. `A` is symmetric positive
//! semi-definite by construction -- `dW/dr` is negative over the whole
//! support -- and that sign is the cheap check on the assembly.
//!
//! `laplacian_rate` is the Cleary--Monaghan second-derivative form written
//! pairwise, so `sum_i V_i dphi_i` telescopes to zero at round-off.

use crate::neighbors::NeighborList;

/// A symmetric 2x2 matrix stored as `[xx, xy, yy]`.
pub type Sym2 = [f64; 3];

/// `(A^T A + eps I)^-1 A^T` for a symmetric 2x2 `A`, SPHinXsys's regularised
/// inverse. At `eps = 0` it is the plain inverse; the default
/// `eps = sqrt(machine eps)` is what their code uses and it costs seven
/// digits of the linear-consistency identity, which is measured rather than
/// assumed in the tests.
pub fn tikhonov_inverse(a: Sym2, eps: f64) -> [[f64; 2]; 2] {
    let [xx, xy, yy] = a;
    // A^T A for symmetric A is A A
    let m00 = xx * xx + xy * xy + eps;
    let m01 = xx * xy + xy * yy;
    let m11 = xy * xy + yy * yy + eps;
    let det = m00 * m11 - m01 * m01;
    let (i00, i01, i11) = (m11 / det, -m01 / det, m00 / det);
    // (M^-1) A, with A symmetric
    [
        [i00 * xx + i01 * xy, i00 * xy + i01 * yy],
        [i01 * xx + i11 * xy, i01 * xy + i11 * yy],
    ]
}

/// Per-particle correction matrices.
pub fn linear_gradient_correction(
    pairs: &NeighborList,
    volumes: &[f64],
    dw: &[f64],
    eps: f64,
) -> Vec<[[f64; 2]; 2]> {
    let mut a = vec![[0.0f64; 3]; pairs.n_particles];
    for k in 0..pairs.len() {
        let (i, j) = (pairs.i[k], pairs.j[k]);
        let w = -dw[k] * pairs.r[k];
        let e = pairs.e[k];
        let ee = [e[0] * e[0], e[0] * e[1], e[1] * e[1]];
        for c in 0..3 {
            a[i][c] += w * volumes[j] * ee[c];
            a[j][c] += w * volumes[i] * ee[c];
        }
    }
    a.into_iter().map(|m| tikhonov_inverse(m, eps)).collect()
}

/// `sum_j V_j W_ij`, without the particle's own `W(0)` term.
pub fn number_density(pairs: &NeighborList, volumes: &[f64], w: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; pairs.n_particles];
    for k in 0..pairs.len() {
        out[pairs.i[k]] += volumes[pairs.j[k]] * w[k];
        out[pairs.j[k]] += volumes[pairs.i[k]] * w[k];
    }
    out
}

/// The corrected SPH gradient of a scalar field.
pub fn gradient(
    pairs: &NeighborList,
    phi: &[f64],
    volumes: &[f64],
    dw: &[f64],
    b: Option<&[[[f64; 2]; 2]]>,
) -> Vec<[f64; 2]> {
    let mut g = vec![[0.0f64; 2]; pairs.n_particles];
    for k in 0..pairs.len() {
        let (i, j) = (pairs.i[k], pairs.j[k]);
        let e = pairs.e[k];
        let dphi = phi[j] - phi[i];
        for d in 0..2 {
            let gw = dw[k] * e[d];
            g[i][d] += volumes[j] * dphi * gw;
            g[j][d] += volumes[i] * dphi * gw;
        }
    }
    match b {
        None => g,
        Some(b) => g
            .iter()
            .zip(b.iter())
            .map(|(v, m)| {
                [
                    m[0][0] * v[0] + m[0][1] * v[1],
                    m[1][0] * v[0] + m[1][1] * v[1],
                ]
            })
            .collect(),
    }
}

/// The edge weight `2 D_ij (B_ij e . e) dW/dr / r`, one per pair.
///
/// Everything in the diffusion operator except `phi` is fixed by the
/// particle positions, so this is built once and the step is then one
/// subtraction and one scatter per pair.
pub fn pair_conductance(
    pairs: &NeighborList,
    dw: &[f64],
    d_ij: &[f64],
    b: Option<&[[[f64; 2]; 2]]>,
) -> Vec<f64> {
    (0..pairs.len())
        .map(|k| {
            let (i, j) = (pairs.i[k], pairs.j[k]);
            let e = pairs.e[k];
            let projection = match b {
                None => 1.0,
                Some(b) => {
                    let m = [
                        [
                            0.5 * (b[i][0][0] + b[j][0][0]),
                            0.5 * (b[i][0][1] + b[j][0][1]),
                        ],
                        [
                            0.5 * (b[i][1][0] + b[j][1][0]),
                            0.5 * (b[i][1][1] + b[j][1][1]),
                        ],
                    ];
                    e[0] * (m[0][0] * e[0] + m[0][1] * e[1])
                        + e[1] * (m[1][0] * e[0] + m[1][1] * e[1])
                }
            };
            2.0 * d_ij[k] * dw[k] * projection / pairs.r[k]
        })
        .collect()
}

/// `d phi / dt` for `div(D grad phi)`, from the edge weights. Conservative
/// by construction: the same weight enters both ends with opposite signs.
pub fn laplacian_rate(
    pairs: &NeighborList,
    phi: &[f64],
    volumes: &[f64],
    conductance: &[f64],
    out: &mut [f64],
) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for k in 0..pairs.len() {
        let (i, j) = (pairs.i[k], pairs.j[k]);
        let flux = conductance[k] * (phi[i] - phi[j]);
        out[i] += flux * volumes[j];
        out[j] -= flux * volumes[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Kernel;
    use crate::neighbors;

    fn lattice(dp: f64) -> (Vec<[f64; 2]>, Vec<f64>) {
        let nx = (2.0 / dp).round() as usize;
        let ny = (0.4 / dp).round() as usize;
        let mut x = Vec::with_capacity(nx * ny);
        for i in 0..nx {
            for j in 0..ny {
                x.push([(i as f64 + 0.5) * dp, (j as f64 + 0.5) * dp]);
            }
        }
        let n = x.len();
        (x, vec![dp * dp; n])
    }

    /// The identity the correction exists for: a linear field, exactly.
    /// Both halves matter -- a correction that improved nothing would pass
    /// the first assertion if the tolerance were loose.
    #[test]
    fn correction_makes_the_gradient_exact_on_a_linear_field() {
        let dp = 0.02;
        let (x, vol) = lattice(dp);
        let h = 1.3 * dp;
        let kernel = Kernel::WendlandC2;
        let pairs = neighbors::build(&x, kernel.support(h), [0.0, 0.0], [2.0, 0.4], [false, false]);
        let dw: Vec<f64> = pairs.r.iter().map(|&r| kernel.dw_dr(r, h)).collect();
        let b = linear_gradient_correction(&pairs, &vol, &dw, 0.0);
        let a = [0.3, -0.7];
        let phi: Vec<f64> = x.iter().map(|p| a[0] * p[0] + a[1] * p[1]).collect();
        let g = gradient(&pairs, &phi, &vol, &dw, Some(&b));
        let g0 = gradient(&pairs, &phi, &vol, &dw, None);
        let err = g
            .iter()
            .map(|v| (v[0] - a[0]).abs().max((v[1] - a[1]).abs()))
            .fold(0.0f64, f64::max);
        let err0 = g0
            .iter()
            .map(|v| (v[0] - a[0]).abs().max((v[1] - a[1]).abs()))
            .fold(0.0f64, f64::max);
        assert!(err < 1e-12, "corrected gradient error {err}");
        assert!(err0 > 1e-2, "uncorrected gradient error {err0} -- too good, check the test");
    }

    #[test]
    fn diffusion_conserves_the_species_exactly() {
        let dp = 0.02;
        let (x, vol) = lattice(dp);
        let h = 1.3 * dp;
        let kernel = Kernel::WendlandC2;
        let pairs = neighbors::build(&x, kernel.support(h), [0.0, 0.0], [2.0, 0.4], [true, false]);
        let dw: Vec<f64> = pairs.r.iter().map(|&r| kernel.dw_dr(r, h)).collect();
        let b = linear_gradient_correction(&pairs, &vol, &dw, 0.0);
        let d_ij = vec![1e-4; pairs.len()];
        let weights = pair_conductance(&pairs, &dw, &d_ij, Some(&b));
        let phi: Vec<f64> = x.iter().map(|p| (3.0 * p[0]).sin() + p[1]).collect();
        let mut rate = vec![0.0; x.len()];
        laplacian_rate(&pairs, &phi, &vol, &weights, &mut rate);
        let drift: f64 = rate.iter().zip(vol.iter()).map(|(r, v)| r * v).sum();
        // relative to the size of the terms being cancelled: the sum is
        // exact in exact arithmetic, so what is left is accumulation
        // round-off and the bound has to be stated that way
        let scale: f64 = rate.iter().zip(vol.iter()).map(|(r, v)| (r * v).abs()).sum();
        assert!(drift.abs() < 1e-14 * scale, "species drift {drift} against scale {scale}");
    }
}
