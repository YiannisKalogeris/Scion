//! SPH smoothing kernels.
//!
//! `q = r/h` and the support is `cutoff_ratio * h`, which for both kernels
//! here is `2h`. The radial shape of Wendland C2 is the same expression in
//! every dimension and its normalising constant is not, which is the easiest
//! way to be wrong by a constant while every qualitative check still passes.
//!
//! Mirrors `scion.discretization.sph.kernels`; the Python implementation is
//! the reference and this must agree with it, not the other way round.

pub const PI: f64 = std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    WendlandC2,
    CubicSpline,
}

impl Kernel {
    pub fn cutoff_ratio(&self) -> f64 {
        2.0
    }

    pub fn support(&self, h: f64) -> f64 {
        self.cutoff_ratio() * h
    }

    /// The normalising constant in two dimensions.
    fn factor_2d(&self) -> f64 {
        match self {
            Kernel::WendlandC2 => 7.0 / (4.0 * PI),
            Kernel::CubicSpline => 10.0 / (7.0 * PI),
        }
    }

    fn shape(&self, q: f64) -> f64 {
        match self {
            Kernel::WendlandC2 => {
                let a = 1.0 - 0.5 * q;
                a * a * a * a * (1.0 + 2.0 * q)
            }
            Kernel::CubicSpline => {
                if q < 1.0 {
                    1.0 - 1.5 * q * q + 0.75 * q * q * q
                } else {
                    let a = 2.0 - q;
                    0.25 * a * a * a
                }
            }
        }
    }

    fn shape_derivative(&self, q: f64) -> f64 {
        match self {
            Kernel::WendlandC2 => {
                let a = q - 2.0;
                0.625 * a * a * a * q
            }
            Kernel::CubicSpline => {
                if q < 1.0 {
                    -3.0 * q + 2.25 * q * q
                } else {
                    let a = 2.0 - q;
                    -0.75 * a * a
                }
            }
        }
    }

    /// `W(r)` in two dimensions.
    pub fn w(&self, r: f64, h: f64) -> f64 {
        let q = r.abs() / h;
        if q >= self.cutoff_ratio() {
            return 0.0;
        }
        self.factor_2d() / (h * h) * self.shape(q)
    }

    /// `dW/dr` in two dimensions; negative over the whole support.
    pub fn dw_dr(&self, r: f64, h: f64) -> f64 {
        let q = r.abs() / h;
        if q >= self.cutoff_ratio() {
            return 0.0;
        }
        self.factor_2d() / (h * h * h) * self.shape_derivative(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 2-D normalisation, by quadrature on a polar grid. This is the
    /// check that a transcribed constant is the right one.
    #[test]
    fn integrates_to_one_in_two_dimensions() {
        for kernel in [Kernel::WendlandC2, Kernel::CubicSpline] {
            let h = 0.013;
            let n = 200_000;
            let r_max = kernel.support(h);
            let dr = r_max / n as f64;
            let mut total = 0.0;
            for i in 0..n {
                let r = (i as f64 + 0.5) * dr;
                total += kernel.w(r, h) * 2.0 * PI * r * dr;
            }
            assert!((total - 1.0).abs() < 1e-6, "{kernel:?} integrates to {total}");
        }
    }

    #[test]
    fn derivative_matches_a_central_difference() {
        let h = 0.013;
        for kernel in [Kernel::WendlandC2, Kernel::CubicSpline] {
            for k in 1..20 {
                let r = k as f64 * 0.1 * h;
                let d = 1e-7 * h;
                let fd = (kernel.w(r + d, h) - kernel.w(r - d, h)) / (2.0 * d);
                let exact = kernel.dw_dr(r, h);
                assert!((fd - exact).abs() <= 1e-4 * exact.abs().max(1.0));
            }
        }
    }

    #[test]
    fn vanishes_at_the_support() {
        for kernel in [Kernel::WendlandC2, Kernel::CubicSpline] {
            assert_eq!(kernel.w(kernel.support(0.013), 0.013), 0.0);
            assert_eq!(kernel.dw_dr(kernel.support(0.013) * 1.5, 0.013), 0.0);
        }
    }
}
