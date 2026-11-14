//! Cell-linked-list neighbour search in two dimensions, with per-axis
//! periodicity.
//!
//! Each unordered pair appears ONCE, with `dx = x_i - x_j` already reduced to
//! its minimum image. Both particles are then updated in a single pass with
//! opposite signs, which is what makes the diffusion operator conserve the
//! species exactly rather than to a tolerance.
//!
//! A periodic axis must span at least three cells, or the three-cell stencil
//! would meet the same cell twice and count those pairs twice. Checked, not
//! assumed -- a doubled pair is a plausible-looking 2x in the diffusivity.

#[derive(Clone, Debug)]
pub struct NeighborList {
    pub i: Vec<usize>,
    pub j: Vec<usize>,
    /// `x_i - x_j`, minimum image
    pub dx: Vec<[f64; 2]>,
    pub r: Vec<f64>,
    /// `dx / r`
    pub e: Vec<[f64; 2]>,
    pub n_particles: usize,
}

impl NeighborList {
    pub fn len(&self) -> usize {
        self.i.len()
    }

    pub fn is_empty(&self) -> bool {
        self.i.is_empty()
    }
}

/// Wrap positions into `[lower, upper)` on the periodic axes.
pub fn wrap(x: &mut [[f64; 2]], lower: [f64; 2], upper: [f64; 2], periodic: [bool; 2]) {
    for p in x.iter_mut() {
        for d in 0..2 {
            if periodic[d] {
                let length = upper[d] - lower[d];
                let mut v = (p[d] - lower[d]) % length;
                if v < 0.0 {
                    v += length;
                }
                p[d] = lower[d] + v;
            }
        }
    }
}

pub fn build(
    positions: &[[f64; 2]],
    cutoff: f64,
    lower: [f64; 2],
    upper: [f64; 2],
    periodic: [bool; 2],
) -> NeighborList {
    let n = positions.len();
    let mut x = positions.to_vec();
    wrap(&mut x, lower, upper, periodic);

    let length = [upper[0] - lower[0], upper[1] - lower[1]];
    let mut n_cells = [0usize; 2];
    for d in 0..2 {
        n_cells[d] = ((length[d] / cutoff).floor() as isize).max(1) as usize;
        if periodic[d] && n_cells[d] < 3 {
            panic!(
                "axis {d} is periodic and only {} cutoffs wide; the minimum image \
                 needs at least three cells",
                length[d] / cutoff
            );
        }
    }

    let cell_of = |p: &[f64; 2]| -> [usize; 2] {
        let mut c = [0usize; 2];
        for d in 0..2 {
            let k = ((p[d] - lower[d]) / length[d] * n_cells[d] as f64).floor();
            c[d] = k.clamp(0.0, (n_cells[d] - 1) as f64) as usize;
        }
        c
    };

    // bucket the particles, counting sort on the linear cell index
    let n_total = n_cells[0] * n_cells[1];
    let lin = |c: [usize; 2]| c[0] + c[1] * n_cells[0];
    let mut counts = vec![0usize; n_total + 1];
    let cells: Vec<[usize; 2]> = x.iter().map(cell_of).collect();
    for c in &cells {
        counts[lin(*c) + 1] += 1;
    }
    for k in 0..n_total {
        counts[k + 1] += counts[k];
    }
    let mut start = counts.clone();
    let mut order = vec![0usize; n];
    for (p, c) in cells.iter().enumerate() {
        let slot = lin(*c);
        order[start[slot]] = p;
        start[slot] += 1;
    }
    let bucket = |c: [usize; 2]| -> &[usize] {
        let s = lin(c);
        &order[counts[s]..counts[s + 1]]
    };

    // the self cell plus a half-space of the shifts, so a pair appears once
    const OFFSETS: [[isize; 2]; 5] = [[0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]];

    let mut out = NeighborList {
        i: Vec::new(),
        j: Vec::new(),
        dx: Vec::new(),
        r: Vec::new(),
        e: Vec::new(),
        n_particles: n,
    };

    for p in 0..n {
        let c = cells[p];
        for offset in OFFSETS {
            let mut target = [0usize; 2];
            let mut inside = true;
            for d in 0..2 {
                let k = c[d] as isize + offset[d];
                if periodic[d] {
                    let m = n_cells[d] as isize;
                    target[d] = (((k % m) + m) % m) as usize;
                } else if k < 0 || k >= n_cells[d] as isize {
                    inside = false;
                } else {
                    target[d] = k as usize;
                }
            }
            if !inside {
                continue;
            }
            for &q in bucket(target) {
                // the self cell is symmetric, so keep one of each pair
                if offset == [0, 0] && q <= p {
                    continue;
                }
                let mut dx = [x[p][0] - x[q][0], x[p][1] - x[q][1]];
                for d in 0..2 {
                    if periodic[d] {
                        dx[d] -= length[d] * (dx[d] / length[d]).round();
                    }
                }
                let r = (dx[0] * dx[0] + dx[1] * dx[1]).sqrt();
                if r >= cutoff || r == 0.0 {
                    continue;
                }
                out.i.push(p);
                out.j.push(q);
                out.dx.push(dx);
                out.r.push(r);
                out.e.push([dx[0] / r, dx[1] / r]);
            }
        }
    }
    out
}

/// The same thing in `O(N^2)`, as the reference the cell list is tested
/// against. Never call it on a production run.
pub fn brute_force(
    positions: &[[f64; 2]],
    cutoff: f64,
    lower: [f64; 2],
    upper: [f64; 2],
    periodic: [bool; 2],
) -> NeighborList {
    let n = positions.len();
    let mut x = positions.to_vec();
    wrap(&mut x, lower, upper, periodic);
    let length = [upper[0] - lower[0], upper[1] - lower[1]];
    let mut out = NeighborList {
        i: Vec::new(),
        j: Vec::new(),
        dx: Vec::new(),
        r: Vec::new(),
        e: Vec::new(),
        n_particles: n,
    };
    for p in 0..n {
        for q in (p + 1)..n {
            let mut dx = [x[p][0] - x[q][0], x[p][1] - x[q][1]];
            for d in 0..2 {
                if periodic[d] {
                    dx[d] -= length[d] * (dx[d] / length[d]).round();
                }
            }
            let r = (dx[0] * dx[0] + dx[1] * dx[1]).sqrt();
            if r < cutoff && r > 0.0 {
                out.i.push(p);
                out.j.push(q);
                out.dx.push(dx);
                out.r.push(r);
                out.e.push([dx[0] / r, dx[1] / r]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_pairs(nl: &NeighborList) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = nl
            .i
            .iter()
            .zip(nl.j.iter())
            .map(|(&a, &b)| if a < b { (a, b) } else { (b, a) })
            .collect();
        v.sort_unstable();
        v
    }

    /// A deterministic pseudo-random cloud; no rand dependency on purpose.
    fn cloud(n: usize) -> Vec<[f64; 2]> {
        let mut s = 12345u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        (0..n).map(|_| [next() * 2.0, next() * 0.4]).collect()
    }

    #[test]
    fn matches_brute_force() {
        for periodic in [[false, false], [true, false], [true, true]] {
            let x = cloud(900);
            let a = build(&x, 0.1, [0.0, 0.0], [2.0, 0.4], periodic);
            let b = brute_force(&x, 0.1, [0.0, 0.0], [2.0, 0.4], periodic);
            assert_eq!(sorted_pairs(&a), sorted_pairs(&b), "periodic = {periodic:?}");
        }
    }

    #[test]
    fn a_pair_appears_once() {
        let x = cloud(400);
        let nl = build(&x, 0.15, [0.0, 0.0], [2.0, 0.4], [true, false]);
        let mut seen = sorted_pairs(&nl);
        let total = seen.len();
        seen.dedup();
        assert_eq!(total, seen.len());
    }

    #[test]
    fn periodic_separation_is_the_short_way_round() {
        let x = vec![[0.005, 0.2], [1.995, 0.2]];
        let nl = build(&x, 0.1, [0.0, 0.0], [2.0, 0.4], [true, false]);
        assert_eq!(nl.len(), 1);
        assert!((nl.r[0] - 0.01).abs() < 1e-15);
    }
}
