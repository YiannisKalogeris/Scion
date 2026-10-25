//! Particle-to-grid scatter-add, the inner loop of every MPM assembly.
//!
//! Replaces `np.add.at(out[:, col:col+k], node_id, vals)`, which is
//! unbuffered and single-threaded.

use rayon::prelude::*;

/// Node-major transpose of a particle stencil, reusable across scatters.
///
/// **Why a transpose rather than atomics.** Each node is summed by exactly one
/// thread, over its entries in ascending entry order. `np.add.at` traverses
/// the same entries in that same order, so the result is bit-identical to it
/// and to a serial accumulation -- the thread count cannot change an answer.
/// An atomic or thread-local scatter is easier to write and loses that, which
/// would silently weaken the checkpoint test asserting a resumed run
/// reproduces the load path at 0.00e+00: an approximate match there is
/// supposed to mean the restored state is incomplete, and a nondeterministic
/// reduction would make it mean nothing.
///
/// The transpose depends only on `node_id`, which is rebuilt once per load
/// step while the scatters run several times per Newton iteration, so it is
/// built once and reused.
pub struct Scatter {
    n_nodes: usize,
    n_entries: usize,
    /// `offsets[n] .. offsets[n + 1]` indexes `entries` for node `n`.
    offsets: Vec<usize>,
    /// Flat stencil entry `e = p * npe + l`, ascending within each node.
    entries: Vec<u32>,
}

impl Scatter {
    /// `node_id` is the flattened `(n_particles, npe)` stencil.
    pub fn new(node_id: &[i64], n_nodes: usize) -> Result<Self, String> {
        let n_entries = node_id.len();
        if n_entries > u32::MAX as usize {
            return Err(format!("{n_entries} stencil entries exceeds the u32 index"));
        }
        // counting sort: count, prefix-sum, then fill by scanning entries in
        // ascending order, which is what makes each node's list ascending.
        let mut offsets = vec![0usize; n_nodes + 1];
        for &i in node_id {
            if i < 0 || i as usize >= n_nodes {
                return Err(format!("node id {i} outside 0..{n_nodes}"));
            }
            offsets[i as usize + 1] += 1;
        }
        for n in 0..n_nodes {
            offsets[n + 1] += offsets[n];
        }
        let mut cursor = offsets.clone();
        let mut entries = vec![0u32; n_entries];
        for (e, &i) in node_id.iter().enumerate() {
            let n = i as usize;
            entries[cursor[n]] = e as u32;
            cursor[n] += 1;
        }
        Ok(Self { n_nodes, n_entries, offsets, entries })
    }

    pub fn n_nodes(&self) -> usize { self.n_nodes }
    pub fn n_entries(&self) -> usize { self.n_entries }

    /// Accumulate `vals` into columns `col .. col + k` of `out`.
    ///
    /// `out` is `(n_nodes, ncomp)` row-major and `vals` is `(n_entries, k)`.
    /// Taking a column offset rather than a sliced array is what keeps this
    /// zero-copy: the call sites scatter into `out[:, :3]` and `out[:, 3:]`,
    /// which are strided views NumPy cannot hand over as a flat buffer.
    pub fn add(&self, out: &mut [f64], ncomp: usize, col: usize,
               vals: &[f64], k: usize) -> Result<(), String> {
        if col + k > ncomp {
            return Err(format!("columns {col}..{} exceed ncomp {ncomp}", col + k));
        }
        if out.len() != self.n_nodes * ncomp {
            return Err(format!("out has {} entries, expected {}",
                               out.len(), self.n_nodes * ncomp));
        }
        if vals.len() != self.n_entries * k {
            return Err(format!("vals has {} entries, expected {}",
                               vals.len(), self.n_entries * k));
        }
        out.par_chunks_mut(ncomp).enumerate().for_each(|(n, row)| {
            for &e in &self.entries[self.offsets[n]..self.offsets[n + 1]] {
                let base = e as usize * k;
                for c in 0..k {
                    row[col + c] += vals[base + c];
                }
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference is a serial accumulation in entry order -- exactly what
    /// `np.add.at` does. Equality here is asserted to the bit, not to a
    /// tolerance, because that is the property the design buys.
    fn serial(node_id: &[i64], n_nodes: usize, ncomp: usize, col: usize,
              vals: &[f64], k: usize) -> Vec<f64> {
        let mut out = vec![0.0; n_nodes * ncomp];
        for (e, &i) in node_id.iter().enumerate() {
            for c in 0..k {
                out[i as usize * ncomp + col + c] += vals[e * k + c];
            }
        }
        out
    }

    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        // values spanning several magnitudes, so a changed summation order
        // shows up in the last bits rather than cancelling
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            (u - 0.5) * 10f64.powi(((s >> 60) % 7) as i32 - 3)
        }).collect()
    }

    #[test]
    fn matches_serial_accumulation_bit_for_bit() {
        let (n_nodes, n_p, npe, ncomp, k, col) = (997usize, 5000usize, 27usize, 6usize, 3usize, 3usize);
        let ids = pseudo_random(n_p * npe, 7);
        let node_id: Vec<i64> =
            ids.iter().map(|v| ((v.abs() * 1e6) as usize % n_nodes) as i64).collect();
        let vals = pseudo_random(n_p * npe * k, 11);

        let sc = Scatter::new(&node_id, n_nodes).unwrap();
        let mut out = vec![0.0; n_nodes * ncomp];
        sc.add(&mut out, ncomp, col, &vals, k).unwrap();

        let want = serial(&node_id, n_nodes, ncomp, col, &vals, k);
        assert_eq!(out, want, "parallel scatter must be bit-identical to serial");
    }

    #[test]
    fn accumulates_onto_existing_values() {
        // the call sites add into a nonzero array (two scatters into the same
        // out), so starting from zero would not test what they do
        let node_id = vec![0i64, 2, 2, 1];
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let sc = Scatter::new(&node_id, 3).unwrap();
        let mut out = vec![10.0, 20.0, 30.0];
        sc.add(&mut out, 1, 0, &vals, 1).unwrap();
        assert_eq!(out, vec![11.0, 24.0, 35.0]);
    }

    #[test]
    fn rejects_an_out_of_range_node() {
        assert!(Scatter::new(&[0i64, 5], 3).is_err());
        assert!(Scatter::new(&[-1i64], 3).is_err());
    }
}
