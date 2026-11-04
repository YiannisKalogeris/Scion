//! The tangent's sparsity pattern, built once per stencil rebuild.
//!
//! `node_id` is rebuilt once per load step, while the tangent is assembled
//! once per Newton iteration -- three times a step on the torsion problem,
//! nine counting the staggered loop. The old path rebuilt the pattern every
//! time: materialise `(rows, cols, vals)` per chunk, sort 3.7M triplets into
//! CSR, then add the chunk matrices together. Measured at 27% of a tangent
//! for the sort and another 3% for the triplets and the merge, all of it
//! recomputing an answer that had not changed.
//!
//! Here the pattern is computed from `node_id` alone and the assembly becomes
//! an accumulation into a fixed `data` array. Two consequences beyond the
//! time: no COO triplets are materialised at all (~96 MB per chunk saved),
//! and the merge tree disappears, so the result no longer depends on how many
//! chunks it happened to be built from.
//!
//! **Bit-identical to a flat ascending COO sum, by construction.** The
//! destination of entry `(p, K, i, L, j)` is a function of the pattern alone,
//! and each node's entries are owned by one thread and walked in ascending
//! `p * npe + K` order -- the same order `coo_to_csr` sums a `(row, col)`
//! group in. So this reproduces a *single-chunk* run of the old path exactly.
//! It differs from a multi-chunk run of it in the last bits, because that one
//! summed each chunk separately and then added the partial sums in merge-tree
//! order; floating-point addition is not associative, so the old path's answer
//! moved with `chunk_entries` and this one does not.

use rayon::prelude::*;

use crate::scatter::Scatter;

/// Fixed CSR pattern for a tangent of `nc x nc` blocks on a particle stencil.
pub struct BlockPattern {
    npe: usize,
    nc: usize,
    n_nodes: usize,
    n_particles: usize,
    sc: Scatter,
    node_id: Vec<i64>,
    /// `nbr[nbr_ptr[n] .. nbr_ptr[n + 1]]`: node `n`'s neighbours, ascending.
    nbr_ptr: Vec<usize>,
    nbr: Vec<i32>,
    /// `n_nodes * nc + 1` row pointers.
    indptr: Vec<i64>,
}

impl BlockPattern {
    /// `node_id` is the flattened `(n_particles, npe)` stencil.
    ///
    /// The pattern carries every `nc x nc` block of a node pair that shares a
    /// particle, including entries that happen to be numerically zero --
    /// `eliminate_zeros` on the result prunes those, and leaving them in is
    /// what makes a block's 36 destinations contiguous and so reachable by
    /// one search per node pair rather than 36.
    pub fn new(node_id: &[i64], n_nodes: usize, npe: usize, nc: usize)
               -> Result<Self, String> {
        if npe == 0 || nc == 0 {
            return Err(format!("npe {npe} and nc {nc} must be positive"));
        }
        if node_id.len() % npe != 0 {
            return Err(format!("{} stencil entries is not a multiple of npe {npe}",
                               node_id.len()));
        }
        let n_particles = node_id.len() / npe;
        // n_dof must stay inside the i32 column index CSR uses
        let n_dof = n_nodes.checked_mul(nc)
            .ok_or_else(|| format!("{n_nodes} nodes x {nc} overflows"))?;
        if n_dof > i32::MAX as usize {
            return Err(format!("{n_dof} dofs exceeds the i32 column index"));
        }
        let sc = Scatter::new(node_id, n_nodes)?;

        // Each node's neighbour set: every node sharing a particle with it.
        let lists: Vec<Vec<i32>> = (0..n_nodes)
            .into_par_iter()
            .map(|n| {
                let mut v: Vec<i32> = Vec::new();
                for &e in sc.entries_of(n) {
                    let p = e as usize / npe;
                    for l in 0..npe {
                        v.push(node_id[p * npe + l] as i32);
                    }
                }
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();

        let mut nbr_ptr = vec![0usize; n_nodes + 1];
        for n in 0..n_nodes {
            nbr_ptr[n + 1] = nbr_ptr[n] + lists[n].len();
        }
        let mut nbr = Vec::with_capacity(nbr_ptr[n_nodes]);
        for l in &lists {
            nbr.extend_from_slice(l);
        }

        // Row `n * nc + i` holds `nc` columns for each neighbour, so every
        // row of a node has the same length and the same column list.
        let mut indptr = vec![0i64; n_dof + 1];
        for n in 0..n_nodes {
            let len = (lists[n].len() * nc) as i64;
            for i in 0..nc {
                indptr[n * nc + i + 1] = len;
            }
        }
        for r in 0..n_dof {
            indptr[r + 1] += indptr[r];
        }

        Ok(Self { npe, nc, n_nodes, n_particles, sc, node_id: node_id.to_vec(),
                  nbr_ptr, nbr, indptr })
    }

    pub fn n_nodes(&self) -> usize { self.n_nodes }
    pub fn n_particles(&self) -> usize { self.n_particles }
    pub fn n_dof(&self) -> usize { self.n_nodes * self.nc }
    pub fn nnz(&self) -> usize { self.indptr[self.n_nodes * self.nc] as usize }
    pub fn indptr(&self) -> &[i64] { &self.indptr }

    /// Column indices, expanded from the neighbour lists.
    ///
    /// Materialised on request rather than stored, since the accumulation
    /// never needs it -- only SciPy does, once per call, and `nnz` i32 is the
    /// largest array here.
    pub fn indices(&self) -> Vec<i32> {
        let nc = self.nc as i32;
        let mut out = vec![0i32; self.nnz()];
        let rows: Vec<(usize, usize)> = (0..self.n_nodes)
            .map(|n| (self.indptr[n * self.nc] as usize,
                      self.indptr[(n + 1) * self.nc] as usize))
            .collect();
        let mut pieces: Vec<&mut [i32]> = Vec::with_capacity(self.n_nodes);
        let mut rest = &mut out[..];
        for &(lo, hi) in &rows {
            let (head, tail) = rest.split_at_mut(hi - lo);
            pieces.push(head);
            rest = tail;
        }
        pieces.into_par_iter().enumerate().for_each(|(n, slab)| {
            let list = &self.nbr[self.nbr_ptr[n]..self.nbr_ptr[n + 1]];
            let width = list.len() * self.nc;
            for i in 0..self.nc {
                let row = &mut slab[i * width..(i + 1) * width];
                for (b, &m) in list.iter().enumerate() {
                    for j in 0..self.nc {
                        row[b * self.nc + j] = m * nc + j as i32;
                    }
                }
            }
        });
        out
    }

    /// Add the per-particle blocks of `lo .. hi` into `data`.
    ///
    /// `blocks` is `(hi - lo, npe, nc, npe, nc)` row-major -- particle, row
    /// slot, row component, column slot, column component -- which is the
    /// shape the NumPy assembly already builds. `data` is the full `nnz`
    /// array and is added into, not overwritten, so the caller zeroes it once
    /// and walks the chunks.
    pub fn accumulate(&self, data: &mut [f64], blocks: &[f64],
                      lo: usize, hi: usize) -> Result<(), String> {
        if data.len() != self.nnz() {
            return Err(format!("data has {} entries, expected {}",
                               data.len(), self.nnz()));
        }
        if hi < lo || hi > self.n_particles {
            return Err(format!("chunk {lo}..{hi} outside 0..{}", self.n_particles));
        }
        let (npe, nc) = (self.npe, self.nc);
        let per_particle = npe * nc * npe * nc;
        if blocks.len() != (hi - lo) * per_particle {
            return Err(format!("blocks has {} entries, expected {}",
                               blocks.len(), (hi - lo) * per_particle));
        }
        if hi == lo {
            return Ok(());
        }
        let (e_lo, e_hi) = ((lo * npe) as u32, (hi * npe) as u32);

        // Node `n` owns rows `n * nc .. (n + 1) * nc`, and those are a
        // contiguous run of `data` disjoint from every other node's -- which
        // is what lets each node go to one thread with no atomics.
        let mut pieces: Vec<&mut [f64]> = Vec::with_capacity(self.n_nodes);
        let mut rest = &mut data[..];
        for n in 0..self.n_nodes {
            let len = (self.indptr[(n + 1) * nc] - self.indptr[n * nc]) as usize;
            let (head, tail) = rest.split_at_mut(len);
            pieces.push(head);
            rest = tail;
        }

        pieces.into_par_iter().enumerate().for_each(|(n, slab)| {
            if slab.is_empty() {
                return;
            }
            let list = &self.nbr[self.nbr_ptr[n]..self.nbr_ptr[n + 1]];
            let width = list.len() * nc;
            let ent = self.sc.entries_of(n);
            // Each node's entries ascend in `p * npe + K`, so a contiguous
            // range of particles is a contiguous slice of them. Scanning the
            // whole list per chunk would be O(entries x chunks), which is
            // 5e9 checks at production size.
            let a = ent.partition_point(|&e| e < e_lo);
            let b = ent.partition_point(|&e| e < e_hi);
            for &e in &ent[a..b] {
                let p = e as usize / npe;
                let k = e as usize % npe;
                let base = (p - lo) * per_particle + k * nc * npe * nc;
                for l in 0..npe {
                    let m = self.node_id[p * npe + l] as i32;
                    // the neighbour list was built from these very ids, so
                    // the search cannot miss
                    let col = match list.binary_search(&m) {
                        Ok(c) => c * nc,
                        Err(_) => continue,
                    };
                    for i in 0..nc {
                        let row = i * width;
                        let src = base + i * npe * nc + l * nc;
                        for j in 0..nc {
                            slab[row + col + j] += blocks[src + j];
                        }
                    }
                }
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::coo_to_csr;

    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            (u - 0.5) * 10f64.powi(((s >> 60) % 7) as i32 - 3)
        }).collect()
    }

    /// The reference is the path this replaces: build every triplet and sort
    /// it. Equality is asserted to the bit because that is the property the
    /// ascending-order ownership buys, not a tolerance it happens to meet.
    fn via_coo(node_id: &[i64], n_nodes: usize, npe: usize, nc: usize,
               blocks: &[f64]) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
        let n_p = node_id.len() / npe;
        let n_dof = n_nodes * nc;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for p in 0..n_p {
            for k in 0..npe {
                for i in 0..nc {
                    for l in 0..npe {
                        for j in 0..nc {
                            rows.push(node_id[p * npe + k] * nc as i64 + i as i64);
                            cols.push(node_id[p * npe + l] * nc as i64 + j as i64);
                            vals.push(blocks[((p * npe + k) * nc + i) * npe * nc
                                             + l * nc + j]);
                        }
                    }
                }
            }
        }
        coo_to_csr(&rows, &cols, &vals, n_dof, n_dof).unwrap()
    }

    #[test]
    fn matches_the_coo_path_bit_for_bit() {
        let (n_nodes, n_p, npe, nc) = (300usize, 400usize, 8usize, 6usize);
        let ids = pseudo_random(n_p * npe, 3);
        let node_id: Vec<i64> =
            ids.iter().map(|v| ((v.abs() * 1e6) as usize % n_nodes) as i64).collect();
        let blocks = pseudo_random(n_p * npe * nc * npe * nc, 5);

        let pat = BlockPattern::new(&node_id, n_nodes, npe, nc).unwrap();
        let mut data = vec![0.0; pat.nnz()];
        pat.accumulate(&mut data, &blocks, 0, n_p).unwrap();

        let (ip, ix, d) = via_coo(&node_id, n_nodes, npe, nc, &blocks);
        assert_eq!(pat.indptr(), &ip[..], "row pointers");
        assert_eq!(pat.indices(), ix, "column indices");
        assert_eq!(data, d, "values must agree to the bit");
    }

    #[test]
    fn chunking_does_not_change_the_answer() {
        // the whole point: the old path's merge tree made the result depend
        // on the chunk count, and this must not
        let (n_nodes, n_p, npe, nc) = (120usize, 350usize, 8usize, 6usize);
        let ids = pseudo_random(n_p * npe, 13);
        let node_id: Vec<i64> =
            ids.iter().map(|v| ((v.abs() * 1e6) as usize % n_nodes) as i64).collect();
        let blocks = pseudo_random(n_p * npe * nc * npe * nc, 17);
        let per = npe * nc * npe * nc;

        let pat = BlockPattern::new(&node_id, n_nodes, npe, nc).unwrap();
        let mut whole = vec![0.0; pat.nnz()];
        pat.accumulate(&mut whole, &blocks, 0, n_p).unwrap();

        for chunk in [1usize, 7, 64, 349, 350] {
            let mut got = vec![0.0; pat.nnz()];
            let mut lo = 0;
            while lo < n_p {
                let hi = (lo + chunk).min(n_p);
                pat.accumulate(&mut got, &blocks[lo * per..hi * per], lo, hi).unwrap();
                lo = hi;
            }
            assert_eq!(got, whole, "chunk size {chunk} changed the answer");
        }
    }

    #[test]
    fn handles_a_duplicated_stencil_node() {
        // CPDI concatenates corner stencils, so the same node can appear
        // twice for one particle and both copies must land
        let node_id = vec![0i64, 1, 1, 2];
        let (npe, nc, n_nodes) = (4usize, 1usize, 3usize);
        let blocks: Vec<f64> = (1..=16).map(|v| v as f64).collect();
        let pat = BlockPattern::new(&node_id, n_nodes, npe, nc).unwrap();
        let mut data = vec![0.0; pat.nnz()];
        pat.accumulate(&mut data, &blocks, 0, 1).unwrap();
        let (_, _, d) = via_coo(&node_id, n_nodes, npe, nc, &blocks);
        assert_eq!(data, d);
    }

    #[test]
    fn unsupported_nodes_give_empty_rows() {
        let node_id = vec![3i64, 4];
        let pat = BlockPattern::new(&node_id, 6, 2, 2).unwrap();
        // nodes 0, 1, 2 and 5 touch no particle
        assert_eq!(pat.indptr()[0..7], [0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(pat.nnz(), 2 * 2 * 2 * 2);
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(BlockPattern::new(&[0i64, 1, 2], 4, 2, 6).is_err());
        assert!(BlockPattern::new(&[0i64, 9], 4, 2, 6).is_err());
        let pat = BlockPattern::new(&[0i64, 1], 2, 2, 2).unwrap();
        let mut data = vec![0.0; pat.nnz()];
        assert!(pat.accumulate(&mut data, &[0.0; 4], 0, 1).is_err());
        assert!(pat.accumulate(&mut data[..1], &[0.0; 16], 0, 1).is_err());
        assert!(pat.accumulate(&mut data, &[0.0; 16], 0, 3).is_err());
    }
}
