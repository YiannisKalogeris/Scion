//! COO -> CSR for the tangent, the second cost in a micropolar Newton step.
//!
//! Replaces `sparse.coo_matrix((v, (r, c))).tocsr()`. SciPy spends its time in
//! `coo_tocsr` plus a per-row `std::sort`, single-threaded; measured at 27% of
//! a staggered step once `optimize=True` had taken the einsums down.
//!
//! **Not bit-identical to SciPy, by one ulp.** SciPy's per-row sort is
//! unstable, so the order in which it adds entries sharing a (row, col) is not
//! reproducible. This sorts by `(key, original index)`, a total order, so the
//! result is deterministic and independent of the thread count -- but it is a
//! different summation order, worth ~1e-16 relative. Deterministic was
//! achievable; identical was not.

use rayon::prelude::*;

/// Sort COO triplets into CSR, summing duplicates.
///
/// `rows` and `cols` are the triplet coordinates, `vals` their values, and
/// `n_rows` the matrix row count. Returns `(indptr, indices, data)`.
pub fn coo_to_csr(rows: &[i64], cols: &[i64], vals: &[f64], n_rows: usize,
                  n_cols: usize) -> Result<(Vec<i64>, Vec<i32>, Vec<f64>), String> {
    let n = rows.len();
    if cols.len() != n || vals.len() != n {
        return Err(format!("rows {}, cols {}, vals {} must agree",
                           n, cols.len(), vals.len()));
    }
    if n == 0 {
        return Ok((vec![0; n_rows + 1], Vec::new(), Vec::new()));
    }
    if n > u32::MAX as usize {
        return Err(format!("{n} triplets exceeds the u32 index"));
    }
    let nc = n_cols as u64;

    // (row * n_cols + col, original index). Including the index makes the key
    // unique, so an unstable parallel sort still yields one fixed order.
    let mut keyed: Vec<(u64, u32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (r, c) = (rows[i], cols[i]);
            ((r as u64) * nc + c as u64, i as u32)
        })
        .collect();
    for i in 0..n {
        if rows[i] < 0 || rows[i] as usize >= n_rows
            || cols[i] < 0 || cols[i] as usize >= n_cols {
            return Err(format!("entry {i} at ({}, {}) outside {n_rows} x {n_cols}",
                               rows[i], cols[i]));
        }
    }
    keyed.par_sort_unstable();

    // one pass to find the distinct (row, col) groups
    let mut starts: Vec<u32> = Vec::with_capacity(n / 4 + 1);
    let mut last = u64::MAX;
    for (j, &(k, _)) in keyed.iter().enumerate() {
        if k != last {
            starts.push(j as u32);
            last = k;
        }
    }
    let nnz = starts.len();
    starts.push(n as u32);

    let mut indices = vec![0i32; nnz];
    let mut data = vec![0.0f64; nnz];
    indices
        .par_iter_mut()
        .zip(data.par_iter_mut())
        .enumerate()
        .for_each(|(g, (ix, dv))| {
            let (lo, hi) = (starts[g] as usize, starts[g + 1] as usize);
            *ix = (keyed[lo].0 % nc) as i32;
            // ascending original index: a fixed order, so the thread count
            // cannot change the last bits
            let mut acc = 0.0;
            for &(_, i) in &keyed[lo..hi] {
                acc += vals[i as usize];
            }
            *dv = acc;
        });

    let mut indptr = vec![0i64; n_rows + 1];
    for g in 0..nnz {
        let r = (keyed[starts[g] as usize].0 / nc) as usize;
        indptr[r + 1] += 1;
    }
    for r in 0..n_rows {
        indptr[r + 1] += indptr[r];
    }
    Ok((indptr, indices, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_duplicates_and_sorts_columns() {
        let rows = vec![1i64, 0, 1, 0, 1];
        let cols = vec![2i64, 1, 0, 1, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (ip, ix, d) = coo_to_csr(&rows, &cols, &vals, 2, 3).unwrap();
        assert_eq!(ip, vec![0, 1, 3]);
        assert_eq!(ix, vec![1, 0, 2]);
        assert_eq!(d, vec![6.0, 3.0, 6.0]);
    }

    #[test]
    fn empty_input_gives_an_empty_matrix() {
        let (ip, ix, d) = coo_to_csr(&[], &[], &[], 3, 3).unwrap();
        assert_eq!(ip, vec![0, 0, 0, 0]);
        assert!(ix.is_empty() && d.is_empty());
    }

    #[test]
    fn rejects_out_of_range() {
        assert!(coo_to_csr(&[5], &[0], &[1.0], 2, 2).is_err());
        assert!(coo_to_csr(&[0], &[-1], &[1.0], 2, 2).is_err());
    }

    #[test]
    fn repeated_calls_agree_to_the_bit() {
        let n = 50_000;
        let mut s = 1u64;
        let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); s };
        let rows: Vec<i64> = (0..n).map(|_| (next() % 97) as i64).collect();
        let cols: Vec<i64> = (0..n).map(|_| (next() % 97) as i64).collect();
        let vals: Vec<f64> = (0..n).map(|_| (next() >> 11) as f64 / 1e12).collect();
        let a = coo_to_csr(&rows, &cols, &vals, 97, 97).unwrap();
        for _ in 0..4 {
            assert_eq!(coo_to_csr(&rows, &cols, &vals, 97, 97).unwrap(), a);
        }
    }
}
