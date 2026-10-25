//! Standalone timing of the scatter, so the kernel can be measured without
//! Python -- and in particular on both the native and the emulated target.
use scion_mpm::Scatter;
use std::time::Instant;

fn main() {
    let (n_p, npe, n_nodes, ncomp, k) = (768_000usize, 27usize, 116_487usize, 6usize, 3usize);
    let mut s = 12345u64;
    let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); s };
    let node_id: Vec<i64> = (0..n_p * npe).map(|_| (next() % n_nodes as u64) as i64).collect();
    let vals: Vec<f64> = (0..n_p * npe * k).map(|_| (next() >> 11) as f64 / 1e15).collect();

    let t = Instant::now();
    let sc = Scatter::new(&node_id, n_nodes).unwrap();
    let build = t.elapsed();

    let mut out = vec![0.0; n_nodes * ncomp];
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        sc.add(&mut out, ncomp, 0, &vals, k).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{:>24}  build {:7.1} ms   add {:6.1} ms",
             std::env::consts::ARCH, build.as_secs_f64() * 1e3, best * 1e3);
}
