//! ParaView output: one `.vtu` per frame plus the `.pvd` that gives them a
//! clock, so a run driven from `cargo` animates exactly as the NumPy one does.
//!
//! The counterpart of `scion.io.particles` and `scion.io.time_series`, and
//! deliberately writing the **same array names into the same directory
//! layout** -- `particles.pvd` beside `particles_000000.vtu`, with `speed`,
//! `pressure`, `velocity`, `vorticity`, `displacement`, `clamped` spelled as
//! Python spells them. That is the whole point: one ParaView state file, or
//! one set of hand-made settings, opens either code's output, and the two can
//! be loaded side by side and compared frame for frame.
//!
//! **ASCII, not base64 binary.** Roughly three times the bytes -- the 20 s dam
//! break lands near 120 MB against the NumPy path's 42 -- and the reason to
//! accept that is that the file stays greppable, so a disagreement with the
//! NumPy series can be read rather than decoded. Values carry nine
//! significant digits, far past anything a renderer resolves and past the
//! ~1e-16 the two implementations agree to anyway; this is output for looking
//! at, not a checkpoint.
//!
//! Particles are `vertex` cells in an `UnstructuredGrid` rather than
//! `PolyData`, for the reason `scion/io/particles.py` records: meshio cannot
//! read `.vtp`, so a `.vtu` is what both this crate's output and Python's can
//! be checked with.

use std::fs::{self, File};
use std::io::{BufWriter, Result, Write};
use std::path::{Path, PathBuf};

const VTK_VERTEX: u8 = 1;
const VTK_LINE: u8 = 3;

/// One point array. Vectors are padded to three components on the way out,
/// because VTK vectors are always 3-D -- the same padding `pad_to_3d` does.
pub enum Field<'a> {
    Scalar(&'a [f64]),
    Vector(&'a [[f64; 2]]),
}

fn write_header(out: &mut impl Write, n_points: usize, n_cells: usize) -> Result<()> {
    writeln!(out, "<?xml version=\"1.0\"?>")?;
    writeln!(
        out,
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(out, " <UnstructuredGrid>")?;
    writeln!(
        out,
        "  <Piece NumberOfPoints=\"{n_points}\" NumberOfCells=\"{n_cells}\">"
    )
}

fn write_points(out: &mut impl Write, points: &[[f64; 2]]) -> Result<()> {
    writeln!(out, "   <Points>")?;
    writeln!(
        out,
        "    <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    for p in points {
        writeln!(out, "{:.9e} {:.9e} 0", p[0], p[1])?;
    }
    writeln!(out, "    </DataArray>")?;
    writeln!(out, "   </Points>")
}

/// Write a particle cloud: one `vertex` cell per point.
pub fn write_particles(
    path: impl AsRef<Path>,
    points: &[[f64; 2]],
    data: &[(&str, Field)],
) -> Result<PathBuf> {
    let path = path.as_ref().to_path_buf();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(File::create(&path)?);
    let n = points.len();
    write_header(&mut out, n, n)?;
    write_points(&mut out, points)?;

    writeln!(out, "   <Cells>")?;
    writeln!(
        out,
        "    <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">"
    )?;
    for i in 0..n {
        writeln!(out, "{i}")?;
    }
    writeln!(out, "    </DataArray>")?;
    writeln!(
        out,
        "    <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">"
    )?;
    for i in 1..=n {
        writeln!(out, "{i}")?;
    }
    writeln!(out, "    </DataArray>")?;
    writeln!(
        out,
        "    <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )?;
    for _ in 0..n {
        writeln!(out, "{VTK_VERTEX}")?;
    }
    writeln!(out, "    </DataArray>")?;
    writeln!(out, "   </Cells>")?;

    writeln!(out, "   <PointData>")?;
    for (name, field) in data {
        match field {
            Field::Scalar(values) => {
                assert_eq!(values.len(), n, "point array '{name}' has the wrong length");
                writeln!(
                    out,
                    // no NumberOfComponents on a scalar: meshio then reads
                    // it back as (n,) exactly as the Python writer's, so the
                    // two codes' files compare element for element
                    "    <DataArray type=\"Float64\" Name=\"{name}\" format=\"ascii\">"
                )?;
                for v in values.iter() {
                    writeln!(out, "{v:.9e}")?;
                }
            }
            Field::Vector(values) => {
                assert_eq!(values.len(), n, "point array '{name}' has the wrong length");
                writeln!(
                    out,
                    "    <DataArray type=\"Float64\" Name=\"{name}\" \
                     NumberOfComponents=\"3\" format=\"ascii\">"
                )?;
                for v in values.iter() {
                    writeln!(out, "{:.9e} {:.9e} 0", v[0], v[1])?;
                }
            }
        }
        writeln!(out, "    </DataArray>")?;
    }
    writeln!(out, "   </PointData>")?;
    writeln!(out, "  </Piece>")?;
    writeln!(out, " </UnstructuredGrid>")?;
    writeln!(out, "</VTKFile>")?;
    Ok(path)
}

/// Write a wireframe of line segments: a tank, an obstacle, a periodic cell,
/// an undeformed outline. Static geometry, written once beside the series.
pub fn write_lines(
    path: impl AsRef<Path>,
    points: &[[f64; 2]],
    segments: &[[usize; 2]],
) -> Result<PathBuf> {
    let path = path.as_ref().to_path_buf();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(File::create(&path)?);
    write_header(&mut out, points.len(), segments.len())?;
    write_points(&mut out, points)?;
    writeln!(out, "   <Cells>")?;
    writeln!(
        out,
        "    <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">"
    )?;
    for s in segments {
        writeln!(out, "{} {}", s[0], s[1])?;
    }
    writeln!(out, "    </DataArray>")?;
    writeln!(
        out,
        "    <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">"
    )?;
    for k in 1..=segments.len() {
        writeln!(out, "{}", 2 * k)?;
    }
    writeln!(out, "    </DataArray>")?;
    writeln!(
        out,
        "    <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )?;
    for _ in segments {
        writeln!(out, "{VTK_LINE}")?;
    }
    writeln!(out, "    </DataArray>")?;
    writeln!(out, "   </Cells>")?;
    writeln!(out, "  </Piece>")?;
    writeln!(out, " </UnstructuredGrid>")?;
    writeln!(out, "</VTKFile>")?;
    Ok(path)
}

/// Build a closed line loop from a polygon's vertices, offset into a shared
/// point list -- what every outline in these examples is made of.
pub fn closed_loop(points: &mut Vec<[f64; 2]>, loop_points: &[[f64; 2]]) -> Vec<[usize; 2]> {
    let start = points.len();
    points.extend_from_slice(loop_points);
    let n = loop_points.len();
    (0..n).map(|i| [start + i, start + (i + 1) % n]).collect()
}

/// Numbered frames and their `.pvd`, rewritten after each one.
///
/// Rewriting the collection every frame rather than at the end is the same
/// choice `TimeSeriesWriter` makes and for the same reason: a run killed at
/// hour three still leaves a series that loads.
pub struct Series {
    directory: PathBuf,
    name: String,
    entries: Vec<(f64, String)>,
}

impl Series {
    pub fn new(directory: impl AsRef<Path>, name: &str) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            name: name.to_string(),
            entries: Vec::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn collection_path(&self) -> PathBuf {
        self.directory.join(format!("{}.pvd", self.name))
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn add(&mut self, time: f64, points: &[[f64; 2]], data: &[(&str, Field)]) -> Result<()> {
        let file_name = format!("{}_{:06}.vtu", self.name, self.entries.len());
        write_particles(self.directory.join(&file_name), points, data)?;
        self.entries.push((time, file_name));
        self.write_collection()
    }

    fn write_collection(&self) -> Result<()> {
        let mut out = BufWriter::new(File::create(self.collection_path())?);
        writeln!(out, "<?xml version=\"1.0\"?>")?;
        writeln!(
            out,
            "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">"
        )?;
        writeln!(out, "<Collection>")?;
        for (t, file_name) in &self.entries {
            writeln!(
                out,
                "<DataSet timestep=\"{t:.17}\" group=\"\" part=\"0\" file=\"{file_name}\" />"
            )?;
        }
        writeln!(out, "</Collection>")?;
        writeln!(out, "</VTKFile>")
    }
}

/// Positional arguments and `--flag value` pairs.
///
/// Lives in the library rather than beside the examples because a Cargo
/// example cannot import from a sibling example, and the four of them have to
/// expose the *same* flags -- `--vtk <dir>` meaning the same thing everywhere
/// is the only way the Rust and Python runs stay interchangeable.
pub struct Args {
    positional: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Default for Args {
    fn default() -> Self {
        Self::parse()
    }
}

impl Args {
    pub fn parse() -> Self {
        let mut positional = Vec::new();
        let mut flags = Vec::new();
        let mut it = std::env::args().skip(1);
        while let Some(token) = it.next() {
            if let Some(name) = token.strip_prefix("--") {
                // every flag here takes a value; a bare one would silently
                // swallow the next positional, so say so instead
                let value = it
                    .next()
                    .unwrap_or_else(|| panic!("--{name} needs a value"));
                flags.push((name.to_string(), value));
            } else {
                positional.push(token);
            }
        }
        Self { positional, flags }
    }

    pub fn flag(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    pub fn positional(&self, index: usize) -> Option<&str> {
        self.positional.get(index).map(String::as_str)
    }

    pub fn parsed<T: std::str::FromStr>(&self, index: usize, fallback: T) -> T {
        self.positional(index)
            .and_then(|v| v.parse().ok())
            .unwrap_or(fallback)
    }
}
