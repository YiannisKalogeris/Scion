"""
Tests for ``scion.io``.

The ``.vti`` writer is hand-rolled XML with an appended binary payload, so the
helper below parses it back *independently* rather than through the library
that produced it. Checking a hand-written format against its own writer only
proves the writer is self-consistent; it cannot catch an offset convention that
is wrong in the same way at both ends.

The mesh and particle writers go through meshio, which is an independent reader
already, so those round-trip through ``meshio.read``.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from scion.io import (
    TimeSeriesWriter,
    flatten_for_vtk,
    pad_to_3d,
    write_image_vti,
    write_particles_vtu,
    write_vtk,
    write_vtk_time_series,
)
from scion.mesh.mesh import ElementBlock, Mesh

meshio = pytest.importorskip("meshio")


# ---------------------------------------------------------------------------
# Independent .vti reader
# ---------------------------------------------------------------------------

_VTK_TO_NUMPY = {
    "Float32": np.float32, "Float64": np.float64,
    "Int32": np.int32, "Int64": np.int64,
    "UInt8": np.uint8, "UInt32": np.uint32, "UInt64": np.uint64,
}


def _read_vti(path: Path) -> dict:
    """Parse a .vti written by ``write_image_vti`` without using scion.

    Returns ``{extent, origin, spacing, arrays: {name: ndarray}}`` with each
    array reshaped to ``(n_points, n_components)`` when it has more than one.
    """
    raw = Path(path).read_bytes()

    # split the XML header from the appended binary block
    marker = raw.index(b"<AppendedData")
    payload_start = raw.index(b"_", marker) + 1
    payload_end = raw.rindex(b"</AppendedData>")

    header = raw[:payload_start].decode("ascii", errors="replace")
    # close the tags the payload interrupted so ElementTree can parse it
    header_xml = header[header.index("<VTKFile"):] + "</AppendedData></VTKFile>"
    root = ET.fromstring(header_xml)

    img = root.find("ImageData")
    piece = img.find("Piece")

    arrays = {}
    for da in piece.find("PointData").findall("DataArray"):
        offset = int(da.attrib["offset"])
        ncomp = int(da.attrib.get("NumberOfComponents", 1))
        dtype = _VTK_TO_NUMPY[da.attrib["type"]]

        at = payload_start + offset
        n_bytes = struct.unpack("<Q", raw[at:at + 8])[0]
        buf = raw[at + 8:at + 8 + n_bytes]
        assert at + 8 + n_bytes <= payload_end, "block runs past the payload"
        vals = np.frombuffer(buf, dtype=dtype)
        arrays[da.attrib["Name"]] = vals.reshape(-1, ncomp) if ncomp > 1 else vals

    return {
        "extent": [int(v) for v in img.attrib["WholeExtent"].split()],
        "origin": [float(v) for v in img.attrib["Origin"].split()],
        "spacing": [float(v) for v in img.attrib["Spacing"].split()],
        "arrays": arrays,
    }


@pytest.fixture
def quad_mesh() -> Mesh:
    """Unit square, two tri3 elements, 2-D coordinates."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    conn = np.array([[0, 1, 2], [0, 2, 3]])
    return Mesh(nodes=nodes, element_blocks=[ElementBlock("tri3", conn)])


# ---------------------------------------------------------------------------
# vtk_writer.py
# ---------------------------------------------------------------------------

class TestMeshWriter:
    def test_round_trip_through_meshio(self, tmp_path, quad_mesh):
        p = tmp_path / "m.vtu"
        T = np.array([1.0, 2.0, 3.0, 4.0])
        write_vtk(p, quad_mesh, point_data={"T": T})

        m = meshio.read(p)
        assert np.allclose(m.points[:, :2], quad_mesh.nodes)
        assert m.cells[0].type == "triangle"
        assert np.array_equal(m.cells[0].data, quad_mesh.element_blocks[0].connectivity)
        assert np.allclose(m.point_data["T"], T)

    def test_2d_nodes_padded_to_three(self, tmp_path, quad_mesh):
        p = tmp_path / "m.vtu"
        write_vtk(p, quad_mesh)
        m = meshio.read(p)
        assert m.points.shape == (4, 3)
        assert np.allclose(m.points[:, 2], 0.0)

    def test_two_component_vectors_padded_to_three(self, tmp_path, quad_mesh):
        """A 2-component array is not a VTK vector.

        ``write_particles_vtu`` has always padded; this writer did not, so the
        same displacement field came back 2-component or 3-component depending
        on which writer produced it.
        """
        u = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        p = tmp_path / "m.vtu"
        write_vtk(p, quad_mesh, point_data={"u": u})

        back = meshio.read(p).point_data["u"]
        assert back.shape == (4, 3)
        assert np.allclose(back[:, :2], u)
        assert np.allclose(back[:, 2], 0.0)

    def test_creates_missing_parent_directories(self, tmp_path, quad_mesh):
        """The other two writers mkdir; this one raised FileNotFoundError."""
        p = tmp_path / "deep" / "nested" / "m.vtu"
        write_vtk(p, quad_mesh)
        assert p.exists()

    def test_returns_the_path(self, tmp_path, quad_mesh):
        p = tmp_path / "m.vtu"
        assert write_vtk(p, quad_mesh) == p

    def test_cell_data_round_trip(self, tmp_path, quad_mesh):
        p = tmp_path / "m.vtu"
        write_vtk(p, quad_mesh, cell_data={"mat": [np.array([1.0, 2.0])]})
        m = meshio.read(p)
        assert np.allclose(np.concatenate(m.cell_data["mat"]), [1.0, 2.0])

    def test_unknown_element_type_names_the_supported_ones(self, tmp_path):
        """meshio's bare ``KeyError: 'wedge6'`` says nothing about the fix."""
        mesh = Mesh(
            nodes=np.zeros((4, 2)),
            element_blocks=[ElementBlock("wedge6", np.array([[0, 1, 2]]))],
        )
        with pytest.raises(KeyError, match="tri3"):
            write_vtk(tmp_path / "bad.vtu", mesh)

    @pytest.mark.parametrize("etype,expected", [
        ("tri3", "triangle"), ("quad4", "quad"), ("tet4", "tetra"),
        ("hex8", "hexahedron"), ("line2", "line"),
    ])
    def test_element_type_mapping(self, tmp_path, etype, expected):
        from scion.io.vtk_writer import _SIM_TO_MESHIO
        assert _SIM_TO_MESHIO[etype] == expected


class TestMeshTimeSeries:
    def test_writes_every_frame_and_a_pvd(self, tmp_path, quad_mesh):
        times = [0.0, 0.5, 1.0]
        fields = [{"T": np.full(4, float(i))} for i in range(3)]
        paths = write_vtk_time_series(tmp_path, "run", quad_mesh, times, fields)

        assert len(paths) == 3
        assert all(p.exists() for p in paths)
        assert [p.name for p in paths] == [
            "run_000000.vtu", "run_000001.vtu", "run_000002.vtu"
        ]

        pvd = (tmp_path / "run.pvd").read_text()
        assert re.findall(r'timestep="([^"]*)"', pvd) == ["0.0", "0.5", "1.0"]
        # every file the collection names must actually be on disk
        for fname in re.findall(r'file="([^"]*)"', pvd):
            assert (tmp_path / fname).exists()

    def test_length_mismatch_raises_instead_of_truncating(self, tmp_path, quad_mesh):
        """zip() wrote min(len, len) frames and dropped the rest silently."""
        times = [0.0, 0.1, 0.2, 0.3, 0.4]
        fields = [{"T": np.zeros(4)}, {"T": np.ones(4)}]
        with pytest.raises(ValueError, match="5 time steps but 2 field dicts"):
            write_vtk_time_series(tmp_path, "run", quad_mesh, times, fields)

    def test_frame_contents_differ(self, tmp_path, quad_mesh):
        fields = [{"T": np.full(4, 7.0)}, {"T": np.full(4, 9.0)}]
        paths = write_vtk_time_series(tmp_path, "run", quad_mesh, [0.0, 1.0], fields)
        assert np.allclose(meshio.read(paths[0]).point_data["T"], 7.0)
        assert np.allclose(meshio.read(paths[1]).point_data["T"], 9.0)


# ---------------------------------------------------------------------------
# particles.py
# ---------------------------------------------------------------------------

class TestPadTo3D:
    @pytest.mark.parametrize("cols", [1, 2])
    def test_pads_narrow_arrays(self, cols):
        a = np.arange(4 * cols, dtype=float).reshape(4, cols)
        out = pad_to_3d(a)
        assert out.shape == (4, 3)
        assert np.allclose(out[:, :cols], a)
        assert np.allclose(out[:, cols:], 0.0)

    def test_three_columns_passes_through_unchanged(self):
        a = np.arange(12.0).reshape(4, 3)
        assert pad_to_3d(a) is a

    def test_one_dimensional_passes_through(self):
        a = np.arange(5.0)
        assert pad_to_3d(a) is a

    def test_dtype_is_preserved(self):
        assert pad_to_3d(np.ones((3, 2), dtype=np.int32)).dtype == np.int32

    def test_more_than_three_columns_explains_itself(self):
        """Used to surface as 'could not broadcast (4,4) into (4,3)'."""
        with pytest.raises(ValueError, match="at most 3"):
            pad_to_3d(np.ones((4, 4)))


class TestParticleWriter:
    def test_round_trip(self, tmp_path):
        x = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        m = np.array([0.5, 1.5, 2.5])

        p = write_particles_vtu(tmp_path / "p.vtu", x, point_data={"v": v, "m": m})
        back = meshio.read(p)

        assert back.points.shape == (3, 3)
        assert np.allclose(back.points[:, :2], x)
        assert back.cells[0].type == "vertex"
        assert np.allclose(back.point_data["m"], m)
        assert back.point_data["v"].shape == (3, 3)
        assert np.allclose(back.point_data["v"][:, :2], v)

    def test_vertex_cells_index_every_particle_once(self, tmp_path):
        x = np.zeros((5, 3))
        p = write_particles_vtu(tmp_path / "p.vtu", x)
        conn = meshio.read(p).cells[0].data
        assert np.array_equal(np.sort(conn.ravel()), np.arange(5))

    def test_row_count_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="3 rows, expected 2"):
            write_particles_vtu(
                tmp_path / "p.vtu", np.zeros((2, 3)),
                point_data={"bad": np.zeros(3)},
            )

    def test_more_than_three_components_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Split it into scalar components"):
            write_particles_vtu(
                tmp_path / "p.vtu", np.zeros((2, 3)),
                point_data={"sigma": np.zeros((2, 9))},
            )

    def test_creates_missing_parent_directories(self, tmp_path):
        p = write_particles_vtu(tmp_path / "a" / "b" / "p.vtu", np.zeros((2, 3)))
        assert p.exists()

    def test_particle_count_may_change_between_frames(self, tmp_path):
        """No connectivity to keep consistent -- that is the point of the module."""
        for i, n in enumerate([3, 7, 2]):
            p = write_particles_vtu(tmp_path / f"p{i}.vtu", np.zeros((n, 3)))
            assert meshio.read(p).points.shape[0] == n


# ---------------------------------------------------------------------------
# uniform_grid.py
# ---------------------------------------------------------------------------

class TestFlattenForVtk:
    def test_scalar_2d_becomes_x_fastest(self):
        a = np.arange(6.0).reshape(2, 3)
        assert np.array_equal(flatten_for_vtk(a), a.T.ravel())

    def test_scalar_3d_reverses_all_axes(self):
        a = np.arange(24.0).reshape(2, 3, 4)
        assert np.array_equal(flatten_for_vtk(a), a.transpose(2, 1, 0).ravel())

    def test_vector_field_keeps_its_components_together(self):
        """Reversing every axis moved the component axis too.

        A (2,3,3) field whose components are uniformly 1, 2, 3 came out as
        [1,1,1,1,1,1,2,2,2,...] -- every value present, every one in the wrong
        place, and nothing about the picture to show it.
        """
        vec = np.zeros((2, 3, 3))
        vec[..., 0], vec[..., 1], vec[..., 2] = 1.0, 2.0, 3.0

        out = flatten_for_vtk(vec, n_components=3)
        assert out.shape == (6, 3)
        assert np.allclose(out[:, 0], 1.0)
        assert np.allclose(out[:, 1], 2.0)
        assert np.allclose(out[:, 2], 3.0)

    def test_vector_spatial_ordering_matches_the_scalar_case(self):
        nx, ny = 2, 3
        ramp = np.arange(nx * ny, dtype=float).reshape(nx, ny)
        vec = np.stack([ramp, ramp + 100.0], axis=-1)

        out = flatten_for_vtk(vec, n_components=2)
        assert np.array_equal(out[:, 0], flatten_for_vtk(ramp))
        assert np.array_equal(out[:, 1], flatten_for_vtk(ramp + 100.0))

    def test_wrong_component_count_is_rejected(self):
        with pytest.raises(ValueError, match="component axis must be last"):
            flatten_for_vtk(np.zeros((2, 3, 3)), n_components=2)

    def test_scalar_path_is_unchanged_by_the_new_parameter(self):
        """The one in-repo caller passes a scalar and must be unaffected."""
        a = np.arange(12.0).reshape(3, 4)
        assert np.array_equal(flatten_for_vtk(a), a.T.ravel())


class TestWriteImageVti:
    def test_geometry_header(self, tmp_path):
        p = write_image_vti(
            tmp_path / "g.vti", origin=[1.0, 2.0, 3.0],
            spacing=[0.5, 0.25, 0.125], dimensions=[3, 4, 5],
        )
        got = _read_vti(p)
        assert got["extent"] == [0, 2, 0, 3, 0, 4]
        assert got["origin"] == [1.0, 2.0, 3.0]
        assert got["spacing"] == [0.5, 0.25, 0.125]

    def test_payload_round_trips_bit_exactly(self, tmp_path):
        data = np.linspace(-1e-9, 1e9, 12)
        p = write_image_vti(tmp_path / "g.vti", [0, 0], 1.0, [3, 4],
                            point_data={"f": data})
        assert np.array_equal(_read_vti(p)["arrays"]["f"], data)

    def test_two_dimensional_grid_gets_nonzero_padded_spacing(self, tmp_path):
        """Spacing 0 on the padded axis makes any gradient filter divide by zero."""
        p = write_image_vti(tmp_path / "g.vti", origin=[0.0, 0.0],
                            spacing=[0.5, 0.25], dimensions=[3, 4])
        got = _read_vti(p)
        assert got["extent"] == [0, 2, 0, 3, 0, 0]
        assert got["spacing"][:2] == [0.5, 0.25]
        assert got["spacing"][2] != 0.0

    def test_scalar_spacing_applies_to_every_axis(self, tmp_path):
        p = write_image_vti(tmp_path / "g.vti", [0, 0, 0], 0.25, [2, 2, 2])
        assert _read_vti(p)["spacing"] == [0.25, 0.25, 0.25]

    def test_multiple_arrays_get_distinct_offsets(self, tmp_path):
        a = np.arange(6.0)
        b = np.arange(6.0) * -3.0
        p = write_image_vti(tmp_path / "g.vti", [0, 0], 1.0, [2, 3],
                            point_data={"a": a, "b": b})
        got = _read_vti(p)["arrays"]
        assert np.array_equal(got["a"], a)
        assert np.array_equal(got["b"], b)

    def test_vector_field_declares_its_component_count(self, tmp_path):
        vec = np.arange(18.0).reshape(6, 3)
        p = write_image_vti(tmp_path / "g.vti", [0, 0], 1.0, [2, 3],
                            point_data={"v": vec})
        got = _read_vti(p)["arrays"]["v"]
        assert got.shape == (6, 3)
        assert np.array_equal(got, vec)

    def test_wrong_tuple_count_raises(self, tmp_path):
        with pytest.raises(ValueError, match="expected 12"):
            write_image_vti(tmp_path / "g.vti", [0, 0], 1.0, [3, 4],
                            point_data={"f": np.zeros(11)})

    def test_unsupported_dtype_falls_back_to_float64(self, tmp_path):
        p = write_image_vti(tmp_path / "g.vti", [0, 0], 1.0, [2, 3],
                            point_data={"f": np.arange(6, dtype=np.float16)})
        assert _read_vti(p)["arrays"]["f"].dtype == np.float64

    def test_integer_dtype_survives(self, tmp_path):
        data = np.arange(6, dtype=np.int32)
        p = write_image_vti(tmp_path / "g.vti", [0, 0], 1.0, [2, 3],
                            point_data={"f": data})
        got = _read_vti(p)["arrays"]["f"]
        assert got.dtype == np.int32
        assert np.array_equal(got, data)

    def test_creates_missing_parent_directories(self, tmp_path):
        p = write_image_vti(tmp_path / "x" / "y" / "g.vti", [0, 0], 1.0, [2, 2])
        assert p.exists()

    def test_declared_block_length_matches_the_data(self, tmp_path):
        """The uint64 header before each block is what a reader trusts."""
        data = np.arange(6.0)
        p = write_image_vti(tmp_path / "g.vti", [0, 0], 1.0, [2, 3],
                            point_data={"f": data})
        raw = p.read_bytes()
        start = raw.index(b"_", raw.index(b"<AppendedData")) + 1
        assert struct.unpack("<Q", raw[start:start + 8])[0] == data.nbytes

    def test_header_is_ascii_and_well_formed(self, tmp_path):
        p = write_image_vti(tmp_path / "g.vti", [0, 0], 1.0, [2, 3],
                            point_data={"f": np.zeros(6)})
        raw = p.read_bytes()
        head = raw[:raw.index(b"<AppendedData")]
        head.decode("ascii")  # raises if the writer emitted anything else
        assert head.startswith(b'<?xml version="1.0"?>')


# ---------------------------------------------------------------------------
# time_series.py
# ---------------------------------------------------------------------------

def _touch(path, text="x"):
    Path(path).write_text(text)


class TestTimeSeriesWriter:
    def test_frames_are_numbered_from_zero(self, tmp_path):
        s = TimeSeriesWriter(tmp_path, "f", suffix=".vtu")
        assert s.add(0.0, _touch).name == "f_000000.vtu"
        assert s.add(1.0, _touch).name == "f_000001.vtu"
        assert len(s) == 2

    def test_creates_its_directory(self, tmp_path):
        d = tmp_path / "not" / "there"
        TimeSeriesWriter(d, "f")
        assert d.is_dir()

    def test_collection_is_rewritten_after_every_frame(self, tmp_path):
        """An interrupted run must still leave a loadable series."""
        s = TimeSeriesWriter(tmp_path, "f")
        s.add(0.0, _touch)
        after_one = (tmp_path / "f.pvd").read_text()
        assert after_one.count("<DataSet") == 1

        s.add(1.0, _touch)
        after_two = (tmp_path / "f.pvd").read_text()
        assert after_two.count("<DataSet") == 2

    def test_timesteps_survive_to_full_precision(self, tmp_path):
        s = TimeSeriesWriter(tmp_path, "f")
        t = 0.1 + 0.2  # 0.30000000000000004
        s.add(t, _touch)
        written = re.findall(r'timestep="([^"]*)"', (tmp_path / "f.pvd").read_text())
        assert float(written[0]) == t

    def test_collection_lists_only_files_that_exist(self, tmp_path):
        s = TimeSeriesWriter(tmp_path, "f")
        for i in range(3):
            s.add(float(i), _touch)
        pvd = (tmp_path / "f.pvd").read_text()
        for fname in re.findall(r'file="([^"]*)"', pvd):
            assert (tmp_path / fname).exists()

    def test_paths_in_the_collection_are_relative(self, tmp_path):
        """An absolute path would break the moment the directory is moved."""
        s = TimeSeriesWriter(tmp_path, "f")
        s.add(0.0, _touch)
        for fname in re.findall(r'file="([^"]*)"', (tmp_path / "f.pvd").read_text()):
            assert not Path(fname).is_absolute()

    def test_a_failed_frame_is_not_recorded(self, tmp_path):
        def boom(path):
            raise RuntimeError("disk full")

        s = TimeSeriesWriter(tmp_path, "f")
        s.add(0.0, _touch)
        with pytest.raises(RuntimeError, match="disk full"):
            s.add(1.0, boom)

        assert len(s) == 1
        pvd = (tmp_path / "f.pvd").read_text()
        assert pvd.count("<DataSet") == 1
        assert "1" not in re.findall(r'timestep="([^"]*)"', pvd)

    def test_index_is_reused_after_a_failure(self, tmp_path):
        """The next good frame takes the failed frame's slot, overwriting any
        partial file it left behind."""
        def boom(path):
            Path(path).write_text("truncated")
            raise RuntimeError("interrupted mid-write")

        s = TimeSeriesWriter(tmp_path, "f")
        with pytest.raises(RuntimeError):
            s.add(0.0, boom)
        assert (tmp_path / "f_000000.vtu").read_text() == "truncated"

        s.add(1.0, _touch, "good")
        assert (tmp_path / "f_000000.vtu").read_text() == "good"
        assert len(s) == 1

    def test_is_writer_agnostic(self, tmp_path, quad_mesh):
        """One class over all three writers is the reason it takes a callable."""
        mesh_s = TimeSeriesWriter(tmp_path / "mesh", "m", suffix=".vtu")
        mesh_s.add(0.0, write_vtk, quad_mesh, point_data={"T": np.zeros(4)})

        part_s = TimeSeriesWriter(tmp_path / "part", "p", suffix=".vtu")
        part_s.add(0.0, write_particles_vtu, np.zeros((3, 2)))

        grid_s = TimeSeriesWriter(tmp_path / "grid", "g", suffix=".vti")
        grid_s.add(0.0, write_image_vti, [0, 0], 1.0, [2, 3])

        for s, name in [(mesh_s, "m"), (part_s, "p"), (grid_s, "g")]:
            assert len(s) == 1
            assert s.frame_path(0).exists()
            assert (s.directory / f"{name}.pvd").exists()

    def test_kwargs_reach_the_writer(self, tmp_path):
        seen = {}

        def spy(path, a, b=None):
            seen.update(a=a, b=b)
            _touch(path)

        TimeSeriesWriter(tmp_path, "f").add(0.0, spy, 1, b=2)
        assert seen == {"a": 1, "b": 2}

    def test_suffix_is_respected(self, tmp_path):
        s = TimeSeriesWriter(tmp_path, "f", suffix=".vti")
        assert s.add(0.0, _touch).name.endswith(".vti")

    def test_repr_reports_the_frame_count(self, tmp_path):
        s = TimeSeriesWriter(tmp_path, "f")
        s.add(0.0, _touch)
        assert "1 frames" in repr(s)


# ---------------------------------------------------------------------------
# Contracts that must hold across all three writers
# ---------------------------------------------------------------------------

class TestWriterConsistency:
    def test_every_writer_returns_its_path(self, tmp_path, quad_mesh):
        assert write_vtk(tmp_path / "a.vtu", quad_mesh) == tmp_path / "a.vtu"
        assert write_particles_vtu(tmp_path / "b.vtu", np.zeros((2, 3))) \
            == tmp_path / "b.vtu"
        assert write_image_vti(tmp_path / "c.vti", [0, 0], 1.0, [2, 2]) \
            == tmp_path / "c.vti"

    def test_every_writer_creates_missing_parents(self, tmp_path, quad_mesh):
        write_vtk(tmp_path / "p1" / "a.vtu", quad_mesh)
        write_particles_vtu(tmp_path / "p2" / "b.vtu", np.zeros((2, 3)))
        write_image_vti(tmp_path / "p3" / "c.vti", [0, 0], 1.0, [2, 2])
        for d in ["p1/a.vtu", "p2/b.vtu", "p3/c.vti"]:
            assert (tmp_path / d).exists()

    def test_vectors_are_three_component_whichever_writer_is_used(self, tmp_path,
                                                                  quad_mesh):
        u = np.array([[1.0, 2.0]] * 4)
        write_vtk(tmp_path / "mesh.vtu", quad_mesh, point_data={"u": u})
        write_particles_vtu(tmp_path / "part.vtu", quad_mesh.nodes,
                            point_data={"u": u})

        for f in ["mesh.vtu", "part.vtu"]:
            assert meshio.read(tmp_path / f).point_data["u"].shape == (4, 3)
