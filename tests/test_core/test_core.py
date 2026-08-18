"""
Verification for the core abstractions.

``core/`` had no tests at all, and writing them turned up six defects. Four are
fixed and five of the tests below are regressions for them:

* ``DOFMap.constrain(-1, 0)`` wrapped and removed a *different* node's DOF from
  the free set, silently
* an out-of-range node left the map inconsistent -- ``n_free`` counted the bogus
  constraint while ``free_dofs`` raised ``IndexError`` somewhere else entirely
* ``Domain.add_material`` accepted a ``region`` argument and discarded it, so
  two regions with the same material name collapsed into one entry
* ``get_logger`` put a handler on every module logger, so any application
  calling ``logging.basicConfig()`` saw every message twice
* ``get_logger`` reset the level on each call, undoing a caller's DEBUG

The sixth -- ``small_linalg.inv`` returning ``inf`` for a singular matrix -- is
intended, and is tested as such rather than fixed, because the callers are
per-particle loops where the check would cost more than the inversion.

The characteristic check for the linear algebra is the round trip:
``inv(F) @ F == I`` and ``det(inv(F)) == 1 / det(F)`` are separate pieces of
algebra from the cofactor expansion, so agreement is a real test of it.

python -m pytest tests/test_core/test_core.py -q
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import scion.core.small_linalg as sla
from scion.core import DOFMap, Domain, Field, get_logger
from scion.core.field import FieldLocation
from scion.core.logging import (
    PACKAGE,
    set_level,
    use_application_logging,
)
from scion.mesh.generators import rectangle_mesh


@pytest.fixture
def mesh():
    return rectangle_mesh((0.0, 1.0), (0.0, 1.0), 2, 2)


def _random_stack(n, dim, seed=0, well_conditioned=True):
    """A stack of invertible matrices: identity plus a small perturbation, so
    the determinants stay well away from zero and the round trip is a test of
    the algebra rather than of conditioning."""
    rng = np.random.default_rng(seed)
    F = np.tile(np.eye(dim), (n, 1, 1))
    if well_conditioned:
        F += 0.3 * rng.standard_normal((n, dim, dim))
    return F


# ---------------------------------------------------------------------------
# DOFMap
# ---------------------------------------------------------------------------

def test_the_numbering_is_node_major():
    """Node ``i``, component ``j`` is ``i * n_comp + j``, so a node's components
    are contiguous. Every ``[0::2]`` slice in the test suite that picks the
    x-components of a boundary depends on this; component-major numbering would
    break all of them and raise nothing."""
    dofs = DOFMap(n_nodes=4, n_components=2)

    assert [dofs.dof(n, c) for n in range(4) for c in range(2)] == list(range(8))
    assert dofs.dof(3, 1) == 7
    assert dofs.n_total == 8


def test_element_dofs_interleaves_the_components():
    dofs = DOFMap(n_nodes=10, n_components=2)
    np.testing.assert_array_equal(dofs.element_dofs([3, 7]), [6, 7, 14, 15])

    scalar = DOFMap(n_nodes=10, n_components=1)
    np.testing.assert_array_equal(scalar.element_dofs([3, 7]), [3, 7])


def test_element_dofs_agrees_with_dof_one_pair_at_a_time():
    """The vectorised path against the scalar one. They are separate pieces of
    arithmetic and only one of them is exercised in the hot loop."""
    dofs = DOFMap(n_nodes=12, n_components=3)
    connectivity = [11, 0, 5, 7]

    expected = [dofs.dof(n, c) for n in connectivity for c in range(3)]
    np.testing.assert_array_equal(dofs.element_dofs(connectivity), expected)


def test_a_negative_node_index_is_refused_rather_than_wrapped():
    """The regression. Python indexing accepts negatives, so this used to
    remove global DOF 8 -- a real DOF belonging to node 4 -- from the free set,
    with nothing to show anything had happened."""
    dofs = DOFMap(n_nodes=5, n_components=2)

    with pytest.raises(IndexError, match="silently wrap"):
        dofs.constrain(-1, 0)
    assert dofs.n_free == dofs.n_total
    np.testing.assert_array_equal(dofs.free_dofs, np.arange(10))


def test_an_out_of_range_node_is_refused_at_the_point_of_use():
    """The other half of the regression: it used to be accepted, counted by
    ``n_free``, and then blow up inside ``free_dofs`` much later."""
    dofs = DOFMap(n_nodes=5, n_components=2)

    with pytest.raises(IndexError, match=r"outside \[0, 5\)"):
        dofs.constrain(99, 0)
    with pytest.raises(IndexError, match=r"outside \[0, 2\)"):
        dofs.constrain(0, 5)

    assert dofs.n_free == 10
    assert dofs.free_dofs.size == 10        # and the map is still consistent


def test_element_dofs_refuses_a_connectivity_off_the_mesh():
    dofs = DOFMap(n_nodes=5, n_components=2)
    with pytest.raises(IndexError, match="outside"):
        dofs.element_dofs([0, 5])
    with pytest.raises(IndexError, match="outside"):
        dofs.element_dofs([-1, 2])


def test_free_and_constrained_partition_the_dofs():
    """The invariant the solvers rely on: every DOF is in exactly one of the
    two sets, and the counts agree with the arrays."""
    dofs = DOFMap(n_nodes=4, n_components=2)
    for node in (0, 2):
        dofs.constrain(node, 1)

    free, fixed = dofs.free_dofs, dofs.constrained_dofs
    assert set(free) | set(fixed) == set(range(8))
    assert not set(free) & set(fixed)
    assert (len(free), len(fixed)) == (dofs.n_free, dofs.n_constrained)
    np.testing.assert_array_equal(fixed, [1, 5])


def test_constraining_twice_is_idempotent():
    """``prescribe`` in the two-field solver re-registers the same DOF on every
    load step, so a double count here would shrink the free set every step."""
    dofs = DOFMap(n_nodes=3, n_components=2)
    for _ in range(5):
        dofs.constrain(1, 0)

    assert dofs.n_constrained == 1
    assert dofs.n_free == 5


def test_a_constraint_can_be_released():
    dofs = DOFMap(n_nodes=3, n_components=1)
    dofs.constrain(1, 0)
    assert dofs.is_constrained(1, 0)

    dofs.release(1, 0)
    assert not dofs.is_constrained(1, 0)
    assert dofs.n_free == 3
    dofs.release(1, 0)                       # idempotent


def test_a_degenerate_map_is_refused():
    with pytest.raises(ValueError, match="n_components"):
        DOFMap(5, 0)
    with pytest.raises(ValueError, match="n_nodes"):
        DOFMap(-1, 2)


def test_the_map_survives_a_realistic_boundary_selection(mesh):
    """The pattern every test and example in the repo uses: pick the nodes on
    an edge, take the x-components with a stride, constrain them."""
    dofs = DOFMap(mesh.n_nodes, 2)
    left = np.flatnonzero(np.isclose(mesh.nodes[:, 0], 0.0))
    x_dofs = dofs.element_dofs(left)[0::2]

    for node in left:
        dofs.constrain(int(node), 0)

    np.testing.assert_array_equal(sorted(dofs.constrained_dofs), sorted(x_dofs))
    assert dofs.n_free == 2 * mesh.n_nodes - left.size


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

def test_a_new_field_is_zero_and_correctly_shaped():
    field = Field("displacement", n_points=6, n_components=2)

    assert field.shape == (6, 2)
    assert field.n_points == 6
    assert not field.values.any()
    assert field.location is FieldLocation.NODAL


def test_flat_is_a_view_and_writes_through():
    """Assembly routines scatter through ``flat`` and expect it to alias
    ``values``. ``ravel`` returns a copy for a non-contiguous array, so the
    guarantee belongs to this class rather than to numpy."""
    field = Field("u", 3, 2)
    flat = field.flat
    flat[0] = 7.0
    flat[-1] = -2.0

    assert np.shares_memory(flat, field.values)
    assert field.values[0, 0] == 7.0
    assert field.values[2, 1] == -2.0


def test_set_accepts_both_a_flat_and_a_blocked_array():
    """A solver hands back a flat vector and a post-processor a block; both
    turn up and neither should need reshaping at the call site."""
    field = Field("u", 3, 2)

    field.set(np.arange(6.0))
    np.testing.assert_array_equal(field.values, [[0, 1], [2, 3], [4, 5]])

    field.set(np.zeros((3, 2)) + 4.0)
    assert np.all(field.values == 4.0)


def test_set_refuses_the_wrong_number_of_values():
    """Reshape alone would accept any array of the right *size*; the failure to
    catch is the wrong size, and the message should say both."""
    field = Field("u", 3, 2)
    with pytest.raises(ValueError, match="holds 6 values"):
        field.set(np.arange(5.0))


def test_a_field_copy_is_independent():
    field = Field("u", 2, 2)
    field.fill(3.0)
    clone = field.copy()
    clone.fill(0.0)

    assert np.all(field.values == 3.0)
    assert clone.name == field.name and clone.location is field.location


def test_a_field_can_be_used_as_an_array():
    field = Field("u", 2, 2)
    field.fill(2.0)
    assert np.asarray(field).sum() == 8.0


def test_a_degenerate_field_is_refused():
    with pytest.raises(ValueError, match="n_components"):
        Field("u", 3, 0)


def test_the_location_distinguishes_fields_of_the_same_length():
    """The one piece of metadata that cannot be recovered from the shape: a
    nodal and a quadrature field can have identical dimensions."""
    nodal = Field("a", 4, 1, FieldLocation.NODAL)
    quad = Field("b", 4, 1, FieldLocation.QUADRATURE)

    assert nodal.shape == quad.shape
    assert nodal.location is not quad.location
    assert "QUADRATURE" in repr(quad)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_a_domain_exposes_the_meshs_counts(mesh):
    domain = Domain(mesh, dim=2)

    assert domain.n_nodes == mesh.n_nodes
    assert domain.n_elements == mesh.n_elements
    assert domain.coordinates is mesh.nodes


def test_a_dimension_that_contradicts_the_mesh_is_refused(mesh):
    """The mesh has 2-D node coordinates. Building a 3-D domain on it produces
    an assembler that runs and returns nonsense, which is the worst of the
    available outcomes."""
    with pytest.raises(ValueError, match="node coordinates"):
        Domain(mesh, dim=3)
    with pytest.raises(ValueError, match="dim must be"):
        Domain(mesh, dim=0)


def test_one_material_can_be_assigned_to_several_regions(mesh):
    """The regression. ``region`` used to be accepted and discarded, so these
    two calls left a single entry and the second model silently won."""
    domain = Domain(mesh, dim=2)
    steel = object()

    domain.add_material("steel", steel, region="flange")
    domain.add_material("steel", steel, region="web")

    assert domain.regions == ("flange", "web")
    assert domain.material_for("flange") is steel
    assert domain.material_for("web") is steel


def test_different_regions_can_hold_different_materials(mesh):
    domain = Domain(mesh, dim=2)
    steel, concrete = object(), object()

    domain.add_material("steel", steel, region="rebar")
    domain.add_material("concrete", concrete, region="slab")

    assert domain.material_for("rebar") is steel
    assert domain.material_for("slab") is concrete
    assert set(domain.materials) == {"steel", "concrete"}


def test_a_name_cannot_be_reused_for_a_different_model(mesh):
    """The case where one of the two calls is a mistake and the old behaviour
    made it unrecoverable."""
    domain = Domain(mesh, dim=2)
    domain.add_material("steel", object(), region="a")

    with pytest.raises(ValueError, match="already registered"):
        domain.add_material("steel", object(), region="b")


def test_a_region_holds_exactly_one_material(mesh):
    """A point of the body is made of one thing, so reassigning a region is a
    mistake rather than an update."""
    domain = Domain(mesh, dim=2)
    domain.add_material("steel", object(), region="web")

    with pytest.raises(ValueError, match="already holds"):
        domain.add_material("concrete", object(), region="web")


def test_a_single_material_problem_needs_no_regions(mesh):
    """``region=None`` is the whole domain, and it is the fallback for any
    region that has no assignment of its own."""
    domain = Domain(mesh, dim=2)
    steel = object()
    domain.add_material("steel", steel)

    assert domain.regions == ()
    assert domain.material_for() is steel
    assert domain.material_for("anything") is steel


def test_asking_for_an_unassigned_region_says_what_is_assigned(mesh):
    domain = Domain(mesh, dim=2)
    domain.add_material("steel", object(), region="web")

    with pytest.raises(KeyError, match="flange"):
        domain.material_for("flange")


def test_boundary_conditions_accumulate(mesh):
    domain = Domain(mesh, dim=2)
    domain.add_bc("left-fixed")
    domain.add_bc("right-load")
    assert domain.bcs == ["left-fixed", "right-load"]


def test_two_domains_do_not_share_their_mutable_defaults(mesh):
    """``dataclasses.field(default_factory=...)`` rather than a bare ``{}``.
    Getting this wrong makes every Domain in the process share one dict."""
    first, second = Domain(mesh, dim=2), Domain(mesh, dim=2)
    first.add_material("steel", object())

    assert second.materials == {}
    assert second.assignments == {}
    assert second.bcs == []


# ---------------------------------------------------------------------------
# small_linalg
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
def test_the_determinant_matches_numpy(dim):
    F = _random_stack(200, dim, seed=1)
    np.testing.assert_allclose(sla.det(F), np.linalg.det(F), rtol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_the_inverse_round_trips_to_the_identity(dim):
    """The characteristic check. The cofactor expansion and the matrix product
    are separate algebra, so ``inv(F) @ F == I`` is a real test of the former --
    unlike comparing against another closed form written the same way."""
    F = _random_stack(200, dim, seed=2)
    identity = np.tile(np.eye(dim), (200, 1, 1))

    np.testing.assert_allclose(sla.inv(F) @ F, identity, atol=1e-10)
    np.testing.assert_allclose(F @ sla.inv(F), identity, atol=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_the_inverse_matches_numpy_entry_by_entry(dim):
    F = _random_stack(100, dim, seed=3)
    np.testing.assert_allclose(sla.inv(F), np.linalg.inv(F), rtol=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_the_determinant_of_the_inverse_is_the_reciprocal(dim):
    F = _random_stack(100, dim, seed=4)
    np.testing.assert_allclose(sla.det(sla.inv(F)), 1.0 / sla.det(F),
                               rtol=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_a_precomputed_determinant_gives_the_same_inverse(dim):
    """``inv`` takes ``J`` to avoid recomputing it in the particle loop; the
    two paths must not drift apart."""
    F = _random_stack(50, dim, seed=5)
    np.testing.assert_allclose(sla.inv(F), sla.inv(F, sla.det(F)), rtol=1e-14)


@pytest.mark.parametrize("dim", [2, 3])
def test_inv_transpose_is_the_transpose_of_the_inverse(dim):
    F = _random_stack(50, dim, seed=6)
    np.testing.assert_allclose(sla.inv_transpose(F),
                               sla.transpose(sla.inv(F)), rtol=1e-14)


@pytest.mark.parametrize("dim", [2, 3])
def test_the_identity_is_its_own_inverse_and_has_unit_determinant(dim):
    identity = sla.identity(7, dim)
    assert identity.shape == (7, dim, dim)
    np.testing.assert_allclose(sla.det(identity), np.ones(7))
    np.testing.assert_allclose(sla.inv(identity), identity)


def test_the_determinant_is_the_volume_ratio_of_a_known_stretch():
    """An external check rather than another matrix identity: a diagonal
    stretch multiplies volume by the product of its entries."""
    F = np.diag([2.0, 3.0, 4.0])[None]
    assert sla.det(F)[0] == pytest.approx(24.0)
    assert sla.det(sla.inv(F))[0] == pytest.approx(1.0 / 24.0)


def test_transpose_swaps_the_last_two_axes():
    F = _random_stack(5, 3, seed=7)
    np.testing.assert_allclose(sla.transpose(F), np.swapaxes(F, 1, 2))
    np.testing.assert_allclose(sla.transpose(sla.transpose(F)), F)


def test_a_singular_matrix_gives_infinities_rather_than_an_exception():
    """Documented behaviour, not an oversight. The callers are per-particle
    loops at 1e5 matrices a step, where the check would cost more than the
    inversion; the physical guard for ``J <= 0`` belongs in the constitutive
    model, which can say that the material has inverted."""
    singular = np.array([[[1.0, 2.0], [2.0, 4.0]]])
    assert sla.det(singular)[0] == 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        out = sla.inv(singular)
    assert not np.all(np.isfinite(out))


def test_an_unsupported_size_is_refused():
    """1x1 and 4x4 are not special-cased, and returning a wrong answer for them
    would be worse than refusing."""
    with pytest.raises(ValueError, match="2x2 or 3x3"):
        sla.det(np.ones((2, 4, 4)))
    with pytest.raises(ValueError, match="2x2 or 3x3"):
        sla.inv(np.ones((2, 4, 4)))


def test_the_routines_handle_a_single_matrix_and_a_large_stack():
    for n in (1, 10000):
        F = _random_stack(n, 3, seed=8)
        np.testing.assert_allclose(sla.inv(F) @ F, sla.identity(n, 3),
                                   atol=1e-9)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_logging():
    """Restore the logging tree, since these tests mutate global state.

    The handler *streams* are saved too: ``_capture`` repoints them at a
    buffer, and leaving them there would silently swallow the log output of
    every later test in the session.

    ``manager.disable`` is reset for the duration, and that is not defensive
    padding. It is process-global, nothing resets it, and an example imported
    by ``tests/examples`` used to set it to CRITICAL inside ``run()`` -- so two
    tests here passed alone and failed in the full suite, with an empty capture
    buffer and no indication why. The example is fixed; resetting it here keeps
    these tests independent of whatever else the session has done.
    """
    package = logging.getLogger(PACKAGE)
    handlers = list(package.handlers)
    streams = [getattr(h, "stream", None) for h in handlers]
    level, propagate = package.level, package.propagate
    disabled = logging.root.manager.disable
    logging.disable(logging.NOTSET)

    yield package

    package.handlers[:] = handlers
    for handler, stream in zip(handlers, streams):
        if stream is not None:
            handler.stream = stream
    package.setLevel(level)
    package.propagate = propagate
    logging.disable(disabled)


def _capture(logger):
    """Point every handler in the subtree at a buffer and return it."""
    import io
    buffer = io.StringIO()
    for handler in logging.getLogger(PACKAGE).handlers:
        handler.stream = buffer
    return buffer


def test_one_message_produces_one_line(clean_logging):
    """The regression. A handler on each module logger meant a record was
    emitted by its own handler and again by every ancestor's."""
    get_logger(PACKAGE)
    child = get_logger(f"{PACKAGE}.solvers.nonlinear")
    buffer = _capture(child)

    child.info("converged in 4 iterations")
    assert buffer.getvalue().count("converged in 4 iterations") == 1


def test_the_handler_lives_only_on_the_package_logger(clean_logging):
    """One handler in the whole library. Anything else duplicates output the
    moment two loggers in the same chain are created."""
    get_logger(f"{PACKAGE}.solvers")
    get_logger(f"{PACKAGE}.optimization.oed")

    assert len(logging.getLogger(PACKAGE).handlers) == 1
    for name in (f"{PACKAGE}.solvers", f"{PACKAGE}.optimization.oed"):
        assert logging.getLogger(name).handlers == []


def test_an_application_calling_basic_config_does_not_double_the_output(
        clean_logging):
    """The failure that actually bites. ``basicConfig`` puts a handler on the
    root logger, and with propagation on, every Scion message appeared
    twice in two different formats."""
    logger = get_logger(f"{PACKAGE}.solvers.linear")
    buffer = _capture(logger)

    root_buffer = logging.StreamHandler()
    import io
    root_stream = io.StringIO()
    root_buffer.stream = root_stream
    logging.getLogger().addHandler(root_buffer)
    try:
        logger.warning("ill-conditioned matrix")
    finally:
        logging.getLogger().removeHandler(root_buffer)

    assert buffer.getvalue().count("ill-conditioned") == 1
    assert root_stream.getvalue().count("ill-conditioned") == 0


def test_a_later_call_does_not_reset_an_explicit_level(clean_logging):
    """The other regression: ``get_logger`` used to set INFO unconditionally,
    so any later import silently undid a caller's DEBUG."""
    name = f"{PACKAGE}.probe"
    get_logger(name, logging.DEBUG)
    assert logging.getLogger(name).level == logging.DEBUG

    get_logger(name)
    assert logging.getLogger(name).level == logging.DEBUG


def test_module_loggers_inherit_the_package_level(clean_logging):
    """Module loggers are left at NOTSET on purpose, so one call to
    ``set_level`` changes the verbosity of the whole library."""
    child = get_logger(f"{PACKAGE}.solvers.eigen")
    assert child.level == logging.NOTSET

    set_level(logging.WARNING)
    assert child.getEffectiveLevel() == logging.WARNING
    set_level(logging.INFO)
    assert child.getEffectiveLevel() == logging.INFO


def test_logging_can_be_handed_back_to_the_application(clean_logging):
    """Turning propagation off is what stops the duplication, and it costs an
    application the ability to capture the output. This is the way back --
    pytest's caplog, among other things, cannot see records that do not
    propagate."""
    get_logger(PACKAGE)
    assert logging.getLogger(PACKAGE).propagate is False

    use_application_logging()
    package = logging.getLogger(PACKAGE)
    assert package.handlers == []
    assert package.propagate is True

    use_application_logging()                      # idempotent
    assert package.handlers == []


def test_a_logger_outside_the_package_gets_its_own_handler(clean_logging):
    """``get_logger`` is used from notebooks and scripts too, where the name
    has nothing to do with this package."""
    logger = get_logger("scratch.analysis")
    try:
        assert logging.getLogger("scratch").handlers
        assert logger.handlers == []
    finally:
        logging.getLogger("scratch").handlers[:] = []


def test_quiet_restores_the_level_it_found(clean_logging):
    """The scoped alternative to ``logging.disable``."""
    from scion.core.logging import quiet

    set_level(logging.INFO)
    with quiet():
        assert logging.getLogger(PACKAGE).level == logging.WARNING
    assert logging.getLogger(PACKAGE).level == logging.INFO

    try:
        with quiet():
            raise RuntimeError("solver blew up")
    except RuntimeError:
        pass
    assert logging.getLogger(PACKAGE).level == logging.INFO


def test_quiet_leaves_other_libraries_alone(clean_logging):
    """The whole reason it exists. ``logging.disable`` is process-global and
    silences every logger in the interpreter, not just this package's."""
    from scion.core.logging import quiet

    # an explicit level, because an unconfigured logger inherits the root's
    # default WARNING and would look "disabled" for reasons of its own
    other = logging.getLogger("some.other.library")
    other.setLevel(logging.INFO)
    try:
        with quiet():
            assert other.isEnabledFor(logging.INFO)
            assert logging.root.manager.disable == 0
    finally:
        other.setLevel(logging.NOTSET)


def test_no_example_disables_logging_process_wide():
    """A regression for the worst defect this module turned up.

    ``examples/plasticity/thick_cylinder.py`` called
    ``logging.disable(logging.CRITICAL)`` inside ``run()`` to keep the solver
    quiet. That is permanent and global: importing and calling it -- which
    ``tests/examples`` does, and which any notebook might -- killed logging for
    the rest of the process, including the caller's own. The symptom here was
    two tests in this file passing alone and failing in the full suite.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    # the call, not the name: the examples now mention it in comments saying
    # why they do not use it
    offenders = [str(path.relative_to(root))
                 for path in (root / "examples").rglob("*.py")
                 if "logging.disable(" in path.read_text()]
    assert offenders == [], (
        f"use scion.core.logging.quiet() or set_level(), which are scoped "
        f"to this package and reversible: {offenders}")


def test_calling_an_example_does_not_break_logging_for_the_caller():
    """The same defect, checked through the behaviour rather than the source."""
    from examples.plasticity.thick_cylinder import run

    caller = logging.getLogger("unrelated.app")
    caller.setLevel(logging.INFO)
    before = logging.root.manager.disable
    try:
        run(n_r=3, n_theta=3, n_steps=2, verbose=False)

        assert logging.root.manager.disable == before
        assert caller.isEnabledFor(logging.INFO)
    finally:
        caller.setLevel(logging.NOTSET)


def test_the_library_modules_all_log_through_one_handler():
    """An integration check on the real modules rather than on invented names:
    importing several of them must not accumulate handlers."""
    import scion.solvers.nonlinear          # noqa: F401
    import scion.solvers.linear             # noqa: F401
    import scion.optimization.oed           # noqa: F401

    assert len(logging.getLogger(PACKAGE).handlers) <= 1
