"""Unit tests for deep_analogical_engine - Hungarian + structure mapping."""

import numpy as np
import pytest

from deep_analogical_engine import (
    DeepAnalogyEngine, DomainGraph, hungarian,
    _jaccard_role, _jaccard_attr, _node_similarity,
)


# --------------------------------------------------------------------------- #
# hungarian assignment
# --------------------------------------------------------------------------- #
def test_hungarian_identity():
    c = np.array([[0, 10, 10], [10, 0, 10], [10, 10, 0]], dtype=float)
    pairs = hungarian(c)
    assert sorted(pairs) == [(0, 0), (1, 1), (2, 2)]
    assert len(pairs) == 3


def test_hungarian_square_min():
    c = np.array([[1, 0], [0, 1]], dtype=float)
    assert sorted(hungarian(c)) == [(0, 1), (1, 0)]


def test_hungarian_rectangular_full_match():
    # 3 rows, 4 cols -> 3 assignments, picks distinct columns
    c = np.array([[1, 2, 3, 100], [4, 5, 6, 100], [100, 100, 100, 0]], dtype=float)
    pairs = hungarian(c)
    assert len(pairs) == 3
    rows = sorted(r for r, _ in pairs); cols = sorted(cc for _, cc in pairs)
    assert rows == [0, 1, 2]
    assert len(set(cols)) == 3


def test_hungarian_empty_and_single():
    assert hungarian(np.zeros((0, 3))) == []
    assert hungarian(np.zeros((0, 0))) == []
    pairs = hungarian(np.array([[5.0]]))
    assert pairs == [(0, 0)]


def test_hungarian_larger_rect():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        c = rng.random((n, m))
        pairs = hungarian(c)
        assert len(pairs) == min(n, m)
        assert len({r for r, _ in pairs}) == len(pairs)
        assert len({cc for _, cc in pairs}) == len(pairs)


# --------------------------------------------------------------------------- #
# similarity helpers
# --------------------------------------------------------------------------- #
def _node(role="central", **attrs):
    return type("N", (), {"role": role, "attributes": attrs})()


def test_jaccard_role():
    assert _jaccard_role(_node("central"), _node("central")) == 1.0
    assert _jaccard_role(_node("central"), _node("satellite")) == 0.0


def test_jaccard_attr():
    assert _jaccard_attr(_node(a=1, b=2), _node(b=3, c=4)) == pytest.approx(1 / 3)
    assert _jaccard_attr(_node(a=1), _node(a=2)) == pytest.approx(1.0)
    assert _jaccard_attr(_node(b=1), _node(a=2)) == pytest.approx(0.0)
    assert _jaccard_attr(_node(), _node()) == pytest.approx(0.5)  # neutral prior on same role


def test_node_similarity_range():
    s = _node_similarity(_node("central", a=1), _node("central", a=1))
    assert 0.0 <= s <= 1.0
    assert s == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# structure mapping
# --------------------------------------------------------------------------- #
@pytest.fixture
def solar_atom():
    solar = DomainGraph("solar")
    solar.add_node("sun", role="central", attracts=1.0)
    for p in ("mercury", "earth", "mars"):
        solar.add_node(p, role="satellite", mass=1.0)
        solar.add_edge(p, "sun", "orbits")
    atom = DomainGraph("atom")
    atom.add_node("nucleus", role="central", charge=1.0)
    for e in ("e1", "e2", "e3"):
        atom.add_node(e, role="satellite", charge=1.0)
        atom.add_edge(e, "nucleus", "orbits")
    return solar, atom


def test_align_central_to_nucleus(solar_atom):
    solar, atom = solar_atom
    ali, consistency, coverage, _ = DeepAnalogyEngine().align(solar, atom)
    assert ali["sun"] == "nucleus"
    assert consistency == pytest.approx(1.0)
    assert coverage == pytest.approx(1.0)


def test_align_empty_graph():
    base = DomainGraph("empty")
    atom = DomainGraph("atom")
    atom.add_node("nucleus", role="central", charge=1.0)
    ali, consistency, coverage, _ = DeepAnalogyEngine().align(base, atom)
    assert ali == {}
    assert coverage == pytest.approx(0.0)


def test_transfer_preserves_edges(solar_atom):
    solar, atom = solar_atom
    engine = DeepAnalogyEngine()
    ali, _, _, _ = engine.align(solar, atom)
    # transfer a property present on the base 'sun' node onto the aligned target
    out = engine.transfer(solar, atom, ali, "attracts")
    assert out == {"nucleus": 1.0}
    pred = engine.predict_structure(solar, atom, "attracts")
    assert pred["values"] == {"nucleus": 1.0}
    assert pred["confidence"] == pytest.approx(1.0)