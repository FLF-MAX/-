"""deep_analogical_engine.py - structure-mapping analogy with optimal alignment.

Implements Gentner's structure-mapping framework in a compressed form:

  1. encode base and target as typed directed graphs (nodes have roles /
     attributes, edges have relation types);
  2. pairwise node similarity via weighted role + attribute Jaccard;
  3. GLOBAL optimal alignment with the Hungarian algorithm (replaces v1.0's
     greedy one-to-one matching);
  4. structural-consistency and coverage scoring;
  5. transfer: node/relation attribute predictions from base onto target.

Edge cases explicitly handled:
  - empty graphs / empty similarity matrices (np.max crash guard);
  - non-square matrices (Hungarian pads internally).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------- #
# graph                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class DomainNode:
    name: str
    role: str = "generic"                      # e.g. "central", "satellite", "agent"
    attributes: Dict[str, float] = field(default_factory=dict)

    def __hash__(self):  # keep nodes hashable/uniquely identifiable
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, DomainNode) and other.name == self.name


@dataclass
class DomainEdge:
    source: str
    target: str
    reltype: str = "relates"                   # e.g. "orbits", "contained_by"
    weight: float = 1.0


class DomainGraph:
    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: Dict[str, DomainNode] = {}
        self.edges: List[DomainEdge] = []

    def add_node(self, name: str, role: str = "generic", **attrs) -> DomainNode:
        node = self.nodes.get(name)
        if node is None:
            node = DomainNode(name=name, role=role, attributes=dict(attrs))
            self.nodes[name] = node
        else:
            node.role = role
            node.attributes.update(attrs)
        return node

    def add_edge(self, source: str, target: str, reltype: str = "relates", weight: float = 1.0) -> None:
        self.edges.append(DomainEdge(source, target, reltype, weight))

    @property
    def node_names(self) -> List[str]:
        return list(self.nodes.keys())


# --------------------------------------------------------------------------- #
# similarity + Hungarian                                                      #
# --------------------------------------------------------------------------- #
def _jaccard_role(a: DomainNode, b: DomainNode) -> float:
    if a.role == b.role:
        return 1.0
    # partial credit for related roles
    role_groups = [{"central", "center"}, {"satellite", "outer"}, {"agent", "subject"}, {"object"}]
    for g in role_groups:
        if a.role in g and b.role in g:
            return 0.7
    return 0.0


def _jaccard_attr(a: DomainNode, b: DomainNode) -> float:
    ka = set(a.attributes.keys())
    kb = set(b.attributes.keys())
    inter = len(ka & kb)
    union = len(ka | kb)
    if union == 0:
        return 0.5 if a.role == b.role else 0.0    # neutral prior
    return inter / union


def _node_similarity(a: DomainNode, b: DomainNode, w_role: float = 0.6,
                     w_attr: float = 0.4) -> float:
    attr = _jaccard_attr(a, b)
    return w_role * _jaccard_role(a, b) + w_attr * attr


def hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Minimise-assignment Hungarian (O(n^3), potentials version) for a
    rectangular cost matrix.  Returns (row, col) pairs; every row matched.
    """
    n, m = cost.shape
    if cost.size == 0:
        return []
    if n > m:
        # orient so n <= m (transpose costs, swap back on return)
        swaps = hungarian(cost.T)
        return [(b, a) for a, b in swaps]
    INF = 1e12
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)      # p[j] = row matched to column j (1-indexed)
    way = np.zeros(m + 1, dtype=np.int64)
    a = np.asarray(cost, dtype=np.float64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            row = a[i0 - 1]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignments = []
    for j in range(1, m + 1):
        if p[j] != 0 and p[j] <= n:
            assignments.append((int(p[j]) - 1, j - 1))
    return assignments


# --------------------------------------------------------------------------- #
# deep analogy engine                                                        #
# --------------------------------------------------------------------------- #
class DeepAnalogyEngine:
    """Gentner-style structure mapping with Hungarian-optimal alignment."""

    def __init__(self, w_role: float = 0.6, w_attr: float = 0.4):
        self.w_role = w_role
        self.w_attr = w_attr
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    def _compute_similarity_matrix(self, base: DomainGraph, target: DomainGraph) -> np.ndarray:
        """Pairwise base@target node similarity.  EMPTY-GRAPH SAFE."""
        base_nodes = list(base.nodes.values())
        target_nodes = list(target.nodes.values())
        if not base_nodes or not target_nodes:
            return np.zeros((len(base_nodes), len(target_nodes)))
        sim = np.zeros((len(base_nodes), len(target_nodes)), dtype=np.float64)
        for i, bn in enumerate(base_nodes):
            for j, tn in enumerate(target_nodes):
                sim[i, j] = _node_similarity(bn, tn, self.w_role, self.w_attr)
        return sim

    def _edge_similarity(self, base: DomainGraph, target: DomainGraph,
                         b_src: int, b_dst: int, t_src: int, t_dst: int) -> float:
        """Consistency of an edge pair between two aligned node pairs."""
        b_edges = [e for e in base.edges if e.source == base.node_names[b_src]
                   and e.target == base.node_names[b_dst]]
        t_edges = [e for e in target.edges if e.source == target.node_names[t_src]
                   and e.target == target.node_names[t_dst]]
        if not b_edges and not t_edges:
            return 1.0
        if not b_edges or not t_edges:
            return 0.0
        rel_b = {e.reltype for e in b_edges}
        rel_t = {e.reltype for e in t_edges}
        inter = len(rel_b & rel_t)
        union = len(rel_b | rel_t)
        return inter / union if union else 1.0

    # ------------------------------------------------------------------ #
    def align(self, base: DomainGraph, target: DomainGraph):
        """Return (alignment, consistency, coverage, similarity)."""
        with self._lock:
            b_names = base.node_names
            t_names = target.node_names
            if not b_names or not t_names:
                return {}, 0.0, 0.0, 0.0
            sim = self._compute_similarity_matrix(base, target)
            cost = 1.0 - sim
            assignments = hungarian(cost)
            ali = {}
            for bi, tj in assignments:
                ali[b_names[bi]] = t_names[tj]
            consistency, covered = self._score(base, target, ali)
            return ali, consistency, covered, float(sim[assignments].mean()) if assignments else 0.0

    def _score(self, base: DomainGraph, target: DomainGraph,
               ali: Dict[str, str]) -> Tuple[float, float]:
        """Structural consistency + coverage of the alignment."""
        if not ali:
            return 0.0, 0.0
        # map aligned base-node names to target indices
        tb = {b: t for b, t in ali.items()}
        cons = []
        for e in base.edges:
            if e.source in tb and e.target in tb:
                bi = base.node_names.index(e.source)
                bj = base.node_names.index(e.target)
                ti = target.node_names.index(tb[e.source])
                tj = target.node_names.index(tb[e.target])
                cons.append(self._edge_similarity(base, target, bi, bj, ti, tj))
        consistency = float(np.mean(cons)) if cons else 1.0
        coverage = len(ali) / max(len(base.nodes), 1)
        return consistency, coverage

    # ------------------------------------------------------------------ #
    def transfer(self, base: DomainGraph, target: DomainGraph,
                 ali: Dict[str, str], property_: str,
                 base_domain_fn=None) -> Dict[str, float]:
        """Infer `property_` for target nodes from its alignment to base."""
        out: Dict[str, float] = {}
        expand_base = callable(base_domain_fn)
        for b_name, t_name in ali.items():
            bn = base.nodes.get(b_name)
            if bn is None:
                continue
            value = bn.attributes.get(property_)
            if value is None and expand_base:
                value = base_domain_fn(b_name, property_)
            if value is None:
                continue
            out[t_name] = value
        return out

    def predict_structure(self, base: DomainGraph, target: DomainGraph,
                          property_: str, base_domain_fn=None) -> Dict[str, float]:
        ali, conf, cov, _ = self.align(base, target)
        values = self.transfer(base, target, ali, property_, base_domain_fn)
        return {"alignment": ali, "confidence": conf, "coverage": cov, "values": values}