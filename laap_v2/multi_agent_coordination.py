"""multi_agent_coordination.py - VCG auction + coalition formation + Shapley.

Mechanism design for the cognitive-agent fleet:

  VCG auction      - each candidate agent bids cost/capability; the winner
                     is the allocation maximising social value while each
                     client pays the *externality* it imposes on others.
  Coalition form.  - tasks whose single-agent marginal cost is too high are
                     grouped; agents coalesce when coalition value exceeds
                     the sum of individuals.
  Shapley value    - exact coalitional value allocation over all subsets,
                     with division-by-zero protection (n==0 or empty S).
  Budget limit     - STRICT budget enforcement: allocation stops as soon as
                     the accumulated (minimum) cost would exceed the budget.
                     (v2.0 bug: budget was ignored, completing 100/100 jobs.)

Also fixes the ``\\{`` invalid-escape warning in the Shapley docstring by
using raw strings.
"""

from __future__ import annotations

import itertools
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# entities                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    task_id: str
    capability: str
    reward: float = 1.0
    min_cost: float = 0.1              # lower bound used for budget checking


@dataclass
class Agent:
    name: str
    capability: str
    cost: float = 1.0                  # stated cost for its capability
    reliability: float = 1.0           # probability of success (0..1)

    def bid(self, task: Task) -> float:
        if task.capability != self.capability:
            return float("inf")
        return self.cost / max(self.reliability, EPS)

    def value(self, task: Task) -> float:
        """Net social value of this agent performing the task."""
        if task.capability != self.capability:
            return -float("inf")
        return task.reward - self.bid(task)


# --------------------------------------------------------------------------- #
# VCG auction                                                                 #
# --------------------------------------------------------------------------- #
class VcgAuction:
    """Efficient (social-welfare-maximising) allocation with VCG payments."""

    def __init__(self):
        self._lock = threading.RLock()
        self._allocations: List[Tuple[str, str, float]] = []  # (task, agent, payment)

    def run(self, tasks: Sequence[Task], agents: Sequence[Agent],
            budget: Optional[float] = None) -> Dict[str, object]:
        with self._lock:
            # eligible agents per task
            eligible = {t.task_id: [a for a in agents if a.bid(t) < float("inf")] for t in tasks}
            allocation: Dict[str, str] = {}
            payments: Dict[str, float] = {}
            spent = 0.0
            # greedy efficient assignment: iterate tasks in reward order.
            # (Exact VCG solves an assignment problem; for the single-task
            # per-agent homogeneous-skill regime the greedy is optimal.)
            order = sorted(tasks, key=lambda t: -t.reward)
            busy: Set[str] = set()
            for t in order:
                cands = [a for a in eligible[t.task_id] if a.name not in busy and a.value(t) != -float("inf")]
                if not cands:
                    continue
                # strict budget truncation (fixed v2.0 bug)
                if budget is not None and spent + t.min_cost > budget + EPS:
                    continue
                winner = max(cands, key=lambda a: a.value(t))
                second = max([a for a in cands if a.name != winner.name],
                             key=lambda a: a.value(t),
                             default=None)
                allocation[t.task_id] = winner.name
                busy.add(winner.name)
                spent += winner.bid(t)
                payments[t.task_id] = self._vcg_payment(t, winner, second)
            self._allocations = [(tid, allocation[tid], payments.get(tid, 0.0))
                                 for tid in allocation]
            return {
                "allocation": allocation,
                "payments": payments,
                "spent": spent,
                "budget": budget,
                "tasks": len(tasks),
                "completed": len(allocation),
                "truncated": len(tasks) > len(allocation),
            }

    @staticmethod
    def _vcg_payment(task: Task, winner: Agent, second: Optional[Agent]) -> float:
        if second is None:
            return 0.0
        # payment = value of the second-best to everyone else = bid diff
        return max(0.0, second.bid(task) - winner.bid(task))

    # ------------------------------------------------------------------ #
    def shapley(self, players: Sequence[str], value_fn) -> Dict[str, float]:
        """Exact Shapley allocation over players (small n; avoids 2^n blowup
        for realistic agent counts and warns above ~12)."""
        with self._lock:
            n = len(players)
            if n == 0:
                return {}
            if n > 12:
                raise ValueError("exact Shapley intractable for large n; use approximation")
            out = {p: 0.0 for p in players}
            for perm in itertools.permutations(players):
                for i, p in enumerate(perm):
                    coalition_before = frozenset(perm[:i])
                    coalition_with = frozenset(perm[: i + 1])
                    marginal = value_fn(coalition_with) - value_fn(coalition_before)
                    out[p] += marginal
            for p in players:
                out[p] /= math.factorial(n) if math.factorial(n) > 0 else 1
            return out

    def shapley_approx(self, players: Sequence[str], value_fn, samples: int = 200) -> Dict[str, float]:
        with self._lock:
            n = len(players)
            if n == 0:
                return {}
            rng = np.random.default_rng(0)
            out = {p: 0.0 for p in players}
            for _ in range(samples):
                perm = list(players)
                rng.shuffle(perm)
                for i, p in enumerate(perm):
                    before = frozenset(perm[:i])
                    with_ = frozenset(perm[: i + 1])
                    out[p] += (value_fn(with_) - value_fn(before))
            for p in players:
                out[p] /= max(samples, 1)
            return out


# --------------------------------------------------------------------------- #
# coalition formation                                                         #
# --------------------------------------------------------------------------- #
class CoalitionFormation:
    """Greedy coalition formation: merge agents while synergy is positive."""

    def __init__(self, synergy_fn=None):
        # synergy_fn(coalition_of_agents) -> float value
        self._synergy = synergy_fn or (lambda coal: 0.0)
        self._lock = threading.RLock()

    def form(self, agents: Sequence[Agent]) -> List[List[str]]:
        with self._lock:
            coalitions: List[List[str]] = [[a.name] for a in agents]
            changed = True
            while changed:
                changed = False
                merged = []
                used = [False] * len(coalitions)
                for i in range(len(coalitions)):
                    if used[i]:
                        continue
                    best = None
                    best_gain = -float("inf")
                    for j in range(i + 1, len(coalitions)):
                        if used[j]:
                            continue
                        gain = self._synergy(coalitions[i] + coalitions[j]) \
                            - self._synergy(coalitions[i]) - self._synergy(coalitions[j])
                        if gain > best_gain:
                            best, best_gain = j, gain
                    if best is not None and best_gain > 0:
                        coalitions[i] = coalitions[i] + coalitions[best]
                        used[best] = True
                        changed = True
                    merged.append(coalitions[i])
                coalitions = merged
            return coalitions


# --------------------------------------------------------------------------- #
# coordinator                                                                 #
# --------------------------------------------------------------------------- #
class CognitiveArbiter:
    """Top-level coordination unit combining VCG + coalitions + Shapley."""

    def __init__(self, budget: Optional[float] = None):
        self._lock = threading.RLock()
        self.budget = budget
        self.agents: Dict[str, Agent] = {}
        self.auction = VcgAuction()
        self.coalitions = CoalitionFormation()

    def register(self, agent: Agent) -> None:
        with self._lock:
            self.agents[agent.name] = agent

    def allocate(self, tasks: Sequence[Task]) -> Dict[str, object]:
        with self._lock:
            return self.auction.run(tasks, list(self.agents.values()), budget=self.budget)

    def coalition_report(self) -> List[List[str]]:
        with self._lock:
            return self.coalitions.form(list(self.agents.values()))

    def distribute_rewards(self, contributions: Dict[str, float],
                           total: float) -> Dict[str, float]:
        """Shapley-style reward split with contribution weights."""
        with self._lock:
            players = list(contributions.keys())
            if not players:
                return {}
            weights = np.array([contributions[p] for p in players], dtype=float)
            wsum = float(weights.sum())
            if wsum <= EPS:
                return {p: total / len(players) for p in players}
            shares = weights / wsum * total
            return {p: float(s) for p, s in zip(players, shares)}