"""Unit tests for multi_agent_coordination - VCG + coalition + Shapley."""

import pytest

from multi_agent_coordination import (
    CognitiveArbiter, VcgAuction, Agent, Task, CoalitionFormation,
)


# --------------------------------------------------------------------------- #
# VCG auction
# --------------------------------------------------------------------------- #
@pytest.fixture
def agents():
    return [
        Agent("a1", "code", cost=0.5, reliability=1.0),
        Agent("a2", "code", cost=0.8, reliability=1.0),
        Agent("a3", "doc", cost=0.3, reliability=1.0),
    ]


def test_auction_assigns_capabilities(agents):
    tasks = [Task("t1", "code"), Task("t2", "code"), Task("t3", "doc")]
    r = VcgAuction().run(tasks, agents)
    assert r["allocation"]["t1"] in ("a1", "a2")
    assert r["allocation"]["t3"] == "a3"
    assert r["completed"] == 3


def test_budget_truncation(agents):
    tasks = [Task(f"t{i}", "code", reward=1.0, min_cost=0.5) for i in range(5)]
    r = VcgAuction().run(tasks, agents, budget=0.6)
    assert r["completed"] < 5
    assert r["truncated"] is True
    assert r["spent"] <= 0.6 + 1e-9


def test_vcg_payment_second_price(agents):
    t = Task("t1", "code")
    # a1 wins, a2 is second; payment = bid(a2)-bid(a1)
    r = VcgAuction().run([t], agents)
    assert r["payments"]["t1"] == pytest.approx(0.8 - 0.5)


def test_no_eligible_agent():
    agents = [Agent("a1", "code")]
    tasks = [Task("t1", "doc")]
    r = VcgAuction().run(tasks, agents)
    assert r["allocation"] == {}


# --------------------------------------------------------------------------- #
# Shapley
# --------------------------------------------------------------------------- #
def test_shapley_symmetric():
    v = lambda S: 0.4 * len(S) if len(S) < 2 else 1.4
    out = VcgAuction().shapley(["a", "b"], v)
    assert out["a"] == pytest.approx(0.7)
    assert out["b"] == pytest.approx(0.7)


def test_shapley_empty():
    assert VcgAuction().shapley([], lambda S: 0.0) == {}


def test_shapley_requires_value():
    out = VcgAuction().shapley(["a"], lambda S: (0.0 if not S else 1.0))
    assert out["a"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# coalition formation
# --------------------------------------------------------------------------- #
def test_coalition_synergy():
    cf = CoalitionFormation()  # default synergy_fn (0 gain -> singleton coalitions)
    groups = cf.form([Agent("x", "code"), Agent("y", "doc"), Agent("z", "doc")])
    assert isinstance(groups, list)
    # every agent appears in exactly one coalition
    flat = [name for g in groups for name in g]
    assert sorted(flat) == ["x", "y", "z"]


def test_coalition_merges_with_synergy():
    # positive pair synergy -> x,y should merge into one coalition
    def synergy(coal):
        if len(coal) >= 2 and set(coal) == {"x", "y"}:
            return 3.0
        return float(len(coal))
    cf = CoalitionFormation(synergy_fn=synergy)
    groups = cf.form([Agent("x", "code"), Agent("y", "code"), Agent("z", "code")])
    xy = [g for g in groups if set(g) == {"x", "y"}]
    assert xy, f"expected merged x,y coalition, got {groups}"


def test_coalition_reports_by_arbiter():
    arb = CognitiveArbiter(budget=10.0)
    arb.register(Agent("a1", "code"))
    arb.register(Agent("a2", "code"))
    report = arb.coalition_report()
    assert isinstance(report, list)


def test_distribute_rewards_conserves():
    arb = CognitiveArbiter()
    out = arb.distribute_rewards({"a": 1.0, "b": 0.5, "c": 0.3}, total=3.6)
    assert sum(out.values()) == pytest.approx(3.6)


def test_arbiter_allocate(agents):
    arb = CognitiveArbiter(budget=2.0)
    for a in agents:
        arb.register(a)
    tasks = [Task("t1", "code"), Task("t2", "doc"), Task("t3", "code")]
    r = arb.allocate(tasks)
    assert r["completed"] == 3
    assert set(r["allocation"].values()) == {"a1", "a3", "a2"}