"""Internal World Model - trajectory simulator"""
import numpy as np
from typing import List, Dict, Optional, Tuple


class InternalWorldModel:
    """Simulates future state trajectories to guide decisions."""

    def __init__(self, dim=1024, n_trajectories=3, horizon=3):
        self.dim = dim
        self.n_traj = n_trajectories
        self.horizon = horizon
        rng = np.random.RandomState(1)
        actions = ["comfort", "explore", "reflect", "play", "help", "create"]
        self.action_vectors = {}
        for a in actions:
            v = rng.randn(dim).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-10)
            self.action_vectors[a] = v * 0.15

    def simulate(self, state, emotion_vec, needs, n_steps=None, n_trajs=None):
        n = n_trajs or self.n_traj
        h = n_steps or self.horizon
        trajectories = []
        for ti in range(n):
            n_act = np.random.randint(2, 4)
            chosen = list(np.random.choice(
                list(self.action_vectors.keys()), n_act, replace=False))
            blend = np.zeros(self.dim, dtype=np.float32)
            for a in chosen:
                blend += self.action_vectors[a] * np.random.uniform(0.5, 1.5)
            bnorm = np.linalg.norm(blend)
            if bnorm > 0:
                blend = blend / bnorm

            sim_state = state.copy()
            trace = [sim_state.copy()]
            for step in range(h):
                sim_state = sim_state + blend * (0.3 ** step)
                for i, ei in enumerate(emotion_vec):
                    if ei > 0.2:
                        sim_state += emotion_vec * ei * 0.05 * (0.5 ** step)
                sim_state = sim_state * 0.95
                nrm = np.linalg.norm(sim_state)
                if nrm > 0:
                    sim_state = sim_state / nrm
                trace.append(sim_state.copy())

            quality = self._evaluate(sim_state, needs)
            trajectories.append({
                "id": ti, "actions": chosen, "quality": quality,
                "final_state": sim_state.copy(), "trace": trace,
            })

        trajectories.sort(key=lambda t: -t["quality"])
        return trajectories

    def _evaluate(self, final_state, needs):
        score = 0.5
        if needs:
            for name, val in needs.items():
                score -= abs(val - 0.7) * 0.2
        entropy = float(-np.sum(final_state * np.log(np.abs(final_state) + 1e-10)))
        if 0.1 < entropy < 2.0:
            score += 0.1
        return float(np.clip(score, 0, 1))

    def best_action(self, trajectories):
        if not trajectories:
            return "reflect", np.zeros(self.dim)
        best = trajectories[0]
        return "+".join(best["actions"]), best["final_state"]

    def to_dict(self):
        return {"n_trajectories": self.n_traj, "horizon": self.horizon,
                "actions": list(self.action_vectors.keys())}
