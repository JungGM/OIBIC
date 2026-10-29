"""Local constrained VPP grouping on complete, out-of-sample residuals.

This is an adaptation of residual cancellation / combinatorial portfolio
selection, not a reproduction of a paper's market model. No missing target
may be imputed to call this optimizer. Optimize on an earlier period and
accept its frozen candidate only on a separate chronological gate period.
"""
from __future__ import annotations

import numpy as np

MIN_CAPACITY_KW = 20_000.0


def _inputs(residuals, weights, capacity, assignment):
    e = np.asarray(residuals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(capacity, dtype=np.float64)
    raw = np.asarray(assignment)
    if e.ndim != 2 or not e.shape[0] or not e.shape[1]:
        raise ValueError("residuals must be a nonempty time x site matrix")
    if w.shape != (e.shape[0],) or c.shape != (e.shape[1],) or raw.shape != c.shape:
        raise ValueError("weights/capacity/assignment dimensions do not match residuals")
    if not all(np.isfinite(x).all() for x in (e, w, c, raw)):
        raise ValueError("All inputs must be finite; never fill missing residuals with zero")
    if np.any(w < 0) or np.any(c <= 0) or not np.isin(raw, [0, 1]).all():
        raise ValueError("Require nonnegative weights, positive capacity and binary assignment")
    a = raw.astype(np.int8, copy=True)
    cap1 = float(c @ a)
    cap2 = float(c.sum() - cap1)
    if min(cap1, cap2) < MIN_CAPACITY_KW:
        raise ValueError("Both groups must have at least 20,000 kW")
    return np.ascontiguousarray(e), w, c, a


def objective(residuals, weights, assignment):
    """Sum SMP*k*absolute GROUP residual; multiply by 0.1 for penalty KRW."""
    e = np.asarray(residuals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    a = np.asarray(assignment)
    if e.ndim != 2 or w.shape != (len(e),) or a.shape != (e.shape[1],):
        raise ValueError("Invalid objective dimensions")
    if not all(np.isfinite(x).all() for x in (e, w, a)):
        raise ValueError("Score inputs must be finite")
    if np.any(w < 0) or not np.isin(a, [0, 1]).all():
        raise ValueError("Invalid weights or assignment")
    g1 = e @ a.astype(float)
    return float(w @ (np.abs(g1) + np.abs(e.sum(axis=1) - g1)))


def optimize(residuals, weights, capacity, reference, seed=20261029,
             steps=20000, restarts=5):
    """Return best feasible assignment and training diagnostics.

    1 means G1, 0 means G2. A deterministic sequence mixes single-site moves
    with cross-group swaps and geometrically cools a Metropolis temperature.
    The exact original objective is retained, so annealing can never make the
    returned incumbent worse than the reference on this training period.
    The report includes at most five distinct restart winners for diagnostics;
    choosing among these on the gate would turn the gate into another search.
    """
    e, w, c, ref = _inputs(residuals, weights, capacity, reference)
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(restarts, (int, np.integer)) or restarts < 1:
        raise ValueError("restarts must be a positive integer")
    rng = np.random.default_rng(seed)
    total_cap = float(c.sum())
    total_e = e.sum(axis=1)
    mass = float(w.sum())
    wn = w / mass if mass else np.zeros_like(w)
    ref_score = objective(e, w, ref)
    best_a, best_score = ref.copy(), ref_score
    winners = {ref.tobytes(): (ref_score, ref.copy())}
    history = []
    accepted_total = feasible_total = 0
    # Contiguous per-site residuals make individual move updates inexpensive.
    site_e = np.ascontiguousarray(e.T)

    def cost(group_e):
        return float(wn @ (np.abs(group_e) + np.abs(total_e - group_e)))

    def proposal(a, cap1):
        ones, zeros = np.flatnonzero(a), np.flatnonzero(1 - a)
        if rng.random() < 0.3:
            i = int(rng.integers(len(a)))
            sign = 1 - 2 * int(a[i])
            cap_new = cap1 + sign * c[i]
            return (i,), sign * site_e[i], cap_new
        i = int(ones[rng.integers(len(ones))])
        j = int(zeros[rng.integers(len(zeros))])
        return (i, j), site_e[j] - site_e[i], cap1 + c[j] - c[i]

    if mass:
        for restart in range(restarts):
            a = ref.copy()
            cap1 = float(c @ a)
            # Feasible diversifications do not replace the global incumbent.
            for _ in range(restart * max(10, len(c) // 4)):
                ix, delta, cap_new = proposal(a, cap1)
                if MIN_CAPACITY_KW <= cap_new <= total_cap - MIN_CAPACITY_KW:
                    a[list(ix)] = 1 - a[list(ix)]
                    cap1 = cap_new
            group_e = e @ a
            current = cost(group_e)
            local_best, local_a = objective(e, w, a), a.copy()
            deltas = []
            for _ in range(80):
                _, delta, cap_new = proposal(a, cap1)
                if MIN_CAPACITY_KW <= cap_new <= total_cap - MIN_CAPACITY_KW:
                    jump = cost(group_e + delta) - current
                    if jump > 0:
                        deltas.append(jump)
            temperature = max(float(np.median(deltas)) * 1.5 if deltas else 0,
                              current * 0.002, 1e-8)
            accepted = feasible = 0
            for step in range(steps):
                ix, delta, cap_new = proposal(a, cap1)
                if not (MIN_CAPACITY_KW <= cap_new <= total_cap - MIN_CAPACITY_KW):
                    continue
                feasible += 1
                proposed_e = group_e + delta
                proposed = cost(proposed_e)
                jump = proposed - current
                temp = temperature * (0.001 ** (step / max(steps - 1, 1)))
                if jump <= 0 or rng.random() < np.exp(-jump / temp):
                    a[list(ix)] = 1 - a[list(ix)]
                    exact_cap = float(c @ a)
                    if not MIN_CAPACITY_KW <= exact_cap <= total_cap - MIN_CAPACITY_KW:
                        # Decimal capacities can accumulate roundoff at the boundary.
                        a[list(ix)] = 1 - a[list(ix)]
                        continue
                    cap1, group_e, current = exact_cap, proposed_e, proposed
                    accepted += 1
                    if current * mass < local_best:
                        # Recompute from original matrix before retaining an incumbent.
                        exact = objective(e, w, a)
                        if exact < local_best:
                            local_best, local_a = exact, a.copy()
                        if exact < best_score:
                            best_score, best_a = exact, a.copy()
                # Bound accumulated floating-point drift in long accepted walks.
                if step % 1000 == 999:
                    group_e = e @ a
                    cap1 = float(c @ a)
                    current = cost(group_e)
            accepted_total += accepted
            feasible_total += feasible
            winners[local_a.tobytes()] = (local_best, local_a.copy())
            history.append({"restart": restart, "accepted": accepted,
                            "feasible_proposals": feasible,
                            "best_weighted_error": local_best})
    # Validate exact capacity and objective independently of incremental updates.
    _inputs(e, w, c, best_a)
    best_score = objective(e, w, best_a)
    ranked = sorted(winners.values(), key=lambda item: item[0])[:5]
    report = {
        "algorithm": "feasible multistart simulated annealing with reference incumbent",
        "seed": int(seed), "steps_per_restart": int(steps), "restarts": int(restarts),
        "hours": len(e), "sites": e.shape[1], "reference_weighted_error": ref_score,
        "best_weighted_error": best_score,
        "training_improvement_fraction": 1 - best_score / ref_score if ref_score else 0.0,
        "irreducible_total_residual_lower_bound": float(w @ np.abs(total_e)),
        "group_capacity_kw": {"G1": float(c @ best_a),
                              "G2": float(c.sum() - c @ best_a)},
        "group_site_count": {"G1": int(best_a.sum()), "G2": int((1 - best_a).sum())},
        "accepted_proposals": accepted_total, "feasible_proposals": feasible_total,
        "restart_history": history,
        "training_candidates": [{"weighted_error": score, "assignment": a.tolist()}
                                for score, a in ranked],
        "selection_warning": "Training search only; gate the single frozen best candidate on a later period.",
    }
    return best_a, report


def daily_gate(reference_daily, candidate_daily, min_improvement=0.01,
               seed=20261029, bootstrap=2000):
    """Conservative paired-day gate, for one previously frozen candidate.

    Require the stated fractional gain and a positive 90% percentile-bootstrap
    lower bound on mean daily savings. This is a diagnostic for this period,
    not a guarantee under seasonality or a future distribution shift.
    """
    r = np.asarray(reference_daily, dtype=float)
    p = np.asarray(candidate_daily, dtype=float)
    if r.ndim != 1 or p.shape != r.shape or not len(r):
        raise ValueError("Gate needs equal nonempty daily score arrays")
    if not np.isfinite(r).all() or not np.isfinite(p).all() or np.any(r < 0) or np.any(p < 0):
        raise ValueError("Daily losses must be finite and nonnegative")
    if bootstrap < 1 or not 0 <= min_improvement < 1:
        raise ValueError("Invalid gate settings")
    savings = r - p
    rng = np.random.default_rng(seed)
    boot = savings[rng.integers(len(r), size=(bootstrap, len(r)))].mean(axis=1)
    lower = float(np.quantile(boot, 0.05))
    gain = float(1 - p.sum() / r.sum()) if r.sum() else 0.0
    accepted = bool(len(r) >= 14 and gain >= min_improvement and lower > 0)
    return {"accepted": accepted, "days": len(r), "improvement_fraction": gain,
            "required_improvement_fraction": min_improvement,
            "mean_daily_savings_lower90": lower, "bootstrap_samples": int(bootstrap),
            "note": "Paired-day bootstrap diagnostic; assumes days are exchangeable, not a seasonal guarantee."}
