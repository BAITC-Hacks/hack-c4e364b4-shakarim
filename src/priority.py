"""Transparent 0–1 review priority score.

The score is the sum of four bounded components: turnover (0.45), connectivity
(0.25), observed turnover imbalance (0.20), and seed status (0.10). Imbalance is
disabled for seeds and censored depth-four leaves; their balance is incomplete.
"""
from __future__ import annotations

import math
from roles import is_censored, is_seed, number


def priority_components(row):
    """Return the actual additive contributions, for transparent TOP explanations."""
    incoming, outgoing = max(0, number(row, "in_amount")), max(0, number(row, "out_amount"))
    senders = max(0, number(row, "unique_senders"))
    receivers = max(0, number(row, "unique_receivers"))
    # Log scale limits domination by a single high-volume node.
    volume = min(1.0, math.log1p(max(0.0, incoming + outgoing)) / math.log1p(10_000_000))
    connectivity = min(1.0, math.log1p(max(0.0, senders + receivers)) / math.log1p(100))
    imbalance = (0.0 if is_seed(row) or is_censored(row)
                 else abs(incoming - outgoing) / max(incoming + outgoing, 1.0))
    return {"volume": .45 * volume, "connectivity": .25 * connectivity,
            "imbalance": .20 * imbalance, "seed": .10 if is_seed(row) else 0.0}


def priority_score(row):
    return round(max(0.0, min(1.0, sum(priority_components(row).values()))), 3)


def priority_reason(role, score, evidence, metrics=None):
    if metrics is None:
        return f"Роль {role}; приоритет {score:.3f}/1. {evidence}"
    parts = priority_components(metrics)
    details = (f"оборот {parts['volume']:.3f}, связи {parts['connectivity']:.3f}, "
               f"дисбаланс {parts['imbalance']:.3f}, seed {parts['seed']:.3f}")
    return f"Приоритет {score:.3f}/1 ({details}). {evidence}"
