"""Transparent 0–1 review priority score.

The score is the sum of four bounded components: turnover (0.45), connectivity
(0.25), turnover imbalance (0.20), and seed status (0.10).
"""
from __future__ import annotations

import math
from roles import is_seed, number


def priority_score(row):
    incoming, outgoing = number(row, "in_amount"), number(row, "out_amount")
    senders = number(row, "unique_senders")
    receivers = number(row, "unique_receivers")
    # Log scale limits domination by a single high-volume node.
    volume = min(1.0, math.log1p(max(0.0, incoming + outgoing)) / math.log1p(10_000_000))
    connectivity = min(1.0, math.log1p(max(0.0, senders + receivers)) / math.log1p(100))
    imbalance = abs(incoming - outgoing) / max(incoming + outgoing, 1.0)
    score = .45 * volume + .25 * connectivity + .20 * imbalance + (.10 if is_seed(row) else 0)
    return round(min(1.0, score), 3)


def priority_reason(role, score, evidence):
    return f"Роль {role}; приоритет {score:.3f}/1. {evidence}"
