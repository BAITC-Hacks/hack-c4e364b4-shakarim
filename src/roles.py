"""Explainable role assignment from one precomputed ``node_metrics`` row.

Rules are evaluated in the listed order, so every gid receives exactly one role:

1. ``peripheral`` — no turnover, or at most two counterparties and two transactions.
2. ``terminal`` — received funds and sent at most 20% of them. A node at ``depth >= 4``
   with no observed outgoing edge is excluded because the extraction boundary censors it.
3. ``coordinator`` — at least five unique senders and five unique receivers, with incoming
   and outgoing turnover within a 2:1 ratio.
4. ``consolidator`` — at least three unique senders and incoming turnover more than 20%
   higher than outgoing turnover.
5. ``distributor`` — at least three unique receivers and outgoing turnover more than 20%
   higher than incoming turnover.
6. ``transit`` — positive incoming and outgoing turnover; sends 75%–125% of received funds.
7. ``peripheral`` — fallback for sparse or inconclusive activity.

``role_score`` is a 0–1 confidence value derived from the same metrics; it never changes
the rule-selected role. The generated evidence states the values that triggered the rule.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import re


def key_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


ALIASES = {
    "in_amount": ("in_amount", "incoming_amount", "total_in", "received", "received_kzt", "sum_in", "in_kzt"),
    "out_amount": ("out_amount", "outgoing_amount", "total_out", "sent", "sent_kzt", "sum_out", "out_kzt"),
    "in_count": ("in_count", "incoming_count", "n_in", "incoming_tx_count", "in_tx_count"),
    "out_count": ("out_count", "outgoing_count", "n_out", "outgoing_tx_count", "out_tx_count"),
    "unique_senders": ("unique_senders", "n_senders", "senders_count", "unique_in", "in_degree"),
    "unique_receivers": ("unique_receivers", "n_receivers", "receivers_count", "unique_out", "out_degree"),
    "seed": ("seed", "is_seed", "seed_flag", "is_suspicious", "flagged"),
    "amount": ("amount", "amount_kzt", "sum_kzt", "transaction_amount", "value", "kzt"),
    "depth": ("depth", "node_depth", "distance_from_seed"),
}

ROLE_RULES = (
    "peripheral: zero turnover or <=2 counterparties and <=2 transactions",
    "terminal: outgoing <=20% of incoming; depth-4 boundary with no outgoing edge is excluded",
    "coordinator: >=5 senders, >=5 receivers, and turnover balance >=50%",
    "consolidator: >=3 senders and incoming >120% of outgoing",
    "distributor: >=3 receivers and outgoing >120% of incoming",
    "transit: outgoing is 75%–125% of incoming",
    "peripheral: fallback",
)

MAX_EVIDENCE_LENGTH = 200


def evidence_text(text: str) -> str:
    """Keep an analyst-facing explanation compact for CSV and UI cards."""
    return text if len(text) <= MAX_EVIDENCE_LENGTH else f"{text[:MAX_EVIDENCE_LENGTH - 1].rstrip()}…"


def number(row: Mapping, metric: str) -> float:
    names = ALIASES.get(metric, (metric,))
    normalized = {key_name(k): v for k, v in row.items()}
    for name in names:
        if name in normalized:
            try:
                value = float(str(normalized[name]).replace(" ", "").replace(",", "."))
                return value if math.isfinite(value) else 0.0
            except (ValueError, TypeError):
                continue
    return 0.0


def is_seed(row: Mapping) -> bool:
    for name in ALIASES["seed"]:
        value = str(next((v for k, v in row.items() if key_name(k) == name), "")).strip().lower()
        if value in {"1", "true", "yes", "да", "seed"}:
            return True
    return False


def classify(row: Mapping) -> tuple[str, float, str]:
    incoming, outgoing = max(0.0, number(row, "in_amount")), max(0.0, number(row, "out_amount"))
    senders, receivers = max(0.0, number(row, "unique_senders")), max(0.0, number(row, "unique_receivers"))
    incount, outcount = max(0.0, number(row, "in_count")), max(0.0, number(row, "out_count"))
    total = incoming + outgoing
    depth = number(row, "depth")
    out_degree = number(row, "out_degree")
    ratio = min(incoming, outgoing) / max(incoming, outgoing) if max(incoming, outgoing) else 0.0
    pass_through = outgoing / incoming if incoming else 0.0
    sparse = total == 0 or (senders + receivers <= 2 and incount + outcount <= 2)
    terminal = incoming > 0 and outgoing <= incoming * .2 and not (depth >= 4 and out_degree == 0)
    coordinator = senders >= 5 and receivers >= 5 and ratio >= .5
    consolidator = senders >= 3 and incoming > outgoing * 1.2
    distributor = receivers >= 3 and outgoing > incoming * 1.2
    transit = incoming > 0 and outgoing > 0 and .75 <= pass_through <= 1.25

    if sparse:
        role = "peripheral"
    elif terminal:
        role = "terminal"
    elif coordinator:
        role = "coordinator"
    elif consolidator:
        role = "consolidator"
    elif distributor:
        role = "distributor"
    elif transit:
        role = "transit"
    else:
        role = "peripheral"

    score_by_role = {
        "coordinator": (math.log1p(senders + receivers) / math.log(11)) * .45 + ratio * .35 + min(1.0, total / 1_000_000) * .20,
        "consolidator": min(1.0, senders / 10) * .45 + min(1.0, incoming / max(outgoing, 1)) * .35 + min(1.0, incoming / 1_000_000) * .20,
        "distributor": min(1.0, receivers / 10) * .45 + min(1.0, outgoing / max(incoming, 1)) * .35 + min(1.0, outgoing / 1_000_000) * .20,
        "transit": ratio * .60 + min(1.0, outcount / max(incount, 1)) * .15 + min(1.0, total / 1_000_000) * .25,
        "terminal": (1 - min(1.0, pass_through)) * .55 + min(1.0, incoming / 500_000) * .25 + min(1.0, senders / 5) * .20,
        "peripheral": .35 + min(.5, 1 / max(senders + receivers, 1)) * .5,
    }
    score = score_by_role[role]
    score = round(max(.05, min(1.0, score)), 3)
    if role == "consolidator":
        evidence = f"Получает средства от {int(senders)} уникальных отправителей; входящий объём {incoming:,.0f} KZT превышает исходящий {outgoing:,.0f} KZT."
    elif role == "distributor":
        evidence = f"Переводит средства {int(receivers)} уникальным получателям; исходящий объём {outgoing:,.0f} KZT превышает входящий {incoming:,.0f} KZT."
    elif role == "transit":
        evidence = f"Получает {incoming:,.0f} KZT и переводит дальше {pass_through:.0%} входящего объёма."
    elif role == "terminal":
        evidence = f"Получает {incoming:,.0f} KZT, исходящий объём составляет {outgoing:,.0f} KZT ({pass_through:.0%} входящего)."
    elif role == "coordinator":
        evidence = f"Связан с {int(senders)} отправителями и {int(receivers)} получателями; объёмы входящих и исходящих средств сопоставимы."
    else:
        evidence = f"Небольшая активность: {int(senders)} уникальных отправителей, {int(receivers)} получателей, оборот {total:,.0f} KZT."
    if depth >= 4 and out_degree == 0:
        evidence += " Глубина 4: 0 исходящих может быть следствием границы выгрузки; terminal не подтверждён."
    return role, score, evidence_text(evidence)
