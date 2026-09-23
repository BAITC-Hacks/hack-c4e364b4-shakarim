"""Explainable role assignment from one precomputed ``node_metrics`` row.

Rules are evaluated in the listed order, so every gid receives exactly one role:

0. Censored leaves (depth >= 4, no observed outgoing transfers) are ``peripheral``
   with low support. Missing outflow is not evidence of retention/consolidation.
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

Seeds have incomplete inflow: terminal/transit and balance-based decisions are
disabled. For seeds coordinator uses >=5 senders and >=5 receivers; consolidator
uses >=3 senders and >=2 times as many senders as receivers (positive inflow);
distributor is symmetric (positive outflow). These are structural hypotheses.

``role_score`` is bounded heuristic support, not a calibrated probability. It
never changes the selected role. Its formula is documented in analytics.md.
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
    "peripheral: censored depth-4 leaf; no retention evidence, support 0.2",
    "peripheral: zero turnover or <=2 counterparties and <=2 transactions",
    "terminal: non-seed, incoming >0, outgoing <=20% of incoming",
    "coordinator: >=5 senders, >=5 receivers; balance >=50% for non-seeds",
    "consolidator: >=3 senders, incoming >0; incoming >120% of outgoing, or seed senders >=2*max(receivers,1)",
    "distributor: >=3 receivers, outgoing >0; outgoing >120% of incoming, or seed receivers >=2*max(senders,1)",
    "transit: non-seed, positive incoming/outgoing, outgoing is 75%–125% of incoming",
    "peripheral: fallback",
)

ROLE_NAMES = frozenset({"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"})

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
        if value in {"1", "1.0", "true", "yes", "да", "seed"}:
            return True
    return False


def is_censored(row: Mapping) -> bool:
    """No observed output at the extraction boundary cannot establish a sink."""
    flagged = str(row.get("truncated_by_depth", "")).strip().lower() in {"true", "1", "1.0"}
    no_output = (
        number(row, "out_amount") <= 0 and number(row, "unique_receivers") <= 0
        and number(row, "out_count") <= 0 and number(row, "out_degree") <= 0
    )
    return flagged or (number(row, "depth") >= 4 and no_output)


def amount_text(value: float) -> str:
    return f"{value:,.0f}" if abs(value) < 1e12 else f"{value:.3g}"


def classify(row: Mapping) -> tuple[str, float, str]:
    incoming, outgoing = max(0.0, number(row, "in_amount")), max(0.0, number(row, "out_amount"))
    senders, receivers = max(0.0, number(row, "unique_senders")), max(0.0, number(row, "unique_receivers"))
    incount, outcount = max(0.0, number(row, "in_count")), max(0.0, number(row, "out_count"))
    total = incoming + outgoing
    seed, censored = is_seed(row), is_censored(row)
    ratio = min(incoming, outgoing) / max(incoming, outgoing) if max(incoming, outgoing) else 0.0
    pass_through = outgoing / incoming if incoming else 0.0
    sparse = total == 0 or (senders + receivers <= 2 and incount + outcount <= 2)
    terminal = not seed and incoming > 0 and outgoing <= incoming * .2
    coordinator = senders >= 5 and receivers >= 5 and (seed or ratio >= .5)
    consolidator = senders >= 3 and incoming > 0 and (
        senders >= 2 * max(receivers, 1) if seed else incoming > outgoing * 1.2
    )
    distributor = receivers >= 3 and outgoing > 0 and (
        receivers >= 2 * max(senders, 1) if seed else outgoing > incoming * 1.2
    )
    transit = not seed and incoming > 0 and outgoing > 0 and .75 <= pass_through <= 1.25

    if censored or sparse:
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

    in_dominance = max(0, incoming - outgoing) / max(incoming, 1)
    out_dominance = max(0, outgoing - incoming) / max(outgoing, 1)
    score_by_role = {
        "coordinator": min(1, math.log1p(senders + receivers) / math.log(101)) * .45 + (0 if seed else ratio) * .35 + min(1.0, total / 1_000_000) * .20,
        "consolidator": min(1.0, senders / 10) * .45 + (0 if seed else in_dominance) * .35 + min(1.0, incoming / 1_000_000) * .20,
        "distributor": min(1.0, receivers / 10) * .45 + (0 if seed else out_dominance) * .35 + min(1.0, outgoing / 1_000_000) * .20,
        "transit": ratio * .60 + min(1.0, outcount / max(incount, 1)) * .15 + min(1.0, total / 1_000_000) * .25,
        "terminal": (1 - min(1.0, pass_through)) * .55 + min(1.0, incoming / 500_000) * .25 + min(1.0, senders / 5) * .20,
        "peripheral": .2 if censored or not sparse else .6,
    }
    score = score_by_role[role]
    score = round(max(.05, min(1.0, score)), 3)
    flows = f"вход {amount_text(incoming)}, выход {amount_text(outgoing)} KZT"
    if role == "consolidator":
        evidence = f"Признаки консолидации: {senders:g} отправителей, {receivers:g} получателей; {flows}."
    elif role == "distributor":
        evidence = f"Признаки распределения: {receivers:g} получателей, {senders:g} отправителей; {flows}."
    elif role == "transit":
        evidence = f"Признаки транзита: вход {amount_text(incoming)} KZT, выход {amount_text(outgoing)} KZT ({pass_through:.0%} входа); {senders:g} отправителей."
    elif role == "terminal":
        evidence = f"Возможный конечный получатель: {flows}; выход {pass_through:.0%} входа, отправителей {senders:g}."
    elif role == "coordinator":
        evidence = f"Признаки координации: {senders:g} отправителей и {receivers:g} получателей; {flows}."
    else:
        evidence = f"Роль не определена: {senders:g} отправителей, {receivers:g} получателей; {flows}."
    if censored:
        evidence += " Обрыв на depth=4: 0 исходящих — эффект границы выгрузки."
    if seed:
        evidence += " Seed: входящие неполны; баланс не используется."
    return role, score, evidence_text(evidence)
