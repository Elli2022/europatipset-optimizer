"""
Inbyggda omgångsanalyser och journal-driven kalibrering för systembyggaren.

Syfte: undvika att optimera bara mot «kolumn-täckning» när Europatipset kräver
minst 10 rätt på *en* rad för utdelning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Referens: trygg 54-kupong (chat) vs utfall vecka 22, 2026-05-27
WEEK_22_OUTCOMES = "121212122X211"
WEEK_22_REFERENCE_SAFE_PICKS = [
    "1",
    "2",
    "2",
    "1X2",
    "1",
    "1",
    "1X",
    "2",
    "1",
    "1X2",
    "1",
    "1X2",
    "1",
]
WEEK_22_PLAYED_PICKS = [
    "1X2",
    "2",
    "2",
    "1",
    "1X",
    "2",
    "1",
    "1",
    "1",
    "X2",
    "1",
    "1",
    "1",
]
WEEK_22_MATCHES = [
    "Crystal Palace - Rayo Vallecano",
    "AIK - Häcken",
    "Eskilstuna - Hammarby",
    "Club Libertad Asunción - UCV FC",
    "CSD Independiente del Valle - Rosario",
    "Club Bolívar - Independiente Rivadavia",
    "Corinthians - Platense",
    "CA Peñarol - Santa Fe",
    "Atletico Mineiro - Academia Puerto Cabello",
    "Caracas FC - Botafogo RJ",
    "Cienciano - Juventud de Las Piedras",
    "Club Olimpia Asunción - Audax Italiano",
    "Vasco da Gama - Barracas Central",
]


def payout_min_rights(game_type: str = "europatipset") -> int:
    return 10 if game_type == "europatipset" else 8


def spike_risk_score(probs: np.ndarray) -> float:
    """
    Högre värde => mer skäl att halvgardera i stället för spik.
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0)
    p = p / p.sum()
    top = float(p.max())
    second = float(np.partition(p, -2)[-2])
    gap = top - second
    miss_single = 1.0 - top
    tight_fav = 1.0 if gap < 0.12 else 0.0
    weak_fav = 1.0 if top < 0.55 else 0.0
    return float(np.clip(0.50 * miss_single + 0.30 * tight_fav + 0.20 * weak_fav, 0.0, 1.0))


def journal_calibration_params(aggregate: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Justera hur aggressivt systembyggaren halvgarderar utifrån spellogg.
    Baslinje inkluderar lärdom från vecka 22 även utan egen logg.
    """
    params = {
        "spike_min_top_prob": 0.52,
        "force_hedge_risk": 0.62,
        "hedge_value_multiplier": 0.85,
        "hedge_boost_threshold": 0.45,
    }
    if not aggregate:
        return params

    settled = int(aggregate.get("settled_rounds", 0))
    sp = int(aggregate.get("miss_on_single_pick", 0))
    fav = int(aggregate.get("miss_on_single_favorite", 0))
    gap_events = int(aggregate.get("column_vs_payout_gap_events", 0))

    if settled >= 1:
        params["spike_min_top_prob"] = 0.54
        params["force_hedge_risk"] = 0.58

    if sp >= 2:
        params["spike_min_top_prob"] = 0.56
        params["force_hedge_risk"] = 0.55
        params["hedge_value_multiplier"] = 1.05

    if sp >= 3 and fav >= max(1, sp // 2):
        params["spike_min_top_prob"] = 0.58
        params["force_hedge_risk"] = 0.50
        params["hedge_value_multiplier"] = 1.25

    if gap_events >= 1:
        params["force_hedge_risk"] = min(params["force_hedge_risk"], 0.52)
        params["hedge_value_multiplier"] = max(params["hedge_value_multiplier"], 1.15)

    return params


def column_coverage_count(picks: List[str], outcomes_13: str) -> int:
    o = list(outcomes_13.strip().upper().replace("TIE", "X"))
    covered = 0
    for pick, outcome in zip(picks[:13], o[:13]):
        if outcome in str(pick).upper().replace("TIE", "X"):
            covered += 1
    return covered


def analyze_week_22_case() -> Dict[str, Any]:
    o = WEEK_22_OUTCOMES
    played_cov = column_coverage_count(WEEK_22_PLAYED_PICKS, o)
    ref_cov = column_coverage_count(WEEK_22_REFERENCE_SAFE_PICKS, o)
    misses_played = []
    misses_ref = []
    for i, (match, pick, ref, outcome) in enumerate(
        zip(WEEK_22_MATCHES, WEEK_22_PLAYED_PICKS, WEEK_22_REFERENCE_SAFE_PICKS, o)
    ):
        ok_p = outcome in pick.upper()
        ok_r = outcome in ref.upper()
        if not ok_p:
            misses_played.append(
                {"match": match, "pick": pick, "outcome": outcome, "reference_pick": ref}
            )
        if not ok_r:
            misses_ref.append({"match": match, "reference_pick": ref, "outcome": outcome})

    return {
        "round_id": "europatipset_v2026-22",
        "outcomes_13": o,
        "played_column_coverage": played_cov,
        "reference_safe_coverage": ref_cov,
        "hits_best_row_played": 8,
        "payout_threshold": 10,
        "payout_kr": 0,
        "misses_played": misses_played,
        "misses_reference_safe": misses_ref,
        "headline_lessons": [
            "8 kolumner rätt ger 0 kr — Europatipset betalar från 10 rätt på en rad.",
            "Smala spikar (AIK–Häcken 2, Libertad 1, Corinthians 1, Peñarol 1, Cienciano 1) kostade utdelning.",
            "Halvgarderingar på osäkra cup-/gruppspelsmatcher gav 9/13 kolumn-täckning i referensraden — fortfarande under utdelningsgränsen utan bättre rad.",
            "Modellen ska prioritera sannolikhet för ≥10 rätt på bästa rad, inte bara flest rätta tecken totalt.",
        ],
    }


def builtin_lessons_summary() -> str:
    case = analyze_week_22_case()
    lines = [
        "**Inbyggd analys (Vecka 22, 2026-05-27)**",
        f"- Utfall: `{case['outcomes_13']}`",
        f"- Ditt spel: {case['played_column_coverage']}/13 kolumner rätt, **{case['hits_best_row_played']} rätt bästa rad**, **0 kr**.",
        f"- Referens «trygg» rad: {case['reference_safe_coverage']}/13 kolumner — visar att även garderingar inte garanterar utdelning.",
        "",
        "**Regler framåt (safe-strategi):**",
        "1. Spika bara när topp-tecken ≥ ~54–58% (höjs om du missar spikar i loggen).",
        "2. Halvgardera cupfinaler, rotationsmatcher och bortafavoriter med tunn marginal.",
        "3. Jämför alltid «kolumn-täckning» med «bästa rad ≥10» i Träffprognos.",
    ]
    return "\n".join(lines)


def lessons_dir(base_dir: Path) -> Path:
    return base_dir / "data" / "user" / "lessons"


def save_builtin_lesson_file(base_dir: Path) -> Path:
    path = lessons_dir(base_dir) / "europatipset_v2026-22.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = analyze_week_22_case()
    payload["played_picks"] = WEEK_22_PLAYED_PICKS
    payload["reference_safe_picks"] = WEEK_22_REFERENCE_SAFE_PICKS
    payload["matches"] = WEEK_22_MATCHES
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_lesson_files(base_dir: Path) -> List[Dict[str, Any]]:
    d = lessons_dir(base_dir)
    if not d.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out
