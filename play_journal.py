"""
Lokal spellogg för kuponger/rekommendationer och rättade omgångar.

OBS: Serverdisk på Streamlit Cloud kan vara tillfällig — spelloggen speglas därför även till
webbläsarens localStorage från UI (se journal_browser_sync). JSON-export är valfri backup.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_journal_path(base_dir: Path) -> Path:
    return base_dir / "data" / "user" / "play_journal.json"


def load_journal(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "bets": [], "aggregate": _empty_aggregate()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "bets" not in data:
            data["bets"] = []
        if "aggregate" not in data:
            data["aggregate"] = _empty_aggregate()
        return data
    except Exception:
        return {"version": 1, "bets": [], "aggregate": _empty_aggregate()}


def save_journal(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _empty_aggregate() -> Dict[str, Any]:
    return {
        "settled_rounds": 0,
        "miss_events_total": 0,
        "miss_on_single_pick": 0,
        "miss_on_single_favorite": 0,
        "miss_on_single_underdog": 0,
    }


def rebuild_aggregate_from_bets(bets: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = _empty_aggregate()
    for bet in bets:
        if bet.get("status") != "settled":
            continue
        ins = bet.get("insights") or {}
        misses = ins.get("misses_detail")
        if not misses:
            continue
        agg["settled_rounds"] = int(agg["settled_rounds"]) + 1
        for m in misses:
            agg["miss_events_total"] = int(agg["miss_events_total"]) + 1
            if m.get("single_pick"):
                agg["miss_on_single_pick"] = int(agg["miss_on_single_pick"]) + 1
                if m.get("favorite_side"):
                    agg["miss_on_single_favorite"] = int(agg["miss_on_single_favorite"]) + 1
                else:
                    agg["miss_on_single_underdog"] = int(agg["miss_on_single_underdog"]) + 1
    return agg


def _pick_newer_duplicate_bet(ba: Dict[str, Any], bb: Dict[str, Any]) -> Dict[str, Any]:
    sa, sb = ba.get("status"), bb.get("status")
    if sa != sb:
        return bb if sb == "settled" else ba
    if sa == "settled" and sb == "settled":
        return bb if bb.get("settled_at", "") >= ba.get("settled_at", "") else ba
    return bb if bb.get("saved_at", "") >= ba.get("saved_at", "") else ba


def merge_journal_data(
    a: Optional[Dict[str, Any]],
    b: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Slår ihop två journaldicts (t.ex. disk + webbläsare); deduplicerar på bet-id."""
    aa = a if isinstance(a, dict) else {"version": 1, "bets": [], "aggregate": _empty_aggregate()}
    bb = b if isinstance(b, dict) else {"version": 1, "bets": [], "aggregate": _empty_aggregate()}
    bets_a = {str(x["id"]): x for x in (aa.get("bets") or []) if x.get("id")}
    bets_b = {str(x["id"]): x for x in (bb.get("bets") or []) if x.get("id")}
    merged_map: Dict[str, Dict[str, Any]] = {}
    for bid in set(bets_a) | set(bets_b):
        ba = bets_a.get(bid)
        bb = bets_b.get(bid)
        if ba is None:
            merged_map[bid] = bb  # type: ignore[assignment]
        elif bb is None:
            merged_map[bid] = ba
        else:
            merged_map[bid] = _pick_newer_duplicate_bet(ba, bb)
    bets_list = sorted(merged_map.values(), key=lambda x: x.get("saved_at", ""), reverse=True)
    ver = max(int(aa.get("version", 1)), int(bb.get("version", 1)))
    return {
        "version": ver,
        "bets": bets_list,
        "aggregate": rebuild_aggregate_from_bets(bets_list),
    }


def _prob_vector(row: Dict[str, Any]) -> List[float]:
    return [float(row.get("P1", 0)), float(row.get("PX", 0)), float(row.get("P2", 0))]


def _favorite_threshold(probs: List[float], thr: float = 0.48) -> bool:
    return max(probs) >= thr


def analyze_column_coverage(
    recommendation_rows: List[Dict[str, Any]],
    outcomes_13: str,
) -> Dict[str, Any]:
    """
    Grov täckning: rätt tecken ingår i förslagssträngen (1/X/2).
    Detta är INTE samma som bästa reducer-rad — men bra för lärande över tid.
    """
    o = list(outcomes_13.strip().upper().replace("TIE", "X"))
    if len(o) != 13:
        raise ValueError("Rätt rad måste vara exakt 13 tecken (1, X, 2).")
    valid = {"1", "X", "2"}
    if any(c not in valid for c in o):
        raise ValueError("Ogiltigt tecken i rätt rad (endast 1, X, 2).")

    misses = []
    covered = 0
    for i, row in enumerate(recommendation_rows[:13]):
        pick = str(row.get("Förslag", "")).upper().replace("TIE", "X")
        outcome = o[i]
        ok = outcome in pick
        if ok:
            covered += 1
        else:
            probs = _prob_vector(row)
            misses.append(
                {
                    "index": i + 1,
                    "match": row.get("Match", ""),
                    "outcome": outcome,
                    "forslag": pick,
                    "max_model_prob": float(max(probs)),
                    "single_pick": len(pick) == 1,
                    "favorite_side": bool(_favorite_threshold(probs)),
                }
            )
    return {"covered_columns": covered, "misses": misses}


def settle_bet(
    journal_path: Path,
    bet_id: str,
    hits_best_row: int,
    outcomes_13: Optional[str],
    after_save: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    data = load_journal(journal_path)
    bet = None
    for b in data["bets"]:
        if b.get("id") == bet_id:
            bet = b
            break
    if not bet:
        raise ValueError("Hittade inte spelet.")
    if bet.get("status") != "pending":
        raise ValueError("Spelet är redan rättat.")

    insights: Dict[str, Any] = {"hits_best_row": int(hits_best_row)}
    if outcomes_13:
        cov = analyze_column_coverage(bet.get("recommendation_rows", []), outcomes_13)
        insights["column_coverage"] = cov["covered_columns"]
        insights["misses_detail"] = cov["misses"]
        agg = data["aggregate"]
        agg["settled_rounds"] = int(agg.get("settled_rounds", 0)) + 1
        for m in cov["misses"]:
            agg["miss_events_total"] = int(agg.get("miss_events_total", 0)) + 1
            if m["single_pick"]:
                agg["miss_on_single_pick"] = int(agg.get("miss_on_single_pick", 0)) + 1
                if m["favorite_side"]:
                    agg["miss_on_single_favorite"] = int(agg.get("miss_on_single_favorite", 0)) + 1
                else:
                    agg["miss_on_single_underdog"] = int(agg.get("miss_on_single_underdog", 0)) + 1

    bet["status"] = "settled"
    bet["settled_at"] = _utc_now_iso()
    bet["insights"] = insights
    if outcomes_13:
        bet["outcomes_13"] = outcomes_13.strip().upper()

    save_journal(journal_path, data)
    if after_save:
        after_save(data)
    return bet


def add_pending_bet(
    journal_path: Path,
    draw_meta: Dict[str, str],
    coupon_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    system_rows: int,
    note: str = "",
    after_save: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> str:
    data = load_journal(journal_path)
    bet_id = f"bet_{uuid.uuid4().hex[:10]}"
    bet = {
        "id": bet_id,
        "status": "pending",
        "saved_at": _utc_now_iso(),
        "draw_number": str(draw_meta.get("draw_number", "")),
        "draw_comment": str(draw_meta.get("draw_comment", "")),
        "reg_close_time": str(draw_meta.get("reg_close_time", "")),
        "game_type": "europatipset",
        "system_rows": int(system_rows),
        "note": note,
        "coupon_rows": coupon_df.to_dict(orient="records"),
        "recommendation_rows": recommendation_df.to_dict(orient="records"),
    }
    data["bets"].insert(0, bet)
    save_journal(journal_path, data)
    if after_save:
        after_save(data)
    return bet_id


def learning_hint(data: Dict[str, Any]) -> str:
    agg = data.get("aggregate") or _empty_aggregate()
    n = int(agg.get("settled_rounds", 0))
    if n < 2:
        return "Spara och rätta minst två omgångar med rätt rad för personlig lärdom."
    sp = int(agg.get("miss_on_single_pick", 0))
    fav = int(agg.get("miss_on_single_favorite", 0))
    parts = [f"Baserat på {n} rättade omgångar i loggen."]
    if sp >= 3 and fav >= sp // 2:
        parts.append(
            "Du missar ofta på **spikar där modellen favoriserar ett tecken** — prova mer halvgardering nästa gång."
        )
    elif sp >= 3:
        parts.append(
            "Du missar ofta på **smala spikar** — överväg halvgardering på osäkra favoriter."
        )
    else:
        parts.append("Mönstret är ännu blandat; fortsätt logga rätt rader för tydligare signal.")
    return " ".join(parts)


def append_outcomes_training_rows(
    base_dir: Path,
    recommendation_rows: List[Dict[str, Any]],
    outcomes_13: str,
) -> None:
    """
    Valfri extra-fil för framtida modell-träning (manuell sammanslagning med history.csv).
    """
    path = base_dir / "data" / "user" / "settled_outcomes_augment.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    o = list(outcomes_13.strip().upper())
    rows = []
    ymap = {"1": 0, "X": 1, "2": 2}
    for i, row in enumerate(recommendation_rows[:13]):
        rows.append(
            {
                "Match": row.get("Match", ""),
                "outcome_sign": o[i],
                "y": ymap.get(o[i], ""),
                "P1": row.get("P1", ""),
                "PX": row.get("PX", ""),
                "P2": row.get("P2", ""),
                "saved_at": _utc_now_iso(),
            }
        )
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, low_memory=False)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
