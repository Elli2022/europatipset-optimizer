"""
Gratis matchkontext för Europatipset-kupongen.

- Ligatabell (position, poäng, målskillnad) via football-data.org **Free** /standings.
- Enkel **form (senaste 5)** från lokal `history_api.csv` (samma källa som API-synken) → inga extra API-anrop.

Skador/elvor och betald «deep data» ingår **inte** — det är medvetet för att hålla allt gratis och stabilt.
"""

from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# Synkad lista med europatipset.py / sync_history_from_free_api (free-tier ligor).
FREE_COMPETITION_CODES: Tuple[str, ...] = ("PL", "ELC", "PD", "BL1", "SA", "FL1", "DED", "PPL")

BASE_URL = "https://api.football-data.org/v4"


def _norm_team(s: str) -> str:
    t = str(s).lower().strip()
    t = t.replace(".", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return max(0.82, SequenceMatcher(None, a, b).ratio())
    return SequenceMatcher(None, a, b).ratio()


def _best_team_match(query: str, roster: Dict[str, Dict]) -> Tuple[Optional[str], float]:
    q = _norm_team(query)
    best_key: Optional[str] = None
    best_sc = 0.0
    for canonical in roster.keys():
        sc = _similarity(q, canonical)
        if sc > best_sc:
            best_sc = sc
            best_key = canonical
    if best_sc < 0.72:
        return None, best_sc
    return best_key, best_sc


def _parse_standings_json(payload: Dict) -> Dict[str, Dict]:
    """canonical normalized team name -> standing facts."""
    blocks = payload.get("standings", []) or []
    preferred = [b for b in blocks if b.get("type") == "TOTAL"]
    iter_blocks = preferred if preferred else blocks
    roster: Dict[str, Dict] = {}
    for block in iter_blocks:
        if block.get("type") in ("HOME", "AWAY"):
            continue
        for row in block.get("table", []) or []:
            team = (row.get("team") or {}).get("name")
            if not team:
                continue
            key = _norm_team(team)
            roster[key] = {
                "display": team,
                "position": int(row.get("position", 0)),
                "played": int(row.get("playedGames", 0)),
                "won": int(row.get("won", 0)),
                "draw": int(row.get("draw", 0)),
                "lost": int(row.get("lost", 0)),
                "points": int(row.get("points", 0)),
                "gf": int(row.get("goalsFor", 0)),
                "ga": int(row.get("goalsAgainst", 0)),
                "gd": int(row.get("goalDifference", 0)),
            }
    return roster


def _load_standings_cache(cache_path: Path, ttl_seconds: float) -> Optional[Dict[str, Dict[str, Dict]]]:
    if not cache_path.exists():
        return None
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        ts = float(raw.get("_cached_at", 0))
        if time.time() - ts > ttl_seconds:
            return None
        data = raw.get("standings") or {}
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_standings_cache(cache_path: Path, data: Dict[str, Dict[str, Dict]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_cached_at": time.time(), "standings": data}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_all_free_standings(api_key: str, cache_path: Path, ttl_hours: float = 12.0) -> Dict[str, Dict[str, Dict]]:
    """
    Hämtar tabeller för alla FREE_COMPETITION_CODES. Resultat: competition_code -> roster dict.
    """
    ttl = max(300.0, float(ttl_hours) * 3600.0)
    cached = _load_standings_cache(cache_path, ttl)
    if cached is not None:
        return cached

    headers = {"X-Auth-Token": api_key}
    out: Dict[str, Dict[str, Dict]] = {}
    for code in FREE_COMPETITION_CODES:
        url = f"{BASE_URL}/competitions/{code}/standings"
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                continue
            roster = _parse_standings_json(r.json())
            if roster:
                out[code] = roster
        except Exception:
            continue

    if out:
        _save_standings_cache(cache_path, out)
    return out


def split_match_cell(raw: str) -> Tuple[str, str]:
    s = str(raw).strip()
    if " - " in s:
        a, b = s.split(" - ", 1)
        return a.strip(), b.strip()
    return "", ""


def _pick_competition_for_pair(
    home_q: str,
    away_q: str,
    standings_by_comp: Dict[str, Dict[str, Dict]],
) -> Tuple[Optional[str], Optional[Dict], Optional[Dict], float]:
    """Välj den liga där båda lagen matchar truppen bäst."""
    best: Tuple[Optional[str], Optional[Dict], Optional[Dict], float] = (None, None, None, -1.0)
    for code, roster in standings_by_comp.items():
        hk, hs = _best_team_match(home_q, roster)
        ak, as_ = _best_team_match(away_q, roster)
        if hk is None or ak is None:
            continue
        score = (hs + as_) / 2.0
        if score > best[3]:
            best = (code, roster.get(hk), roster.get(ak), score)
    return best


def _short_comment(home: Dict, away: Dict) -> str:
    pdiff = int(home["points"]) - int(away["points"])
    pos_gap = int(away["position"]) - int(home["position"])  # positivt om hemma högre i tabellen
    if pos_gap >= 6 and pdiff >= 6:
        return "Tabell starkt för hemmalaget"
    if pos_gap <= -6 and pdiff <= -6:
        return "Tabell starkt för bortalaget"
    if abs(pos_gap) <= 2 and abs(pdiff) <= 3:
        return "Tabell relativt jämn"
    return "Tabell lutar åt hemmalaget" if pos_gap > 0 else "Tabell lutar åt bortalaget"


def _load_history_api(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path or not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.empty or "HomeTeam" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _fuzzy_names_equal(a: str, b: str) -> bool:
    return _similarity(_norm_team(a), _norm_team(b)) >= 0.78


def _form_last_n(team_display: str, competition: str, hist: pd.DataFrame, n: int = 5) -> str:
    """Returnerar t.ex. 'WWDLX' över senaste n matcherna för laget i serien."""
    rows = []
    if "Date" in hist.columns:
        work = hist.copy()
        work["_d"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work.dropna(subset=["_d"]).sort_values("_d", ascending=False)
    else:
        work = hist.iloc[::-1].copy()

    for _, r in work.iterrows():
        if str(r.get("Competition", "")).strip() != competition:
            continue
        h = str(r.get("HomeTeam", ""))
        aw = str(r.get("AwayTeam", ""))
        ftr = str(r.get("FTR", "")).upper()
        if ftr not in {"H", "D", "A"}:
            continue
        if _fuzzy_names_equal(team_display, h):
            if ftr == "H":
                rows.append("W")
            elif ftr == "D":
                rows.append("D")
            else:
                rows.append("L")
        elif _fuzzy_names_equal(team_display, aw):
            if ftr == "A":
                rows.append("W")
            elif ftr == "D":
                rows.append("D")
            else:
                rows.append("L")
        else:
            continue
        if len(rows) >= n:
            break
    return "".join(rows) if rows else ""


def build_free_context_for_coupon(
    coupon_df: pd.DataFrame,
    api_key: Optional[str],
    history_csv: Optional[Path],
    cache_path: Path,
    ttl_hours: float = 12.0,
) -> Tuple[pd.DataFrame, str]:
    """
    Returnerar (df_med_extra_kolumner_samma_index, status_text).
    """
    if api_key is None or str(api_key).strip() == "":
        empty = pd.DataFrame(index=coupon_df.index)
        return empty, "Saknar FOOTBALL_DATA_API_KEY — ingen ligatabell hämtad (övrigt gratis-läge funkar ändå via lokal historik om den finns)."

    standings = fetch_all_free_standings(str(api_key).strip(), cache_path, ttl_hours=ttl_hours)
    if not standings:
        empty = pd.DataFrame(index=coupon_df.index)
        return empty, "Kunde inte hämta ligatabeller (nyckel ogiltig, kvot eller nätverksfel)."

    hist = _load_history_api(history_csv)

    rows_out = []
    matched = 0
    for idx, row in coupon_df.iterrows():
        home_q, away_q = split_match_cell(row.get("Match", ""))
        code, home_i, away_i, conf = _pick_competition_for_pair(home_q, away_q, standings)
        if code is None or home_i is None or away_i is None:
            rows_out.append(
                {
                    "Ctx_liga": "",
                    "Ctx_match_conf": round(conf, 2),
                    "Ctx_h_pos": "",
                    "Ctx_b_pos": "",
                    "Ctx_poängdiff": "",
                    "Ctx_plac_diff": "",
                    "Ctx_form_H": "",
                    "Ctx_form_B": "",
                    "Ctx_tabell": "",
                }
            )
            continue
        matched += 1
        hp, ap = home_i["position"], away_i["position"]
        pts_h, pts_a = home_i["points"], away_i["points"]
        pdiff = pts_h - pts_a
        pos_diff = ap - hp  # positivt om hemma har bättre placering
        comment = _short_comment(home_i, away_i)

        form_h = ""
        form_b = ""
        if hist is not None:
            form_h = _form_last_n(home_i["display"], code, hist)
            form_b = _form_last_n(away_i["display"], code, hist)

        rows_out.append(
            {
                "Ctx_liga": code,
                "Ctx_match_conf": round(conf, 2),
                "Ctx_h_pos": hp,
                "Ctx_b_pos": ap,
                "Ctx_poängdiff": pdiff,
                "Ctx_plac_diff": pos_diff,
                "Ctx_form_H": form_h or "-",
                "Ctx_form_B": form_b or "-",
                "Ctx_tabell": comment,
            }
        )

    extra = pd.DataFrame(rows_out, index=coupon_df.index)
    msg = (
        f"Ligatabell-kontext: {matched}/{len(coupon_df)} matcher kopplade till free-tier ligor "
        f"({', '.join(FREE_COMPETITION_CODES)}). Övriga matcher finns inte i samma gratis urval."
    )
    return extra, msg
