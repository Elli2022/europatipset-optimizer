import argparse
import json
import math
import os
import re
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from coupon_enrichment import add_streck_volatility_column, enrich_coupon_with_history_form
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression


LEAGUE_CODES = [
    "E0",   # Premier League
    "E1",   # Championship
    "SP1",  # La Liga
    "D1",   # Bundesliga
    "I1",   # Serie A
    "F1",   # Ligue 1
    "N1",   # Eredivisie
    "P1",   # Primeira Liga
    "B1",   # Belgian Pro League
    "T1",   # Super Lig
]

load_dotenv()

COMPETITION_CODES = [
    "PL",   # Premier League
    "ELC",  # Championship
    "PD",   # La Liga
    "BL1",  # Bundesliga
    "SA",   # Serie A
    "FL1",  # Ligue 1
    "DED",  # Eredivisie
    "PPL",  # Primeira Liga
]

GAME_TYPES = {
    "europatipset": {"match_count": 13, "row_price": 1.0},
    "topptipset": {"match_count": 8, "row_price": 1.0},
}


def default_signal_backtest_profiles() -> List[Dict[str, object]]:
    """
    Ablationprofiler för backtest: jämför ROI mot rå träffstatistik per «signal»-gren
    (så man inte optimerar bara 13-rätt i sim utan avkastning).
    """
    return [
        {
            "SignalProfile": "allt_på",
            "enrich_history_form": True,
            "use_form_probability_adjustment": True,
            "use_volatility_probability_adjustment": True,
            "include_favourites_in_blend": True,
            "include_tio_in_blend": True,
        },
        {
            "SignalProfile": "utan_form_sannolikhetsjustering",
            "enrich_history_form": True,
            "use_form_probability_adjustment": False,
            "use_volatility_probability_adjustment": True,
            "include_favourites_in_blend": True,
            "include_tio_in_blend": True,
        },
        {
            "SignalProfile": "utan_vol_sannolikhetsjustering",
            "enrich_history_form": True,
            "use_form_probability_adjustment": True,
            "use_volatility_probability_adjustment": False,
            "include_favourites_in_blend": True,
            "include_tio_in_blend": True,
        },
        {
            "SignalProfile": "utan_form_och_vol_i_sannolikheter",
            "enrich_history_form": True,
            "use_form_probability_adjustment": False,
            "use_volatility_probability_adjustment": False,
            "include_favourites_in_blend": True,
            "include_tio_in_blend": True,
        },
        {
            "SignalProfile": "utan_historikform_kolumner",
            "enrich_history_form": False,
            "use_form_probability_adjustment": True,
            "use_volatility_probability_adjustment": True,
            "include_favourites_in_blend": True,
            "include_tio_in_blend": True,
        },
        {
            "SignalProfile": "utan_fav_i_blend",
            "enrich_history_form": True,
            "use_form_probability_adjustment": True,
            "use_volatility_probability_adjustment": True,
            "include_favourites_in_blend": False,
            "include_tio_in_blend": True,
        },
        {
            "SignalProfile": "utan_tio_i_blend",
            "enrich_history_form": True,
            "use_form_probability_adjustment": True,
            "use_volatility_probability_adjustment": True,
            "include_favourites_in_blend": True,
            "include_tio_in_blend": False,
        },
    ]


def season_codes(back_years: int = 6) -> List[str]:
    # Example season code for 2025/26 is "2526"
    now = pd.Timestamp.now("UTC")
    start_year = now.year if now.month >= 7 else now.year - 1
    return [f"{(start_year - i) % 100:02d}{(start_year - i + 1) % 100:02d}" for i in range(back_years)]


def download_historical_data(out_file: Path, back_years: int = 6) -> pd.DataFrame:
    rows = []
    for season in season_codes(back_years):
        for league in LEAGUE_CODES:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            try:
                response = requests.get(url, timeout=20)
                if response.status_code != 200:
                    continue
                df = pd.read_csv(pd.io.common.StringIO(response.text))
                if "FTR" not in df.columns:
                    continue
                df["Season"] = season
                df["League"] = league
                rows.append(df)
            except Exception:
                continue

    if not rows:
        raise RuntimeError("Ingen historik kunde laddas ner från football-data.co.uk.")

    full = pd.concat(rows, ignore_index=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_file, index=False)
    return full


def sync_history_from_free_api(
    out_file: Path,
    days_back: int = 120,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Pull finished matches from football-data.org (free tier).
    Stores a simple historical dataset for continuous updates.
    """
    key = api_key or os.getenv("FOOTBALL_DATA_API_KEY")
    if not key:
        raise RuntimeError("Sätt FOOTBALL_DATA_API_KEY för att hämta historik från API.")

    base_url = "https://api.football-data.org/v4"
    headers = {"X-Auth-Token": key}
    date_from = (pd.Timestamp.now("UTC") - pd.Timedelta(days=days_back)).date().isoformat()
    date_to = pd.Timestamp.now("UTC").date().isoformat()

    rows: List[Dict] = []
    for code in COMPETITION_CODES:
        url = f"{base_url}/competitions/{code}/matches"
        params = {"status": "FINISHED", "dateFrom": date_from, "dateTo": date_to}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                continue
            payload = response.json()
            matches = payload.get("matches", [])
            for m in matches:
                home = (m.get("homeTeam") or {}).get("name")
                away = (m.get("awayTeam") or {}).get("name")
                score = (m.get("score") or {}).get("fullTime") or {}
                hg = score.get("home")
                ag = score.get("away")
                if home is None or away is None or hg is None or ag is None:
                    continue
                if hg > ag:
                    ftr = "H"
                elif hg == ag:
                    ftr = "D"
                else:
                    ftr = "A"
                rows.append(
                    {
                        "Date": m.get("utcDate", "")[:10],
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "FTHG": hg,
                        "FTAG": ag,
                        "FTR": ftr,
                        "Competition": code,
                        "Source": "football-data.org",
                    }
                )
        except Exception:
            continue

    if not rows:
        raise RuntimeError("Ingen historik hämtades från football-data.org.")

    new_df = pd.DataFrame(rows).drop_duplicates()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        old_df = pd.read_csv(out_file, low_memory=False)
        merged = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
            subset=["Date", "HomeTeam", "AwayTeam", "Competition"]
        )
    else:
        merged = new_df
    merged = merged.sort_values(["Date", "Competition", "HomeTeam"], ascending=True)
    merged.to_csv(out_file, index=False)
    return merged


def download_upcoming_coupon(out_file: Path, n_matches: int = 13) -> pd.DataFrame:
    season = season_codes(1)[0]
    rows = []
    for league in LEAGUE_CODES:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
        try:
            response = requests.get(url, timeout=20)
            if response.status_code != 200:
                continue
            df = pd.read_csv(pd.io.common.StringIO(response.text))
            if "FTR" not in df.columns:
                continue
            df["League"] = league
            rows.append(df)
        except Exception:
            continue

    if not rows:
        raise RuntimeError("Kunde inte hämta kommande matcher.")

    data = pd.concat(rows, ignore_index=True)
    odds = pd.DataFrame(
        {
            "odd1": data.apply(lambda r: _pick_odds(r, ["B365H", "PSH", "WHH", "VCH", "AvgH", "MaxH"]), axis=1),
            "oddx": data.apply(lambda r: _pick_odds(r, ["B365D", "PSD", "WHD", "VCD", "AvgD", "MaxD"]), axis=1),
            "odd2": data.apply(lambda r: _pick_odds(r, ["B365A", "PSA", "WHA", "VCA", "AvgA", "MaxA"]), axis=1),
        }
    )
    data = pd.concat([data, odds], axis=1)

    upcoming = data[data["FTR"].isna()].copy()
    if "Date" in upcoming.columns:
        upcoming["DateParsed"] = pd.to_datetime(upcoming["Date"], errors="coerce", dayfirst=True)
        upcoming = upcoming.sort_values(["DateParsed"], ascending=True)

    upcoming = upcoming.dropna(subset=["odd1", "oddx", "odd2", "HomeTeam", "AwayTeam"])
    if len(upcoming) < n_matches:
        # Fallback in offseason periods: use latest available played matches with odds.
        fallback = data.dropna(subset=["odd1", "oddx", "odd2", "HomeTeam", "AwayTeam"]).copy()
        if "Date" in fallback.columns:
            fallback["DateParsed"] = pd.to_datetime(fallback["Date"], errors="coerce", dayfirst=True)
            fallback = fallback.sort_values(["DateParsed"], ascending=False)
        upcoming = fallback.head(n_matches).copy()
        if len(upcoming) < n_matches:
            raise RuntimeError(f"Hittade bara {len(upcoming)} matcher med odds.")
    else:
        upcoming = upcoming.head(n_matches).copy()

    out = pd.DataFrame(
        {
            "Match": upcoming["HomeTeam"].astype(str) + " - " + upcoming["AwayTeam"].astype(str),
            "Odd1": upcoming["odd1"],
            "OddX": upcoming["oddx"],
            "Odd2": upcoming["odd2"],
            "Streck1": 100 / 3,
            "StreckX": 100 / 3,
            "Streck2": 100 / 3,
        }
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)
    return out


def _extract_json_assignment(html: str, variable_name: str) -> dict:
    marker = f"{variable_name}="
    start = html.find(marker)
    if start == -1:
        raise RuntimeError(f"Kunde inte hitta {variable_name} i HTML.")
    i = start + len(marker)
    while i < len(html) and html[i].isspace():
        i += 1
    if i >= len(html) or html[i] != "{":
        raise RuntimeError(f"Ogiltig JSON-start för {variable_name}.")

    depth = 0
    in_string = False
    escaped = False
    j = i
    while j < len(html):
        ch = html[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_text = html[i : j + 1]
                    return json.loads(json_text)
        j += 1
    raise RuntimeError(f"Kunde inte extrahera JSON för {variable_name}.")


def _odd_mv_triplet(curr: List[object], start: List[object], _to_float) -> Tuple[float, float, float]:
    """Relativ odds-rörelse (curr-start)/start per utfall."""
    out: List[float] = []
    for i in range(3):
        try:
            c = _to_float(curr[i], np.nan)
            s = _to_float(start[i], np.nan)
            if np.isnan(c) or np.isnan(s) or s < 1.01:
                out.append(float("nan"))
            else:
                out.append(float((c - s) / s))
        except Exception:
            out.append(float("nan"))
    return float(out[0]), float(out[1]), float(out[2])


def _streck_mv_triplet(curr_pct: List[object], prev_pct: List[object], _to_float) -> Tuple[float, float, float]:
    """Streck-rörelse i andelsenheter (t.ex. 0.03 = +3 procentenheter)."""
    out: List[float] = []
    for i in range(3):
        try:
            c = _to_float(curr_pct[i], np.nan)
            p = _to_float(prev_pct[i], np.nan)
            if np.isnan(c) or np.isnan(p):
                out.append(float("nan"))
            else:
                out.append(float((c - p) / 100.0))
        except Exception:
            out.append(float("nan"))
    return float(out[0]), float(out[1]), float(out[2])


def fetch_official_coupon_state() -> Tuple[pd.DataFrame, Dict[str, str]]:
    url = "https://spela.svenskaspel.se/europatipset/statistik"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    state = _extract_json_assignment(response.text, "_svs.tipsen.data.preloadedState")

    draw_ids = state.get("Draws", {}).get("ids", [])
    if not draw_ids:
        raise RuntimeError("Ingen aktiv Europatipset-omgång hittades.")

    draw_id = draw_ids[0]
    draw = state["Draws"]["entities"][draw_id]
    draw_number = str(draw["drawNumber"])
    events = sorted(draw.get("drawEvents", []), key=lambda e: e.get("eventNumber", 0))
    event_stats = state.get("EventTypeStatistic", {})

    rows = []
    for ev in events[:13]:
        event_number = ev.get("eventNumber")
        match_id = ev.get("matchId")
        match = ev.get("eventDescription", "")
        stat_key = f"{match_id}_1_{event_number}"
        stat = event_stats.get(stat_key, {})

        odds_values = (
            stat.get("odds", {})
            .get("current", {})
            .get("value", [None, None, None])
        )
        if len(odds_values) != 3:
            odds_values = [None, None, None]
        if any(v in [None, ""] for v in odds_values):
            # After/near kickoff, current odds may be missing; fallback to opening odds.
            start_odds = (
                stat.get("startOdds", {})
                .get("current", {})
                .get("value", [None, None, None])
            )
            if len(start_odds) == 3 and not any(v in [None, ""] for v in start_odds):
                odds_values = start_odds

        distributions = stat.get("distributions", {})
        dist_draw = distributions.get("2", {}).get(draw_number, {})
        dist_current = dist_draw.get("current", {}).get("value", [None, None, None])
        if len(dist_current) != 3:
            dist_current = (
                distributions.get("2", {})
                .get("Global", {})
                .get("current", {})
                .get("value", [None, None, None])
            )
        if len(dist_current) != 3:
            dist_current = [33.33, 33.33, 33.33]

        def _to_float(v, default):
            try:
                return float(str(v).replace(",", "."))
            except Exception:
                return default

        def _triplet_probs_percent(stat_block: Dict, key: str) -> Tuple[float, float, float]:
            """Läser tre procent-strängar (0–100) och normaliserar till sannolikheter."""
            vals = (
                stat_block.get(key, {})
                .get("current", {})
                .get("value", [None, None, None])
            )
            if len(vals) != 3 or any(v in [None, ""] for v in vals):
                return (np.nan, np.nan, np.nan)
            a = _to_float(vals[0], np.nan)
            b = _to_float(vals[1], np.nan)
            c = _to_float(vals[2], np.nan)
            if any(np.isnan([a, b, c])):
                return (np.nan, np.nan, np.nan)
            tot = a + b + c
            if tot <= 0:
                return (np.nan, np.nan, np.nan)
            return float(a / tot), float(b / tot), float(c / tot)

        def _triplet_probs_counts(stat_block: Dict, key: str) -> Tuple[float, float, float]:
            """Tolkar tre heltalsröster (t.ex. tidningstips) som frekvenser."""
            vals = (
                stat_block.get(key, {})
                .get("current", {})
                .get("value", [None, None, None])
            )
            if len(vals) != 3 or any(v is None for v in vals):
                return (np.nan, np.nan, np.nan)
            try:
                ia, ib, ic = int(vals[0]), int(vals[1]), int(vals[2])
            except Exception:
                return (np.nan, np.nan, np.nan)
            tot = ia + ib + ic
            if tot <= 0:
                return (np.nan, np.nan, np.nan)
            return ia / tot, ib / tot, ic / tot

        fav1, favx, fav2 = _triplet_probs_percent(stat, "favourites")
        tio1, tiox, tio2 = _triplet_probs_counts(stat, "tioTidningarTips")

        start_odds_vals = stat.get("startOdds", {}).get("current", {}).get("value", [None, None, None])
        if len(start_odds_vals) != 3:
            start_odds_vals = [None, None, None]

        odd_mv1, odd_mv_x, odd_mv_2 = _odd_mv_triplet(list(odds_values), list(start_odds_vals), _to_float)

        dist_prev = dist_draw.get("previous", {}).get("value", [None, None, None])
        if len(dist_prev) != 3:
            dist_prev = (
                distributions.get("2", {})
                .get("Global", {})
                .get("previous", {})
                .get("value", [None, None, None])
            )
        if len(dist_prev) != 3:
            dist_prev = [None, None, None]
        streck_mv1, streck_mv_x, streck_mv_2 = _streck_mv_triplet(dist_current, dist_prev, _to_float)

        rows.append(
            {
                "Match": match,
                "Odd1": _to_float(odds_values[0], np.nan),
                "OddX": _to_float(odds_values[1], np.nan),
                "Odd2": _to_float(odds_values[2], np.nan),
                "Streck1": _to_float(dist_current[0], 33.33),
                "StreckX": _to_float(dist_current[1], 33.33),
                "Streck2": _to_float(dist_current[2], 33.33),
                "Fav1": fav1,
                "FavX": favx,
                "Fav2": fav2,
                "Tio1": tio1,
                "TioX": tiox,
                "Tio2": tio2,
                "OddMv1": odd_mv1,
                "OddMvX": odd_mv_x,
                "OddMv2": odd_mv_2,
                "StreckMv1": streck_mv1,
                "StreckMvX": streck_mv_x,
                "StreckMv2": streck_mv_2,
            }
        )

    out = pd.DataFrame(rows).dropna(subset=["Odd1", "OddX", "Odd2"]).head(13)
    if len(out) < 13:
        raise RuntimeError(f"Hittade bara {len(out)} matcher med odds i officiell kupong.")

    meta = {
        "draw_number": str(draw.get("drawNumber", "")),
        "draw_comment": str(draw.get("drawComment", "")),
        "reg_close_time": str(draw.get("regCloseTime", "")),
        "reg_close_description": str(draw.get("regCloseDescription", "")),
        "current_net_sale": str(draw.get("currentNetSale", "")),
        "row_price": str(draw.get("rowPrice", "")),
    }
    return out, meta


def _fetch_tipsen_preloaded_state(url: str) -> Dict:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return _extract_json_assignment(response.text, "_svs.tipsen.data.preloadedState")


def outcome_to_sign(outcomes: object) -> Optional[str]:
    """
    Tolka Svenska Spels tips-data `outcomes`-fält till 1/X/2 när möjligt.
    Strukturen varierar — hantera vanliga varianter defensivt.
    """
    if outcomes is None:
        return None
    if isinstance(outcomes, str):
        t = outcomes.strip().upper().replace("TIE", "X")
        return t if t in {"1", "X", "2"} else None
    if isinstance(outcomes, dict):
        for key in ("sign", "label", "type", "outcomeSign"):
            s = outcome_to_sign(outcomes.get(key))
            if s:
                return s
        # hem/oavgjort/borta-flaggor
        if outcomes.get("homeWin") is True or outcomes.get("one") is True:
            return "1"
        if outcomes.get("draw") is True or outcomes.get("tie") is True or outcomes.get("x") is True:
            return "X"
        if outcomes.get("awayWin") is True or outcomes.get("two") is True:
            return "2"
        return None
    if isinstance(outcomes, list):
        chosen = None
        for item in outcomes:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).upper()
            won = item.get("won")
            if won is True or status in {"WON", "WINNING"}:
                chosen = item
                break
        if chosen is None:
            return None
        if chosen.get("outcome"):
            s = outcome_to_sign(chosen.get("outcome"))
            if s:
                return s
        return outcome_to_sign(chosen.get("label")) or outcome_to_sign(chosen.get("sign"))
    return None


def score_text_to_sign(text: Optional[str]) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
    if not m:
        return None
    home_goals, away_goals = int(m.group(1)), int(m.group(2))
    if home_goals > away_goals:
        return "1"
    if home_goals < away_goals:
        return "2"
    return "X"


def extract_sign_from_draw_event(ev: Dict) -> Optional[str]:
    s = outcome_to_sign(ev.get("outcomes"))
    if s:
        return s
    return score_text_to_sign(ev.get("extraInfo")) or score_text_to_sign(ev.get("eventComment"))


def extract_correct_row_from_draw(draw: Dict) -> Optional[str]:
    events = sorted(draw.get("drawEvents") or [], key=lambda e: int(e.get("eventNumber", 0)))
    if len(events) < 13:
        return None
    sigs: List[str] = []
    for ev in events[:13]:
        if ev.get("cancelled"):
            return None
        sign = extract_sign_from_draw_event(ev)
        if sign is None:
            return None
        sigs.append(sign)
    return "".join(sigs)


def merged_draw_entities_from_tipsen_urls(urls: List[str]) -> Tuple[Dict[str, Dict], List[str]]:
    merged: Dict[str, Dict] = {}
    errors: List[str] = []
    for url in urls:
        try:
            state = _fetch_tipsen_preloaded_state(url)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        if not isinstance(state, dict):
            continue
        ents = state.get("Draws", {}).get("entities", {})
        if isinstance(ents, dict):
            for draw_key, draw_val in ents.items():
                if isinstance(draw_val, dict):
                    merged[str(draw_key)] = draw_val
    return merged, errors


def fetch_correct_row_for_draw_number(draw_number: str) -> Tuple[Optional[str], str]:
    """
    Försök läsa en komplett 13-teckensrad för given omgång ur publika tips-sidor (ingen inloggning).
    Fungerar först när Svenska Spel bundit utfall till drawEvents (ofta efter avslutade matcher).
    """
    urls = [
        "https://www.svenskaspel.se/europatipset/resultat",
        "https://spela.svenskaspel.se/europatipset/statistik",
    ]
    merged, errs = merged_draw_entities_from_tipsen_urls(urls)
    if not merged:
        hint = " | ".join(errs) if errs else "okänt fel"
        return None, f"Kunde inte läsa tips-data från Svenska Spel ({hint})."

    target = str(draw_number).strip()
    candidates: List[Dict] = []
    for draw in merged.values():
        if str(draw.get("drawNumber", "")).strip() == target:
            candidates.append(draw)

    if not candidates:
        return (
            None,
            "Hittade ingen omgång med detta draw-nummer i den publika datan just nu.",
        )

    for draw in candidates:
        row = extract_correct_row_from_draw(draw)
        if row:
            return row, "Läste referensrad från Svenska Spels publika tips-data (utan skador/elvor)."

    return (
        None,
        "Omgången finns men saknar kompletta utfall i publik JSON än — prova igen efter matcher eller ange rad manuellt.",
    )


def auto_refresh_official_snapshot(
    out_coupon_csv: Path,
    out_meta_json: Path,
    hours_before_close: int = 2,
    min_interval_minutes: int = 15,
) -> Tuple[bool, str]:
    """
    Refresh official coupon/meta only when we are close to stop time.
    Returns (did_refresh, message).
    """
    coupon_df, meta = fetch_official_coupon_state()
    close_raw = meta.get("reg_close_time", "")
    if not close_raw:
        return False, "Kunde inte avgöra spelstoppstid."

    close_dt = pd.to_datetime(close_raw, utc=True, errors="coerce")
    if pd.isna(close_dt):
        return False, "Ogiltig spelstoppstid."

    now_utc = pd.Timestamp.now("UTC")
    hours_left = (close_dt - now_utc).total_seconds() / 3600.0
    if hours_left > float(hours_before_close):
        return False, f"För tidigt för refresh ({hours_left:.2f}h kvar till spelstopp)."

    if out_meta_json.exists():
        try:
            old = json.loads(out_meta_json.read_text(encoding="utf-8"))
            last_refresh = pd.to_datetime(old.get("last_refresh_utc"), utc=True, errors="coerce")
            if pd.notna(last_refresh):
                mins_since = (now_utc - last_refresh).total_seconds() / 60.0
                if mins_since < float(min_interval_minutes):
                    return False, f"Refresh nyligen gjord ({mins_since:.1f} min sedan)."
        except Exception:
            pass

    out_coupon_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    coupon_df.to_csv(out_coupon_csv, index=False)
    meta_out = {
        **meta,
        "last_refresh_utc": now_utc.isoformat(),
        "hours_before_close_window": hours_before_close,
        "min_interval_minutes": min_interval_minutes,
    }
    out_meta_json.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return True, "Refresh genomförd."


def sync_official_snapshot_smart(
    out_coupon_csv: Path,
    out_meta_json: Path,
    min_interval_minutes: float = 10.0,
) -> Tuple[bool, str]:
    """
    Synka officiell kupong/meta till disk utan att fastna på gamla omgångar.

    Uppdaterar alltid direkt om draw_number ändrats (ny öppen omgång).
    Annars throttle för att inte överbelasta Svenska Spel.
    """
    coupon_df, meta = fetch_official_coupon_state()
    now_utc = pd.Timestamp.now("UTC")
    old_draw = None
    last_refresh = None
    if out_meta_json.exists():
        try:
            old = json.loads(out_meta_json.read_text(encoding="utf-8"))
            old_draw = str(old.get("draw_number", "") or "")
            last_refresh = pd.to_datetime(old.get("last_refresh_utc"), utc=True, errors="coerce")
        except Exception:
            pass

    new_draw = str(meta.get("draw_number", "") or "")
    draw_changed = bool(old_draw) and old_draw != new_draw

    mins_since = float("inf")
    if pd.notna(last_refresh):
        mins_since = (now_utc - last_refresh).total_seconds() / 60.0

    first_run = not old_draw
    should_refresh = first_run or draw_changed or mins_since >= float(min_interval_minutes)

    if not should_refresh:
        return False, f"Synk nyligen ({mins_since:.1f} min sedan)."

    out_coupon_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    coupon_df.to_csv(out_coupon_csv, index=False)
    meta_out = {**meta, "last_refresh_utc": now_utc.isoformat()}
    out_meta_json.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    msg = "Ny omgång hämtad." if draw_changed else "Omgång uppdaterad."
    return True, msg


def download_official_coupon_from_svenskaspel(out_file: Path) -> pd.DataFrame:
    out, _ = fetch_official_coupon_state()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)
    return out


def _pick_odds(row: pd.Series, keys: List[str]) -> float:
    for key in keys:
        val = row.get(key)
        if pd.notna(val):
            try:
                odds = float(val)
                if odds > 1.01:
                    return odds
            except Exception:
                pass
    return np.nan


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["odd1"] = df.apply(lambda r: _pick_odds(r, ["B365H", "PSH", "WHH", "VCH", "AvgH", "MaxH"]), axis=1)
    out["oddx"] = df.apply(lambda r: _pick_odds(r, ["B365D", "PSD", "WHD", "VCD", "AvgD", "MaxD"]), axis=1)
    out["odd2"] = df.apply(lambda r: _pick_odds(r, ["B365A", "PSA", "WHA", "VCA", "AvgA", "MaxA"]), axis=1)
    out["ftr"] = df.get("FTR")
    out = out.dropna()
    out = out[out["ftr"].isin(["H", "D", "A"])].copy()

    inv = 1 / out[["odd1", "oddx", "odd2"]].to_numpy()
    probs = inv / inv.sum(axis=1, keepdims=True)
    out["p1_raw"] = probs[:, 0]
    out["px_raw"] = probs[:, 1]
    out["p2_raw"] = probs[:, 2]

    # Log-probabilities work well for calibration of market probabilities.
    out["x1"] = np.log(np.clip(out["p1_raw"], 1e-6, 1))
    out["xx"] = np.log(np.clip(out["px_raw"], 1e-6, 1))
    out["x2"] = np.log(np.clip(out["p2_raw"], 1e-6, 1))
    out["y"] = out["ftr"].map({"H": 0, "D": 1, "A": 2})

    if "HomeTeam" in df.columns and "AwayTeam" in df.columns:
        out["HomeTeam"] = df.loc[out.index, "HomeTeam"].astype(str).values
        out["AwayTeam"] = df.loc[out.index, "AwayTeam"].astype(str).values
    return out


def train_model(history_csv: Path, model_file: Path) -> None:
    df = pd.read_csv(history_csv, low_memory=False)
    train = prepare_training_frame(df)
    X = train[["x1", "xx", "x2"]].to_numpy()
    y = train["y"].to_numpy()

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)

    model_file.parent.mkdir(parents=True, exist_ok=True)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)


@dataclass
class MatchSuggestion:
    match: str
    odd1: float
    oddx: float
    odd2: float
    p1: float
    px: float
    p2: float
    streck1: float
    streckx: float
    streck2: float
    selection: str


def validate_coupon_data(df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []
    required = {"Match", "Odd1", "OddX", "Odd2"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"Saknar kolumner: {', '.join(sorted(missing))}")
        return warnings

    odd_cols = ["Odd1", "OddX", "Odd2"]
    for col in odd_cols:
        if (pd.to_numeric(df[col], errors="coerce") <= 1.01).any():
            warnings.append(f"Upptäckte orimliga odds i {col} (<= 1.01).")
        if pd.to_numeric(df[col], errors="coerce").isna().any():
            warnings.append(f"Upptäckte saknade/ogiltiga odds i {col}.")

    streck_cols = ["Streck1", "StreckX", "Streck2"]
    if all(c in df.columns for c in streck_cols):
        streck = df[streck_cols].copy()
        for c in streck_cols:
            if streck[c].max() > 1.5:
                streck[c] = streck[c] / 100.0
        sums = streck.sum(axis=1)
        if ((sums < 0.97) | (sums > 1.03)).any():
            warnings.append("Streck 1/X/2 summerar inte nära 100% på vissa matcher.")
    else:
        warnings.append("Streckkolumner saknas; default 33.33% används.")

    if df["Match"].duplicated().any():
        warnings.append("Duplicerade matcher upptäckta i kupongen.")

    return warnings


def parse_coupon(coupon_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(coupon_csv)
    required = {"Match", "Odd1", "OddX", "Odd2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Saknade kolumner i kupongfil: {', '.join(sorted(missing))}")

    for col in ["Streck1", "StreckX", "Streck2"]:
        if col not in df.columns:
            df[col] = 100 / 3

    # Handle both 45 and 0.45 input formats.
    for col in ["Streck1", "StreckX", "Streck2"]:
        if df[col].max() > 1.5:
            df[col] = df[col] / 100.0

    for col in ["Fav1", "FavX", "Fav2", "Tio1", "TioX", "Tio2"]:
        if col not in df.columns:
            df[col] = np.nan

    for col in [
        "OddMv1",
        "OddMvX",
        "OddMv2",
        "StreckMv1",
        "StreckMvX",
        "StreckMv2",
        "FormH_pts5",
        "FormB_pts5",
        "StreckVol",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def enforce_game_type(df: pd.DataFrame, game_type: str = "europatipset") -> pd.DataFrame:
    cfg = GAME_TYPES.get(game_type)
    if not cfg:
        raise ValueError(f"Okänd omgångstyp: {game_type}")
    needed = int(cfg["match_count"])
    if len(df) < needed:
        raise ValueError(f"{game_type} kräver {needed} matcher, men kupongen har {len(df)}.")
    return df.head(needed).copy()


def calibrated_probs(model, odd1: float, oddx: float, odd2: float) -> Tuple[float, float, float]:
    inv = np.array([1 / odd1, 1 / oddx, 1 / odd2], dtype=float)
    p_raw = inv / inv.sum()
    x = np.log(np.clip(p_raw, 1e-6, 1)).reshape(1, -1)
    p = model.predict_proba(x)[0]
    return float(p[0]), float(p[1]), float(p[2])


def blend_probability_views(
    p_cal: Tuple[float, float, float],
    streck: Tuple[float, float, float],
    fav: Optional[Tuple[float, float, float]] = None,
    tio: Optional[Tuple[float, float, float]] = None,
    *,
    w_cal: float = 0.62,
    w_streck: float = 0.23,
    w_fav: float = 0.10,
    w_tio: float = 0.05,
    include_favourites: bool = True,
    include_tio: bool = True,
) -> Tuple[float, float, float]:
    """
    Kombinerar kalibrerad odds-modell med streck, Svenska Spels «favourites»-fördelning
    och tidnings-röster (tioTidningarTips) när de finns i kupongens JSON/spreadsheet.

    Detta ersätter inte riktiga skador/elvor — bara extra signaler ur samma publika tips-flöde.
    """
    pc = np.clip(np.array(p_cal, dtype=float), 1e-9, 1.0)
    pc = pc / pc.sum()
    st = np.clip(np.array(streck, dtype=float), 1e-9, 1.0)
    st = st / st.sum()

    wf_cal = float(w_cal)
    wf_st = float(w_streck)
    wf_fav = float(w_fav)
    wf_tio = float(w_tio)

    use_fav = (
        include_favourites
        and fav is not None
        and len(fav) == 3
        and all(not (isinstance(x, float) and np.isnan(x)) for x in fav)
    )
    use_tio_blend = (
        include_tio
        and tio is not None
        and len(tio) == 3
        and all(not (isinstance(x, float) and np.isnan(x)) for x in tio)
    )
    if not use_fav:
        wf_cal += wf_fav
        wf_fav = 0.0
    if not use_tio_blend:
        wf_cal += wf_tio
        wf_tio = 0.0

    tot_w = wf_cal + wf_st + wf_fav + wf_tio
    if tot_w <= 0:
        return float(pc[0]), float(pc[1]), float(pc[2])

    mix = (wf_cal / tot_w) * pc + (wf_st / tot_w) * st
    if wf_fav > 0 and fav is not None:
        fv = np.clip(np.array(fav, dtype=float), 1e-9, 1.0)
        fv = fv / fv.sum()
        mix += (wf_fav / tot_w) * fv
    if wf_tio > 0 and tio is not None and use_tio_blend:
        tv = np.clip(np.array(tio, dtype=float), 1e-9, 1.0)
        tv = tv / tv.sum()
        mix += (wf_tio / tot_w) * tv

    mix = np.clip(mix, 1e-9, 1.0)
    mix = mix / mix.sum()
    return float(mix[0]), float(mix[1]), float(mix[2])


def adjust_probs_form_and_volatility(
    p1: float,
    px: float,
    p2: float,
    form_h_pts: float,
    form_b_pts: float,
    streck_vol: float,
    *,
    enable_form: bool = True,
    enable_vol: bool = True,
    form_strength: float = 0.038,
    form_scale_pts: float = 12.0,
    vol_gamma: float = 2.6,
    vol_cap: float = 0.22,
) -> Tuple[float, float, float]:
    """
    Mild efterjustering efter blend: flytta lite massa 1↔2 utifrån formdiff,
    platta ut mot 1/3 vid hög streck-volatilitet (osäker marknad).
    """
    p = np.clip(np.array([p1, px, p2], dtype=float), 1e-9, 1.0)
    p = p / p.sum()

    if enable_form and not (np.isnan(form_h_pts) or np.isnan(form_b_pts)):
        diff = float(form_h_pts) - float(form_b_pts)
        nd = float(np.clip(diff / max(form_scale_pts, 1e-6), -1.0, 1.0))
        adj = np.array([form_strength * nd, 0.0, -form_strength * nd], dtype=float)
        p = p + adj
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum()

    if enable_vol and not np.isnan(streck_vol) and streck_vol >= 0:
        lam = min(vol_cap, vol_gamma * float(streck_vol))
        u = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
        p = (1.0 - lam) * p + lam * u
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum()

    return float(p[0]), float(p[1]), float(p[2])


def optimize_system(
    matches: List[MatchSuggestion],
    max_rows: int,
    strategy: str = "balanced",
) -> Tuple[List[str], int]:
    options = ["1", "X", "2"]
    probs = [np.array([m.p1, m.px, m.p2]) for m in matches]
    order = [list(np.argsort(-p)) for p in probs]

    # Start with one sign each.
    current = [[order[i][0]] for i in range(len(matches))]
    rows = 1

    def cover(idx: int, selected: List[int]) -> float:
        return float(probs[idx][selected].sum())

    def score(state: List[List[int]]) -> float:
        # Proxy for chance to have at least one winning row.
        base = float(sum(math.log(max(1e-9, cover(i, state[i]))) for i in range(len(state))))
        if strategy == "safe":
            # Bias slightly toward favorites (higher max sign probability per match).
            safe_bonus = float(
                sum(max(probs[i][state[i]]) for i in range(len(state))) / max(1, len(state))
            )
            return base + 0.90 * safe_bonus
        if strategy == "value":
            # Bias slightly toward wider coverage in uncertain matches.
            width_bonus = float(sum(len(state[i]) for i in range(len(state))) / max(1, len(state)))
            entropy_bonus = float(
                sum((-np.sum(probs[i] * np.log(np.clip(probs[i], 1e-9, 1.0)))) * len(state[i]) for i in range(len(state)))
                / max(1, len(state))
            )
            return base + 0.28 * width_bonus + 0.12 * entropy_bonus
        return base

    while True:
        best_delta = None
        best_new_state = None
        best_new_rows = None

        current_score = score(current)
        for i in range(len(matches)):
            selected = current[i]
            if len(selected) >= 3:
                continue
            if len(selected) == 1:
                candidate = selected + [order[i][1]]
            else:
                candidate = [order[i][0], order[i][1], order[i][2]]

            new_rows = rows // len(selected) * len(candidate)
            if new_rows > max_rows:
                continue

            new_state = [s[:] for s in current]
            new_state[i] = candidate
            delta = score(new_state) - current_score
            cost = new_rows - rows
            value = delta / max(1, cost)

            if best_delta is None or value > best_delta:
                best_delta = value
                best_new_state = new_state
                best_new_rows = new_rows

        if best_new_state is None:
            break
        current = best_new_state
        rows = best_new_rows

    picks = []
    for i, selected in enumerate(current):
        signs = "".join(options[j] for j in sorted(selected))
        picks.append(signs)
    return picks, rows


def expand_system_rows(picks: List[str], max_expand_rows: int = 10000) -> np.ndarray:
    sign_to_int = {"1": 0, "X": 1, "2": 2}
    rows: List[List[int]] = [[]]
    for p in picks:
        options = [sign_to_int[s] for s in p]
        rows = [r + [o] for r in rows for o in options]
        if len(rows) > max_expand_rows:
            rows = rows[:max_expand_rows]
            break
    return np.array(rows, dtype=np.int8)


def simulate_rights_distribution(
    result_df: pd.DataFrame,
    n_sim: int = 20000,
    seed: int = 42,
) -> Dict:
    picks = result_df["Förslag"].tolist()
    row_matrix = expand_system_rows(picks)
    if len(row_matrix) == 0:
        return {}

    probs = result_df[["P1", "PX", "P2"]].to_numpy(dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    n_matches = probs.shape[0]
    rng = np.random.default_rng(seed)

    outcomes = np.zeros((n_sim, n_matches), dtype=np.int8)
    for i in range(n_matches):
        outcomes[:, i] = rng.choice([0, 1, 2], size=n_sim, p=probs[i])

    max_hits = np.zeros(n_sim, dtype=np.int16)
    for i in range(n_sim):
        hits = (row_matrix == outcomes[i]).sum(axis=1)
        max_hits[i] = hits.max()

    dist = {k: float((max_hits == k).mean()) for k in sorted(set(max_hits.tolist()))}
    dist["most_likely"] = int(Counter(max_hits.tolist()).most_common(1)[0][0])
    return dist


def assess_forecast_confidence(result_df: pd.DataFrame) -> Dict[str, float | str]:
    probs = result_df[["P1", "PX", "P2"]].to_numpy(dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)
    max_entropy = math.log(3.0)
    normalized_uncertainty = float(np.mean(entropy / max_entropy))
    sorted_probs = np.sort(probs, axis=1)
    avg_top_gap = float(np.mean(sorted_probs[:, 2] - sorted_probs[:, 1]))

    # Entropy tends to be high in 1X2 markets; use a more practical calibration.
    if normalized_uncertainty < 0.88 and avg_top_gap > 0.18:
        label = "Hög"
    elif normalized_uncertainty < 0.96 and avg_top_gap > 0.10:
        label = "Medel"
    else:
        label = "Låg"
    return {
        "confidence_label": label,
        "uncertainty_score": normalized_uncertainty,
        "avg_top_gap": avg_top_gap,
    }


def recommend_max_stake(
    forecast_dist: Dict,
    payout_by_rights: Dict[int, float],
    margin_pct: float = 15.0,
    bankroll_cap: float = 0.0,
) -> Dict[str, float]:
    expected_gross = 0.0
    for k, payout in payout_by_rights.items():
        expected_gross += float(forecast_dist.get(k, 0.0)) * float(payout)
    max_break_even = max(0.0, expected_gross)
    max_conservative = max_break_even * (1 - margin_pct / 100.0)
    recommended = min(max_conservative, bankroll_cap) if bankroll_cap > 0 else max_conservative
    return {
        "expected_gross": expected_gross,
        "max_break_even": max_break_even,
        "max_conservative": max_conservative,
        "recommended_max": recommended,
    }


def suggest_system(
    coupon_csv: Path,
    model_file: Path,
    max_rows: int,
    out_csv: Path,
    strategy: str = "balanced",
    game_type: str = "europatipset",
    *,
    enrich_history_form: bool = True,
    use_form_probability_adjustment: bool = True,
    use_volatility_probability_adjustment: bool = True,
    include_favourites_in_blend: bool = True,
    include_tio_in_blend: bool = True,
) -> pd.DataFrame:
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    coupon = enforce_game_type(parse_coupon(coupon_csv), game_type=game_type)
    if enrich_history_form:
        coupon = enrich_coupon_with_history_form(coupon, coupon_csv)
    coupon = add_streck_volatility_column(coupon)
    matches: List[MatchSuggestion] = []

    for _, row in coupon.iterrows():
        p1, px, p2 = calibrated_probs(model, float(row["Odd1"]), float(row["OddX"]), float(row["Odd2"]))
        streck_trip = (float(row["Streck1"]), float(row["StreckX"]), float(row["Streck2"]))
        fav_trip: Optional[Tuple[float, float, float]] = None
        tio_trip: Optional[Tuple[float, float, float]] = None
        try:
            if all(pd.notna(row.get(c)) for c in ("Fav1", "FavX", "Fav2")):
                fav_trip = (float(row["Fav1"]), float(row["FavX"]), float(row["Fav2"]))
        except Exception:
            fav_trip = None
        try:
            if all(pd.notna(row.get(c)) for c in ("Tio1", "TioX", "Tio2")):
                tio_trip = (float(row["Tio1"]), float(row["TioX"]), float(row["Tio2"]))
        except Exception:
            tio_trip = None

        p1, px, p2 = blend_probability_views(
            (p1, px, p2),
            streck_trip,
            fav_trip,
            tio_trip,
            include_favourites=include_favourites_in_blend,
            include_tio=include_tio_in_blend,
        )
        try:
            fh = float(row["FormH_pts5"])
        except Exception:
            fh = float("nan")
        try:
            fb = float(row["FormB_pts5"])
        except Exception:
            fb = float("nan")
        try:
            sv = float(row["StreckVol"])
        except Exception:
            sv = float("nan")
        p1, px, p2 = adjust_probs_form_and_volatility(
            p1,
            px,
            p2,
            fh,
            fb,
            sv,
            enable_form=use_form_probability_adjustment,
            enable_vol=use_volatility_probability_adjustment,
        )
        m = MatchSuggestion(
            match=str(row["Match"]),
            odd1=float(row["Odd1"]),
            oddx=float(row["OddX"]),
            odd2=float(row["Odd2"]),
            p1=p1,
            px=px,
            p2=p2,
            streck1=float(row["Streck1"]),
            streckx=float(row["StreckX"]),
            streck2=float(row["Streck2"]),
            selection="",
        )
        matches.append(m)

    picks, rows = optimize_system(matches, max_rows=max_rows, strategy=strategy)

    out = coupon.copy()
    out["P1"] = [m.p1 for m in matches]
    out["PX"] = [m.px for m in matches]
    out["P2"] = [m.p2 for m in matches]
    out["Value1"] = out["P1"] - out["Streck1"]
    out["ValueX"] = out["PX"] - out["StreckX"]
    out["Value2"] = out["P2"] - out["Streck2"]
    out["Förslag"] = picks
    out["Systemrader"] = rows
    out["Strategi"] = strategy
    out["GameType"] = game_type

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def backtest_strategies(
    history_csv: Path,
    model_file: Path,
    budgets: List[int],
    strategies: List[str],
    game_type: str = "europatipset",
    n_coupons: int = 50,
    seed: int = 7,
    *,
    compare_signal_profiles: bool = False,
) -> pd.DataFrame:
    """
    Historiskt skiss-backtest på slumpade «kuponger» ur odds-historik.

    Sätt ``compare_signal_profiles=True`` för att köra samma sweep per signal-ablation
    (form-/vol-justering, historikberikning, fav/tio i blend).     Jämför då **ROI** mot **FullHitRate** (andel kuponger där bästa raden får alla matcher rätt) —
    högre «fullträff»-andel utan högre ROI är ofta brus / överanpassning.
    """
    if not history_csv.exists() or history_csv.stat().st_size == 0:
        raise RuntimeError("Historikfil saknas eller är tom för backtest.")
    df = pd.read_csv(history_csv, low_memory=False)
    train = prepare_training_frame(df)
    rng = np.random.default_rng(seed)
    match_count = GAME_TYPES[game_type]["match_count"]
    row_price = GAME_TYPES[game_type]["row_price"]

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    samples = train.sample(n=min(len(train), max(match_count * n_coupons, match_count)), random_state=seed).reset_index(drop=True)
    chunks = [samples.iloc[i : i + match_count] for i in range(0, len(samples), match_count)]
    chunks = [c for c in chunks if len(c) == match_count][:n_coupons]
    if not chunks:
        raise RuntimeError("För lite historik för backtest.")

    profile_defs = default_signal_backtest_profiles()
    if not compare_signal_profiles:
        profile_defs = [profile_defs[0]]

    results: List[Dict[str, object]] = []
    for profile in profile_defs:
        prof_name = str(profile["SignalProfile"])
        enrich_hist = bool(profile["enrich_history_form"])
        use_form_adj = bool(profile["use_form_probability_adjustment"])
        use_vol_adj = bool(profile["use_volatility_probability_adjustment"])
        incl_fav = bool(profile["include_favourites_in_blend"])
        incl_tio = bool(profile["include_tio_in_blend"])

        for budget in budgets:
            for strategy in strategies:
                total_cost = 0.0
                total_return = 0.0
                rights_counter: Counter = Counter()

                for chunk in chunks:
                    names: List[str] = []
                    for i in range(match_count):
                        ht = chunk.iloc[i].get("HomeTeam") if "HomeTeam" in chunk.columns else None
                        at = chunk.iloc[i].get("AwayTeam") if "AwayTeam" in chunk.columns else None
                        if ht is not None and at is not None:
                            hs = str(ht).strip()
                            ass = str(at).strip()
                            if hs and ass:
                                names.append(f"{hs} - {ass}")
                                continue
                        names.append(f"M{i+1}")

                    coupon = pd.DataFrame(
                        {
                            "Match": names,
                            "Odd1": chunk["odd1"].to_numpy(),
                            "OddX": chunk["oddx"].to_numpy(),
                            "Odd2": chunk["odd2"].to_numpy(),
                        }
                    )
                    inv = 1 / coupon[["Odd1", "OddX", "Odd2"]].to_numpy()
                    p_hat = inv / inv.sum(axis=1, keepdims=True)
                    prev_streck = np.clip(p_hat + rng.normal(0, 0.055, size=p_hat.shape), 0.02, 0.98)
                    prev_streck /= prev_streck.sum(axis=1, keepdims=True)
                    curr_streck = np.clip(p_hat + rng.normal(0, 0.038, size=p_hat.shape), 0.02, 0.98)
                    curr_streck /= curr_streck.sum(axis=1, keepdims=True)
                    coupon["Streck1"] = curr_streck[:, 0]
                    coupon["StreckX"] = curr_streck[:, 1]
                    coupon["Streck2"] = curr_streck[:, 2]
                    coupon["StreckMv1"] = curr_streck[:, 0] - prev_streck[:, 0]
                    coupon["StreckMvX"] = curr_streck[:, 1] - prev_streck[:, 1]
                    coupon["StreckMv2"] = curr_streck[:, 2] - prev_streck[:, 2]

                    shock = rng.uniform(0.94, 1.06, size=(match_count, 3))
                    start_odds = coupon[["Odd1", "OddX", "Odd2"]].to_numpy(dtype=float) * shock
                    curr_odds = coupon[["Odd1", "OddX", "Odd2"]].to_numpy(dtype=float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        odd_mv = (curr_odds - start_odds) / np.maximum(start_odds, 1.02)
                    coupon["OddMv1"] = odd_mv[:, 0]
                    coupon["OddMvX"] = odd_mv[:, 1]
                    coupon["OddMv2"] = odd_mv[:, 2]

                    tmp_coupon = history_csv.parent / "_tmp_backtest_coupon.csv"
                    tmp_out = history_csv.parent / "_tmp_backtest_out.csv"
                    coupon.to_csv(tmp_coupon, index=False)
                    out = suggest_system(
                        coupon_csv=tmp_coupon,
                        model_file=model_file,
                        max_rows=int(budget),
                        out_csv=tmp_out,
                        strategy=strategy,
                        game_type=game_type,
                        enrich_history_form=enrich_hist,
                        use_form_probability_adjustment=use_form_adj,
                        use_volatility_probability_adjustment=use_vol_adj,
                        include_favourites_in_blend=incl_fav,
                        include_tio_in_blend=incl_tio,
                    )

                    picks = out["Förslag"].tolist()
                    rows_matrix = expand_system_rows(picks)
                    actual = chunk["y"].to_numpy(dtype=int)
                    hits = (rows_matrix == actual).sum(axis=1)
                    max_hit = int(hits.max())
                    rights_counter[max_hit] += 1

                    base_pool = float(budget) * 1500.0
                    upset_level = float(np.mean(1.0 - curr_streck[np.arange(match_count), actual]))
                    payout_map = {
                        10: base_pool * 0.02 * (1 + upset_level * 1.2),
                        11: base_pool * 0.09 * (1 + upset_level * 1.8),
                        12: base_pool * 0.25 * (1 + upset_level * 2.4),
                        13: base_pool * 0.64 * (1 + upset_level * 3.2),
                    }
                    payout = payout_map.get(max_hit, 0.0)
                    rows = int(out["Systemrader"].iloc[0])
                    cost = rows * row_price
                    total_cost += cost
                    total_return += payout

                n_ch = len(chunks)
                roi = (total_return - total_cost) / total_cost if total_cost > 0 else 0.0
                row: Dict[str, object] = {
                    "GameType": game_type,
                    "Strategy": strategy,
                    "BudgetRows": int(budget),
                    "CouponsTested": n_ch,
                    "ROI": roi,
                    "AvgReturnPerCoupon": total_return / max(1, n_ch),
                    "Hit10PlusRate": sum(v for k, v in rights_counter.items() if k >= 10) / max(1, n_ch),
                    "Hit12PlusRate": sum(v for k, v in rights_counter.items() if k >= 12) / max(1, n_ch),
                    "FullHitRate": float(rights_counter.get(match_count, 0)) / max(1, n_ch),
                }
                if compare_signal_profiles:
                    row["SignalProfile"] = prof_name
                results.append(row)

    out_df = pd.DataFrame(results)
    if compare_signal_profiles:
        return out_df.sort_values(["Strategy", "BudgetRows", "SignalProfile"], ascending=True).reset_index(drop=True)
    return out_df.sort_values(["ROI", "Hit12PlusRate"], ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Europatipset optimizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Ladda ner historisk data")
    d.add_argument("--out", default="data/raw/history.csv")
    d.add_argument("--years", type=int, default=6)

    fa = sub.add_parser("sync-free-api-history", help="Synka historik från gratis football-data.org API")
    fa.add_argument("--out", default="data/raw/history_api.csv")
    fa.add_argument("--days-back", type=int, default=120)

    t = sub.add_parser("train", help="Träna sannolikhetsmodell")
    t.add_argument("--history", default="data/raw/history.csv")
    t.add_argument("--model", default="data/models/calibration.pkl")

    r = sub.add_parser("recommend", help="Få systemförslag för kupong")
    r.add_argument("--coupon", required=True, help="CSV med Match,Odd1,OddX,Odd2 och ev Streck1,StreckX,Streck2")
    r.add_argument("--model", default="data/models/calibration.pkl")
    r.add_argument("--max-rows", type=int, default=64)
    r.add_argument("--out", default="data/output/recommendation.csv")
    r.add_argument("--strategy", choices=["balanced", "safe", "value"], default="balanced")
    r.add_argument("--game-type", choices=list(GAME_TYPES.keys()), default="europatipset")

    c = sub.add_parser("build-coupon", help="Bygg kupong automatiskt från kommande matcher")
    c.add_argument("--out", default="data/input/auto_coupon.csv")
    c.add_argument("--n-matches", type=int, default=13)

    o = sub.add_parser("build-official-coupon", help="Bygg kupong från Svenska Spel statistik")
    o.add_argument("--out", default="data/input/official_coupon.csv")

    ar = sub.add_parser("auto-refresh-official", help="Auto-refresh nära spelstopp för omsättning/kupong")
    ar.add_argument("--coupon-out", default="data/input/official_coupon.csv")
    ar.add_argument("--meta-out", default="data/input/official_meta.json")
    ar.add_argument("--hours-before-close", type=int, default=2)
    ar.add_argument("--min-interval-minutes", type=int, default=15)

    ss = sub.add_parser(
        "sync-official-smart",
        help="Synka officiell kupong (ny omgång direkt, annars throttle)",
    )
    ss.add_argument("--coupon-out", default="data/input/official_coupon.csv")
    ss.add_argument("--meta-out", default="data/input/official_meta.json")
    ss.add_argument("--min-interval-minutes", type=float, default=12.0)

    bt = sub.add_parser("backtest", help="Kör historiskt backtest för strategier/budgetar")
    bt.add_argument("--history", default="data/raw/history.csv")
    bt.add_argument("--model", default="data/models/calibration.pkl")
    bt.add_argument("--out", default="data/output/backtest.csv")
    bt.add_argument("--budgets", default="32,64,128")
    bt.add_argument("--strategies", default="balanced,safe,value")
    bt.add_argument("--game-type", choices=list(GAME_TYPES.keys()), default="europatipset")
    bt.add_argument("--n-coupons", type=int, default=50)
    bt.add_argument(
        "--signal-ablation",
        action="store_true",
        help="Kör backtest per signal-profil (form/vol/fav/tio) och rapportera bl.a. FullHitRate vs ROI.",
    )

    args = parser.parse_args()

    if args.cmd == "download":
        df = download_historical_data(Path(args.out), back_years=args.years)
        print(f"Nedladdat: {len(df)} matcher -> {args.out}")
    elif args.cmd == "sync-free-api-history":
        df = sync_history_from_free_api(Path(args.out), days_back=args.days_back)
        print(f"API-historik synkad: {len(df)} matcher totalt -> {args.out}")
    elif args.cmd == "train":
        train_model(Path(args.history), Path(args.model))
        print(f"Modell sparad: {args.model}")
    elif args.cmd == "recommend":
        out = suggest_system(
            Path(args.coupon),
            Path(args.model),
            args.max_rows,
            Path(args.out),
            strategy=args.strategy,
            game_type=args.game_type,
        )
        print(f"Förslag sparat: {args.out}")
        print(out[["Match", "Förslag", "P1", "PX", "P2", "Value1", "ValueX", "Value2"]].to_string(index=False))
    elif args.cmd == "build-coupon":
        out = download_upcoming_coupon(Path(args.out), args.n_matches)
        print(f"Kupong skapad: {args.out}")
        print(out.to_string(index=False))
    elif args.cmd == "build-official-coupon":
        out = download_official_coupon_from_svenskaspel(Path(args.out))
        print(f"Officiell kupong skapad: {args.out}")
        print(out.to_string(index=False))
    elif args.cmd == "auto-refresh-official":
        changed, message = auto_refresh_official_snapshot(
            out_coupon_csv=Path(args.coupon_out),
            out_meta_json=Path(args.meta_out),
            hours_before_close=args.hours_before_close,
            min_interval_minutes=args.min_interval_minutes,
        )
        print(f"Auto-refresh: {'JA' if changed else 'NEJ'} - {message}")
    elif args.cmd == "sync-official-smart":
        changed, message = sync_official_snapshot_smart(
            out_coupon_csv=Path(args.coupon_out),
            out_meta_json=Path(args.meta_out),
            min_interval_minutes=args.min_interval_minutes,
        )
        print(f"Smart-sync: {'JA' if changed else 'NEJ'} - {message}")
    elif args.cmd == "backtest":
        budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
        strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
        out = backtest_strategies(
            history_csv=Path(args.history),
            model_file=Path(args.model),
            budgets=budgets,
            strategies=strategies,
            game_type=args.game_type,
            n_coupons=args.n_coupons,
            compare_signal_profiles=bool(args.signal_ablation),
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"Backtest sparat: {args.out}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
