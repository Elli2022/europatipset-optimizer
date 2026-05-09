"""
Hämta Europatipset-kontext från Svenska Spels webb (samma JSON som sidan laddar in i HTML).

Detta är **inte** klassisk DOM-scraping: vi läser `_svs.tipsen.data.preloadedState` ur sidan,
som redan innehåller omgång, matcher, tider, ligor, deltagare och odds/streck-statistik.

**Begränsning:** xStats, nyhetsflöden, tabell-vyer och laguppställningar som visas i UI
byggs ofta upp med **ytterligare** anrop (iframes, Sportradar/Kambi, interna API:er) som
kräver reverse engineering, session cookies eller headless browser — de ingår inte här.
Vi exponerar däremot `sportradarId` / `kambiId` så du kan följa länkar manuellt eller bygga
vidare med tillåtna API:er.

Respektera Svenska Spels användarvillkor och undvik onödigt höga förfrågningsfrekvenser.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from europatipset import _fetch_tipsen_preloaded_state

EUROPATIPSET_PAGE = "https://spela.svenskaspel.se/europatipset/"
EUROPATIPSET_STATISTIK = "https://spela.svenskaspel.se/europatipset/statistik"


def fetch_preloaded_state_with_fallback(
    urls: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Försök läsa tipsen-state från första URL som svarar med giltig JSON.
    Returnerar (state_dict, url_used).
    """
    order = urls or [EUROPATIPSET_PAGE, EUROPATIPSET_STATISTIK]
    errors: List[str] = []
    for url in order:
        try:
            return _fetch_tipsen_preloaded_state(url), url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Kunde inte läsa tipsen-state. " + " | ".join(errors))


def _participant_map(participants_block: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(participants_block, dict):
        return out
    for k, v in participants_block.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def _league_map(leagues_block: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(leagues_block, dict):
        return out
    for k, v in leagues_block.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def _country_map(countries_block: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(countries_block, dict):
        return out
    for k, v in countries_block.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def _sport_events_map(sport_events: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(sport_events, dict):
        return out
    for k, v in sport_events.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def build_round_context_bundle(state: Dict[str, Any], *, source_url: str) -> Dict[str, Any]:
    """Platta ut det som finns i Redux-state till ett JSON-vänligt paket."""
    draws = state.get("Draws") or {}
    draw_ids = draws.get("ids") or []
    entities = draws.get("entities") or {}
    if not draw_ids:
        raise RuntimeError("Draws.ids saknas i state.")
    draw_id = str(draw_ids[0])
    draw = entities.get(draw_id) or {}
    bet_events = state.get("BetEvents") or {}
    participants = _participant_map(state.get("Participants") or {})
    leagues = _league_map(state.get("Leagues") or {})
    countries = _country_map(state.get("Countries") or {})
    sport_events = _sport_events_map(state.get("SportEvents") or {})

    rows: List[Dict[str, Any]] = []
    items: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(bet_events, dict):
        for k, v in bet_events.items():
            if isinstance(v, dict):
                items.append((str(k), v))
    items.sort(key=lambda kv: int(kv[1].get("eventNumber") or 0))

    for _, be in items:
        evn = int(be.get("eventNumber") or 0)
        match_id = be.get("matchId")
        mid = str(match_id) if match_id is not None else ""
        league_id = str(be.get("leagueId") or "")
        league = leagues.get(league_id, {})
        country_id = str(league.get("countryId") or "")
        country = countries.get(country_id, {})
        ext = be.get("externalId") or {}
        if not isinstance(ext, dict):
            ext = {}

        pids = be.get("participants") or []
        home_name = ""
        away_name = ""
        if isinstance(pids, list) and len(pids) >= 2:
            hp = participants.get(str(pids[0]), {})
            ap = participants.get(str(pids[1]), {})
            if str(hp.get("type", "")).lower() == "home":
                home_name = str(hp.get("name") or "")
                away_name = str(ap.get("name") or "")
            else:
                home_name = str(hp.get("name") or "")
                away_name = str(ap.get("name") or "")

        label = str(be.get("eventDescription") or "").strip()
        if not label and home_name and away_name:
            label = f"{home_name} - {away_name}"

        se = sport_events.get(mid, {}) if mid else {}
        match_start = se.get("matchStart")
        status = se.get("sportEventStatus") or se.get("dataState")

        draw_ev = None
        for ev in draw.get("drawEvents") or []:
            if isinstance(ev, dict) and int(ev.get("eventNumber") or 0) == evn:
                draw_ev = ev
                break
        event_comment = ""
        if isinstance(draw_ev, dict):
            event_comment = str(draw_ev.get("eventComment") or "").strip()

        rows.append(
            {
                "eventNumber": evn,
                "matchLabel": label,
                "homeTeam": home_name,
                "awayTeam": away_name,
                "leagueName": str(league.get("name") or ""),
                "countryName": str(country.get("name") or ""),
                "matchStart": match_start,
                "sportEventStatus": status,
                "eventComment": event_comment,
                "matchId": match_id,
                "sportradarId": ext.get("sportradarId"),
                "kambiId": ext.get("Kambi"),
                "eventTypeStatisticId": be.get("eventTypeStatisticId"),
            }
        )

    draw_meta = {
        "drawNumber": draw.get("drawNumber"),
        "drawComment": draw.get("drawComment"),
        "drawState": draw.get("drawState"),
        "regCloseTime": draw.get("regCloseTime"),
        "regCloseDescription": draw.get("regCloseDescription"),
        "currentNetSale": draw.get("currentNetSale"),
        "rowPrice": draw.get("rowPrice"),
        "productName": draw.get("productName"),
    }

    return {
        "fetchedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceUrl": source_url,
        "draw": draw_meta,
        "matches": rows,
        "disclaimer": (
            "Källa: inbäddad tipsen-state i HTML. xStats/tabell/nyheter/elvor i full UI "
            "kan kräva ytterligare källor eller tillstånd från Svenska Spel / Sportradar."
        ),
    }


def fetch_europatipset_round_context(
    urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    state, url_used = fetch_preloaded_state_with_fallback(urls=urls)
    bundle = build_round_context_bundle(state, source_url=url_used)
    return bundle


def context_matches_dataframe(bundle: Dict[str, Any]) -> pd.DataFrame:
    rows = bundle.get("matches") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def save_round_context_bundle(bundle: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")


def load_round_context_bundle(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
