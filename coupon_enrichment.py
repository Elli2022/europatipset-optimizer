"""
Gratis berikning av kupongrader från lokala historikfiler (history.csv / history_api.csv).

Bygger lag-tidslinjer för snabb lookup av senaste matchernas poäng (W=3,D=1,L=0).
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def _standardize_history(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    colmap = {c.lower(): c for c in df.columns}
    need = ["hometeam", "awayteam", "date", "ftr"]
    if not all(k in colmap for k in need):
        return None
    out = pd.DataFrame(
        {
            "HomeTeam": df[colmap["hometeam"]].astype(str),
            "AwayTeam": df[colmap["awayteam"]].astype(str),
            "Date": pd.to_datetime(df[colmap["date"]], errors="coerce"),
            "FTR": df[colmap["ftr"]].astype(str).str.upper(),
        }
    )
    out = out.dropna(subset=["Date"])
    out = out[out["FTR"].isin(["H", "D", "A"])].copy()
    return out.sort_values("Date", ascending=True).reset_index(drop=True)


def _points_home(ftr: str) -> int:
    return 3 if ftr == "H" else (1 if ftr == "D" else 0)


def _points_away(ftr: str) -> int:
    return 3 if ftr == "A" else (1 if ftr == "D" else 0)


def build_team_point_timelines(hist: pd.DataFrame) -> Dict[str, List[Tuple[pd.Timestamp, int]]]:
    """
    Per normaliserat lagnamn: lista (datum, poäng för laget den matchen), äldst→nyast.
    """
    timelines: DefaultDict[str, List[Tuple[pd.Timestamp, int]]] = defaultdict(list)
    for _, r in hist.iterrows():
        d = r["Date"]
        ftr = r["FTR"]
        h = _norm_team(str(r["HomeTeam"]))
        a = _norm_team(str(r["AwayTeam"]))
        timelines[h].append((d, _points_home(ftr)))
        timelines[a].append((d, _points_away(ftr)))
    # Sortera varje lista på datum
    for k in list(timelines.keys()):
        timelines[k].sort(key=lambda x: x[0])
    return dict(timelines)


def _resolve_timeline_key(query: str, timelines: Dict[str, List[Tuple[pd.Timestamp, int]]]) -> Optional[str]:
    q = _norm_team(query)
    best_k = None
    best_s = 0.0
    for k in timelines.keys():
        s = _similarity(q, k)
        if s > best_s:
            best_s = s
            best_k = k
    if best_k is None or best_s < 0.72:
        return None
    return best_k


def last_n_points(timeline: List[Tuple[pd.Timestamp, int]], n: int = 5) -> float:
    if not timeline:
        return float("nan")
    tail = timeline[-n:]
    return float(sum(p for _, p in tail))


def load_combined_histories(paths: List[Path]) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            raw = pd.read_csv(p, low_memory=False)
            std = _standardize_history(raw)
            if std is not None and len(std) > 0:
                frames.append(std)
        except Exception:
            continue
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return merged.sort_values("Date", ascending=True).reset_index(drop=True)


def enrich_coupon_with_history_form(coupon_df: pd.DataFrame, coupon_csv: Path, n: int = 5) -> pd.DataFrame:
    """
    Lägger till FormH_pts{n}, FormB_pts{n} om historik finns.
    """
    base = coupon_csv.resolve().parent.parent / "raw"
    paths = [base / "history.csv", base / "history_api.csv"]
    hist = load_combined_histories(paths)
    out = coupon_df.copy()
    out["FormH_pts5"] = np.nan
    out["FormB_pts5"] = np.nan
    if hist is None or hist.empty:
        return out

    timelines = build_team_point_timelines(hist)

    new_h: List[float] = []
    new_b: List[float] = []
    for _, row in out.iterrows():
        raw = str(row.get("Match", ""))
        if " - " not in raw:
            new_h.append(float("nan"))
            new_b.append(float("nan"))
            continue
        home_q, away_q = raw.split(" - ", 1)
        home_q, away_q = home_q.strip(), away_q.strip()
        hk = _resolve_timeline_key(home_q, timelines)
        ak = _resolve_timeline_key(away_q, timelines)
        hp = last_n_points(timelines.get(hk, []), n=n) if hk else float("nan")
        ap = last_n_points(timelines.get(ak, []), n=n) if ak else float("nan")
        new_h.append(hp)
        new_b.append(ap)

    out["FormH_pts5"] = new_h
    out["FormB_pts5"] = new_b
    return out


def add_streck_volatility_column(df: pd.DataFrame) -> pd.DataFrame:
    """Summa absolut streck-rörelse (andelar 0–1) som enkel volatilitetsproxy."""
    cols = ["StreckMv1", "StreckMvX", "StreckMv2"]
    if not all(c in df.columns for c in cols):
        df = df.copy()
        df["StreckVol"] = np.nan
        return df
    out = df.copy()
    vol = []
    for _, row in out.iterrows():
        vals = []
        for c in cols:
            v = row.get(c)
            if pd.notna(v):
                try:
                    vals.append(abs(float(v)))
                except Exception:
                    pass
        vol.append(float(sum(vals)) if vals else float("nan"))
    out["StreckVol"] = vol
    return out
