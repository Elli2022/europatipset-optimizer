import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
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


def download_official_coupon_from_svenskaspel(out_file: Path) -> pd.DataFrame:
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

        rows.append(
            {
                "Match": match,
                "Odd1": _to_float(odds_values[0], np.nan),
                "OddX": _to_float(odds_values[1], np.nan),
                "Odd2": _to_float(odds_values[2], np.nan),
                "Streck1": _to_float(dist_current[0], 33.33),
                "StreckX": _to_float(dist_current[1], 33.33),
                "Streck2": _to_float(dist_current[2], 33.33),
            }
        )

    out = pd.DataFrame(rows).dropna(subset=["Odd1", "OddX", "Odd2"]).head(13)
    if len(out) < 13:
        raise RuntimeError(f"Hittade bara {len(out)} matcher med odds i officiell kupong.")

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
    return df


def calibrated_probs(model, odd1: float, oddx: float, odd2: float) -> Tuple[float, float, float]:
    inv = np.array([1 / odd1, 1 / oddx, 1 / odd2], dtype=float)
    p_raw = inv / inv.sum()
    x = np.log(np.clip(p_raw, 1e-6, 1)).reshape(1, -1)
    p = model.predict_proba(x)[0]
    return float(p[0]), float(p[1]), float(p[2])


def optimize_system(matches: List[MatchSuggestion], max_rows: int) -> Tuple[List[str], int]:
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
        return float(sum(math.log(max(1e-9, cover(i, state[i]))) for i in range(len(state))))

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


def suggest_system(coupon_csv: Path, model_file: Path, max_rows: int, out_csv: Path) -> pd.DataFrame:
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    coupon = parse_coupon(coupon_csv)
    matches: List[MatchSuggestion] = []

    for _, row in coupon.iterrows():
        p1, px, p2 = calibrated_probs(model, float(row["Odd1"]), float(row["OddX"]), float(row["Odd2"]))
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

    picks, rows = optimize_system(matches, max_rows=max_rows)

    out = coupon.copy()
    out["P1"] = [m.p1 for m in matches]
    out["PX"] = [m.px for m in matches]
    out["P2"] = [m.p2 for m in matches]
    out["Value1"] = out["P1"] - out["Streck1"]
    out["ValueX"] = out["PX"] - out["StreckX"]
    out["Value2"] = out["P2"] - out["Streck2"]
    out["Förslag"] = picks
    out["Systemrader"] = rows

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Europatipset optimizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Ladda ner historisk data")
    d.add_argument("--out", default="data/raw/history.csv")
    d.add_argument("--years", type=int, default=6)

    t = sub.add_parser("train", help="Träna sannolikhetsmodell")
    t.add_argument("--history", default="data/raw/history.csv")
    t.add_argument("--model", default="data/models/calibration.pkl")

    r = sub.add_parser("recommend", help="Få systemförslag för kupong")
    r.add_argument("--coupon", required=True, help="CSV med Match,Odd1,OddX,Odd2 och ev Streck1,StreckX,Streck2")
    r.add_argument("--model", default="data/models/calibration.pkl")
    r.add_argument("--max-rows", type=int, default=64)
    r.add_argument("--out", default="data/output/recommendation.csv")

    c = sub.add_parser("build-coupon", help="Bygg kupong automatiskt från kommande matcher")
    c.add_argument("--out", default="data/input/auto_coupon.csv")
    c.add_argument("--n-matches", type=int, default=13)

    o = sub.add_parser("build-official-coupon", help="Bygg kupong från Svenska Spel statistik")
    o.add_argument("--out", default="data/input/official_coupon.csv")

    args = parser.parse_args()

    if args.cmd == "download":
        df = download_historical_data(Path(args.out), back_years=args.years)
        print(f"Nedladdat: {len(df)} matcher -> {args.out}")
    elif args.cmd == "train":
        train_model(Path(args.history), Path(args.model))
        print(f"Modell sparad: {args.model}")
    elif args.cmd == "recommend":
        out = suggest_system(Path(args.coupon), Path(args.model), args.max_rows, Path(args.out))
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


if __name__ == "__main__":
    main()
