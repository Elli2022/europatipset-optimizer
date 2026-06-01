import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from europatipset import (
    GAME_TYPES,
    assess_forecast_confidence,
    backtest_strategies,
    download_historical_data,
    fetch_correct_row_for_draw_number,
    fetch_official_coupon_state,
    recommend_max_stake,
    simulate_rights_distribution,
    suggest_system,
    sync_history_from_free_api,
    sync_official_snapshot_smart,
    train_model,
    validate_coupon_data,
)
from free_context import build_free_context_for_coupon
from journal_browser_sync import ensure_journal_merged_once_session, sync_journal_to_browser
from svenskaspel_context import (
    context_matches_dataframe,
    fetch_europatipset_round_context,
    load_round_context_bundle,
    save_round_context_bundle,
)
from play_journal import (
    add_pending_bet,
    append_outcomes_training_rows,
    default_journal_path,
    ensure_seed_week_22,
    learning_hint,
    load_journal,
    settle_bet,
)
from round_lessons import builtin_lessons_summary, payout_min_rights


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
MODEL_PATH = DATA_DIR / "models" / "calibration.pkl"
OFFICIAL_COUPON_PATH = INPUT_DIR / "official_coupon.csv"
OFFICIAL_COUPON_EDITED_PATH = INPUT_DIR / "official_coupon_edited.csv"
RECOMMENDATION_PATH = OUTPUT_DIR / "recommendation_official.csv"
API_HISTORY_PATH = DATA_DIR / "raw" / "history_api.csv"
SS_CONTEXT_PATH = DATA_DIR / "raw" / "svenskaspel_round_context.json"
OFFICIAL_META_PATH = INPUT_DIR / "official_meta.json"
HISTORY_CSV = DATA_DIR / "raw" / "history.csv"
USER_DATA_DIR = DATA_DIR / "user"
JOURNAL_PATH = default_journal_path(BASE_DIR)
FD_STANDINGS_CACHE = DATA_DIR / "cache" / "fd_standings_free.json"

CTX_COLUMN_LABELS = {
    "Ctx_liga": "FD-liga",
    "Ctx_match_conf": "Namn-match",
    "Ctx_h_pos": "H tab",
    "Ctx_b_pos": "B tab",
    "Ctx_poängdiff": "Poäng Δ(H−B)",
    "Ctx_plac_diff": "Plac Δ (+ för hem)",
    "Ctx_form_H": "Form hem",
    "Ctx_form_B": "Form borta",
    "Ctx_tabell": "Tabell (gratis)",
}


def _inject_streamlit_secrets_into_env() -> None:
    """Streamlit Cloud: secrets.toml → os.environ (så befintlig kod med getenv fungerar)."""
    try:
        sec = st.secrets
        if "FOOTBALL_DATA_API_KEY" in sec and not os.getenv("FOOTBALL_DATA_API_KEY"):
            os.environ["FOOTBALL_DATA_API_KEY"] = str(sec["FOOTBALL_DATA_API_KEY"])
    except Exception:
        pass


def _bootstrap_calibration_model_if_missing(history_years: int = 3) -> tuple[bool, str]:
    """På Streamlit Cloud finns ingen committad .pkl – bygg modell första gången."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return True, ""
    try:
        download_historical_data(HISTORY_CSV, back_years=history_years)
        train_model(HISTORY_CSV, MODEL_PATH)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _ensure_backtest_history(history_years: int = 6) -> tuple[bool, str]:
    """
    Ensure history CSV exists and is parseable for backtests.
    """
    try:
        if HISTORY_CSV.exists():
            # Validate it's not empty/corrupt.
            pd.read_csv(HISTORY_CSV, low_memory=False)
            return True, ""
    except Exception:
        pass

    try:
        download_historical_data(HISTORY_CSV, back_years=history_years)
        pd.read_csv(HISTORY_CSV, low_memory=False)
        return True, ""
    except Exception as exc:
        return False, str(exc)


WEEKDAYS_SV = {
    0: "Måndag",
    1: "Tisdag",
    2: "Onsdag",
    3: "Torsdag",
    4: "Fredag",
    5: "Lördag",
    6: "Söndag",
}


def _parse_swe_number(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "").replace("\xa0", "")
    # Handle "502836,00" style numbers.
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except Exception:
        return default


def _format_draw_meta(meta: dict) -> tuple[str, str]:
    close_raw = meta.get("reg_close_time", "")
    try:
        close_dt = datetime.fromisoformat(close_raw)
        close_dt = close_dt.astimezone(ZoneInfo("Europe/Stockholm"))
        week = close_dt.isocalendar().week
        weekday = WEEKDAYS_SV[close_dt.weekday()]
        date_sv = close_dt.strftime("%Y-%m-%d")
        header = f"Vecka {week} - {weekday} {date_sv}"
        details = f"Omgång: {meta.get('draw_comment', '-')}, spelstopp: {close_dt.strftime('%Y-%m-%d %H:%M')}"
        return header, details
    except Exception:
        return "Vecka/datum saknas", f"Omgång: {meta.get('draw_comment', '-')}"


def _run_recommendation(
    max_rows: int,
    *,
    coupon_path: Path | None = None,
    use_manual_context_adjustment: bool = True,
    manual_context_strength: float = 0.08,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    journal = load_journal(JOURNAL_PATH)
    return suggest_system(
        coupon_csv=coupon_path or OFFICIAL_COUPON_PATH,
        model_file=MODEL_PATH,
        max_rows=max_rows,
        out_csv=RECOMMENDATION_PATH,
        strategy=st.session_state.get("strategy", "balanced"),
        game_type=st.session_state.get("game_type", "europatipset"),
        use_manual_context_adjustment=use_manual_context_adjustment,
        manual_context_strength=manual_context_strength,
        journal_aggregate=journal.get("aggregate"),
    )


def _history_status(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "Ingen API-historik synkad ännu.", "Klicka 'Synka API-historik nu' i sidpanelen."
    df = pd.read_csv(path, low_memory=False)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=ZoneInfo("Europe/Stockholm"))
    return (
        f"{len(df)} matcher i API-historik",
        f"Senast synkad: {mtime.strftime('%Y-%m-%d %H:%M:%S')}",
    )


def _hours_to_close(meta: dict) -> float | None:
    close_raw = meta.get("reg_close_time", "")
    if not close_raw:
        return None
    try:
        close_dt = pd.to_datetime(close_raw, utc=True, errors="coerce")
        if pd.isna(close_dt):
            return None
        now_utc = pd.Timestamp.now("UTC")
        return float((close_dt - now_utc).total_seconds() / 3600.0)
    except Exception:
        return None


def _last_hour_warning(meta: dict) -> None:
    hours_left = _hours_to_close(meta)
    if hours_left is None:
        return
    if hours_left <= 0:
        st.info(
            "Spelstopp har passerat för den omgång som visas i metadata just nu. "
            "Appen synkar automatiskt nästa öppna omgång enligt intervall — "
            "tryck **Hämta officiell kupong och beräkna förslag** om datum och matcher inte uppdaterats än."
        )
        return
    minutes_left = int(hours_left * 60)
    if minutes_left <= 15:
        st.markdown(
            f"""
            <div style="background:#b91c1c;color:white;padding:14px 16px;border-radius:10px;
                        font-weight:700;font-size:1.1rem;text-align:center;margin:8px 0 12px 0;">
                LAGG SPEL NU - {minutes_left} minuter kvar till spelstopp
            </div>
            """,
            unsafe_allow_html=True,
        )
    if hours_left <= 1:
        st.warning(
            f"Sista timmen: {minutes_left} minuter kvar till spelstopp. "
            "Uppdatera kupong och lägg spel nu för senast möjliga beslut."
        )
    elif hours_left <= 2:
        st.info(f"{minutes_left} minuter kvar till spelstopp. Förbered sista uppdatering.")


def _load_cached_meta() -> dict:
    if not OFFICIAL_META_PATH.exists():
        return {}
    try:
        return json.loads(OFFICIAL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _risk_level(confidence_label: str, warning_count: int) -> tuple[str, str, str]:
    """
    Returns (label, color, reason)
    """
    if confidence_label == "Hög" and warning_count == 0:
        return ("GRON", "#16a34a", "Hög konfidens och inga datavarningar.")
    if confidence_label == "Låg" or warning_count >= 2:
        return ("ROD", "#b91c1c", "Låg konfidens eller flera datavarningar.")
    return ("GUL", "#ca8a04", "Viss osäkerhet - spela mer försiktigt.")


def _render_my_spel_page() -> None:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    st.subheader("Mina spel")
    st.caption(
        "Spara kupong och förslag innan du lägger spelet. Efter rättning kan du ange "
        "13-teckensrad (1/X/2) för enkel miss-analys."
    )
    st.success(
        "Historiken sparas **automatiskt i din webbläsare** (localStorage) och följer med "
        "vid omstart av Streamlit Cloud — du behöver inte exportera JSON i normalfallet."
    )
    st.info(
        "Byter du webbläsare eller rensar sajtdatum försvinner lokala kopior; använd export nedan "
        "som backup eller spela i samma webbläsare på den enhet du brukar."
    )

    data = load_journal(JOURNAL_PATH)
    pending = [b for b in data["bets"] if b.get("status") == "pending"]
    settled = [b for b in data["bets"] if b.get("status") == "settled"]

    raw_json = json.dumps(data, ensure_ascii=False, indent=2)
    ej_tab, rt_tab = st.tabs(["Ej rättade", "Rättade"])
    with ej_tab:
        if not pending:
            st.info(
                "Inga osparade kuponger. Under **Analyser** → fliken **Systemförslag**, "
                "tryck **Spara till Mina spel** efter du hämtat kupongen."
            )
        for bet in pending:
            title = f"{bet.get('draw_comment', 'Omgång')} · rader {bet.get('system_rows', '-')}"
            with st.expander(title, expanded=False):
                st.caption(f"ID `{bet['id']}` · spelstopp `{bet.get('reg_close_time', '-')}`")
                dn = str(bet.get("draw_number", "")).strip()
                ref_row, ref_msg = fetch_correct_row_for_draw_number(dn)
                st.caption(ref_msg)
                if ref_row:
                    st.code(ref_row, language=None)
                    if st.button("Lägg in referensrad i fältet", key=f"apply_ref_{bet['id']}"):
                        st.session_state[f"out_{bet['id']}"] = ref_row
                        st.rerun()
                rec_rows = bet.get("recommendation_rows") or []
                if rec_rows:
                    cols = [c for c in ["Match", "Förslag", "P1", "PX", "P2"] if c in rec_rows[0]]
                    st.dataframe(
                        pd.DataFrame(rec_rows)[cols],
                        use_container_width=True,
                        hide_index=True,
                    )
                hits_best = st.number_input(
                    "Bästa rad — antal rätt (0–13)",
                    min_value=0,
                    max_value=13,
                    value=0,
                    key=f"hits_{bet['id']}",
                )
                outcomes = st.text_input(
                    "Valfritt: rätt rad, exakt 13 tecken (1, X, 2)",
                    max_chars=13,
                    key=f"out_{bet['id']}",
                    placeholder="t.ex. 1X212X1212122",
                )
                if st.button("Spara rättning", key=f"settle_{bet['id']}", type="primary"):
                    o = outcomes.strip().replace(" ", "").upper() if outcomes.strip() else ""
                    try:
                        settle_bet(
                            JOURNAL_PATH,
                            bet["id"],
                            int(hits_best),
                            o if len(o) == 13 else None,
                            after_save=sync_journal_to_browser,
                        )
                        if len(o) == 13 and rec_rows:
                            append_outcomes_training_rows(BASE_DIR, rec_rows, o)
                        st.success("Rättning sparad.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    with rt_tab:
        if not settled:
            st.info("Inga rättade spel ännu.")
        else:
            rows_out = []
            for bet in settled:
                ins = bet.get("insights") or {}
                rows_out.append(
                    {
                        "Omgång": bet.get("draw_comment", ""),
                        "Bästa rad": ins.get("hits_best_row", ""),
                        "Modell kolumner": ins.get("column_coverage", ""),
                        "Spelat kolumner": ins.get("played_column_coverage", ""),
                        "Under 10 rätt": "Ja" if ins.get("below_payout_threshold") else "Nej",
                        "Datum": bet.get("settled_at", ""),
                    }
                )
            st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)
            with st.expander("Detaljer — senaste rättade"):
                bet = settled[0]
                st.json(bet)

    st.markdown("##### Valfri backup (JSON)")
    st.download_button(
        "Exportera spellogg (JSON)",
        data=raw_json.encode("utf-8"),
        file_name="play_journal_export.json",
        mime="application/json",
        use_container_width=True,
    )


st.set_page_config(page_title="Europatipset Optimizer", page_icon="⚽", layout="wide")
_inject_streamlit_secrets_into_env()

_bootstrap_years = 3
try:
    _bootstrap_years = int(st.secrets.get("BOOTSTRAP_HISTORY_YEARS", 3))
except Exception:
    pass

if not MODEL_PATH.exists():
    with st.spinner(
        "Första starten: hämtar matchhistorik och tränar modell (kan ta 1–4 minuter på Streamlit Cloud)..."
    ):
        ok_bootstrap, bootstrap_err = _bootstrap_calibration_model_if_missing(history_years=_bootstrap_years)
    if not ok_bootstrap:
        st.error(
            "Kunde inte skapa modellen automatiskt på servern.\n\n"
            f"**Fel:** `{bootstrap_err}`\n\n"
            "**Gör så här:** Kör lokalt:\n"
            "`python europatipset.py download --years 3 --out data/raw/history.csv` och "
            "`python europatipset.py train --history data/raw/history.csv --model data/models/calibration.pkl`, "
            "lägg sedan till filen `data/models/calibration.pkl` i repot och pusha — eller försök deploy igen."
        )
        st.stop()

with st.sidebar:
    page_section = st.radio("Sektion", ["Analyser", "Mina spel"], index=0)

ensure_journal_merged_once_session(JOURNAL_PATH)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
if ensure_seed_week_22(JOURNAL_PATH, BASE_DIR):
    sync_journal_to_browser(load_journal(JOURNAL_PATH))

if page_section == "Mina spel":
    st.title("Mina spel")
    _render_my_spel_page()
    st.stop()

st.title("Europatipset Optimizer")
st.caption("Användarvänligt stöd för att välja tecken med sannolikhet, streckvärde och scenarios.")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 1rem;}
      @media (max-width: 768px) {
        .block-container {padding-left: 0.7rem; padding-right: 0.7rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Inställningar")
    view_mode = st.selectbox(
        "Visningsläge",
        options=["Desktop (tabell)", "Mobil (kort)"],
        index=0,
    )
    mobile_mode = view_mode == "Mobil (kort)"
    st.session_state["game_type"] = st.selectbox(
        "Omgångstyp",
        options=list(GAME_TYPES.keys()),
        format_func=lambda x: f"{x} ({GAME_TYPES[x]['match_count']} matcher)",
    )
    max_rows = st.number_input("Max antal rader", min_value=1, max_value=4096, value=64, step=1)
    st.session_state["strategy"] = st.selectbox(
        "Strategi",
        options=["balanced", "safe", "value"],
        format_func=lambda x: {
            "balanced": "Balanserad",
            "safe": "Säker (favorit-fokus)",
            "value": "Värde (mer gardering)",
        }[x],
    )
    compare_budgets = st.multiselect(
        "Jämför budgetscenarier",
        options=[16, 32, 64, 128, 256, 512],
        default=[32, 64, 128],
    )
    days_back = st.slider("API-historik (dagar bakåt)", min_value=30, max_value=365, value=120, step=10)
    auto_turnover_refresh = st.toggle(
        "Auto-synka officiell kupong (Svenska Spel, smart intervall)", value=True
    )
    _journal_for_hint = load_journal(JOURNAL_PATH)
    _agg = _journal_for_hint.get("aggregate") or {}
    with st.expander("Lärdom & kalibrering (spellogg)", expanded=int(_agg.get("settled_rounds", 0)) >= 1):
        st.markdown(builtin_lessons_summary())
        st.caption(learning_hint(_journal_for_hint))
    if st.button("Synka API-historik nu", use_container_width=True):
        if not os.getenv("FOOTBALL_DATA_API_KEY"):
            st.error("Saknar FOOTBALL_DATA_API_KEY. Lägg till i miljövariabler/Streamlit Secrets.")
        else:
            with st.spinner("Synkar historik från football-data.org..."):
                try:
                    df_api = sync_history_from_free_api(API_HISTORY_PATH, days_back=days_back)
                    st.success(f"API-historik synkad: {len(df_api)} matcher totalt.")
                except Exception as exc:
                    st.error(f"Kunde inte synka API-historik: {exc}")

status_line, status_detail = _history_status(API_HISTORY_PATH)
st.info(f"{status_line}  |  {status_detail}")
cached_meta = _load_cached_meta()
if auto_turnover_refresh:
    try:
        sync_official_snapshot_smart(
            out_coupon_csv=OFFICIAL_COUPON_PATH,
            out_meta_json=OFFICIAL_META_PATH,
            min_interval_minutes=12.0,
        )
    except Exception:
        pass
cached_meta = _load_cached_meta()

if cached_meta:
    _last_hour_warning(cached_meta)

st.write("1) Hämta officiell kupong  2) Välj strategi/budget  3) Analysera och exportera")

with st.expander("Om modellen — vad som ingår (och inte)"):
    st.markdown(
        """
Förslagen bygger på **matchodds**, **streck**, **kalibrerad modell** och en **viktad blandning** med extra fält som finns i Svenska Spels **publika**
tips-JSON (bl.a. «favourites» och tidningsröster `tioTidningarTips` när de inte är tomma). Det är **inte** startelvor,
xG-detaljer, nyhetsartiklar eller Oddset-kors — den datan levereras inte i samma öppna payload som statistiksidan,
och kräver i praktiken andra produkter/API:er eller licenser.

Gratis **ligatabell + form** under Matchanalys hämtas separat via football-data.org (gratisplan + din API-historik).

Vid hämtning av kupong försöker appen även läsa en **publik referensrad** när utfallet finns i tips-data.

Det betyder att modellen kan missa sent nyheter; använd sunda marginaler och egen matchkunskap vid osäkerhet.
        """
    )

if "coupon_df" not in st.session_state:
    st.session_state["coupon_df"] = None
if "result_df" not in st.session_state:
    st.session_state["result_df"] = None
if "meta" not in st.session_state:
    st.session_state["meta"] = {}
if "backtest_df" not in st.session_state:
    st.session_state["backtest_df"] = None
if "ss_context_bundle" not in st.session_state:
    st.session_state["ss_context_bundle"] = None
if "use_manual_context_adjustment" not in st.session_state:
    st.session_state["use_manual_context_adjustment"] = True
if "manual_context_strength" not in st.session_state:
    st.session_state["manual_context_strength"] = 0.08
if "use_edited_coupon_for_reco" not in st.session_state:
    st.session_state["use_edited_coupon_for_reco"] = False

if st.button("Hämta officiell kupong och beräkna förslag", type="primary", use_container_width=True):
    with st.spinner("Hämtar officiell kupong och räknar fram förslag..."):
        try:
            st.session_state.pop("sv_ref", None)
            coupon_df, meta = fetch_official_coupon_state()
            INPUT_DIR.mkdir(parents=True, exist_ok=True)
            coupon_df.to_csv(OFFICIAL_COUPON_PATH, index=False)
            chosen_coupon = (
                OFFICIAL_COUPON_EDITED_PATH
                if st.session_state.get("use_edited_coupon_for_reco") and OFFICIAL_COUPON_EDITED_PATH.exists()
                else OFFICIAL_COUPON_PATH
            )
            result_df = _run_recommendation(
                max_rows=max_rows,
                coupon_path=chosen_coupon,
                use_manual_context_adjustment=bool(st.session_state.get("use_manual_context_adjustment", True)),
                manual_context_strength=float(st.session_state.get("manual_context_strength", 0.08)),
            )
            st.session_state["coupon_df"] = coupon_df
            st.session_state["result_df"] = result_df
            st.session_state["meta"] = meta
            st.session_state["forecast_dist"] = None
            rr, rm = fetch_correct_row_for_draw_number(str(meta.get("draw_number", "")))
            st.session_state["sv_ref"] = {"dn": str(meta.get("draw_number", "")), "row": rr, "msg": rm}
        except Exception as exc:
            st.error(f"Kunde inte hämta eller beräkna kupong: {exc}")

if st.session_state["result_df"] is not None and st.session_state["coupon_df"] is not None:
    try:
                coupon_df = st.session_state["coupon_df"]
                result_df = st.session_state["result_df"]
                meta = st.session_state.get("meta") or {}
                header, details = _format_draw_meta(meta)
                st.success("Klart!")
                st.subheader(header)
                st.write(details)
                st.caption(f"Senast uppdaterad: {datetime.now(ZoneInfo('Europe/Stockholm')).strftime('%Y-%m-%d %H:%M:%S')}")
                _last_hour_warning(meta)

                pack = st.session_state.get("sv_ref") or {}
                if pack.get("dn") == str(meta.get("draw_number", "")):
                    if pack.get("row"):
                        st.info(
                            "Publik **referensrad** för denna omgången (läst från Svenska Spels tips-sidor när utfallet "
                            "finns i deras JSON — inte skador eller elvor): "
                            f"`{pack['row']}`"
                        )
                    elif pack.get("msg"):
                        st.caption(str(pack["msg"]))

                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                    [
                        "Systemförslag",
                        "Matchanalys",
                        "Budgetjämförelse",
                        "Träffprognos",
                        "Utdelningskalkyl",
                        "Data & export",
                        "Backtest",
                    ]
                )

                data_warnings = validate_coupon_data(coupon_df)
                confidence = assess_forecast_confidence(result_df)
                if data_warnings:
                    st.warning("Datakvalitetsflaggor: " + " | ".join(data_warnings))
                st.caption(
                    f"Prognoskonfidens: {confidence['confidence_label']} "
                    f"(osäkerhet={confidence['uncertainty_score']:.2f}, topp-gap={confidence['avg_top_gap']:.2f})"
                )
                risk_label, risk_color, risk_reason = _risk_level(
                    confidence_label=str(confidence["confidence_label"]),
                    warning_count=len(data_warnings),
                )
                st.markdown(
                    f"""
                    <div style="background:{risk_color};color:white;padding:10px 14px;border-radius:10px;
                                font-weight:700;text-align:center;margin:6px 0 10px 0;">
                        Trafikljus: {risk_label}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Orsak: {risk_reason}")

                with tab1:
                    show_cols = ["Match", "Förslag", "P1", "PX", "P2", "Value1", "ValueX", "Value2"]
                    formatted = result_df[show_cols].copy()
                    for c in ["P1", "PX", "P2", "Value1", "ValueX", "Value2"]:
                        formatted[c] = (formatted[c] * 100).map(lambda v: f"{v:.1f}%")
                    if mobile_mode:
                        for _, row in formatted.iterrows():
                            st.markdown(f"**{row['Match']}**")
                            c1, c2 = st.columns(2)
                            c1.metric("Förslag", row["Förslag"])
                            c2.metric("P1 / PX / P2", f"{row['P1']} / {row['PX']} / {row['P2']}")
                            st.caption(
                                f"Värde: 1={row['Value1']} | X={row['ValueX']} | 2={row['Value2']}"
                            )
                            st.divider()
                    else:
                        st.dataframe(formatted, use_container_width=True, hide_index=True)
                    st.metric("Systemrader", int(result_df["Systemrader"].iloc[0]))
                    st.caption(f"Strategi: {st.session_state.get('strategy', 'balanced')}")
                    _gt = st.session_state.get("game_type", "europatipset")
                    _mp = payout_min_rights(_gt)
                    st.info(
                        f"Europatipset betalar från **{_mp} rätt på en rad**. "
                        f"Kolumn-täckning (rätt tecken någonstans i raden) räcker inte — se **Träffprognos**."
                    )
                    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
                    if st.button("Spara till Mina spel", key="save_play_journal", use_container_width=True):
                        bid = add_pending_bet(
                            JOURNAL_PATH,
                            meta,
                            coupon_df,
                            result_df,
                            int(result_df["Systemrader"].iloc[0]),
                            "",
                            after_save=sync_journal_to_browser,
                        )
                        st.success(
                            f"Sparat som `{bid}`. Byt till **Mina spel** i sidopanelen för att rätta efter omgången."
                        )

                with tab2:
                    st.markdown(
                        "#### Matchanalys — gratis ligatabell + form"
                        "\n\n"
                        "Här visas **ligatabell** (football-data.org free, cache ~12 h) och **form** "
                        "från din synkade **API-historik** när den finns. "
                        "Extra kolumner (om de finns i kupongen): **odds- och streck-rörelser** från Svenska Spel, "
                        "**StreckVol** (summa absolut streckförskjutning), samt **FormH/B_pts5** från lokala "
                        "`data/raw/history*.csv` — de påverkar **P1/PX/P2** i systemförslaget lätt. "
                        "Det är **inte** skador eller officiella elvor — bara kontext. "
                        "Kupongmatcher utanför de åtta gratisligorna (se `free_context.py`) får ofta tomma tabellkolumner där."
                    )
                    analysis = result_df.copy()
                    analysis["Troligaste tecken"] = analysis[["P1", "PX", "P2"]].idxmax(axis=1).map(
                        {"P1": "1", "PX": "X", "P2": "2"}
                    )
                    analysis["Bäst värde"] = analysis[["Value1", "ValueX", "Value2"]].idxmax(axis=1).map(
                        {"Value1": "1", "ValueX": "X", "Value2": "2"}
                    )
                    _sig_cols = [
                        c
                        for c in (
                            "OddMv1",
                            "OddMvX",
                            "OddMv2",
                            "StreckMv1",
                            "StreckMvX",
                            "StreckMv2",
                            "StreckVol",
                            "FormH_pts5",
                            "FormB_pts5",
                        )
                        if c in analysis.columns
                    ]
                    analysis_view = analysis[
                        ["Match", "Troligaste tecken", "Bäst värde", "Förslag", "P1", "PX", "P2"]
                        + _sig_cols
                    ]
                    hist_path = API_HISTORY_PATH if API_HISTORY_PATH.exists() else None
                    ctx_df, ctx_note = build_free_context_for_coupon(
                        coupon_df,
                        os.getenv("FOOTBALL_DATA_API_KEY"),
                        hist_path,
                        FD_STANDINGS_CACHE,
                        ttl_hours=12.0,
                    )
                    ctx_show = ctx_df.rename(columns=CTX_COLUMN_LABELS)
                    analysis_disp = pd.concat([analysis_view, ctx_show], axis=1)
                    st.caption(ctx_note)
                    if mobile_mode:
                        for _, row in analysis_disp.iterrows():
                            with st.expander(row["Match"]):
                                st.write(f"Troligaste tecken: **{row['Troligaste tecken']}**")
                                st.write(f"Bäst värde: **{row['Bäst värde']}**")
                                st.write(f"Förslag: **{row['Förslag']}**")
                                st.write(
                                    f"Sannolikhet 1/X/2: {row['P1']:.3f} / {row['PX']:.3f} / {row['P2']:.3f}"
                                )
                                if _sig_cols:
                                    st.caption("Signaler: " + ", ".join(_sig_cols))
                                    for sc in _sig_cols:
                                        v = row.get(sc)
                                        if v is None or (isinstance(v, float) and pd.isna(v)):
                                            v = "-"
                                        elif isinstance(v, (int, float)) and sc.startswith("OddMv"):
                                            v = f"{float(v) * 100:.1f}% rel."
                                        elif isinstance(v, (int, float)) and sc.startswith("StreckMv"):
                                            v = f"{float(v) * 100:.1f} pp"
                                        st.caption(f"{sc}: {v}")
                                st.divider()
                                st.markdown("**Gratis tabell/form:**")
                                for _, lab in CTX_COLUMN_LABELS.items():
                                    if lab in row.index:
                                        val = row[lab]
                                        if val is None or (isinstance(val, float) and pd.isna(val)):
                                            val = "-"
                                        st.caption(f"{lab}: {val}")
                    else:
                        st.dataframe(
                            analysis_disp,
                            use_container_width=True,
                            hide_index=True,
                        )

                with tab3:
                    scenarios = []
                    scenario_coupon = (
                        OFFICIAL_COUPON_EDITED_PATH
                        if st.session_state.get("use_edited_coupon_for_reco") and OFFICIAL_COUPON_EDITED_PATH.exists()
                        else OFFICIAL_COUPON_PATH
                    )
                    for budget in compare_budgets:
                        tmp_out = OUTPUT_DIR / f"scenario_{budget}.csv"
                        df_budget = suggest_system(
                            coupon_csv=scenario_coupon,
                            model_file=MODEL_PATH,
                            max_rows=int(budget),
                            out_csv=tmp_out,
                            strategy=st.session_state.get("strategy", "balanced"),
                            game_type=st.session_state.get("game_type", "europatipset"),
                            use_manual_context_adjustment=bool(
                                st.session_state.get("use_manual_context_adjustment", True)
                            ),
                            manual_context_strength=float(st.session_state.get("manual_context_strength", 0.08)),
                        )
                        scenarios.append(
                            {
                                "Budget": int(budget),
                                "Rader": int(df_budget["Systemrader"].iloc[0]),
                                "Helgarderingar": int((df_budget["Förslag"].str.len() == 3).sum()),
                                "Halvgarderingar": int((df_budget["Förslag"].str.len() == 2).sum()),
                                "Spikar": int((df_budget["Förslag"].str.len() == 1).sum()),
                            }
                        )
                    st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

                with tab4:
                    st.markdown("#### Sannolikhetsprognos för antal rätt")
                    n_sim = st.slider("Antal simuleringar", min_value=5000, max_value=50000, value=20000, step=5000)
                    if st.button("Beräkna prognos", key="run_prob_forecast"):
                        with st.spinner("Simulerar utfallsfördelning..."):
                            st.session_state["forecast_dist"] = simulate_rights_distribution(
                                result_df,
                                n_sim=n_sim,
                                game_type=st.session_state.get("game_type", "europatipset"),
                            )
                    dist = st.session_state.get("forecast_dist")
                    if dist:
                        min_pay = int(dist.get("min_payout_rights", payout_min_rights("europatipset")))
                        p_pay = float(dist.get(f"p_ge_{min_pay}", 0.0))
                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        c1.metric("10 rätt", f"{dist.get(10, 0)*100:.1f}%")
                        c2.metric("11 rätt", f"{dist.get(11, 0)*100:.1f}%")
                        c3.metric("12 rätt", f"{dist.get(12, 0)*100:.1f}%")
                        c4.metric("13 rätt", f"{dist.get(13, 0)*100:.2f}%")
                        c5.metric(f"≥{min_pay} rätt (utdelning)", f"{p_pay*100:.1f}%")
                        c6.metric("Mest sannolikt", f"{dist['most_likely']} rätt")
                        st.caption(
                            "Prognosen är Monte Carlo på **bästa rad** i systemet — inte bara antal rätta kolumner."
                        )

                with tab5:
                    st.markdown("#### Jämför vinst mot insats")
                    meta_for_calc = meta if meta else cached_meta
                    row_price = _parse_swe_number(meta_for_calc.get("row_price") or meta_for_calc.get("rowPrice"), default=1.0)
                    net_sale = _parse_swe_number(
                        meta_for_calc.get("current_net_sale") or meta_for_calc.get("currentNetSale"),
                        default=0.0,
                    )
                    rows_count = int(result_df["Systemrader"].iloc[0])
                    system_cost = rows_count * row_price

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        omsattning = st.number_input(
                            "Omsättning (kr)",
                            min_value=0.0,
                            value=float(net_sale),
                            step=1000.0,
                        )
                    with c2:
                        aterbetalning = st.slider(
                            "Återbetalning till vinstpool (%)",
                            min_value=30,
                            max_value=100,
                            value=65,
                            step=1,
                        )
                    with c3:
                        st.metric("Systemkostnad", f"{system_cost:,.0f} kr".replace(",", " "))

                    st.caption("Ange uppskattat antal vinnande rader per nivå för att räkna möjlig utdelning.")
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        p13 = st.slider("Andel pott 13 rätt (%)", 0, 100, 40, 1)
                        w13 = st.number_input("Vinnande rader 13 rätt", min_value=1, value=10, step=1)
                    with k2:
                        p12 = st.slider("Andel pott 12 rätt (%)", 0, 100, 20, 1)
                        w12 = st.number_input("Vinnande rader 12 rätt", min_value=1, value=150, step=1)
                    with k3:
                        p11 = st.slider("Andel pott 11 rätt (%)", 0, 100, 20, 1)
                        w11 = st.number_input("Vinnande rader 11 rätt", min_value=1, value=1200, step=1)
                    with k4:
                        p10 = st.slider("Andel pott 10 rätt (%)", 0, 100, 20, 1)
                        w10 = st.number_input("Vinnande rader 10 rätt", min_value=1, value=7000, step=1)

                    share_sum = p13 + p12 + p11 + p10
                    if share_sum != 100:
                        st.warning(f"Pottandelarna summerar till {share_sum}%. Justera till 100% för full fördelning.")

                    prize_pool = omsattning * (aterbetalning / 100.0)
                    pay13 = (prize_pool * (p13 / 100.0)) / max(1, w13)
                    pay12 = (prize_pool * (p12 / 100.0)) / max(1, w12)
                    pay11 = (prize_pool * (p11 / 100.0)) / max(1, w11)
                    pay10 = (prize_pool * (p10 / 100.0)) / max(1, w10)

                    out_df = pd.DataFrame(
                        [
                            {"Nivå": "13 rätt", "Uppskattad utdelning/rad (kr)": round(pay13)},
                            {"Nivå": "12 rätt", "Uppskattad utdelning/rad (kr)": round(pay12)},
                            {"Nivå": "11 rätt", "Uppskattad utdelning/rad (kr)": round(pay11)},
                            {"Nivå": "10 rätt", "Uppskattad utdelning/rad (kr)": round(pay10)},
                        ]
                    )
                    st.dataframe(out_df, use_container_width=True, hide_index=True)

                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Netto vid 13 rätt", f"{(pay13 - system_cost):,.0f} kr".replace(",", " "))
                    g2.metric("Netto vid 12 rätt", f"{(pay12 - system_cost):,.0f} kr".replace(",", " "))
                    g3.metric("Netto vid 11 rätt", f"{(pay11 - system_cost):,.0f} kr".replace(",", " "))
                    g4.metric("Netto vid 10 rätt", f"{(pay10 - system_cost):,.0f} kr".replace(",", " "))

                    st.markdown("#### Rekommenderad maxinsats")
                    forecast = st.session_state.get("forecast_dist")
                    if not forecast:
                        st.info("Kör först `Beräkna prognos` i fliken `Träffprognos` för att få maxinsats-rekommendation.")
                    else:
                        payout_model = {13: pay13, 12: pay12, 11: pay11, 10: pay10}
                        margin = st.slider(
                            "Säkerhetsmarginal för maxinsats (%)",
                            min_value=0,
                            max_value=50,
                            value=15,
                            step=1,
                        )
                        bankroll_cap = st.number_input(
                            "Frivilligt eget tak (kr)",
                            min_value=0.0,
                            value=200.0,
                            step=50.0,
                        )
                        rec = recommend_max_stake(
                            forecast_dist=forecast,
                            payout_by_rights=payout_model,
                            margin_pct=float(margin),
                            bankroll_cap=float(bankroll_cap),
                        )

                        r1, r2, r3 = st.columns(3)
                        r1.metric("Break-even max (kr)", f"{rec['max_break_even']:,.0f}".replace(",", " "))
                        r2.metric("Konservativ max (kr)", f"{rec['max_conservative']:,.0f}".replace(",", " "))
                        r3.metric("Rekommenderad maxinsats (kr)", f"{rec['recommended_max']:,.0f}".replace(",", " "))
                        st.caption(
                            "Beräkningen baseras på simulerade sannolikheter (10-13 rätt) och din utdelningsmodell. "
                            "Använd rekommenderad maxinsats som riskstyrning, inte garanti."
                        )

                with tab6:
                    st.markdown("#### Inför omgången — Svenska Spel (publik JSON)")
                    st.caption(
                        "Hämtar samma **inbäddade tipsen-state** som webbsidan (matcher, tider, ligor, "
                        "Sportradar/Kambi-id, ev. `eventComment`). Full **xStats, tabell, nyheter och elvor** "
                        "ligger ofta i separata anrop — inte i denna JSON. Respektera Svenska Spels villkor."
                    )

                    c_ss1, c_ss2 = st.columns(2)
                    with c_ss1:
                        if st.button("Hämta / uppdatera från spela.svenskaspel.se", key="fetch_ss_context"):
                            with st.spinner("Hämtar Europatipset-sida…"):
                                b = fetch_europatipset_round_context()
                                save_round_context_bundle(b, SS_CONTEXT_PATH)
                                st.session_state["ss_context_bundle"] = b
                            st.success(f"Sparat: `{SS_CONTEXT_PATH}`")
                    with c_ss2:
                        if st.button("Läs sparad kontext från disk", key="reload_ss_context_disk"):
                            st.session_state["ss_context_bundle"] = load_round_context_bundle(SS_CONTEXT_PATH)

                    bundle = st.session_state.get("ss_context_bundle")
                    if bundle is None:
                        bundle = load_round_context_bundle(SS_CONTEXT_PATH)
                    if bundle:
                        st.caption(
                            f"Hämtad (UTC): `{bundle.get('fetchedAtUtc', '')}` · Källa: `{bundle.get('sourceUrl', '')}`"
                        )
                        df_ctx = context_matches_dataframe(bundle)
                        if not df_ctx.empty:
                            st.dataframe(df_ctx, use_container_width=True, hide_index=True)
                        with st.expander("Omgångs-meta + disclaimer"):
                            st.json(
                                {
                                    "draw": bundle.get("draw"),
                                    "disclaimer": bundle.get("disclaimer"),
                                }
                            )
                    else:
                        st.info("Ingen kontext sparad än — tryck på «Hämta / uppdatera».")

                    st.markdown("#### Officiell kupong")
                    if mobile_mode:
                        for _, row in coupon_df.iterrows():
                            with st.expander(row["Match"]):
                                st.write(f"Odds 1/X/2: {row['Odd1']} / {row['OddX']} / {row['Odd2']}")
                                st.write(
                                    f"Streck 1/X/2: {row['Streck1']} / {row['StreckX']} / {row['Streck2']}"
                                )
                    else:
                        st.dataframe(coupon_df, use_container_width=True, hide_index=True)
                    st.markdown("#### Redigerbar kupong (vad-om)")
                    st.caption(
                        "Tips: fyll `ManualHomeAdj` / `ManualDrawAdj` / `ManualAwayAdj` per match i intervallet "
                        "`[-1, +1]` för sen info (elvor/nyheter). + = stärker utfallet, - = försvagar."
                    )
                    coupon_edit_base = coupon_df.copy()
                    for c in ["ManualHomeAdj", "ManualDrawAdj", "ManualAwayAdj"]:
                        if c not in coupon_edit_base.columns:
                            coupon_edit_base[c] = 0.0
                    edited = st.data_editor(
                        coupon_edit_base,
                        num_rows="fixed",
                        use_container_width=True,
                        hide_index=True,
                    )
                    edited_path = OFFICIAL_COUPON_EDITED_PATH
                    pd.DataFrame(edited).to_csv(edited_path, index=False)
                    c_m1, c_m2, c_m3 = st.columns(3)
                    with c_m1:
                        st.session_state["use_edited_coupon_for_reco"] = st.checkbox(
                            "Använd redigerad kupong i beräkning",
                            value=bool(st.session_state.get("use_edited_coupon_for_reco", False)),
                        )
                    with c_m2:
                        st.session_state["use_manual_context_adjustment"] = st.checkbox(
                            "Aktivera manuell sen-info-justering",
                            value=bool(st.session_state.get("use_manual_context_adjustment", True)),
                        )
                    with c_m3:
                        st.session_state["manual_context_strength"] = st.slider(
                            "Styrka manuell justering",
                            min_value=0.0,
                            max_value=0.30,
                            value=float(st.session_state.get("manual_context_strength", 0.08)),
                            step=0.01,
                        )
                    if st.button("Räkna om förslag med dessa manuella signaler", key="rerun_manual_context"):
                        coupon_choice = (
                            OFFICIAL_COUPON_EDITED_PATH
                            if st.session_state.get("use_edited_coupon_for_reco")
                            else OFFICIAL_COUPON_PATH
                        )
                        new_result = _run_recommendation(
                            max_rows=max_rows,
                            coupon_path=coupon_choice,
                            use_manual_context_adjustment=bool(
                                st.session_state.get("use_manual_context_adjustment", True)
                            ),
                            manual_context_strength=float(st.session_state.get("manual_context_strength", 0.08)),
                        )
                        st.session_state["result_df"] = new_result
                        st.success("Nytt systemförslag beräknat med manuella signaler.")
                    st.caption(f"Redigerad kupong sparad: {edited_path}")
                    st.caption(f"Rekommendation sparad i {RECOMMENDATION_PATH}")

                with tab7:
                    st.markdown("#### Historiskt backtest")
                    c1, c2 = st.columns(2)
                    with c1:
                        bt_budgets = st.multiselect(
                            "Budgetar (rader)",
                            options=[16, 32, 64, 128, 256],
                            default=[32, 64, 128],
                        )
                    with c2:
                        bt_n = st.slider("Antal historiska kuponger", min_value=10, max_value=120, value=40, step=10)
                    bt_signal_ablation = st.checkbox(
                        "Jämför per signal (ablation, långsammare)",
                        value=False,
                        help="Kör samma sweep för varje signal-profil och jämför ROI mot 13-rätt-andel — undvik att «tuna» bara mot hög träff i sim.",
                    )

                    if st.button("Kör backtest", key="run_backtest"):
                        with st.spinner("Kör backtest över historik..."):
                            ok_hist, hist_err = _ensure_backtest_history(history_years=6)
                            if not ok_hist:
                                st.error(f"Kunde inte förbereda backtest-historik: {hist_err}")
                            else:
                                bt = backtest_strategies(
                                    history_csv=DATA_DIR / "raw" / "history.csv",
                                    model_file=MODEL_PATH,
                                    budgets=[int(x) for x in bt_budgets],
                                    strategies=["balanced", "safe", "value"],
                                    game_type=st.session_state.get("game_type", "europatipset"),
                                    n_coupons=int(bt_n),
                                    compare_signal_profiles=bool(bt_signal_ablation),
                                )
                                st.session_state["backtest_df"] = bt
                                st.session_state["backtest_signal_ablation"] = bool(bt_signal_ablation)
                    if st.session_state.get("backtest_df") is not None:
                        bt = st.session_state["backtest_df"]
                        bt_ablation = bool(st.session_state.get("backtest_signal_ablation"))
                        if bt_ablation and "SignalProfile" in bt.columns:
                            st.warning(
                                "Signal-ablation: **prioritera ROI / nettoresultat** i denna tabell. "
                                "Om **FullHitRate** (alla rätt på kupongen) stiger för en profil men **ROI** sjunker jämfört med `allt_på` "
                                "är det ofta brus — då är sannolikhetsgrenen sannolikt inte värd pengarna i denna modell."
                            )
                        bt_numeric = bt.copy()
                        if bt_ablation and "SignalProfile" in bt_numeric.columns:
                            ref = bt_numeric[bt_numeric["SignalProfile"] == "allt_på"].copy()
                            if ref.empty:
                                ref = bt_numeric.copy()
                        else:
                            ref = bt_numeric.copy()
                        ref["Score"] = (ref["ROI"] * 0.65) + (ref["Hit12PlusRate"] * 0.35)
                        best = ref.sort_values(["Score", "ROI"], ascending=False).iloc[0]
                        st.success(
                            f"Rekommenderad budget (baserat på profilen **allt_på**): {int(best['BudgetRows'])} rader "
                            f"med strategi `{best['Strategy']}` "
                            f"(balans mellan avkastning och chans till 12+ rätt)."
                        )
                        st.caption(
                            "Tolkning: högre budget ger ofta högre träffchans, men inte alltid bättre avkastning per krona."
                        )

                        if not bt_ablation:
                            chart_df = (
                                bt_numeric.groupby("BudgetRows", as_index=False)
                                .agg({"ROI": "mean", "Hit12PlusRate": "mean"})
                                .sort_values("BudgetRows")
                            )
                            st.markdown("##### Chans vs kostnad")
                            st.line_chart(
                                chart_df.set_index("BudgetRows")[["ROI", "Hit12PlusRate"]],
                                use_container_width=True,
                            )
                        else:
                            st.caption(
                                "Linjediagram döljs vid ablation (medel över profiler vore missvisande). "
                                "Sortera tabellen på ROI och jämför **FullHitRate** mot ROI per SignalProfile."
                            )

                        bt_show = bt.copy()
                        bt_show["ROI"] = (bt_show["ROI"] * 100).map(lambda v: f"{v:.1f}%")
                        bt_show["Hit10PlusRate"] = (bt_show["Hit10PlusRate"] * 100).map(lambda v: f"{v:.1f}%")
                        bt_show["Hit12PlusRate"] = (bt_show["Hit12PlusRate"] * 100).map(lambda v: f"{v:.1f}%")
                        if "FullHitRate" in bt_show.columns:
                            bt_show["FullHitRate"] = (bt_show["FullHitRate"] * 100).map(lambda v: f"{v:.2f}%")
                        st.dataframe(bt_show, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Kunde inte visa resultat: {exc}")
else:
    st.info("Tryck på knappen för att hämta aktuell kupong och få förslag.")
