"""
Synkar spelloggen till webbläsarens localStorage så historiken överlever Streamlit Cloud-omstarter.

Använder streamlit-js-eval + zlib + base64 (ingen manuell JSON-export krävs för normal användning).
"""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any, Dict, Optional

import streamlit as st

META_KEY = "europatipset_journal_meta_v1"
PAYLOAD_KEY = "europatipset_journal_payload_v1"
CHUNK_PREFIX = "europatipset_journal_c_v1_"
MAX_CHUNK = 450_000  # under vanliga URL/stränggränser i JS-motorn


def encode_journal_blob(data: Dict[str, Any]) -> str:
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    z = zlib.compress(raw, level=9)
    return base64.urlsafe_b64encode(z).decode("ascii")


def decode_journal_blob(blob: str) -> Dict[str, Any]:
    z = base64.urlsafe_b64decode(blob.encode("ascii"))
    raw = zlib.decompress(z).decode("utf-8")
    return json.loads(raw)


def _pull_js_expression() -> str:
    return f"""
(() => {{
  const metaRaw = localStorage.getItem("{META_KEY}");
  if (!metaRaw) return "";
  let meta;
  try {{ meta = JSON.parse(metaRaw); }} catch (e) {{ return ""; }}
  if (meta.mode === "single") {{
    return localStorage.getItem("{PAYLOAD_KEY}") || "";
  }}
  if (meta.mode === "chunk") {{
    let out = "";
    for (let i = 0; i < meta.n; i++) {{
      out += localStorage.getItem("{CHUNK_PREFIX}" + i) || "";
    }}
    return out;
  }}
  return "";
}})()
"""


def _push_js_expression(payload: str) -> str:
    if len(payload) <= MAX_CHUNK:
        return """(() => {{
  localStorage.setItem("{META}", {META_JSON});
  localStorage.setItem("{PAYLOAD}", {PAYLOAD_JSON});
  return "single";
}})()""".format(
            META=META_KEY,
            PAYLOAD=PAYLOAD_KEY,
            META_JSON=json.dumps(json.dumps({"mode": "single"})),
            PAYLOAD_JSON=json.dumps(payload),
        )
    chunks = [payload[i : i + MAX_CHUNK] for i in range(0, len(payload), MAX_CHUNK)]
    lines = [
        f'localStorage.setItem("{META_KEY}", {json.dumps(json.dumps({"mode": "chunk", "n": len(chunks)}))});'
    ]
    for idx, ch in enumerate(chunks):
        lines.append(f'localStorage.setItem("{CHUNK_PREFIX}{idx}", {json.dumps(ch)});')
    lines.append('return "chunk";')
    return "(() => {\n" + "\n".join(lines) + "\n})()\n"


def pull_journal_from_browser() -> Optional[Dict[str, Any]]:
    try:
        from streamlit_js_eval import streamlit_js_eval
    except Exception:
        return None
    try:
        raw = streamlit_js_eval(js_expressions=_pull_js_expression(), key="europatip_journal_ls_pull_v4")
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, str):
        blob = raw.strip()
        if not blob:
            return None
    else:
        return None
    try:
        return decode_journal_blob(blob)
    except Exception:
        return None


def push_journal_to_browser(data: Dict[str, Any]) -> None:
    try:
        from streamlit_js_eval import streamlit_js_eval
    except Exception:
        return
    try:
        blob = encode_journal_blob(data)
        expr = _push_js_expression(blob)
        digest = str(abs(hash(blob)))[-8:]
        streamlit_js_eval(js_expressions=expr, key=f"europatip_journal_ls_push_{digest}")
    except Exception:
        return


def merge_disk_and_browser_journal(path) -> Dict[str, Any]:
    from play_journal import load_journal, merge_journal_data, save_journal

    disk = load_journal(path)
    browser = pull_journal_from_browser()
    merged = merge_journal_data(disk, browser)
    try:
        sd = json.dumps(disk, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        sm = json.dumps(merged, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if sd != sm:
            save_journal(path, merged)
    except Exception:
        save_journal(path, merged)
    push_journal_to_browser(merged)
    return merged


def ensure_journal_merged_once_session(journal_path) -> None:
    """Kör en gång per Streamlit-session före första läsning av journal på disk."""
    if st.session_state.get("_eu_journal_browser_merged_v1"):
        return
    merge_disk_and_browser_journal(journal_path)
    st.session_state["_eu_journal_browser_merged_v1"] = True


def sync_journal_to_browser(data: Dict[str, Any]) -> None:
    """Anropa efter save_journal så webbläsaren speglar senaste loggen."""
    push_journal_to_browser(data)
