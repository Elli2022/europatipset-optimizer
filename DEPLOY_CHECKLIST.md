# Streamlit Cloud Deploy Checklist

## 1) Repository

- `main` branch contains `streamlit_app.py`
- `requirements.txt` is pinned
- `.streamlit/config.toml` exists

## 2) Streamlit app settings

- Repository: `Elli2022/europatipset-optimizer`
- Branch: `main`
- Main file path: `streamlit_app.py`

## 3) Secrets (API-nyckel – var klistrar du in den?)

På **Streamlit Community Cloud**:

1. Öppna din deployade app på `share.streamlit.io`
2. Klicka på **⋮** (tre prickar) uppe till höger → **Manage app** (eller liknande meny)
3. Välj fliken **Secrets** (ibland under **Settings**)
4. Klistra in:

```toml
FOOTBALL_DATA_API_KEY = "din_nyckel_fran_football_data_org"
```

5. Spara och välj **Reboot app** om den inte startar om själv

**OBS:** Nyckeln behövs för knappen *Synka API-historik*. Den används **inte** för att skapa `calibration.pkl` – modellen hämtar historik från football-data.co.uk vid första start.

Valfritt (färre säsonger = snabbare första träning på Cloud):

```toml
BOOTSTRAP_HISTORY_YEARS = "3"
```

## 4) Post-deploy smoke tests

- Open app URL — första gången kan sidan visa spinner **1–4 min** medan modellen tränas
- Click `Synka API-historik nu`
- Click `Hämta officiell kupong och beräkna förslag`
- Verify week/day/date header appears
- Verify recommendation table renders

## 5) Ongoing operations

- If sync fails, confirm API key is valid and free-tier quota is not exhausted
- Reboot app after major dependency updates

## 6) Ephemeral disk (Mina spel)

Spelloggen (`data/user/play_journal.json`) sparas på serverns lokala disk. På Streamlit Community Cloud kan den **nollställas** vid cold start eller omstart. Använd **Exportera spellogg (JSON)** i appen om du behöver beständighet, eller kör verktyget lokalt med committad/backupad `data/user/`.
