# Streamlit Cloud Deploy Checklist

## 1) Repository

- `main` branch contains `streamlit_app.py`
- `requirements.txt` is pinned
- `.streamlit/config.toml` exists

## 2) Streamlit app settings

- Repository: `Elli2022/europatipset-optimizer`
- Branch: `main`
- Main file path: `streamlit_app.py`

## 3) Secrets / environment variables

Add in Streamlit app **Settings -> Secrets**:

```toml
FOOTBALL_DATA_API_KEY="din_api_nyckel"
```

## 4) Post-deploy smoke tests

- Open app URL
- Click `Synka API-historik nu`
- Click `Hämta officiell kupong och beräkna förslag`
- Verify week/day/date header appears
- Verify recommendation table renders

## 5) Ongoing operations

- If sync fails, confirm API key is valid and free-tier quota is not exhausted
- Reboot app after major dependency updates
