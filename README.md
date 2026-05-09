# Europatipset Optimizer (MVP)

En Python-app som:

1. Laddar ner historisk fotbollsdata och odds från webben.
2. Tränar en kalibreringsmodell för 1X2-sannolikheter.
3. Tar en Europatipset-kupong och föreslår ett system under en radbudget.

## Varför detta upplägg?

- Marknadsodds är normalt den starkaste baseline-signalen.
- Historik används för att kalibrera rå odds-sannolikheter till bättre träffsäkerhet.
- Optimeringen prioriterar tecken/garderingar som ger bäst täckning per extra systemrad.

## Kom igång

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Hämta historik från nätet

```bash
python europatipset.py download --out data/raw/history.csv --years 6
```

## 1b) Synka kontinuerlig historik från gratis API

Vi använder `football-data.org` (gratisnivå). Skapa API-nyckel och sätt miljövariabel:

```bash
export FOOTBALL_DATA_API_KEY="din_nyckel"
python europatipset.py sync-free-api-history --out data/raw/history_api.csv --days-back 120
```

Tips: kör detta dagligen/veckovis för kontinuerlig uppdatering.

Exempel lokal dagskörning (cron):

```bash
chmod +x scripts/sync_history.sh
crontab -e
```

Lägg till rad:

```bash
0 6 * * * /Users/elli/europatipset-optimizer/scripts/sync_history.sh >> /Users/elli/europatipset-optimizer/data/sync.log 2>&1
```

Smart synk av officiell kupong (ny omgång direkt när `draw_number` ändrats, annars högst var 15:e minut):

```bash
chmod +x scripts/refresh_before_stop.sh
crontab -e
```

Lägg till rad (var 15:e minut):

```bash
*/15 * * * * /Users/elli/europatipset-optimizer/scripts/refresh_before_stop.sh >> /Users/elli/europatipset-optimizer/data/refresh.log 2>&1
```

CLI motsvarighet:

```bash
python europatipset.py sync-official-smart \
  --coupon-out data/input/official_coupon.csv \
  --meta-out data/input/official_meta.json \
  --min-interval-minutes 15
```

Kommandot `auto-refresh-official` finns kvar om du uttryckligen vill begränsa uppdateringar till fönstret nära spelstopp.

## 2) Träna modell

```bash
python europatipset.py train --history data/raw/history.csv --model data/models/calibration.pkl
```

## 3) Kör systemförslag på kupong

Utgå från `coupon_template.csv` och fyll med aktuell kupong.

```bash
python europatipset.py recommend \
  --coupon coupon_template.csv \
  --model data/models/calibration.pkl \
  --max-rows 64 \
  --game-type europatipset \
  --out data/output/recommendation.csv
```

## Alternativ: bygg kupong automatiskt från nätet

```bash
python europatipset.py build-coupon --out data/input/auto_coupon.csv --n-matches 13
python europatipset.py recommend \
  --coupon data/input/auto_coupon.csv \
  --model data/models/calibration.pkl \
  --max-rows 64 \
  --out data/output/recommendation_auto.csv
```

## Officiell kupong + streck från Svenska Spel

```bash
python europatipset.py build-official-coupon --out data/input/official_coupon.csv
python europatipset.py recommend \
  --coupon data/input/official_coupon.csv \
  --model data/models/calibration.pkl \
  --max-rows 64 \
  --out data/output/recommendation_official.csv
```

## Webb-UI (användarvänligt)

Starta appen:

```bash
streamlit run streamlit_app.py
```

I UI:t kan du:

- hämta officiell kupong och streck med en knapp
- välja radbudget och strategi (balanserad/säker/värde)
- se systemförslag i tabell
- jämföra flera budgetscenarier
- göra manuell "vad-om"-redigering av kupong
- synka gratis API-historik direkt i UI
- se status för senaste historiksynk
- se tydligt vilken vecka, veckodag och datum omgången gäller
- räkna uppskattad utdelning/netto per rättnivå i utdelningskalkyl
- se datakvalitetsvarningar för odds/streck
- se prognoskonfidens (hög/medel/låg)
- köra historiskt backtest (ROI + träffnivåer)
- välja omgångstyp (`europatipset`/`topptipset`)
- i **Matchanalys**: gratis **ligatabell + form** för matcher som går att koppla till football-data.orgs free-tier ligor (via API-nyckel), kompletterat med form från lokal API-historik — inte skador/elvor
- spara spelade kuponger under **Mina spel**, rätta med antal rätt och valfri 13-teckensrad för enkel lärdom

**Streamlit Cloud:** serverdisk kan vara **tillfällig**, men **Mina spel** speglas automatiskt till din webbläsares **localStorage** (ingen JSON-export krävs i normalfallet). Byter du webbläsare eller rensar du sajtdatum kan historiken ändå försvinna — då är JSON-export en bra backup.

## Deploy

Se `DEPLOY_CHECKLIST.md` för snabb och stabil Streamlit Community Cloud-deploy.

## Backtest via CLI

```bash
python europatipset.py backtest \
  --history data/raw/history.csv \
  --model data/models/calibration.pkl \
  --out data/output/backtest.csv \
  --budgets 32,64,128 \
  --strategies balanced,safe,value \
  --game-type europatipset \
  --n-coupons 50
```

## Test

```bash
pytest
```

## Kupongformat (CSV)

Obligatoriska kolumner:

- `Match`
- `Odd1`
- `OddX`
- `Odd2`

Valfria kolumner (streckfördelning):

- `Streck1`
- `StreckX`
- `Streck2`

`Streck*` kan anges som `52` eller `0.52`. Om de saknas antas 33.33% vardera.

## Begränsningar i MVP

- `build-coupon` hämtar kommande matcher med odds från football-data.co.uk, inte exakt officiell Europatipset-kupong.
- `build-official-coupon` läser officiell kupong, odds och streck från Svenska Spels statistik-sida.
- Historiken kommer från europeiska ligor (football-data.co.uk), inte enbart Europatipset.
- `sync-free-api-history` hämtar resultat-historik från football-data.org (gratis API), men innehåller inte bookmaker-odds.
- Målet i optimeringen är bra sannolikhetstäckning under radbudget, inte direkt utdelningssimulering.

## Nästa steg

- Automatisk import av aktuell kupong (matcher, odds, streck).
- Monte Carlo-simulering av utdelning.
- Reduceringsregler (spikar utifrån värde, krav på antal skrällar m.m.).
- En enkel webb-UI.
