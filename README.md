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

- Hämtar inte automatiskt veckans Europatipset-kupong från Svenska Spel (det varierar beroende på tillgänglighet och åtkomst).
- `build-coupon` hämtar kommande matcher med odds från football-data.co.uk, inte exakt officiell Europatipset-kupong.
- `build-official-coupon` läser officiell kupong, odds och streck från Svenska Spels statistik-sida.
- Historiken kommer från europeiska ligor (football-data.co.uk), inte enbart Europatipset.
- Målet i optimeringen är bra sannolikhetstäckning under radbudget, inte direkt utdelningssimulering.

## Nästa steg

- Automatisk import av aktuell kupong (matcher, odds, streck).
- Monte Carlo-simulering av utdelning.
- Reduceringsregler (spikar utifrån värde, krav på antal skrällar m.m.).
- En enkel webb-UI.
