# Climate Change Effects on Moroccan Coastal Upwelling (1978–2024)

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![ERA5](https://img.shields.io/badge/data-ERA5-red.svg)](https://cds.climate.copernicus.eu/)

This repository contains the analysis code, data processing scripts, and figures for the study:

> **"Effets du changement climatique sur l’upwelling côtier Marocain : Relations entre la NAO, l’indice d’upwelling et la température de surface de la mer (1978-2024)"**  
> *Mohammed El Abdirou Abdellah Msaadi, supervised by Rachid Ilmen*  
> École Hassania des Travaux Publics, Casablanca, Morocco – October 2025

The project investigates how climate change and atmospheric variability (North Atlantic Oscillation) affect the intensity and seasonality of coastal upwelling along the Moroccan Atlantic coast, using 42 years of reanalysis data.

## Table of Contents

- [Overview](#overview)
- [Key Questions](#key-questions)
- [Data Sources](#data-sources)
- [Methods](#methods)
- [Main Results](#main-results)

## Overview

The Moroccan Atlantic margin (≈3500 km) hosts one of the world’s most dynamic coastal upwelling systems, where cold, nutrient‑rich deep waters rise to the surface, supporting high biological productivity and local fisheries. Using ERA5 reanalysis data (1978–2024), we compute:

- **NAO index** (North Atlantic Oscillation) from sea‑level pressure differences (Azores – Iceland)
- **Coastal Upwelling Index (CUI)** from Ekman transport based on 10 m wind components
- **Sea Surface Temperature (SST)** trends and anomalies

We analyse seasonal and spatial variability, long‑term trends, breakpoints, and Granger causality to understand the ocean‑atmosphere interactions in a warming climate.

## Key Questions

1. How have the NAO, CUI, and SST evolved seasonally and interannually over 1978–2024?
2. What are the dominant correlations between these variables in different seasons?
3. Does the NAO drive upwelling intensity, or are local processes more important?
4. Can we detect regime shifts in upwelling or SST?
5. What are the socio‑economic implications for artisanal fisheries, offshore fishing, and aquaculture?

## Data Sources

All data come from the **ERA5 reanalysis** (ECMWF) accessed via the Copernicus Climate Data Store (CDS):

- **Variables**:
  - 10 m u/v wind components (for Ekman transport)
  - Mean sea level pressure (for NAO)
  - Sea surface temperature
- **Temporal coverage**: 1978 – 2024 (monthly means, 00:00 UTC)
- **Spatial resolution**: 0.25° × 0.25° grid (~28 km), covering the North Atlantic and the Moroccan coast (Dakhla to Tangier)
- **File format**: NetCDF4

## Methods

### Indices calculation

- **NAO**: normalized pressure difference between the Azores (37°N, 23°W) and Iceland (65°N, 22°W), referenced to 1981–2010.
- **CUI**: Ekman transport  
  `Mx = (ρ_air * Cd * |V| * u10) / (ρ_w * f)`  
  `My = (ρ_air * Cd * |V| * v10) / (ρ_w * f)`  
  `UI = -My * 100`  (m³/s per 100 m of coastline)
- **SST**: monthly means from ERA5.

### Statistical analyses

- **Seasonal aggregation**: DJF (winter), MAM (spring), JJA (summer), SON (autumn)
- **Correlation matrices** (Pearson) between NAO, CUI, and SST per season
- **Linear trend estimation** (per decade, significance at p < 0.05)
- **Change point detection** (binary segmentation algorithm, `ruptures` package)
- **Granger causality** (VAR models, optimal lags by AIC/BIC) to infer directional relationships

### Software & libraries

- **Python 3.9** + Jupyter Notebook
- `xarray`, `netCDF4` – NetCDF handling
- `numpy`, `pandas` – numerical / tabular data
- `scipy`, `statsmodels` – statistical tests, Granger causality
- `matplotlib`, `seaborn` – visualisation

## Main Results

### Seasonal relationships

| Season | NAO–CUI | CUI–SST | NAO–SST |
|--------|---------|---------|---------|
| DJF    | +0.65   | –0.41   | –0.37   |
| MAM    | +0.07   | –0.45   | –0.40   |
| JJA    | +0.06   | –0.46   | –0.25   |
| SON    | +0.32   | –0.57   | –0.25   |

- Strongest **NAO–upwelling coupling** in winter (DJF): a positive NAO intensifies trade winds and enhances upwelling.
- **Local upwelling forcing** dominates in summer (JJA) and autumn (SON), with CUI–SST correlations up to –0.57.
- **Spatial gradient**: quasi‑permanent upwelling in the south (Dakhla), seasonal peak in the centre (Safi–Essaouira), and stronger NAO teleconnection in the north (Casablanca–Tangier).

### Long‑term trends (1978–2024)

| Variable | Trend (units/decade) | Significance |
|----------|----------------------|--------------|
| SST (annual) | +0.0736 | p < 0.05 |
| SST (SON)    | +0.0846 | p < 0.05 |
| CUI (JJA)    | –0.0635 | p < 0.05 |
| NAO (all seasons) | +0.0214 | not significant |

- **Widespread warming** of the coastal ocean (+0.149 °C/decade on average, 98% of area warming).
- **Significant summer upwelling decline** (JJA), despite global warming.
- **Regional paradox**: warming coexists with localised cooling in intense upwelling cells (Bakun hypothesis partially supported).

### Granger causality

| Causal link | Season | Optimal lag | Direction |
|-------------|--------|-------------|-----------|
| CUI → SST   | JJA    | 1–3 months  | Local forcing |
| NAO → SST   | SON    | 2–4 months  | Teleconnection |
| SST → NAO   | SON    | 7–11 months | Ocean memory feedback |

The SST‑to‑NAO feedback (lag 7–11 months) confirms that oceanic heat anomalies in the Canary‑Moroccan region can influence the following winter’s NAO phase.


