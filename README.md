# Optical Turbulence Estimation with Tabular Foundation Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Sukanta Basu, University at Albany  
**Last Updated:** March 1, 2026

---

## Associated Publication

This repository contains the full reproducible code for:

> S. Basu, "Leveraging deep learning-based foundation models for optical turbulence (Cn2) estimation under data scarcity," *Applied Optics*, in press. https://doi.org/10.1364/AO.585045

---

## Overview

The pipeline estimates the refractive index structure parameter (Cn2) using tabular foundation models (TabPFNv2, TabDPT) applied to tower-based meteorological observations. The central finding of the paper is that foundation models pretrained on entirely unrelated tabular datasets can effectively transfer to optical turbulence prediction tasks, even under severe data scarcity  without any task-specific fine-tuning or hyperparameter optimization.

The pipeline is validated using data from the **Mauna Loa Observatory ISFF tower campaign (2006)**, with June and August used for in-context learning (training) and July held out for testing.

---

## Repository Structure

```
├── Step1_Preprocess.py       # Data loading, despiking, feature engineering
├── Step2_TFM.py              # Foundation model training, ensemble prediction, SHAP analysis
├── Step3_TimeSeriesPlots.m   # 4-panel weekly time-series plot of observed vs. predicted Cn2 (Figure 1)
├── Step4_ScatterPlots.m      # Density-colored scatter plots of observed vs. predicted log₁₀(Cn2) (Figure 2)
├── DATA/                     # Raw NetCDF input files (not included; see Data section)
├── ExtractedDATA/            # Preprocessed CSV outputs from Step 1
└── FinalResults/             # Model predictions, R² summaries, and SHAP outputs
```

---

## Pipeline Summary

### Step 1 — Preprocessing and Feature Engineering (`Step1_Preprocess.py`)

Loads raw 5-minute meteorological and Cn2 observations from NetCDF files, applies quality control, derives physically motivated predictors, and saves a clean CSV for use in Step 2. Implements the data processing described in Section 3 of the paper.

Key operations:
- **Despiking** of tower temperature and wind components using z-score filtering (3σ threshold; Eq. 3 in the paper), with options for Local Outlier Factor and Isolation Forest
- **Potential temperature** calculation at 6 m, 15 m, and 25 m tower levels using standard thermodynamic relationship
- **Vertical gradient** computation (potential temperature gradient `dTHdz`, wind shear `S`) via a non-uniform centered finite difference scheme (Eq. 4 in the paper)
- **Cyclical feature encoding** for time of day, day of year, and wind direction using sine/cosine transformations (Eqs. 1 and 5 in the paper)

Output: `mauna_loa_processed_data.csv`

---

### Step 2 — Foundation Model Training and Analysis (`Step2_TFM.py`)

Trains TabPFNv2 or TabDPT regressors on variable-length training subsets (1 day to maximum available days) and evaluates predictive skill on the July test set. Reproduces the results reported in Section 4 of the paper.

Key operations:
- **Variable sample size sensitivity analysis**: systematically tests training lengths from 1 day upward in 288-sample (1-day) increments, reproducing Table 1 of the paper
- **Ensemble predictions**: 10 runs with independently shuffled training data; ensemble median is used as the final prediction, consistent with the methodology in Section 4 of the paper
- **Timing diagnostics**: records in-context learning (fit) and inference times separately
- **SHAP interpretability** (optional, `optSHAP = 1`): KernelExplainer-based main-effect and pairwise interaction analysis with stratified sampling (200 test samples, 50 background samples), reproducing Table 2 of the paper

Output files:
- `SampleSize_Sensitivity_<model>.csv` — R2 and timing statistics per training length (reproduces Table 1)
- `predictions_<model>.csv` — observed and predicted LCn2 for each training length tested
- `SHAP_1st_order_<model>.csv` — mean absolute SHAP values per feature (reproduces Table 2)
- `SHAP_2nd_order_<model>.csv` — pairwise interaction strengths
- `SHAP_values_<model>.pkl` — raw SHAP arrays for custom plotting

---

## Input Features

The following 10 variables are used as input features (Section 3E of the paper), with `LCn2_15m` as the prediction target:

| Feature | Description |
|---|---|
| `sinHR`, `cosHR` | Cyclical hour-of-day encoding |
| `P_2m` | Surface pressure at 2 m (hPa) |
| `T_2m` | Air temperature at 2 m (K) |
| `Tdew_2m` | Dew point temperature at 2 m (K) |
| `Spd_10m` | Wind speed at 10 m (m/s) |
| `sinWD`, `cosWD` | Cyclical wind direction encoding |
| `dTHdz_15m` | Potential temperature gradient at 15 m (K/m) |
| `S_15m` | Wind shear magnitude at 15 m (s⁻¹) |

**Target:** `LCn2_15m` — log₁₀(Cn2) at 15 m height

---

## Key Results (from the paper)

- With only **1 day** of training data, TabPFNv2 achieves R² = 0.666–0.682, already substantially outperforming TabDPT (R² = 0.419–0.550) in the few-shot regime
- With **18 days** of training data, TabPFNv2 reaches R² = 0.860–0.861
- TabPFNv2 performance plateaus around **10 days** (~2880 samples), demonstrating strong data efficiency
- In-context learning completes in under **0.3 s** and inference in under **1.25 s** on a single NVIDIA A6000 Ada GPU
- SHAP analysis identifies the **potential temperature gradient** (∂Θ/∂z|₁₅ₘ) as by far the dominant predictor (normalized importance = 0.378), consistent with established surface layer physics

---

## Requirements

```
numpy
pandas
netCDF4
scikit-learn
tabpfn          # TabPFNv2; used when optMod = 0
tabdpt          # TabDPT; used when optMod = 1
shap            # optional, for SHAP interpretability analysis
```

Install dependencies:

```bash
pip install numpy pandas netCDF4 scikit-learn tabpfn shap
```

---

## Configuration

In `Step2_TFM.py`, two flags control model selection and analysis scope:

```python
optMod  = 0    # 0 = TabPFNv2, 1 = TabDPT
optSHAP = 0    # 0 = skip SHAP analysis, 1 = run SHAP for longest training scenario
```

Update `ROOT_DIR` in both scripts to match your local data path before running.

---

## Data

Raw data are 5-minute NetCDF files from the NSF NCAR/EOL ISFS Mauna Loa Cn2 field campaign (June–August 2006). Files follow the naming convention `isff_2006MMDD.nc` and should be placed in `DATA/HAWAII2006/Mauna_Loa_ISFF/`. The dataset is publicly available at https://www.eol.ucar.edu/field_projects/mlocn2.

---

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{basu2026tfm,
  author  = {Sukanta Basu},
  title   = {Leveraging deep learning-based foundation models for optical
             turbulence ({C$_n^2$}) estimation under data scarcity},
  journal = {Applied Optics},
  volume  = {},
  number  = {},
  year    = {2026},
  doi     = {10.1364/AO.585045}
}
```

---

## AI Assistance

Claude AI (Anthropic) was used for documentation, code restructuring, and performance optimization.

---

## License

MIT License. Copyright (c) 2026 Sukanta Basu. See [LICENSE](LICENSE) for details.
