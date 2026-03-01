# Copyright (c) 2026 Sukanta Basu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
File: Step2_TFM.py
==================
:Author: Sukanta Basu (University at Albany)
:Date: March 1, 2026
:Description: Variable training sample size sensitivity analysis using
    tabular foundation models (TabPFNv2 or TabDPT) for Cn2 prediction at
    the Mauna Loa Observatory 15 m tower level.

Associated Publication:
-----------------------
S. Basu, "Leveraging deep learning-based foundation models for optical
turbulence (Cn²) estimation under data scarcity," Applied Optics,
https://doi.org/10.1364/AO.585045

This script implements the modeling experiments and interpretability
analysis described in Section 4 of the paper (Sections 4A–4D), and
reproduces Tables 1 and 2 and Figures 1 and 2.

Overall Strategy:
-----------------
Step 2a — Data preparation (Section 3F of the paper):
    Load the preprocessed CSV from Step 1. Partition into training months
    (June + August) and a held-out test set (July, comprising 7731 samples).
    Drop rows with any missing feature or target values.

Step 2b — Variable sample size loop (Section 4C, Table 1 of the paper):
    Systematically subset the training data in increments of 288 samples
    (one day of 5-minute observations), from 1 day up to the maximum
    available training days (~18 days). For each subset size, fit an
    ensemble of TabPFNv2 (or TabDPT) regressors on independently shuffled
    draws from the training pool. Training data are added sequentially from
    the start of the training period to avoid temporal leakage.

Step 2c — Ensemble prediction and evaluation (Section 4A–4C of the paper):
    Predict log10(Cn2) on the July test set for each ensemble member.
    Compute ensemble median and mean predictions. Evaluate R2 for median,
    mean, and each individual member. Record in-context learning (fit) and
    inference times per repeat. The ensemble median across 10 members is
    used as the final reported prediction.

Step 2d — SHAP interpretability (optional; Section 4D, Table 2 of the paper):
    For the longest training scenario only, run KernelExplainer-based SHAP
    analysis across all ensemble models. Use 50 background samples and
    200 test samples for computational efficiency. Compute first-order
    (mean absolute SHAP) and second-order (pairwise interaction) feature
    importances aggregated across all ensemble members.

Configuration:
--------------
    optMod  = 0    # 0 = TabPFNv2 (paper primary model), 1 = TabDPT
    optSHAP = 0    # 0 = skip SHAP analysis, 1 = run SHAP (reproduces Table 2)

Output Files:
-------------
SampleSize_Sensitivity_<model>.csv — R2 statistics and timing per training
    length; reproduces Table 1 of the paper. Columns: multiplier,
    actual_days, actual_sample_size, n_models, r2_median, r2_mean,
    r2_min, r2_max, icl_time_mean, icl_time_std, inference_time_mean,
    inference_time_std.
predictions_<model>.csv — date, observed LCn2_15m, and ensemble median
    predictions for each training length tested; used for Figures 1 and 2.
SHAP_1st_order_<model>.csv — mean absolute SHAP value per feature;
    reproduces Table 2 of the paper.
SHAP_2nd_order_<model>.csv — pairwise SHAP interaction strengths.
SHAP_values_<model>.pkl — raw SHAP arrays and metadata for custom plotting.

AI Assistance: Claude AI (Anthropic) was used for documentation, code
    restructuring, and performance optimization.
"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import pickle
import os
import time

optMod = 0
optSHAP = 0

if optMod == 0:
    import tabpfn
    from tabpfn import TabPFNRegressor
else:
    from tabdpt import TabDPTRegressor


# =============================================================================
# DIRECTORIES AND CONFIGURATION
# =============================================================================

ROOT_DIR = "/data/Sukanta/Works_Ongoing/2025_HawaiiCn2_TabPFN/"
INPUT_DIR = ROOT_DIR + "DATA/HAWAII2006/Mauna_Loa_ISFF/"
OUTPUT_DIR = ROOT_DIR + "ExtractedDATA/Input/OBS/"
RESULTS_DIR = ROOT_DIR + "FinalResults/"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training sample sizes to test (in multiples of 288 - one day of 5-min data)
BASE_SAMPLES = 288  # 5-minute data: 288 samples per day
N_REPEATS = 1  # Number of ensemble predictions with shuffled training data

# Feature and target columns
FEATURE_COLS = ['sinHR', 'cosHR',
                'P_2m', 'T_2m', 'Tdew_2m',
                'Spd_10m', 'sinWD', 'cosWD',
                'dTHdz_15m', 'S_15m']
TARGET_COLS = ['LCn2_15m']

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

print("Loading and preparing data...")
df = pd.read_csv(OUTPUT_DIR + "mauna_loa_processed_data.csv")

# Convert TIME column to datetime for time series plotting
df['TIME'] = pd.to_datetime(df['TIME'])

# Filter and clean data
df_june = df.query('MONTH == 6').dropna()
df_july = df.query('MONTH == 7').dropna()
df_august = df.query('MONTH == 8').dropna()

print(f"June data: {len(df_june)} clean samples")
print(f"July data: {len(df_july)} clean samples")
print(f"August data: {len(df_august)} clean samples")

# Concatenate DataFrames
df_Trn = pd.concat([df_june, df_august], ignore_index=True)
df_Tst = df_july

# Calculate maximum possible training days and create sample multipliers
max_possible_days = len(df_Trn) // BASE_SAMPLES
print(f"Maximum possible training days: {max_possible_days}")

# Create sample multipliers from 1 day up to maximum possible days
SAMPLE_MULTIPLIERS = list(range(1, max_possible_days + 1))
print(f"Will test training sample sizes: {SAMPLE_MULTIPLIERS} days")

# Prepare test data (July - remains constant across all experiments)
XTst = df_Tst[FEATURE_COLS].values
yTst = df_Tst[TARGET_COLS].values.ravel()

# Determine model name based on optMod (needed for file naming)
model_name = 'TabPFN' if optMod == 0 else 'TabDPT'

# Initialize predictions DataFrame with date and observed columns
predictions_df = pd.DataFrame({
    'date': df_Tst['TIME'].values,
    'observed': yTst
})


# =============================================================================
# ANALYSIS LOOP - VARIABLE TRAINING SAMPLE SIZES
# =============================================================================

# Set random seed for reproducible shuffling
np.random.seed(42)

# Initialize list to store results for DataFrame
results_list = []

for multiplier in SAMPLE_MULTIPLIERS:
    sample_size = multiplier * BASE_SAMPLES

    # Ensure we don't exceed available data
    max_available_samples = len(df_Trn)
    actual_sample_size = min(sample_size, max_available_samples)
    actual_days = actual_sample_size / BASE_SAMPLES

    print(
        f"\n--- Training with {actual_sample_size} samples ({actual_days:.1f} days) ---")

    # Prepare training data with limited sample size
    XTrn = df_Trn[FEATURE_COLS].values[:actual_sample_size, :]
    yTrn = df_Trn[TARGET_COLS].values.ravel()[:actual_sample_size]

    print(f"Training shape: X={XTrn.shape}, y={yTrn.shape}")

    # -------------------------------------------------------------------------
    # ENSEMBLE PREDICTIONS WITH SHUFFLED TRAINING DATA
    # -------------------------------------------------------------------------

    print(f"Making {N_REPEATS} ensemble predictions...")
    all_predictions = []
    all_models = []
    icl_times = []  # In-Context Learning (fit) times
    inference_times = []  # Prediction times

    for repeat in range(N_REPEATS):
        # Shuffle training data
        shuffle_indices = np.random.permutation(len(XTrn))
        XTrn_shuffled = XTrn[shuffle_indices]
        yTrn_shuffled = yTrn[shuffle_indices]

        try:
            if optMod == 0:
                regressor = TabPFNRegressor()
            else:
                regressor = TabDPTRegressor()

            # Time the ICL (fitting) phase
            icl_start = time.time()
            regressor.fit(XTrn_shuffled, yTrn_shuffled)
            icl_end = time.time()
            icl_time = icl_end - icl_start
            icl_times.append(icl_time)

            # Time the inference (prediction) phase
            inference_start = time.time()
            yPred_repeat = regressor.predict(XTst)
            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_times.append(inference_time)

            all_predictions.append(yPred_repeat)
            all_models.append(regressor)

            print(
                f"  Completed prediction {repeat + 1}/{N_REPEATS} - ICL: {icl_time:.3f}s, Inference: {inference_time:.3f}s")

        except Exception as e:
            print(f"  Error in prediction {repeat + 1}/{N_REPEATS}: {e}")
            # Skip this iteration if fitting fails
            continue

    # Check if we have any successful predictions
    if len(all_predictions) == 0:
        print("No successful predictions - skipping this sample size")
        continue

    # Convert to numpy array for easier manipulation
    all_predictions = np.array(
        all_predictions)  # Shape: (N_successful_repeats, n_test_samples)
    print('shape of all_predictions:', all_predictions.shape)

    # Calculate ensemble statistics
    yPred_median = np.median(all_predictions, axis=0)
    yPred_mean = np.mean(all_predictions, axis=0)

    # Calculate R² scores for different ensemble methods
    r2_median = r2_score(yTst, yPred_median)
    r2_mean = r2_score(yTst, yPred_mean)
    r2_individual = [r2_score(yTst, pred) for pred in all_predictions]

    print(f"R² scores - Median: {r2_median:.4f}, Mean: {r2_mean:.4f}")
    print(
        f"Individual R² range: {min(r2_individual):.4f} - {max(r2_individual):.4f}")

    # Use median for main analysis
    yPred = yPred_median
    r2 = r2_median

    # -------------------------------------------------------------------------
    # SHAP ANALYSIS (ONLY FOR LONGEST TRAINING DATA)
    # -------------------------------------------------------------------------
    if (multiplier == max(SAMPLE_MULTIPLIERS)) and (optMod == 0) and (optSHAP == 1):
        print("\n*** Running SHAP analysis for longest training scenario ***")

        try:
            import shap

            # Run SHAP for each model in the ensemble
            all_shap_values = []

            for model_idx, model in enumerate(all_models):
                print(
                    f"  Computing SHAP values for model {model_idx + 1}/{len(all_models)}...")

                shap_start = time.time()

                # Create SHAP explainer
                # Use a subset of training data as background for faster computation
                background_size = min(50, len(XTrn))
                background_indices = np.random.choice(len(XTrn),
                                                      background_size,
                                                      replace=False)
                background_data = XTrn[background_indices]

                # KernelExplainer is model-agnostic and works with any model
                explainer = shap.KernelExplainer(model.predict,
                                                 background_data)

                # Compute SHAP values for test set
                # Use a subset for computational efficiency if needed
                n_samples_to_explain = min(200, len(XTst))
                sample_indices = np.random.choice(len(XTst),
                                                  n_samples_to_explain,
                                                  replace=False)
                XTst_subset = XTst[sample_indices]

                print(
                    f"    Explaining {n_samples_to_explain} test samples with {background_size} background samples...")
                shap_values = explainer.shap_values(XTst_subset)

                shap_end = time.time()
                print(
                    f"    SHAP computation time: {shap_end - shap_start:.2f}s")

                all_shap_values.append(shap_values)

            # Aggregate SHAP results across models and samples
            print("  Aggregating SHAP results...")

            # Concatenate all SHAP values from all models
            all_shap_values = np.concatenate(all_shap_values, axis=0)

            # Extract feature names
            feature_names = FEATURE_COLS

            # Calculate 1st order: Mean absolute SHAP values for each feature
            shapley_1st_order = {}
            for i, feat in enumerate(feature_names):
                shapley_1st_order[feat] = np.mean(
                    np.abs(all_shap_values[:, i]))

            # Calculate 2nd order: Pairwise interaction strength
            # Use correlation of SHAP values as a proxy for interaction strength
            shapley_2nd_order = {}
            for i, feat_i in enumerate(feature_names):
                for j, feat_j in enumerate(feature_names):
                    if i < j:  # Only upper triangle to avoid duplicates
                        pair_name = f"{feat_i} x {feat_j}"
                        # Interaction strength as correlation of SHAP values
                        correlation = np.abs(np.corrcoef(
                            all_shap_values[:, i],
                            all_shap_values[:, j]
                        )[0, 1])
                        # Multiply by product of importances for interaction magnitude
                        interaction = correlation * shapley_1st_order[feat_i] * \
                                      shapley_1st_order[feat_j]
                        shapley_2nd_order[pair_name] = interaction

            # Create DataFrames
            shapley_1st_df = pd.DataFrame([
                {'feature': k, 'importance': v, 'order': 1}
                for k, v in shapley_1st_order.items()
            ]).sort_values('importance', ascending=False)

            shapley_2nd_df = pd.DataFrame([
                {'feature_pair': k, 'interaction_strength': v, 'order': 2}
                for k, v in shapley_2nd_order.items()
            ]).sort_values('interaction_strength', ascending=False)

            # Save SHAP results
            shapley_1st_csv = f'{RESULTS_DIR}SHAP_1st_order_{model_name}.csv'
            shapley_2nd_csv = f'{RESULTS_DIR}SHAP_2nd_order_{model_name}.csv'

            shapley_1st_df.to_csv(shapley_1st_csv, index=False)
            shapley_2nd_df.to_csv(shapley_2nd_csv, index=False)

            print(f"  SHAP 1st order saved to: {shapley_1st_csv}")
            print(f"  SHAP 2nd order saved to: {shapley_2nd_csv}")

            # Save raw SHAP values for future plotting
            shap_values_file = f'{RESULTS_DIR}SHAP_values_{model_name}.pkl'
            with open(shap_values_file, 'wb') as f:
                pickle.dump({
                    'shap_values': all_shap_values,
                    'feature_names': feature_names,
                    'X_explained': XTst_subset
                }, f)
            print(f"  Raw SHAP values saved to: {shap_values_file}")

            # Display top results
            print("\n  Top 5 Main Effects (1st order):")
            print(shapley_1st_df.head().to_string(index=False))
            print("\n  Top 5 Interactions (2nd order):")
            print(shapley_2nd_df.head().to_string(index=False))

        except ImportError as ie:
            print(
                f"  Warning: shap library not found. Install with: pip install shap")
            print(f"  Import error details: {ie}")
        except Exception as e:
            print(f"  Error in SHAP analysis: {e}")
            import traceback

            print("  Full traceback:")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # STORE RESULTS FOR DATAFRAME
    # -------------------------------------------------------------------------

    result_row = {
        'multiplier': multiplier,
        'actual_days': actual_days,
        'actual_sample_size': actual_sample_size,
        'n_models': len(all_models),
        'r2_median': r2_median,
        'r2_mean': r2_mean,
        'r2_min': min(r2_individual),
        'r2_max': max(r2_individual),
        'icl_time_mean': np.mean(icl_times),
        'icl_time_std': np.std(icl_times),
        'inference_time_mean': np.mean(inference_times),
        'inference_time_std': np.std(inference_times),
        'optMod': optMod,
        'n_repeats': N_REPEATS
    }

    results_list.append(result_row)

    # Add predictions as a column to predictions DataFrame
    column_name = f'pred_{int(actual_days)}days'
    predictions_df[column_name] = yPred_median
    print(f"Added predictions column: {column_name}")

# =============================================================================
# SAVE RESULTS TO DATAFRAME
# =============================================================================

# Convert results list to DataFrame
results_df = pd.DataFrame(results_list)

# Save results summary to CSV with model-specific filename
results_csv = f'{RESULTS_DIR}SampleSize_Sensitivity_{model_name}.csv'
results_df.to_csv(results_csv, index=False)
print(f"\nResults DataFrame saved to: {results_csv}")

# Save predictions DataFrame with model-specific filename
predictions_csv = f'{RESULTS_DIR}predictions_{model_name}.csv'
predictions_df.to_csv(predictions_csv, index=False)
print(f"Predictions DataFrame saved to: {predictions_csv}")

# Display summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))
print("\n" + "=" * 60)
print("PREDICTIONS DATAFRAME SHAPE")
print("=" * 60)
print(f"Shape: {predictions_df.shape}")
print(f"Columns: {list(predictions_df.columns)}")

# =============================================================================
# SUMMARY ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Tested training sample sizes from 1 to {max_possible_days} days")
if optMod == 0:
    print(
        f"Used ensemble of {N_REPEATS} TabPFN models with shuffled training data")
else:
    print(
        f"Used ensemble of {N_REPEATS} TabDPT models with shuffled training "
        f"data")
print(
    f"Total sample sizes: {[int(m * BASE_SAMPLES) for m in SAMPLE_MULTIPLIERS]}")
print(f"\nOutput files saved to: {RESULTS_DIR}")
print(f"  1. {results_csv}")
print(
    f"     - Columns: multiplier, actual_days, actual_sample_size, n_models,")
print(f"                r2_median, r2_mean, r2_min, r2_max,")
print(
    f"                icl_time_mean, icl_time_std, inference_time_mean, inference_time_std")
print(f"  2. {predictions_csv}")
print(
    f"     - Columns: date, observed, pred_1days, pred_2days, ..., pred_{max_possible_days}days")
print(f"  3. SHAP_1st_order_{model_name}.csv (feature importance)")
print(f"  4. SHAP_2nd_order_{model_name}.csv (pairwise interactions)")
print(f"  5. SHAP_values_{model_name}.pkl (raw SHAP values for plotting)")
print(
    f"\nSHAP analysis performed for longest training scenario ({max_possible_days} days)")
print("=" * 60)