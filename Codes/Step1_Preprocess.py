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
File: Step1_Preprocess.py
==========================
:Author: Sukanta Basu (University at Albany)
:Date: March 1, 2026
:Description: Data loading, quality control, and feature engineering for
    Mauna Loa ISFF tower observations (June–August 2006).

Associated Publication:
-----------------------
S. Basu, "Leveraging deep learning-based foundation models for optical
turbulence (Cn2) estimation under data scarcity," Applied Optics,
https://doi.org/10.1364/AO.585045

This script implements the data preprocessing and feature engineering
described in Section 3 of the paper (Sections 3A–3E).

Overall Strategy:
-----------------
Step 1a — Raw data loading:
    Read 5-minute NetCDF files for each day in the campaign period
    (June–August 2006). Extract Cn2 at 6, 15, and 25 m tower levels
    (converted to log10 per Eq. 2 of the paper), surface meteorology
    (pressure, temperature, dew point, wind speed and direction), and
    tower-level wind components and temperatures.

Step 1b — Despiking (Section 3B, Eq. 3 of the paper):
    Apply z-score quality control (3σ threshold) to raw tower temperatures
    (T_6m, T_15m, T_25m) and wind components (u, v at 6, 15, 25 m).
    Alternative methods (Local Outlier Factor, Isolation Forest) are also
    implemented but not used by default.

Step 1c — Derived physical quantities (Section 3C, Eq. 4 of the paper):
    Compute potential temperature (TH) at each tower level from despiked
    temperatures using the barometric formula and surface pressure.
    Compute vertical gradients (dTHdz, dUdz, dVdz) at the 15 m midpoint
    using a non-uniform centered finite difference scheme, and combine wind
    component gradients into scalar wind shear magnitude (S_15m).

Step 1d — Cyclical feature encoding (Section 3D, Eqs. 1 and 5 of the paper):
    Encode time of day, day of year, and wind direction as sine/cosine pairs
    to preserve their circular continuity for machine learning models.

Output:
-------
mauna_loa_processed_data.csv — cleaned and feature-enriched dataset ready
    for use in Step 2. Contains the 10 input features and LCn2_15m target
    described in Section 3E of the paper.

AI Assistance: Claude AI (Anthropic) was used for documentation, code
    restructuring, and performance optimization.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4
from pathlib import Path
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


# =============================================================================
# INPUT & OUTPUT DIRECTORIES
# =============================================================================

ROOT_DIR = "/data/Sukanta/Works_Ongoing/2025_HawaiiCn2_TabPFN/"
INPUT_DIR = ROOT_DIR + "DATA/HAWAII2006/Mauna_Loa_ISFF/"
OUTPUT_DIR = ROOT_DIR + "ExtractedDATA/Input/OBS/"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def filter_spike(y, method=0):
    """
    Despike time series using different methods.

    Parameters:
    -----------
    y : array-like
        Input time series
    method : int
        0: Z-score (3σ threshold)
        1: Local Outlier Factor
        2: Isolation Forest

    Returns:
    --------
    y_filtered : array
        Filtered time series with outliers replaced by NaN
    """
    # Convert to numeric array, forcing non-numeric values to NaN
    y = pd.to_numeric(y, errors='coerce')
    y = np.asarray(y, dtype=float)

    # Check if all values are NaN
    if np.all(np.isnan(y)):
        return y

    # Mask out NaNs for fitting
    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask].reshape(-1, 1)
    final_mask = np.full_like(y, False, dtype=bool)

    # Check if we have enough valid data points
    if len(y_valid) < 2:
        return y

    if method == 0:
        scaler = StandardScaler()
        z = scaler.fit_transform(y_valid)
        outlier_mask = np.abs(z) < 3  # keep values within 3σ
        final_mask[valid_mask] = outlier_mask.ravel()

    elif method == 1:
        lof = LocalOutlierFactor(n_neighbors=10)
        pred = lof.fit_predict(y_valid)
        final_mask[valid_mask] = pred == 1  # keep inliers

    elif method == 2:
        iso = IsolationForest()
        pred = iso.fit_predict(y_valid)
        final_mask[valid_mask] = pred == 1  # keep inliers

    # Replace outliers with NaN
    y_filtered = np.where(final_mask, y, np.nan)
    return y_filtered


# -----------------------------------------------------------------------------
def gradient_nonuniform(x, y):
    """
    Calculate gradient dy/dx for non-uniformly spaced x.

    Parameters:
    -----------
    x : array-like
        1D array of x values (non-uniform spacing allowed)
    y : array-like
        1D array of y values, same length as x

    Returns:
    --------
    dydx : ndarray
        Gradient approximation at interior points (length N-2)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    f1 = x[1:-1] - x[:-2]  # x(i)   - x(i-1)
    f2 = x[2:] - x[1:-1]  # x(i+1) - x(i)
    a = f2 / f1

    num = y[2:] + (a ** 2 - 1) * y[1:-1] - (a ** 2) * y[:-2]
    den = a * (a + 1) * f1

    dydx = num / den
    return dydx


# -----------------------------------------------------------------------------
def calculate_pressure_at_height(p_surface, t_surface, t_height, height_diff):
    """Calculate pressure at different height using barometric formula."""
    return p_surface * np.exp(
        -height_diff * 9.81 / 287 / (0.5 * (t_surface + t_height)))


# -----------------------------------------------------------------------------
def calculate_potential_temperature(temperature, pressure):
    """Calculate potential temperature."""
    return temperature * (1000 / pressure) ** 0.286


# =============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# =============================================================================

def load_and_process_data():
    """Load and process meteorological and Cn2 data from NetCDF files."""

    # Column names for the combined dataset
    columns = [
        'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND',
        'LCn2_6m', 'LCn2_15m', 'LCn2_25m',
        'Spd_10m', 'Dir_10m', 'P_2m', 'T_2m', 'Tdew_2m',
        'T_6m', 'T_15m', 'T_25m',  # Changed from T_06m to T_6m
        'u_6m', 'u_15m', 'u_25m',  # Changed from u_06m to u_6m
        'v_6m', 'v_15m', 'v_25m'   # Changed from v_06m to v_6m
    ]

    all_data = []

    # Loop through months and days
    for month in range(6, 9):  # June to August
        for day in range(1, 32):  # 1 to 31
            filename = f'isff_2006{month:02d}{day:02d}.nc'
            filepath = Path(INPUT_DIR + filename)

            if filepath.is_file():
                print(f"Processing {filename}")

                with netCDF4.Dataset(str(filepath), 'r') as dataset:
                    # Extract time and Cn2 data
                    time_data = np.squeeze(dataset.variables['time'][:])
                    cn2_vars = ['Cn2_6m', 'Cn2_15m', 'Cn2_25m']
                    misc_vars = ['Spd_10m', 'Dir_10m', 'P_2m', 'T_2m',
                                'Tdew_2m']
                    wind_vars = ['u_6m', 'u_15m', 'u_25m', 'v_6m', 'v_15m',
                                 'v_25m']
                    temp_vars = ['tc_6m', 'tc_15m', 'tc_25m']

                    # Create daily dataframe
                    n_records = len(time_data)
                    daily_data = pd.DataFrame(index=range(n_records),
                                              columns=columns)

                    # Fill time columns
                    daily_data['YEAR'] = 2006
                    daily_data['MONTH'] = month
                    daily_data['DAY'] = day
                    daily_data['HOUR'] = 0
                    daily_data['MINUTE'] = 0
                    daily_data['SECOND'] = time_data

                    # Extract and process Cn2 data (convert to log10)
                    for i, var in enumerate(cn2_vars):
                        data = np.ma.filled(dataset.variables[var][:],
                                            fill_value=np.nan)
                        # Use consistent naming without leading zeros
                        height = var.split("_")[1]  # This gives '6m', '15m', '25m'
                        daily_data[f'LCn2_{height}'] = np.log10(data)

                    # Extract meteorological data
                    for var in misc_vars:
                        data = np.ma.filled(dataset.variables[var][:],
                                            fill_value=np.nan)
                        daily_data[var] = data

                        # Convert temperatures from C to K
                        if var in ['T_2m', 'Tdew_2m']:
                            daily_data[var] += 273.16

                    # Extract wind data at towers
                    for var in wind_vars:
                        data = np.ma.filled(dataset.variables[var][:],
                                            fill_value=np.nan)
                        daily_data[var] = data

                    # Extract temperature data at towers
                    for var in temp_vars:
                        data = np.ma.filled(dataset.variables[var][:],
                                            fill_value=np.nan)
                        # Store as raw temperature for despiking
                        height = var.split('_')[1]  # This gives '6m', '15m', '25m'
                        daily_data[f'T_{height}'] = data

                    all_data.append(daily_data)

    # Combine all daily data
    df = pd.concat(all_data, ignore_index=True)
    print(f"Total records loaded: {len(df)}")

    return df


# -----------------------------------------------------------------------------
def despike_columns(df, columns_to_despike, method=0):
    """Apply despiking to specified columns."""
    df_clean = df.copy()

    for col in columns_to_despike:
        if col in df_clean.columns:
            print(f"Despiking {col}")

            # Check data type and basic stats before despiking
            data_info = df_clean[col].dtype
            non_null_count = df_clean[col].notna().sum()
            print(
                f"  Data type: {data_info}, Non-null values: {non_null_count}/{len(df_clean)}")

            # Apply despiking
            df_clean[col] = filter_spike(df_clean[col].values, method)
        else:
            print(f"Warning: Column {col} not found in dataframe")

    return df_clean


# -----------------------------------------------------------------------------
def calculate_potential_temperatures(df):
    """Calculate potential temperatures at tower levels."""
    df_processed = df.copy()

    # Surface conditions
    p_surface = df_processed['P_2m']
    t_surface = df_processed['T_2m']

    # Heights (m) - use consistent naming
    heights = {'6m': 6, '15m': 15, '25m': 25}
    surface_height = 2

    for height_label, height in heights.items():
        temp_col = f'T_{height_label}'  # Raw temperature column
        th_col = f'TH_{height_label}'  # Potential temperature column

        if temp_col in df_processed.columns:
            # Convert temperature to Kelvin first
            temp_kelvin = df_processed[temp_col] + 273.16

            # Calculate pressure at tower height
            height_diff = height - surface_height
            p_height = calculate_pressure_at_height(
                p_surface, t_surface, temp_kelvin, height_diff
            )

            # Calculate potential temperature
            df_processed[th_col] = calculate_potential_temperature(
                temp_kelvin, p_height
            )

    return df_processed


# =============================================================================
# GRADIENT CALCULATION FUNCTIONS
# =============================================================================

def calculate_gradients(df):
    """Calculate vertical gradients of wind and potential temperature."""
    df_with_gradients = df.copy()

    # Heights for gradient calculation
    heights = np.array([6, 15, 25], dtype=float)

    # Initialize gradient columns
    gradient_cols = ['dTHdz_15m', 'dUdz_15m', 'dVdz_15m', 'S_15m']
    for col in gradient_cols:
        df_with_gradients[col] = np.nan

    # Calculate gradients for each time step
    th_cols = ['TH_6m', 'TH_15m', 'TH_25m']   # Updated column names
    u_cols = ['u_6m', 'u_15m', 'u_25m']       # Updated column names
    v_cols = ['v_6m', 'v_15m', 'v_25m']       # Updated column names

    for idx in df_with_gradients.index:
        # Extract profiles for this time step and convert to numeric
        th_profile = pd.to_numeric(df_with_gradients.loc[idx, th_cols], errors='coerce').values
        u_profile = pd.to_numeric(df_with_gradients.loc[idx, u_cols], errors='coerce').values
        v_profile = pd.to_numeric(df_with_gradients.loc[idx, v_cols], errors='coerce').values

        # Skip if any values are NaN
        if (not np.isnan(th_profile).any() and
                not np.isnan(u_profile).any() and
                not np.isnan(v_profile).any()):

            # Calculate gradients (returns single value for middle point)
            dth_dz = gradient_nonuniform(heights, th_profile)
            du_dz = gradient_nonuniform(heights, u_profile)
            dv_dz = gradient_nonuniform(heights, v_profile)

            if len(dth_dz) > 0:  # Check if gradient calculation succeeded
                df_with_gradients.loc[idx, 'dTHdz_15m'] = dth_dz[0]
                df_with_gradients.loc[idx, 'dUdz_15m'] = du_dz[0]
                df_with_gradients.loc[idx, 'dVdz_15m'] = dv_dz[0]

                # Calculate wind shear magnitude
                s_15m = np.sqrt(du_dz[0] ** 2 + dv_dz[0] ** 2)
                df_with_gradients.loc[idx, 'S_15m'] = s_15m

    return df_with_gradients


# =============================================================================
# TEMPORAL FEATURE FUNCTIONS
# =============================================================================

def add_temporal_features(df):
    """Add cyclical temporal features."""
    df_temporal = df.copy()

    # Calculate day of year
    df_temporal['DOY'] = pd.to_datetime(
        df_temporal[['YEAR', 'MONTH', 'DAY']]
    ).dt.dayofyear

    # Calculate hour from seconds
    df_temporal['HOUR_DECIMAL'] = df_temporal['SECOND'] / 3600

    # Add cyclical features
    df_temporal['sinDY'] = np.sin(2 * np.pi * df_temporal['DOY'] / 365.25)
    df_temporal['cosDY'] = np.cos(2 * np.pi * df_temporal['DOY'] / 365.25)
    df_temporal['sinHR'] = np.sin(2 * np.pi * df_temporal['HOUR_DECIMAL'] / 24)
    df_temporal['cosHR'] = np.cos(2 * np.pi * df_temporal['HOUR_DECIMAL'] / 24)
    df_temporal['sinWD'] = np.sin(2 * np.pi * df_temporal['Dir_10m'] / 360)
    df_temporal['cosWD'] = np.cos(2 * np.pi * df_temporal['Dir_10m'] / 360)

    return df_temporal


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def main():
    """Main processing function."""
    print("Loading and processing Mauna Loa data...")

    # Load raw data (temperatures in Celsius)
    df = load_and_process_data()

    # Define columns that need despiking (raw temperature and wind measurements at towers)
    columns_to_despike = [
        'T_6m', 'T_15m', 'T_25m',    # Updated column names (no leading zeros)
        'u_6m', 'u_15m', 'u_25m',   # Updated column names (no leading zeros)
        'v_6m', 'v_15m', 'v_25m'    # Updated column names (no leading zeros)
    ]

    # Apply despiking to raw data FIRST
    print("Applying despiking algorithm to raw data...")
    df_clean = despike_columns(df, columns_to_despike, method=0)

    # Calculate potential temperatures from despiked raw temperatures
    print("Calculating potential temperatures...")
    df_processed = calculate_potential_temperatures(df_clean)

    # Calculate gradients
    print("Calculating vertical gradients...")
    df_gradients = calculate_gradients(df_processed)

    # Add temporal features
    print("Adding temporal features...")
    df_final = add_temporal_features(df_gradients)

    # Create TIME column for easier handling
    df_final['TIME'] = pd.to_datetime(df_final[['YEAR', 'MONTH', 'DAY']]) + \
                       pd.to_timedelta(df_final['SECOND'], unit='s')

    print(f"Final dataset shape: {df_final.shape}")
    print(f"Available columns: {list(df_final.columns)}")

    # Display summary
    print("\nData Summary:")
    print(df_final.describe())

    df_final.to_csv(OUTPUT_DIR + "mauna_loa_processed_data.csv", index=False)

    return df_final


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    df_result = main()