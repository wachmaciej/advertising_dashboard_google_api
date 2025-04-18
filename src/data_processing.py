# src/data_processing.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import re
import warnings

# Import constants for column names!
from src.config import (DATE_COL, YEAR_COL, WEEK_COL, NUMERIC_COLS_TO_PREPROCESS,
                        CATEGORICAL_COLS_TO_ENSURE)

# Filter warnings for a clean output - applied when module is imported
warnings.filterwarnings("ignore")

@st.cache_data
def preprocess_ad_data(df):
    """Preprocess advertising data for analysis. Includes numeric cleaning."""
    if df is None or df.empty:
        return pd.DataFrame()
    df_processed = df.copy() # Work on a copy

    # --- Date Handling ---
    if DATE_COL not in df_processed.columns:
        st.error(f"Input data is missing the required '{DATE_COL}' column.")
        return pd.DataFrame()

    try:
        # GSheet reads as string, convert first before specific format attempt
        df_processed[DATE_COL] = df_processed[DATE_COL].astype(str).replace(['', 'nan', 'None', 'NULL'], np.nan)
        # Try common date formats first
        temp_dates_specific = pd.to_datetime(df_processed[DATE_COL], format="%d/%m/%Y", dayfirst=True, errors='coerce')
        mask_failed_specific = temp_dates_specific.isnull()
        if mask_failed_specific.any(): # If first format failed for some, try another
            # Try letting pandas infer if the first format didn't work for all rows
            temp_dates_infer = pd.to_datetime(df_processed.loc[mask_failed_specific, DATE_COL], errors='coerce')
            df_processed[DATE_COL] = temp_dates_specific.fillna(temp_dates_infer) # Combine results
        else:
            df_processed[DATE_COL] = temp_dates_specific # Use specific format if it worked for all

        # Drop rows where date conversion failed completely
        original_rows = len(df_processed)
        df_processed.dropna(subset=[DATE_COL], inplace=True)
        if len(df_processed) < original_rows:
            st.warning(f"Dropped {original_rows - len(df_processed)} rows due to invalid '{DATE_COL}' format.")

        if df_processed.empty: # Added check
            st.error(f"No valid rows remaining after '{DATE_COL}' cleaning.")
            return pd.DataFrame()

        df_processed = df_processed.sort_values(DATE_COL)

    except Exception as e:
        st.error(f"Error processing '{DATE_COL}': {e}")
        return pd.DataFrame()

    # --- Year/Week Creation (Do this *after* date is valid datetime) ---
    if pd.api.types.is_datetime64_any_dtype(df_processed[DATE_COL]):
         if YEAR_COL not in df_processed.columns: df_processed[YEAR_COL] = df_processed[DATE_COL].dt.year
         if WEEK_COL not in df_processed.columns: df_processed[WEEK_COL] = df_processed[DATE_COL].dt.isocalendar().week
    else:
         st.error(f"Cannot create Year/Week columns because '{DATE_COL}' is invalid.")
         return pd.DataFrame()

    # --- Numeric Conversion (Includes Cleaning) ---
    for col in NUMERIC_COLS_TO_PREPROCESS:
        if col in df_processed.columns:
            # Convert to string first
            df_processed[col] = df_processed[col].astype(str).replace(['', 'nan', 'None', 'NULL', '#N/A', 'N/A'], np.nan)

            # --- ADDED CLEANING STEP ---
            # Remove currency symbols, commas, percentage signs and potential extra whitespace
            if df_processed[col].notna().any():
                # Use .loc to modify inplace safely
                non_na_mask = df_processed[col].notna()
                # Chain replacements: remove $, then ,, then %, then strip whitespace
                cleaned_col = df_processed.loc[non_na_mask, col].astype(str) \
                                    .str.replace(r'[$,]', '', regex=True) \
                                    .str.replace('%', '', regex=False) \
                                    .str.strip()
                df_processed.loc[non_na_mask, col] = cleaned_col
            # --- END ADDED STEP ---

            # Convert to numeric, coercing errors to NaN
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")


    # --- Year/Week Conversion (Ensure Integer) ---
    for col in [YEAR_COL, WEEK_COL]:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            pre_drop_len = len(df_processed)
            df_processed.dropna(subset=[col], inplace=True)
            if len(df_processed) < pre_drop_len:
                st.warning(f"Dropped {pre_drop_len - len(df_processed)} rows due to invalid values in '{col}' column.")
            if not df_processed.empty:
                try:
                    df_processed[col] = df_processed[col].astype(int)
                except ValueError:
                    st.error(f"Could not convert '{col}' to integer after cleaning. Check data.")
                    return pd.DataFrame()
            elif col in [YEAR_COL, WEEK_COL]: # If Year/Week were critical and now df is empty
                st.warning(f"No valid data remaining after cleaning '{col}' column.")
                return pd.DataFrame()
        else:
            # This case should be less likely now Year/Week are created above if Date is valid
            st.error(f"Required column '{col}' is missing (this shouldn't happen if {DATE_COL} was valid).")
            return pd.DataFrame()

    # --- Categorical Column Handling ---
    # Ensure essential categorical columns exist, fill NaNs
    for col in CATEGORICAL_COLS_TO_ENSURE:
        if col not in df_processed.columns:
            st.warning(f"Column '{col}' not found. Adding placeholder 'Unknown...' column. Analysis using this column may be inaccurate.")
            df_processed[col] = "Unknown..."
        else:
            # Basic fillna for existing columns
            df_processed[col] = df_processed[col].fillna("Unknown...").astype(str)


    return df_processed


@st.cache_data
def filter_data_by_timeframe(df, selected_years, selected_timeframe, selected_week):
    """
    Filters data for selected years based on timeframe.
    - "Specific Week": Filters all selected years for that specific week number.
    # ... (rest of docstring same as original)
    """
    if not isinstance(selected_years, list) or not selected_years:
        return pd.DataFrame()
    if df is None or df.empty:
        st.warning("Input DataFrame to filter_data_by_timeframe is empty.")
        return pd.DataFrame()

    try:
        selected_years_int = [int(y) for y in selected_years]
    except ValueError:
        st.error("Selected years must be numeric.")
        return pd.DataFrame()

    df_copy = df.copy()

    # Check required columns created during preprocessing (Use Constants)
    required_cols = {DATE_COL, YEAR_COL, WEEK_COL}
    if not required_cols.issubset(df_copy.columns):
        missing = required_cols - set(df_copy.columns)
        st.error(f"Required '{', '.join(missing)}' columns missing for timeframe filtering. Check preprocessing.")
        return pd.DataFrame()

    # Ensure types are correct (should be handled by preprocess, but double check)
    if not pd.api.types.is_datetime64_any_dtype(df_copy[DATE_COL]): df_copy[DATE_COL] = pd.to_datetime(df_copy[DATE_COL], errors='coerce')
    if not pd.api.types.is_integer_dtype(df_copy.get(YEAR_COL)): df_copy[YEAR_COL] = pd.to_numeric(df_copy[YEAR_COL], errors='coerce').astype('Int64')
    if not pd.api.types.is_integer_dtype(df_copy.get(WEEK_COL)): df_copy[WEEK_COL] = pd.to_numeric(df_copy[WEEK_COL], errors='coerce').astype('Int64')
    df_copy.dropna(subset=[DATE_COL, YEAR_COL, WEEK_COL], inplace=True)
    if df_copy.empty: return pd.DataFrame()
    df_copy[YEAR_COL] = df_copy[YEAR_COL].astype(int) # Convert back to standard int after dropna
    df_copy[WEEK_COL] = df_copy[WEEK_COL].astype(int)


    df_filtered_years = df_copy[df_copy[YEAR_COL].isin(selected_years_int)].copy()
    if df_filtered_years.empty:
        return pd.DataFrame() # No data for selected years

    if selected_timeframe == "Specific Week":
        if selected_week is not None:
            try:
                selected_week_int = int(selected_week)
                return df_filtered_years[df_filtered_years[WEEK_COL] == selected_week_int]
            except ValueError:
                st.error(f"Invalid 'selected_week': {selected_week}. Must be a number.")
                return pd.DataFrame()
        else:
            return pd.DataFrame() # No specific week selected
    else: # Last X Weeks
        try:
            match = re.search(r'\d+', selected_timeframe)
            if match:
                weeks_to_filter = int(match.group(0))
            else:
                raise ValueError("Could not find number in timeframe string")
        except Exception as e:
            st.error(f"Could not parse weeks from timeframe: '{selected_timeframe}': {e}")
            return pd.DataFrame()

        if df_filtered_years.empty: return pd.DataFrame()
        # Need to handle potential error if df_filtered_years[YEAR_COL] is empty or all NaN
        if df_filtered_years[YEAR_COL].isnull().all() or df_filtered_years.empty:
            st.warning("No valid 'Year' data found in filtered data to determine latest year.")
            return pd.DataFrame()
        latest_year_with_data = int(df_filtered_years[YEAR_COL].max())

        df_latest_year = df_filtered_years[df_filtered_years[YEAR_COL] == latest_year_with_data]
        if df_latest_year.empty:
            st.warning(f"No data found for the latest selected year ({latest_year_with_data}) to determine week range.")
            return pd.DataFrame()

        # Need to handle potential error if df_latest_year[WEEK_COL] is empty or all NaN
        if df_latest_year[WEEK_COL].isnull().all() or df_latest_year.empty:
            st.warning(f"No valid 'Week' data found for latest year ({latest_year_with_data}) to determine week range.")
            return pd.DataFrame()
        global_max_week = int(df_latest_year[WEEK_COL].max())
        start_week = max(1, global_max_week - weeks_to_filter + 1)
        target_weeks = list(range(start_week, global_max_week + 1))

        final_filtered_df = df_filtered_years[df_filtered_years[WEEK_COL].isin(target_weeks)]
        return final_filtered_df