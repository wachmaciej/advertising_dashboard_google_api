# CONVERTED GSHEET INPUT CODE - CHUNK 1/3 - MODIFIED FOR SECRETS
# Logic based identically on File Uploader Version (April 4, 2025)
# Data input mechanism changed to load GSheet URL/Name from secrets.toml
# Modified on: Thursday, April 10, 2025 at 1:43:13 PM BST

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
from functools import reduce
import gspread # Added for Google Sheet
from google.oauth2.service_account import Credentials # Added for Google Sheet
# import os # No longer needed if only using st.secrets

# Filter warnings for a clean output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="YOY Dashboard - Advertising Data (GSheet Input)", # Modified title slightly
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# =============================================================================
# Google Sheet Data Loading Function (Standard - needed for GSheet connection)
# (REMAINS UNCHANGED)
# =============================================================================

# Cache data for 10 minutes
@st.cache_data(ttl=600)
def load_data_from_gsheet(sheet_url, worksheet_name):
    """Loads data from a Google Sheet using service account credentials stored in Streamlit secrets."""
    # (This function is copied from the working GSheet example as it's required for the connection)

    # Note: Warnings/Errors from this function will now appear in the sidebar
    # or main area depending on where error messages are displayed by default.
    # Removed the original check/warning here as URL/Name now come from secrets/constants.
    # if not sheet_url or not worksheet_name:
    #     st.sidebar.warning("Please provide both Google Sheet URL/ID and Worksheet Name.")
    #     return pd.DataFrame()

    try:
        creds_dict = st.secrets["gcp_service_account"]
        scopes = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)

        spreadsheet = client.open_by_url(sheet_url)
        worksheet = spreadsheet.worksheet(worksheet_name)

        # Read all as strings initially to handle potential mixed types better
        data = worksheet.get_all_records(numericise_ignore=['all'])

        if not data:
            st.sidebar.warning(f"Worksheet '{worksheet_name}' appears to be empty or header row is missing.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        # Do not print success message here, handle in main logic
        return df

    except gspread.exceptions.SpreadsheetNotFound:
        # Modify error message slightly as URL comes from secrets
        st.sidebar.error(f"Error: Google Sheet not found using URL/ID from secrets.")
        st.sidebar.info(f"Check 'url_or_id' in secrets and sheet sharing permissions.")
        return pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        st.sidebar.error(f"Error: Worksheet '{worksheet_name}' (from secrets) not found. Check name (it's case-sensitive).")
        return pd.DataFrame()
    except KeyError as e:
        if "gcp_service_account" in str(e):
             st.sidebar.error("Error: Missing `[gcp_service_account]` secrets. Please configure Streamlit secrets.")
        elif "gsheet_config" in str(e): # Added check for new secrets section
             st.sidebar.error("Error: Missing `[gsheet_config]` section in secrets.toml. Cannot load sheet details.")
        else:
             st.sidebar.error(f"A configuration error occurred accessing secrets: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.sidebar.error(f"An error occurred accessing Google Sheet: {e}")
        st.sidebar.info("Tips: Ensure service account email has 'Viewer' access. Verify URL/Name in secrets.")
        return pd.DataFrame()


# =============================================================================
# Common Functions for Advertising Data (FROM PROVIDED FILE UPLOADER VERSION)
# THESE ARE KEPT IDENTICAL TO THE FILE UPLOADER VERSION AS REQUESTED
# (ALL FUNCTIONS BELOW REMAIN UNCHANGED FROM YOUR ORIGINAL CODE)
# =============================================================================

@st.cache_data
def preprocess_ad_data(df):
    """Preprocess advertising data for analysis (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    if df is None or df.empty:
         return pd.DataFrame()
    df_processed = df.copy() # Work on a copy

    # --- Date Handling (IDENTICAL TO FILE UPLOADER VERSION) ---
    # Attempt to convert 'WE Date', handle errors gracefully
    date_col_name = "WE Date" # Assuming this name from FU version
    if date_col_name not in df_processed.columns:
        st.error(f"Input data is missing the required '{date_col_name}' column.")
        return pd.DataFrame()

    try:
        # GSheet reads as string, so convert first before specific format attempt
        df_processed[date_col_name] = df_processed[date_col_name].astype(str).replace(['', 'nan', 'None', 'NULL'], np.nan)
        # Try common date formats first (FU Logic)
        temp_dates_specific = pd.to_datetime(df_processed[date_col_name], format="%d/%m/%Y", dayfirst=True, errors='coerce')
        mask_failed_specific = temp_dates_specific.isnull()
        if mask_failed_specific.any(): # If first format failed for some, try another (FU Logic)
            # Try letting pandas infer if the first format didn't work for all rows
             temp_dates_infer = pd.to_datetime(df_processed.loc[mask_failed_specific, date_col_name], errors='coerce')
             df_processed[date_col_name] = temp_dates_specific.fillna(temp_dates_infer) # Combine results
        else:
             df_processed[date_col_name] = temp_dates_specific # Use specific format if it worked for all

        # Drop rows where date conversion failed completely (FU Logic)
        original_rows = len(df_processed)
        df_processed.dropna(subset=[date_col_name], inplace=True)
        if len(df_processed) < original_rows:
             st.warning(f"Dropped {original_rows - len(df_processed)} rows due to invalid '{date_col_name}' format.") # FU Warning

        if df_processed.empty: # Added check
            st.error(f"No valid rows remaining after '{date_col_name}' cleaning.")
            return pd.DataFrame()

        df_processed = df_processed.sort_values(date_col_name) # FU Logic

    # FU version only caught KeyError specifically here. Broadening slightly for GSheet context.
    except Exception as e:
        st.error(f"Error processing '{date_col_name}': {e}")
        return pd.DataFrame()

    # --- Numeric Conversion (IDENTICAL TO FILE UPLOADER VERSION) ---
    # NOTE: This does NOT strip symbols/commas. Assumes GSheet data is clean numeric
    # or relies on pd.to_numeric coercion.
    numeric_cols = [
        "Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "Total Sales", # Added Total Sales (from FU version)
        "CTR", "CVR", "Orders %", "Spend %", "Sales %", "ACOS", "ROAS" # From FU version
    ]
    # Ensure Year and Week columns are added *after* WE Date is confirmed valid datetime (FU Logic)
    if date_col_name in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed[date_col_name]):
        if 'Year' not in df_processed.columns: df_processed["Year"] = df_processed[date_col_name].dt.year
        if 'Week' not in df_processed.columns: df_processed["Week"] = df_processed[date_col_name].dt.isocalendar().week
    else:
        st.error(f"Cannot create Year/Week columns because '{date_col_name}' is invalid.")
        return pd.DataFrame()

    for col in numeric_cols:
        if col in df_processed.columns:
            # GSheet reads as string, convert first. FU version assumed correct type from pd.read_csv.
            df_processed[col] = df_processed[col].astype(str).replace(['', 'nan', 'None', 'NULL'], np.nan)
            # Convert to numeric, coercing errors to NaN (FU Logic)
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

    # --- Year/Week Conversion (IDENTICAL TO FILE UPLOADER VERSION) ---
    # Ensure Year/Week are integer types after potential creation/conversion, handle NaNs
    for col in ['Year', 'Week']:
        if col in df_processed.columns:
            # FU logic used pd.to_numeric then dropna then astype(int)
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            # Drop rows if Year/Week conversion failed AFTER numeric conversion attempt
            pre_drop_len = len(df_processed)
            df_processed.dropna(subset=[col], inplace=True)
            if len(df_processed) < pre_drop_len:
                 st.warning(f"Dropped {pre_drop_len - len(df_processed)} rows due to invalid values in '{col}' column.") # FU Warning
            # Check if dataframe is empty after dropna before astype
            if not df_processed.empty:
                 try:
                      df_processed[col] = df_processed[col].astype(int) # FU Logic
                 except ValueError:
                      st.error(f"Could not convert '{col}' to integer after cleaning. Check data.") # FU Error
                      return pd.DataFrame() # FU Logic
            elif col in ['Year', 'Week']: # If Year/Week were critical and now df is empty
                 st.warning(f"No valid data remaining after cleaning '{col}' column.") # FU Warning
                 return pd.DataFrame()
        else:
            # Added check as Year/Week might not exist if date failed earlier
            st.error(f"Required column '{col}' is missing.")
            return pd.DataFrame()

    # --- Minimal Column Checks (Based on FU Version's implicit handling) ---
    # The FU version didn't explicitly check/fill many categoricals here.
    # Add placeholder columns if essential ones used later are missing, mirroring FU's potential behavior.
    cols_to_ensure = ["Product", "Portfolio Name", "Marketplace", "Match Type", "RTW/Prospecting", "Campaign Name"]
    for col in cols_to_ensure:
        if col not in df_processed.columns:
            st.warning(f"Column '{col}' not found. Adding placeholder 'Unknown...' column. Analysis using this column may be inaccurate.")
            df_processed[col] = "Unknown..."
        else:
            # Basic fillna for existing columns, similar to implicit FU handling
            df_processed[col] = df_processed[col].fillna("Unknown...").astype(str)


    return df_processed

@st.cache_data
def filter_data_by_timeframe(df, selected_years, selected_timeframe, selected_week):
    """
    Filters data for selected years based on timeframe. (LOGIC IDENTICAL TO FILE UPLOADER VERSION)
    - "Specific Week": Filters all selected years for that specific week number.
    - "Last X Weeks": Determines the last X weeks based on the *latest* selected year's max week,
                      and filters *all* selected years for those *same* week numbers.
    Returns a concatenated dataframe across the selected years.
    """
    # This function's logic is identical to the provided FU version
    if not isinstance(selected_years, list) or not selected_years: # Added check from FU comparison
        # FU version returned empty, replicating that
        # st.warning("No years selected for timeframe filtering.")
        return pd.DataFrame()
    if df is None or df.empty:
        st.warning("Input DataFrame to filter_data_by_timeframe is empty.") # FU Warning
        return pd.DataFrame()

    try:
        selected_years_int = [int(y) for y in selected_years] # FU used selected_years directly after try/except
    except ValueError:
        st.error("Selected years must be numeric.") # FU Error
        return pd.DataFrame()

    df_copy = df.copy()
    date_col_name = "WE Date" # Assuming from FU version

    # Check required columns created during preprocessing (FU Check)
    required_cols = {date_col_name, "Year", "Week"}
    if not required_cols.issubset(df_copy.columns):
        missing = required_cols - set(df_copy.columns) # Calculate missing
        st.error(f"Required '{', '.join(missing)}' columns missing for timeframe filtering. Check preprocessing.") # FU Error adjusted slightly
        return pd.DataFrame()

    # Ensure types are correct (should be handled by preprocess, but double check) (FU Check)
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col_name]): df_copy[date_col_name] = pd.to_datetime(df_copy[date_col_name], errors='coerce')
    if not pd.api.types.is_integer_dtype(df_copy.get('Year')): df_copy['Year'] = pd.to_numeric(df_copy['Year'], errors='coerce').astype('Int64')
    if not pd.api.types.is_integer_dtype(df_copy.get('Week')): df_copy['Week'] = pd.to_numeric(df_copy['Week'], errors='coerce').astype('Int64')
    df_copy.dropna(subset=[date_col_name, "Year", "Week"], inplace=True) # FU Logic
    if df_copy.empty: return pd.DataFrame() # FU Logic
    df_copy['Year'] = df_copy['Year'].astype(int) # Convert back to standard int after dropna (FU Logic)
    df_copy['Week'] = df_copy['Week'].astype(int) # FU Logic


    df_filtered_years = df_copy[df_copy["Year"].isin(selected_years_int)].copy() # FU used selected_years directly, use int version here
    if df_filtered_years.empty:
        return pd.DataFrame() # FU Logic

    if selected_timeframe == "Specific Week":
        if selected_week is not None:
            try:
                selected_week_int = int(selected_week)
                return df_filtered_years[df_filtered_years["Week"] == selected_week_int] # FU Logic
            except ValueError:
                st.error(f"Invalid 'selected_week': {selected_week}. Must be a number.") # FU Error
                return pd.DataFrame()
        else:
            return pd.DataFrame() # No specific week selected (FU Logic)
    else: # Last X Weeks (FU Logic)
        try:
            match = re.search(r'\d+', selected_timeframe)
            if match:
                 weeks_to_filter = int(match.group(0))
            else:
                 raise ValueError("Could not find number in timeframe string") # FU Logic
        except Exception as e:
            st.error(f"Could not parse weeks from timeframe: '{selected_timeframe}': {e}") # FU Error
            return pd.DataFrame()

        if df_filtered_years.empty: return pd.DataFrame() # FU Logic
        # Need to handle potential error if df_filtered_years["Year"] is empty or all NaN
        if df_filtered_years["Year"].isnull().all() or df_filtered_years.empty:
             st.warning("No valid 'Year' data found in filtered data to determine latest year.")
             return pd.DataFrame()
        latest_year_with_data = int(df_filtered_years["Year"].max()) # FU Logic

        df_latest_year = df_filtered_years[df_filtered_years["Year"] == latest_year_with_data]
        if df_latest_year.empty:
            st.warning(f"No data found for the latest selected year ({latest_year_with_data}) to determine week range.") # FU Warning
            return pd.DataFrame()

        # Need to handle potential error if df_latest_year["Week"] is empty or all NaN
        if df_latest_year["Week"].isnull().all() or df_latest_year.empty:
             st.warning(f"No valid 'Week' data found for latest year ({latest_year_with_data}) to determine week range.")
             return pd.DataFrame()
        global_max_week = int(df_latest_year["Week"].max()) # FU Logic
        start_week = max(1, global_max_week - weeks_to_filter + 1) # FU Logic
        target_weeks = list(range(start_week, global_max_week + 1)) # FU Logic

        final_filtered_df = df_filtered_years[df_filtered_years["Week"].isin(target_weeks)] # FU Logic
        return final_filtered_df


# =============================================================================
# Basic Chart/Table/Insight/Styling Helpers (IDENTICAL TO FILE UPLOADER VERSION)
# (UNCHANGED)
# =============================================================================

@st.cache_data
def create_metric_comparison_chart(df, metric, portfolio_name=None, campaign_type="Sponsored Products"):
    """Creates a bar chart comparing a metric by Portfolio Name. Now calculates CPC if possible. (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # This function's logic is identical to the provided FU version
    required_cols_base = {"Product", "Portfolio Name"}
    required_cols_metric = {metric} # Base requirement is the metric itself or components

    # Define base components needed if metric needs calculation
    base_components = {}
    if metric == "CTR": base_components = {"Clicks", "Impressions"}
    elif metric == "CVR": base_components = {"Orders", "Clicks"}
    elif metric == "ACOS": base_components = {"Spend", "Sales"}
    elif metric == "ROAS": base_components = {"Sales", "Spend"}
    elif metric == "CPC": base_components = {"Spend", "Clicks"} # Added CPC components

    if df is None or df.empty:
        return go.Figure()

    # Check base columns first
    if not required_cols_base.issubset(df.columns):
        missing = required_cols_base - set(df.columns)
        st.warning(f"Comparison chart missing base columns: {missing}")
        return go.Figure()

    filtered_df = df[df["Product"] == campaign_type].copy()
    if filtered_df.empty:
        return go.Figure()

    # Check if metric exists OR if base components for calculation exist
    metric_col_exists = metric in filtered_df.columns
    can_calculate_metric = bool(base_components) and base_components.issubset(filtered_df.columns)

    if not metric_col_exists and not can_calculate_metric:
        missing = {metric} if not base_components else base_components - set(filtered_df.columns)
        st.warning(f"Comparison chart cannot display '{metric}'. Missing required columns: {missing} in {campaign_type} data.")
        return go.Figure()

    # Handle portfolio filtering (FU Logic)
    filtered_df["Portfolio Name"] = filtered_df["Portfolio Name"].fillna("Unknown Portfolio")
    if portfolio_name and portfolio_name != "All Portfolios":
        if portfolio_name in filtered_df["Portfolio Name"].unique():
             filtered_df = filtered_df[filtered_df["Portfolio Name"] == portfolio_name]
        else:
             st.warning(f"Portfolio '{portfolio_name}' not found for {campaign_type}. Showing all.")
             portfolio_name = "All Portfolios" # Reset variable to reflect change

    if filtered_df.empty: # Check again after potential portfolio filter
        return go.Figure()

    # Calculation logic (FU Logic)
    grouped_df = pd.DataFrame() # Initialize
    group_col = "Portfolio Name"
    try:
        # Handle metrics calculated from aggregated base components
        if metric in ["CTR", "CVR", "ACOS", "ROAS", "CPC"]: # Added CPC here
            if metric == "CTR":
                 agg_df = filtered_df.groupby(group_col).agg(Nominator=("Clicks", "sum"), Denominator=("Impressions", "sum")).reset_index()
                 agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"] * 100) if r["Denominator"] else 0, axis=1).round(2)
            elif metric == "CVR":
                 agg_df = filtered_df.groupby(group_col).agg(Nominator=("Orders", "sum"), Denominator=("Clicks", "sum")).reset_index()
                 agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"] * 100) if r["Denominator"] else 0, axis=1).round(2)
            elif metric == "ACOS":
                 agg_df = filtered_df.groupby(group_col).agg(Nominator=("Spend", "sum"), Denominator=("Sales", "sum")).reset_index()
                 agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"] * 100) if r["Denominator"] else np.nan, axis=1).round(2)
            elif metric == "ROAS":
                 agg_df = filtered_df.groupby(group_col).agg(Nominator=("Sales", "sum"), Denominator=("Spend", "sum")).reset_index()
                 agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"]) if r["Denominator"] else np.nan, axis=1).round(2)
            elif metric == "CPC": # Added CPC calculation logic
                 agg_df = filtered_df.groupby(group_col).agg(Nominator=("Spend", "sum"), Denominator=("Clicks", "sum")).reset_index()
                 agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"]) if r["Denominator"] else np.nan, axis=1).round(2) # Use np.nan if clicks are 0

            agg_df[metric] = agg_df[metric].replace([np.inf, -np.inf], np.nan) # Handle division errors
            grouped_df = agg_df[[group_col, metric]].copy()

        # Handle metrics that are directly aggregatable (like sum)
        elif metric_col_exists:
             # Ensure the column is numeric before attempting sum aggregation
             if pd.api.types.is_numeric_dtype(filtered_df[metric]):
                 grouped_df = filtered_df.groupby(group_col).agg(**{metric: (metric, "sum")}).reset_index()
             else:
                 st.warning(f"Comparison chart cannot aggregate non-numeric column '{metric}'.")
                 return go.Figure()
        else:
            # This case should be caught earlier, but as a fallback:
            st.warning(f"Comparison chart cannot display '{metric}'. Column not found and no calculation rule defined.")
            return go.Figure()

    except Exception as e:
        st.warning(f"Error aggregating comparison chart for {metric}: {e}")
        return go.Figure()

    grouped_df = grouped_df.dropna(subset=[metric])
    if grouped_df.empty:
        # st.info(f"No valid data points for metric '{metric}' after aggregation.") # Can be noisy
        return go.Figure()

    grouped_df = grouped_df.sort_values(metric, ascending=False)

    title_suffix = f" - {portfolio_name}" if portfolio_name and portfolio_name != "All Portfolios" else ""
    chart_title = f"{metric} by Portfolio ({campaign_type}){title_suffix}"

    fig = px.bar(grouped_df, x=group_col, y=metric, title=chart_title, text_auto=True) # Use group_col variable

    # Apply formatting (FU Logic)
    if metric in ["Spend", "Sales"]:
        fig.update_traces(texttemplate='%{y:$,.0f}')
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f")
    elif metric in ["CTR", "CVR", "ACOS"]:
        fig.update_traces(texttemplate='%{y:.1f}%')
        fig.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".1f")
    elif metric == "ROAS":
        fig.update_traces(texttemplate='%{y:.2f}')
        fig.update_layout(yaxis_tickformat=".2f")
    elif metric == "CPC": # Added CPC formatting
        fig.update_traces(texttemplate='%{y:$,.2f}') # Currency format for text on bars
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f") # Currency format for y-axis
    else: # Default formatting for Impressions, Clicks, Orders, Units (summed metrics)
        fig.update_traces(texttemplate='%{y:,.0f}')
        fig.update_layout(yaxis_tickformat=",.0f")

    fig.update_layout(margin=dict(t=50, b=50, l=20, r=20), height=400) # Adjust margins/height
    return fig

# <<< End of create_metric_comparison_chart >>>


# Note: create_performance_metrics_table is intentionally kept as it was in the FU code
@st.cache_data
def create_performance_metrics_table(df, portfolio_name=None, campaign_type="Sponsored Products"):
    """Creates portfolio breakdown and total summary tables (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # THIS FUNCTION IS NO LONGER DIRECTLY USED FOR SP/SB/SD TABS but kept for potential future use / other parts
    # This function's logic is identical to the provided FU version
    required_cols = {"Product", "Portfolio Name", "Impressions", "Clicks", "Spend", "Sales", "Orders"}
    if df is None or df.empty:
       return pd.DataFrame(), pd.DataFrame()

    if not required_cols.issubset(df.columns):
       missing = required_cols - set(df.columns)
       st.warning(f"Performance table missing required columns: {missing}")
       return pd.DataFrame(), pd.DataFrame()

    filtered_df = df[df["Product"] == campaign_type].copy()
    filtered_df["Portfolio Name"] = filtered_df["Portfolio Name"].fillna("Unknown Portfolio")

    if portfolio_name and portfolio_name != "All Portfolios":
       if portfolio_name in filtered_df["Portfolio Name"].unique():
            filtered_df = filtered_df[filtered_df["Portfolio Name"] == portfolio_name]
       else:
            st.warning(f"Portfolio '{portfolio_name}' not found for {campaign_type} in performance table.")
            return pd.DataFrame(), pd.DataFrame()

    if filtered_df.empty:
       return pd.DataFrame(), pd.DataFrame()

    try:
       metrics_by_portfolio = filtered_df.groupby("Portfolio Name").agg(
           Impressions=("Impressions", "sum"),
           Clicks=("Clicks", "sum"),
           Spend=("Spend", "sum"),
           Sales=("Sales", "sum"),
           Orders=("Orders", "sum")
       ).reset_index()
    except Exception as e:
       st.warning(f"Error aggregating performance table: {e}")
       return pd.DataFrame(), pd.DataFrame()

    metrics_by_portfolio["CTR"] = metrics_by_portfolio.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r.get("Impressions") else 0, axis=1)
    metrics_by_portfolio["CVR"] = metrics_by_portfolio.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r.get("Clicks") else 0, axis=1)
    metrics_by_portfolio["ACOS"] = metrics_by_portfolio.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r.get("Sales") else np.nan, axis=1)
    metrics_by_portfolio["ROAS"] = metrics_by_portfolio.apply(lambda r: (r["Sales"] / r["Spend"]) if r.get("Spend") else np.nan, axis=1)
    metrics_by_portfolio = metrics_by_portfolio.replace([np.inf, -np.inf], np.nan)

    for col in ["CTR", "CVR", "ACOS"]:
        if col in metrics_by_portfolio.columns: metrics_by_portfolio[col] = metrics_by_portfolio[col].round(1)
    for col in ["Spend", "Sales", "ROAS"]:
        if col in metrics_by_portfolio.columns: metrics_by_portfolio[col] = metrics_by_portfolio[col].round(2)

    total_summary = pd.DataFrame()
    if not filtered_df.empty:
        sum_cols = ["Impressions", "Clicks", "Spend", "Sales", "Orders"]
        numeric_summary_data = filtered_df.copy()
        for col in sum_cols:
             if col in numeric_summary_data.columns:
                  numeric_summary_data[col] = pd.to_numeric(numeric_summary_data[col], errors='coerce').fillna(0)
             else: numeric_summary_data[col] = 0

        total_impressions = numeric_summary_data["Impressions"].sum()
        total_clicks = numeric_summary_data["Clicks"].sum()
        total_spend = numeric_summary_data["Spend"].sum()
        total_sales = numeric_summary_data["Sales"].sum()
        total_orders = numeric_summary_data["Orders"].sum()
        total_ctr = (total_clicks / total_impressions * 100) if total_impressions else 0
        total_cvr = (total_orders / total_clicks * 100) if total_clicks else 0
        total_acos = (total_spend / total_sales * 100) if total_sales else np.nan
        total_roas = (total_sales / total_spend) if total_spend else np.nan
        total_acos = np.nan if total_acos in [np.inf, -np.inf] else total_acos
        total_roas = np.nan if total_roas in [np.inf, -np.inf] else total_roas
        total_summary_data = {
            "Metric": ["Total"],
            "Impressions": [total_impressions], "Clicks": [total_clicks], "Orders": [total_orders],
            "Spend": [round(total_spend, 2)], "Sales": [round(total_sales, 2)],
            "CTR": [round(total_ctr, 1)], "CVR": [round(total_cvr, 1)],
            "ACOS": [total_acos], "ROAS": [total_roas]
        }
        total_summary = pd.DataFrame(total_summary_data)

    metrics_by_portfolio = metrics_by_portfolio.rename(columns={"Portfolio Name": "Portfolio"})
    return metrics_by_portfolio, total_summary

@st.cache_data
def create_metric_over_time_chart(data, metric, portfolio, product_type, show_yoy=True, weekly_total_sales_data=None): # Added weekly_total_sales_data
    """Create a chart showing metric over time with optional YoY comparison (Weekly YoY Overlay with Month Annotations). (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # This function's logic is identical to the provided FU version
    if data is None or data.empty:
        return go.Figure()

    date_col_name = "WE Date" # Assuming from FU version
    base_required = {"Product", "Portfolio Name", date_col_name, "Year", "Week"}
    if not base_required.issubset(data.columns):
        missing = base_required - set(data.columns)
        st.warning(f"Metric over time chart missing required columns: {missing}")
        return go.Figure()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col_name]):
        st.warning(f"{date_col_name} column is not datetime type for time chart.")
        return go.Figure()

    data_copy = data.copy() # Work on a copy

    filtered_data = data_copy[data_copy["Product"] == product_type].copy()
    # FU logic filledna here
    filtered_data["Portfolio Name"] = filtered_data["Portfolio Name"].fillna("Unknown Portfolio")
    if portfolio != "All Portfolios":
        if portfolio in filtered_data["Portfolio Name"].unique():
            filtered_data = filtered_data[filtered_data["Portfolio Name"] == portfolio]
        else:
            st.warning(f"Portfolio '{portfolio}' not found for {product_type}. Showing all.")
            portfolio = "All Portfolios" # Update variable to reflect change

    if filtered_data.empty:
        return go.Figure()

    # --- Define required base components for derived metrics (FU Logic) ---
    metric_required_cols = {metric}
    base_needed_for_metric = set()
    is_derived_metric = False
    if metric == "CTR": base_needed_for_metric.update({"Clicks", "Impressions"}); is_derived_metric = True
    elif metric == "CVR": base_needed_for_metric.update({"Orders", "Clicks"}); is_derived_metric = True
    elif metric == "ACOS": base_needed_for_metric.update({"Spend", "Sales"}); is_derived_metric = True
    elif metric == "ROAS": base_needed_for_metric.update({"Sales", "Spend"}); is_derived_metric = True
    elif metric == "CPC": base_needed_for_metric.update({"Spend", "Clicks"}); is_derived_metric = True
    elif metric == "Ad % Sale": base_needed_for_metric.update({"Sales"}); is_derived_metric = True # Also needs external denom

    # --- Check if necessary columns exist for the selected metric (FU Logic) ---
    metric_exists_in_input = metric in filtered_data.columns
    base_components_exist = base_needed_for_metric.issubset(filtered_data.columns)

    ad_sale_check_passed = True # Assume pass unless specific checks fail
    if metric == "Ad % Sale":
        if not base_components_exist: # Check 'Sales' column exists
             st.warning(f"Metric chart requires 'Sales' column for 'Ad % Sale'.") # FU Warning
             ad_sale_check_passed = False
        if weekly_total_sales_data is None or weekly_total_sales_data.empty:
             st.info(f"Denominator data (weekly total sales) not available for 'Ad % Sale' calculation.") # FU Info
             ad_sale_check_passed = False
        elif not {"Year", "Week", "Weekly_Total_Sales"}.issubset(weekly_total_sales_data.columns):
             st.warning(f"Passed 'weekly_total_sales_data' is missing required columns (Year, Week, Weekly_Total_Sales).") # FU Warning
             ad_sale_check_passed = False

    # If it's a derived metric, we MUST have its base components (FU Logic)
    if is_derived_metric and not base_components_exist:
        # Specific check for Ad % Sale denominator availability
        if metric == "Ad % Sale": # Check the flag here
             if not ad_sale_check_passed: # If check failed above
                 st.warning(f"Cannot calculate 'Ad % Sale'. Check required 'Sales' column and denominator data source.") # FU Warning
                 return go.Figure()
             # If base component ('Sales') was missing, it should trigger below anyway
        # For other derived metrics if base components are missing
        else:
             missing_bases = base_needed_for_metric - set(filtered_data.columns)
             st.warning(f"Cannot calculate derived metric '{metric}'. Missing required base columns: {missing_bases}") # FU Warning
             return go.Figure()

    # If it's NOT a derived metric, it MUST exist in the input data (FU Logic)
    if not is_derived_metric and not metric_exists_in_input:
         st.warning(f"Metric chart requires column '{metric}' in the data.") # FU Warning
         return go.Figure()

    # --- Start Plotting (FU Logic) ---
    years = sorted(filtered_data["Year"].dropna().unique().astype(int))
    fig = go.Figure()

    if metric in ["CTR", "CVR", "ACOS", "Ad % Sale"]: hover_suffix = "%"; hover_format = ".1f"
    elif metric in ["Spend", "Sales"]: hover_suffix = ""; hover_format = "$,.2f"
    elif metric in ["ROAS", "CPC"]: hover_suffix = ""; hover_format = ".2f"
    else: hover_suffix = ""; hover_format = ",.0f" # Impressions, Clicks, Orders, Units
    base_hover_template = f"Date: %{{customdata[1]|%Y-%m-%d}}<br>Week: %{{customdata[0]}}<br>{metric}: %{{y:{hover_format}}}{hover_suffix}<extra></extra>"

    processed_years = []
    colors = px.colors.qualitative.Plotly

    # ========================
    # YoY Plotting Logic (FU Logic)
    # ========================
    if show_yoy and len(years) > 1:
        # Define columns needed for aggregation: base components + WE Date
        cols_to_agg_yoy = list(base_needed_for_metric | {metric} | {date_col_name})
        # Ensure only columns actually present in the data are included
        actual_cols_to_agg_yoy = list(set(cols_to_agg_yoy) & set(filtered_data.columns))

        if date_col_name not in actual_cols_to_agg_yoy: # WE Date is critical for hover
             st.warning(f"Missing '{date_col_name}' for aggregation (YoY).") # FU Warning adjusted
             return go.Figure()

        try:
            agg_dict_yoy = {}
            numeric_aggregated = False
            for col in actual_cols_to_agg_yoy:
                 if col == date_col_name:
                     agg_dict_yoy[col] = 'min' # Get earliest date within the week for hover
                 # Aggregate ONLY the base components or the original metric IF it's not derived
                 elif pd.api.types.is_numeric_dtype(filtered_data[col]) and \
                      (col in base_needed_for_metric or (col == metric and not is_derived_metric)):
                     agg_dict_yoy[col] = "sum"
                     numeric_aggregated = True

            if not numeric_aggregated and not is_derived_metric:
                 st.warning(f"No numeric column found for metric '{metric}' to aggregate for the YoY chart.") # FU Warning
                 return go.Figure()
            elif not base_components_exist and is_derived_metric:
                 # This case should have been caught earlier, but double-check
                 st.warning(f"Cannot proceed with YoY chart for derived metric '{metric}' due to missing base columns.") # FU Warning
                 return go.Figure()

            # Aggregate: Sum up base components (and original metric if not derived) by Year/Week
            grouped = filtered_data.groupby(["Year", "Week"], as_index=False).agg(agg_dict_yoy)
            grouped[date_col_name] = pd.to_datetime(grouped[date_col_name])

        except Exception as e:
            st.warning(f"Could not group data by week for YoY chart: {e}") # FU Warning
            return go.Figure()

        # --- *** ALWAYS RECALCULATE DERIVED METRICS POST-AGGREGATION *** (FU Logic)---
        metric_calculated_successfully = False # Flag
        if is_derived_metric:
            if metric == "CTR":
                 if {"Clicks", "Impressions"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r.get("Impressions") else 0, axis=1).round(1) # FU added rounding
                     metric_calculated_successfully = True
            elif metric == "CVR":
                 if {"Orders", "Clicks"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r.get("Clicks") else 0, axis=1).round(1) # FU added rounding
                     metric_calculated_successfully = True
            elif metric == "ACOS":
                 if {"Spend", "Sales"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r.get("Sales") else np.nan, axis=1).round(1) # FU added rounding
                     metric_calculated_successfully = True
            elif metric == "ROAS":
                 if {"Sales", "Spend"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Sales"] / r["Spend"]) if r.get("Spend") else np.nan, axis=1).round(2) # FU added rounding
                     metric_calculated_successfully = True
            elif metric == "CPC":
                 if {"Spend", "Clicks"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Clicks"]) if r.get("Clicks") else np.nan, axis=1).round(2) # FU added rounding
                     metric_calculated_successfully = True
            elif metric == "Ad % Sale":
                 if {"Sales"}.issubset(grouped.columns) and ad_sale_check_passed: # Use flag
                     try:
                         temp_denom = weekly_total_sales_data.copy()
                         # Ensure data types match for merge
                         if 'Year' in grouped.columns and 'Year' in temp_denom.columns: temp_denom['Year'] = temp_denom['Year'].astype(grouped['Year'].dtype)
                         if 'Week' in grouped.columns and 'Week' in temp_denom.columns: temp_denom['Week'] = temp_denom['Week'].astype(grouped['Week'].dtype)
                         # Perform merge safely
                         grouped_merged = pd.merge(grouped, temp_denom[['Year', 'Week', 'Weekly_Total_Sales']], on=['Year', 'Week'], how='left')
                         grouped_merged[metric] = grouped_merged.apply(lambda r: (r['Sales'] / r['Weekly_Total_Sales'] * 100) if pd.notna(r['Weekly_Total_Sales']) and r['Weekly_Total_Sales'] > 0 else np.nan, axis=1).round(1) # FU added rounding
                         grouped = grouped_merged.drop(columns=['Weekly_Total_Sales'], errors='ignore') # Drop temp col
                         metric_calculated_successfully = True
                     except Exception as e:
                         st.warning(f"Failed to merge/calculate Ad % Sale for YoY chart: {e}") # FU Warning
                         grouped[metric] = np.nan # Ensure column exists even if calculation fails
                 else:
                     # This handles case where 'Sales' exists but denom check failed
                     grouped[metric] = np.nan # Ensure column exists if calculation wasn't possible
            # Check flag after attempting all derived metric calculations
            if not metric_calculated_successfully:
                 # This condition might be hit if a derived metric was selected but its base cols were missing *after* grouping
                 st.error(f"Internal Error: Failed to recalculate derived metric '{metric}' (YoY). Check base columns post-aggregation.") # FU Error
                 return go.Figure()
        else:
             # If metric wasn't derived, it should have been aggregated directly
             if metric not in grouped.columns:
                 st.warning(f"Metric column '{metric}' not found after aggregation (YoY).") # FU Warning
                 return go.Figure()
             metric_calculated_successfully = True # Treat direct aggregation as success

        # --- End Recalculation Block ---

        # Handle Inf/-Inf values after calculation/aggregation
        if metric_calculated_successfully: # Only attempt replace if metric should exist
             grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)
        else: # Should have returned earlier if calc failed, but as safety
             return go.Figure()

        # --- Plotting YoY data (FU Logic) ---
        min_week_data, max_week_data = 53, 0
        for i, year in enumerate(years):
            year_data = grouped[grouped["Year"] == year].sort_values("Week")
            if year_data.empty or year_data[metric].isnull().all(): continue

            processed_years.append(year)
            min_week_data = min(min_week_data, year_data["Week"].min())
            max_week_data = max(max_week_data, year_data["Week"].max())
            custom_data_hover = year_data[['Week', date_col_name]] # WE Date from 'min' aggregation

            fig.add_trace(
                go.Scatter(x=year_data["Week"], y=year_data[metric], mode="lines+markers", name=f"{year}",
                           line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=6),
                           customdata=custom_data_hover, hovertemplate=base_hover_template)
            )

        # Add month annotations if data was plotted
        if processed_years:
            month_approx_weeks = { 1: 2.5, 2: 6.5, 3: 10.5, 4: 15, 5: 19.5, 6: 24, 7: 28, 8: 32.5, 9: 37, 10: 41.5, 11: 46, 12: 50.5 }
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            for month_num, week_val in month_approx_weeks.items():
                 if week_val >= min_week_data - 1 and week_val <= max_week_data + 1:
                     fig.add_annotation(x=week_val, y=-0.12, xref="x", yref="paper", text=month_names[month_num-1], showarrow=False, font=dict(size=10, color="grey"))
            fig.update_layout(xaxis_range=[max(0, min_week_data - 1), min(54, max_week_data + 1)])

        fig.update_layout(xaxis_title="Week of Year", xaxis_showticklabels=True, legend_title="Year", margin=dict(b=70))

    # ========================
    # Non-YoY Plotting Logic (FU Logic)
    # ========================
    else:
        # Define columns needed for aggregation: base components + WE Date, Year, Week
        cols_to_agg_noyoy = list(base_needed_for_metric | {metric} | {date_col_name, "Year", "Week"})
        # Ensure only columns actually present in the data are included
        actual_cols_to_agg_noyoy = list(set(cols_to_agg_noyoy) & set(filtered_data.columns))

        if not {date_col_name, "Year", "Week"}.issubset(actual_cols_to_agg_noyoy): # Critical keys
             st.warning(f"Missing '{date_col_name}', 'Year', or 'Week' for aggregation (non-YoY).") # FU Warning adjusted
             return go.Figure()

        try:
            agg_dict_noyoy = {}
            numeric_aggregated = False
            grouping_keys_noyoy = [date_col_name, "Year", "Week"] # Group by specific date point
            for col in actual_cols_to_agg_noyoy:
                 if col not in grouping_keys_noyoy and pd.api.types.is_numeric_dtype(filtered_data[col]) and \
                   (col in base_needed_for_metric or (col == metric and not is_derived_metric)):
                    agg_dict_noyoy[col] = "sum"
                    numeric_aggregated = True

            if not numeric_aggregated and not is_derived_metric:
                 st.warning(f"No numeric column found for metric '{metric}' to aggregate for the time chart (non-YoY).") # FU Warning
                 return go.Figure()
            elif not base_components_exist and is_derived_metric:
                 st.warning(f"Cannot proceed with time chart for derived metric '{metric}' due to missing base columns (non-YoY).") # FU Warning
                 return go.Figure()

            # Aggregate: Sum up base components (and original metric if not derived) by Date/Year/Week
            grouped = filtered_data.groupby(grouping_keys_noyoy, as_index=False).agg(agg_dict_noyoy)
            grouped[date_col_name] = pd.to_datetime(grouped[date_col_name]) # Ensure datetime type

        except Exception as e:
            st.warning(f"Could not group data for time chart (non-YoY): {e}") # FU Warning
            return go.Figure()

        # --- *** ALWAYS RECALCULATE DERIVED METRICS POST-AGGREGATION *** (FU Logic) ---
        metric_calculated_successfully = False # Flag
        if is_derived_metric:
             if metric == "CTR":
                 if {"Clicks", "Impressions"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r.get("Impressions") else 0, axis=1).round(1)
                     metric_calculated_successfully = True
             elif metric == "CVR":
                  if {"Orders", "Clicks"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r.get("Clicks") else 0, axis=1).round(1)
                     metric_calculated_successfully = True
             elif metric == "ACOS":
                 if {"Spend", "Sales"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r.get("Sales") else np.nan, axis=1).round(1)
                     metric_calculated_successfully = True
             elif metric == "ROAS":
                 if {"Sales", "Spend"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Sales"] / r["Spend"]) if r.get("Spend") else np.nan, axis=1).round(2)
                     metric_calculated_successfully = True
             elif metric == "CPC":
                 if {"Spend", "Clicks"}.issubset(grouped.columns):
                     grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Clicks"]) if r.get("Clicks") else np.nan, axis=1).round(2)
                     metric_calculated_successfully = True
             elif metric == "Ad % Sale":
                 if {"Sales"}.issubset(grouped.columns) and ad_sale_check_passed: # Use flag
                     try:
                         temp_denom = weekly_total_sales_data.copy()
                         if 'Year' in grouped.columns and 'Year' in temp_denom.columns: temp_denom['Year'] = temp_denom['Year'].astype(grouped['Year'].dtype)
                         if 'Week' in grouped.columns and 'Week' in temp_denom.columns: temp_denom['Week'] = temp_denom['Week'].astype(grouped['Week'].dtype)
                         grouped_merged = pd.merge(grouped, temp_denom[['Year', 'Week', 'Weekly_Total_Sales']], on=['Year', 'Week'], how='left')
                         grouped_merged[metric] = grouped_merged.apply(lambda r: (r['Sales'] / r['Weekly_Total_Sales'] * 100) if pd.notna(r['Weekly_Total_Sales']) and r['Weekly_Total_Sales'] > 0 else np.nan, axis=1).round(1)
                         grouped = grouped_merged.drop(columns=['Weekly_Total_Sales'], errors='ignore')
                         metric_calculated_successfully = True
                     except Exception as e:
                         st.warning(f"Failed to merge/calculate Ad % Sale for non-YoY chart: {e}") # FU Warning
                         grouped[metric] = np.nan
                 else:
                      grouped[metric] = np.nan # Ensure column exists if calculation wasn't possible

             if not metric_calculated_successfully:
                  st.error(f"Internal Error: Failed to recalculate derived metric '{metric}' (non-YoY). Check base columns post-aggregation.") # FU Error
                  return go.Figure()
        else:
             # If metric wasn't derived, it should exist
             if metric not in grouped.columns:
                  st.warning(f"Metric column '{metric}' not found after aggregation (non-YoY).") # FU Warning
                  return go.Figure()
             metric_calculated_successfully = True # Treat direct aggregation as success


        # Handle Inf/-Inf values (FU Logic)
        if metric_calculated_successfully:
             grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)
        else: # Should have returned if calculation failed
             return go.Figure()

        # --- Plotting Non-YoY data (FU Logic) ---
        if grouped[metric].isnull().all():
            st.info(f"No valid data points for metric '{metric}' over time (non-YoY).") # FU Info Message
            return go.Figure() # Return empty figure if all values are NaN

        grouped = grouped.sort_values(date_col_name)
        custom_data_hover_noyoy = grouped[['Week', date_col_name]]
        fig.add_trace(
            go.Scatter(x=grouped[date_col_name], y=grouped[metric], mode="lines+markers", name=metric,
                       line=dict(color="#1f77b4", width=2), marker=dict(size=6),
                       customdata=custom_data_hover_noyoy, hovertemplate=base_hover_template)
        )
        fig.update_layout(xaxis_title="Date", showlegend=False)

    # --- Final Chart Layout (FU Logic) ---
    portfolio_title = f" for {portfolio}" if portfolio != "All Portfolios" else " for All Portfolios"
    years_in_plot = processed_years if (show_yoy and len(years) > 1 and processed_years) else years # Get years actually plotted
    final_chart_title = f"{metric} "

    if show_yoy and len(years_in_plot) > 1:
        final_chart_title += f"Weekly Comparison {portfolio_title} ({product_type})"
        final_xaxis_title = "Week of Year"
    else:
        final_chart_title += f"Over Time (Weekly) {portfolio_title} ({product_type})"
        final_xaxis_title = "Week Ending Date"

    final_margin = dict(t=80, b=70, l=70, r=30)
    fig.update_layout(
        title=final_chart_title, xaxis_title=final_xaxis_title, yaxis_title=metric,
        hovermode="x unified", template="plotly_white", yaxis=dict(rangemode="tozero"), margin=final_margin
    )

    # Apply Y-axis formatting based on the metric (FU Logic)
    if metric in ["Spend", "Sales", "CPC"]: fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f")
    elif metric in ["CTR", "CVR", "ACOS", "Ad % Sale"]: fig.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".1f")
    elif metric == "ROAS": fig.update_layout(yaxis_tickformat=".2f")
    else: fig.update_layout(yaxis_tickformat=",.0f") # Impressions, Clicks, Orders, Units

    return fig

# --- End of Chunk 1 (Modified) ---

# CONVERTED GSHEET INPUT CODE - CHUNK 2/3 - MODIFIED FOR SECRETS
# Logic based identically on File Uploader Version (April 4, 2025)
# Data input mechanism changed to load GSheet URL/Name from secrets.toml
# NOTE: NO FUNCTIONAL CHANGES IN THIS CHUNK FOR SECRETS IMPLEMENTATION.
# Modified on: Thursday, April 10, 2025 at 1:54:13 PM BST

# [Imports, load_data_from_gsheet, preprocess_ad_data, filter_data_by_timeframe,
#  create_metric_comparison_chart, create_performance_metrics_table,
#  create_metric_over_time_chart Functions from Chunk 1]
# =============================================================================

# --- Styling & Insight Helpers (IDENTICAL TO FILE UPLOADER VERSION) ---
# (UNCHANGED FROM YOUR ORIGINAL CODE)

def style_dataframe(df, format_dict, highlight_cols=None, color_map_func=None, text_align='right', na_rep='N/A'):
    """Generic styling function for dataframes with alignment and NaN handling. (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # This function's logic is identical to the provided FU version
    if df is None or df.empty: return None
    df_copy = df.copy().replace([np.inf, -np.inf], np.nan)
    valid_format_dict = {k: v for k, v in format_dict.items() if k in df_copy.columns}

    try:
        styled = df_copy.style.format(valid_format_dict, na_rep=na_rep)
    except Exception as e:
        st.error(f"Error applying format: {e}. Formatting dictionary: {valid_format_dict}")
        return df_copy.style # Basic styler on error

    if highlight_cols and color_map_func:
        if len(highlight_cols) == len(color_map_func):
            for col, func in zip(highlight_cols, color_map_func):
                 if col in df_copy.columns:
                     try: styled = styled.applymap(func, subset=[col]) # FU used applymap
                     except Exception as e: st.warning(f"Styling failed for column '{col}': {e}")
        else:
             st.error("Mismatch between highlight_cols and color_map_func in style_dataframe.") # FU Error

    cols_to_align = df_copy.columns
    if len(cols_to_align) > 0:
        try:
            # FU version used set_table_styles with CSS selectors
            first_col_idx = df_copy.columns.get_loc(cols_to_align[0])
            styles = [
                {'selector': 'th', 'props': [('text-align', text_align), ('white-space', 'nowrap')]},
                {'selector': 'td', 'props': [('text-align', text_align)]},
                {'selector': f'th.col_heading.level0.col{first_col_idx}', 'props': [('text-align', 'left')]},
                {'selector': f'td.col{first_col_idx}', 'props': [('text-align', 'left')]}
            ]
            # Attempting to replicate FU alignment approach
            styled = styled.set_table_styles(styles, overwrite=False)
        except Exception as e:
            st.warning(f"Could not apply specific alignment: {e}. Using general alignment.") # FU Warning
            # Fallback using set_properties if set_table_styles fails (might be more robust)
            try:
                 styled = styled.set_properties(**{'text-align': text_align})
                 first_col_name = df_copy.columns[0]
                 styled = styled.set_properties(subset=[first_col_name], **{'text-align': 'left'})
            except Exception as e2:
                 st.warning(f"Fallback alignment also failed: {e2}")

    else: # If no columns, just apply general alignment potentially
        styled = styled.set_properties(**{'text-align': text_align})

    return styled

def style_total_summary(df):
    """Styles the single-row total summary table (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # THIS FUNCTION IS NO LONGER DIRECTLY USED FOR SP/SB/SD TABS but kept
    # This function's logic is identical to the provided FU version
    format_dict = {
        "Impressions": "{:,.0f}", "Clicks": "{:,.0f}", "Orders": "{:,.0f}",
        "Spend": "${:,.2f}", "Sales": "${:,.2f}",
        "CTR": "{:.1f}%", "CVR": "{:.1f}%",
        "ACOS": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        "ROAS": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    }
    def color_acos(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f <= 15 else ("color: orange" if val_f <= 20 else "color: red")
        except (ValueError, TypeError): return "color: grey"
    def color_roas(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f > 3 else "color: red"
        except (ValueError, TypeError): return "color: grey"

    # FU version called the generic style_dataframe
    styled = style_dataframe(df, format_dict, highlight_cols=["ACOS", "ROAS"], color_map_func=[color_acos, color_roas], na_rep="N/A")
    if styled: return styled.set_properties(**{"font-weight": "bold"})
    return None

def style_metrics_table(df):
    """Styles the multi-row performance metrics table (by Portfolio) (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # THIS FUNCTION IS NO LONGER DIRECTLY USED FOR SP/SB/SD TABS but kept
    # This function's logic is identical to the provided FU version
    format_dict = {
        "Impressions": "{:,.0f}", "Clicks": "{:,.0f}", "Orders": "{:,.0f}",
        "Spend": "${:,.2f}", "Sales": "${:,.2f}",
        "CTR": "{:.1f}%", "CVR": "{:.1f}%",
        "ACOS": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        "ROAS": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    }
    if "Units" in df.columns: format_dict["Units"] = "{:,.0f}"
    if "CPC" in df.columns: format_dict["CPC"] = "${:,.2f}"
    id_cols = ["Portfolio", "Match Type", "RTW/Prospecting", "Campaign"]
    id_col_name = next((col for col in df.columns if col in id_cols), None)
    if id_col_name: format_dict[id_col_name] = "{}"

    def color_acos(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f <= 15 else ("color: orange" if val_f <= 20 else "color: red")
        except (ValueError, TypeError): return "color: grey"
    def color_roas(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f > 3 else "color: red"
        except (ValueError, TypeError): return "color: grey"

    # FU version called the generic style_dataframe
    styled = style_dataframe(df, format_dict, highlight_cols=["ACOS", "ROAS"], color_map_func=[color_acos, color_roas], na_rep="N/A")
    return styled

@st.cache_data
def generate_insights(total_metrics_series, campaign_type):
    """Generates text insights based on a summary row (Pandas Series) (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # This function's logic is identical to the provided FU version
    # --- Define Your Specific Thresholds Here ---
    acos_target = 15.0    # Acceptable ACOS is <= 15%
    roas_target = 5.0     # Good ROAS is >= 5
    ctr_target = 0.35   # Good CTR is >= 0.35%
    cvr_target = 10.0     # Good CVR is >= 10%
    # --- End of Threshold Definitions ---

    insights = []
    # Get the metric values from the input series
    acos = total_metrics_series.get("ACOS", np.nan)
    roas = total_metrics_series.get("ROAS", np.nan)
    ctr = total_metrics_series.get("CTR", np.nan)
    cvr = total_metrics_series.get("CVR", np.nan)
    sales = total_metrics_series.get("Sales", 0)
    spend = total_metrics_series.get("Spend", 0)

    if pd.isna(sales): sales = 0
    if pd.isna(spend): spend = 0

    # --- Insight Logic using the defined thresholds (FU Logic) ---
    if spend > 0 and sales == 0:
        insights.append("⚠️ **Immediate Attention:** Spend occurred with zero attributed sales. Review targeting, keywords, and product pages urgently.")
        if pd.notna(ctr): insights.append(f"ℹ️ Click-through rate was {ctr:.2f}%.") # FU used 2 decimals
    else:
        # ACOS Insight
        if pd.isna(acos):
            if spend == 0 and sales == 0: insights.append("ℹ️ No spend or sales recorded for ACOS calculation.")
            elif sales == 0 and spend > 0: insights.append("ℹ️ ACOS is not applicable (No Sales from Spend).")
            elif spend == 0 and sales > 0: insights.append(f"✅ **ACOS:** ACOS is effectively 0% (Sales with no spend), which is below the target (≤{acos_target}%).") # FU Clarified message
            elif spend == 0: insights.append("ℹ️ ACOS is not applicable (No Spend).")
        elif acos > acos_target: # Compare with acos_target
            insights.append(f"📈 **High ACOS:** Overall ACOS ({acos:.1f}%) is above the target ({acos_target}%). Consider optimizing bids, keywords, or targeting.")
        else: # ACOS is <= acos_target
            insights.append(f"✅ **ACOS:** Overall ACOS ({acos:.1f}%) is within the acceptable range (≤{acos_target}%).")

        # ROAS Insight
        if pd.isna(roas):
            if spend == 0 and sales == 0: insights.append("ℹ️ No spend or sales recorded for ROAS calculation.")
            elif spend == 0 and sales > 0 : insights.append("✅ **ROAS:** ROAS is effectively infinite (Sales with No Spend).")
            elif spend > 0 and sales == 0: insights.append("ℹ️ ROAS is 0 (No Sales from Spend).")
        elif roas < roas_target: # Compare with roas_target
            insights.append(f"📉 **Low ROAS:** Overall ROAS ({roas:.2f}) is below the target of {roas_target}. Review performance and strategy.")
        else: # ROAS is >= roas_target
            insights.append(f"✅ **ROAS:** Overall ROAS ({roas:.2f}) is good (≥{roas_target}).")

        # CTR Insight
        if pd.isna(ctr):
            insights.append("ℹ️ Click-Through Rate (CTR) could not be calculated (likely no impressions).")
        elif ctr < ctr_target: # Compare with ctr_target
            insights.append(f"📉 **Low CTR:** Click-through rate ({ctr:.2f}%) is low (<{ctr_target}%). Review ad creative, relevance, or placement.") # FU Display CTR with 2 decimals
        else: # CTR is >= ctr_target
            insights.append(f"✅ **CTR:** Click-through rate ({ctr:.2f}%) is satisfactory (≥{ctr_target}%).")

        # CVR Insight
        if pd.isna(cvr):
            insights.append("ℹ️ Conversion Rate (CVR) could not be calculated (likely no clicks).")
        elif cvr < cvr_target: # Compare with cvr_target
            insights.append(f"📉 **Low CVR:** Conversion rate ({cvr:.1f}%) is below the target ({cvr_target}%). Review product listing pages and targeting.")
        else: # CVR is >= cvr_target
            insights.append(f"✅ **CVR:** Conversion rate ({cvr:.1f}%) is good (≥{cvr_target}%).")

    return insights

@st.cache_data
def create_yoy_grouped_table(df_filtered_period, group_by_col, selected_metrics, years_to_process, display_col_name=None):
    """Creates a merged YoY comparison table grouped by a specific column. (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # This function's logic is identical to the provided FU version
    # Includes internal Ad % Sale denominator calculation
    if df_filtered_period is None or df_filtered_period.empty: return pd.DataFrame()
    if group_by_col not in df_filtered_period.columns: st.warning(f"Grouping column '{group_by_col}' not found."); return pd.DataFrame() # FU Warning
    if not isinstance(selected_metrics, list) or not selected_metrics: st.warning("No metrics selected."); return pd.DataFrame() # FU Warning + type check

    date_col = "WE Date" # Assume from FU version
    # Check required columns for Ad % Sale based on FU logic
    ad_sale_possible = ("Ad % Sale" in selected_metrics and {"Sales", "Total Sales", date_col}.issubset(df_filtered_period.columns))

    if "Ad % Sale" in selected_metrics and not ad_sale_possible:
        st.warning("Cannot calculate 'Ad % Sale'. Requires 'Sales', 'Total Sales', and 'WE Date' columns.") # FU Warning
        selected_metrics = [m for m in selected_metrics if m != "Ad % Sale"]
        if not selected_metrics: return pd.DataFrame()

    # FU logic filled NaNs in grouping column here
    df_filtered_period[group_by_col] = df_filtered_period[group_by_col].fillna(f"Unknown {group_by_col}")
    yearly_tables = []

    for yr in years_to_process:
        df_year = df_filtered_period[df_filtered_period["Year"] == yr].copy()
        if df_year.empty: continue

        # Determine base metrics needed (FU Logic)
        base_metrics_to_sum_needed = set()
        for metric in selected_metrics:
            if metric in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]: base_metrics_to_sum_needed.add(metric)
            elif metric == "CTR": base_metrics_to_sum_needed.update(["Clicks", "Impressions"])
            elif metric == "CVR": base_metrics_to_sum_needed.update(["Orders", "Clicks"])
            elif metric == "CPC": base_metrics_to_sum_needed.update(["Spend", "Clicks"])
            elif metric == "ACOS": base_metrics_to_sum_needed.update(["Spend", "Sales"])
            elif metric == "ROAS": base_metrics_to_sum_needed.update(["Sales", "Spend"])
            elif metric == "Ad % Sale": base_metrics_to_sum_needed.add("Sales") # Only numerator base needed here

        actual_base_present = {m for m in base_metrics_to_sum_needed if m in df_year.columns}
        # FU logic check: skip year if no base metrics needed are present AND no selected metrics (that aren't derived) are present
        if not actual_base_present and not any(m in df_year.columns for m in selected_metrics if m not in ["CTR","CVR","CPC","ACOS","ROAS","Ad % Sale"]): continue

        # Determine which of the selected metrics can actually be calculated/displayed for this year (FU Logic)
        calculable_metrics_for_year = []
        for metric in selected_metrics:
             can_calc_yr = False
             if metric in df_year.columns and pd.api.types.is_numeric_dtype(df_year[metric]): can_calc_yr = True # Check if directly present and numeric
             # Check if derivable based on *present* base metrics
             elif metric == "CTR" and {"Clicks", "Impressions"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "CVR" and {"Orders", "Clicks"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "CPC" and {"Spend", "Clicks"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "ACOS" and {"Spend", "Sales"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "ROAS" and {"Sales", "Spend"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "Ad % Sale" and ad_sale_possible and "Sales" in actual_base_present: can_calc_yr = True
             if can_calc_yr: calculable_metrics_for_year.append(metric)

        if not calculable_metrics_for_year: continue # Skip year if none of the selected metrics can be handled

        # Calculate Ad % Sale Denominator for the year (FU Logic - calculated internally)
        total_sales_for_period = 0
        if "Ad % Sale" in calculable_metrics_for_year:
            try:
                # FU logic for calculating denominator for the specific year/period
                df_year_valid_dates_total = df_year.dropna(subset=[date_col, 'Total Sales']) # Ensure Total Sales is not NaN for calc
                df_year_valid_dates_total['Total Sales'] = pd.to_numeric(df_year_valid_dates_total['Total Sales'], errors='coerce')
                df_year_valid_dates_total.dropna(subset=['Total Sales'], inplace=True)
                if not df_year_valid_dates_total.empty:
                     unique_subset = [date_col] # FU logic used date_col for uniqueness here
                     if "Marketplace" in df_year_valid_dates_total.columns: unique_subset.append("Marketplace") # Check if MP exists
                     # Ensure unique_subset columns actually exist before using them
                     unique_subset = [col for col in unique_subset if col in df_year_valid_dates_total.columns]
                     if unique_subset: # Only drop duplicates if columns exist
                         unique_weekly_totals = df_year_valid_dates_total.drop_duplicates(subset=unique_subset)
                         total_sales_for_period = unique_weekly_totals['Total Sales'].sum()
                     else: # Fallback if no valid columns for uniqueness
                          total_sales_for_period = df_year_valid_dates_total['Total Sales'].sum()


            except Exception as e: st.warning(f"Could not calculate total sales denominator for year {yr}: {e}") # FU Warning

        # Aggregate necessary base metrics (FU Logic)
        agg_dict_final = {m: 'sum' for m in actual_base_present if m in df_year.columns and pd.api.types.is_numeric_dtype(df_year[m])} # Ensure numeric check
        if not agg_dict_final:
            # If no numeric base metrics to aggregate, create empty pivot with group names
             df_pivot = pd.DataFrame({group_by_col: df_year[group_by_col].unique()})
        else:
            try:
                 # Ensure columns are numeric before aggregation
                 for col_to_agg in agg_dict_final:
                      df_year[col_to_agg] = pd.to_numeric(df_year[col_to_agg], errors='coerce')
                 df_pivot = df_year.groupby(group_by_col).agg(agg_dict_final).reset_index()
            except Exception as e: st.warning(f"Error aggregating data for {group_by_col} in year {yr}: {e}"); continue # FU Warning

        # Calculate derived metrics post-aggregation (FU Logic)
        if "CTR" in calculable_metrics_for_year: df_pivot["CTR"] = df_pivot.apply(lambda r: (r.get("Clicks",0) / r.get("Impressions",0) * 100) if r.get("Impressions") else 0, axis=1)
        if "CVR" in calculable_metrics_for_year: df_pivot["CVR"] = df_pivot.apply(lambda r: (r.get("Orders",0) / r.get("Clicks",0) * 100) if r.get("Clicks") else 0, axis=1)
        if "CPC" in calculable_metrics_for_year: df_pivot["CPC"] = df_pivot.apply(lambda r: (r.get("Spend",0) / r.get("Clicks",0)) if r.get("Clicks") else np.nan, axis=1)
        if "ACOS" in calculable_metrics_for_year: df_pivot["ACOS"] = df_pivot.apply(lambda r: (r.get("Spend",0) / r.get("Sales",0) * 100) if r.get("Sales") else np.nan, axis=1)
        if "ROAS" in calculable_metrics_for_year: df_pivot["ROAS"] = df_pivot.apply(lambda r: (r.get("Sales",0) / r.get("Spend",0)) if r.get("Spend") else np.nan, axis=1)
        if "Ad % Sale" in calculable_metrics_for_year: df_pivot["Ad % Sale"] = df_pivot.apply( lambda r: (r.get("Sales", 0) / total_sales_for_period * 100) if total_sales_for_period > 0 else np.nan, axis=1 )

        # Clean up calculated metrics (FU Logic)
        df_pivot = df_pivot.replace([np.inf, -np.inf], np.nan)
        # Select only columns needed and rename for the specific year (FU Logic)
        final_cols_for_year = [group_by_col] + [m for m in calculable_metrics_for_year if m in df_pivot.columns]
        df_pivot_final = df_pivot[final_cols_for_year].rename(columns={m: f"{m} {yr}" for m in calculable_metrics_for_year})
        yearly_tables.append(df_pivot_final)

    # Merge yearly tables (FU Logic)
    if not yearly_tables: return pd.DataFrame()
    try: merged_table = reduce(lambda left, right: pd.merge(left, right, on=group_by_col, how="outer"), yearly_tables)
    except Exception as e: st.error(f"Error merging yearly {group_by_col} tables: {e}"); return pd.DataFrame() # FU Error

    # Fill NaNs for base metrics (FU Logic)
    base_sum_metrics = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]
    cols_to_fill_zero = [f"{m} {yr}" for yr in years_to_process for m in base_sum_metrics if f"{m} {yr}" in merged_table.columns]
    if cols_to_fill_zero: merged_table[cols_to_fill_zero] = merged_table[cols_to_fill_zero].fillna(0)

    # Order columns and calculate % change (FU Logic - single change column)
    ordered_cols = [group_by_col]
    # Get actual years present in merged table columns
    actual_years_in_data = sorted(list(set(int(y.group(1)) for col in merged_table.columns if (y := re.search(r'(\d{4})$', col)) is not None)))


    if len(actual_years_in_data) >= 2:
        current_year_sel, prev_year_sel = actual_years_in_data[-1], actual_years_in_data[-2]
        percentage_metrics = {"CTR", "CVR", "ACOS", "Ad % Sale"} # FU definition
        for metric in selected_metrics: # Iterate through originally selected metrics
            col_current, col_prev = f"{metric} {current_year_sel}", f"{metric} {prev_year_sel}"
            change_col_name = f"{metric} % Change" # FU naming

            # Append year columns if they exist
            if col_prev in merged_table.columns: ordered_cols.append(col_prev)
            if col_current in merged_table.columns: ordered_cols.append(col_current)

            # Calculate change only if both columns exist
            if col_current in merged_table.columns and col_prev in merged_table.columns:
                 # Convert to numeric safely before calculation
                 val_curr = pd.to_numeric(merged_table[col_current], errors='coerce')
                 val_prev = pd.to_numeric(merged_table[col_prev], errors='coerce')

                 if metric in percentage_metrics: # Absolute change for % metrics
                     merged_table[change_col_name] = val_curr - val_prev
                 else: # Percentage change for others
                      # Use abs() for denominator, handle division by zero
                      merged_table[change_col_name] = np.where(
                          (val_prev.notna()) & (val_prev != 0) & (val_curr.notna()), # Ensure prev notna/0 and curr notna
                          ((val_curr - val_prev) / val_prev.abs()) * 100,
                          np.nan # NaN if prev is 0 or NaN, or curr is NaN
                      )
                      # Handle 0 to 0 change -> 0%
                      mask_zero_to_zero = (val_prev == 0) & (val_curr == 0)
                      merged_table.loc[mask_zero_to_zero, change_col_name] = 0.0
                      # Handle NaN to 0 change -> 0% ? (FU code didn't explicitly, np.where handles NaN input -> NaN output)

                 merged_table[change_col_name] = merged_table[change_col_name].replace([np.inf, -np.inf], np.nan) # Handle potential Inf/-Inf
                 ordered_cols.append(change_col_name) # Append the change column

    elif actual_years_in_data: # Only one year of data
        yr_single = actual_years_in_data[0]
        ordered_cols.extend([f"{m} {yr_single}" for m in selected_metrics if f"{m} {yr_single}" in merged_table.columns])

    # Final column selection and renaming (FU Logic)
    ordered_cols = [col for col in ordered_cols if col in merged_table.columns] # Ensure columns exist
    merged_table_display = merged_table[ordered_cols].copy()
    final_display_col = display_col_name or group_by_col # Use display name if provided
    if group_by_col in merged_table_display.columns:
        merged_table_display = merged_table_display.rename(columns={group_by_col: final_display_col})

    # Sorting (FU Logic)
    if len(actual_years_in_data) >= 1:
        last_yr = actual_years_in_data[-1]
        # Sort by the first selected metric's value in the last year
        sort_col_metric = selected_metrics[0] if selected_metrics else None
        sort_col = f"{sort_col_metric} {last_yr}" if sort_col_metric else None

        if sort_col and sort_col in merged_table_display.columns:
            try:
                # Ensure column is numeric before sorting
                merged_table_display[sort_col] = pd.to_numeric(merged_table_display[sort_col], errors='coerce')
                merged_table_display = merged_table_display.sort_values(sort_col, ascending=False, na_position='last')
            except Exception as e: st.warning(f"Could not sort table by column '{sort_col}': {e}") # FU Warning

    return merged_table_display


# Note: style_yoy_comparison_table is identical to FU version
def style_yoy_comparison_table(df):
    """Styles the YoY comparison table with formats and % change coloring using applymap. (LOGIC IDENTICAL TO FILE UPLOADER VERSION)
       Inverts colors for 'ACOS % Change'.
    """
    # This function's logic is identical to the provided FU version
    if df is None or df.empty: return None
    df_copy = df.copy().replace([np.inf, -np.inf], np.nan)

    format_dict = {}
    highlight_change_cols = []
    percentage_metrics = {"CTR", "CVR", "ACOS", "Ad % Sale"} # Metrics where change is absolute difference

    # --- Determine Formats and Identify Change Columns ---
    for col in df_copy.columns:
        base_metric_match = re.match(r"([a-zA-Z\s%]+)", col)
        base_metric = base_metric_match.group(1).strip() if base_metric_match else ""
        is_change_col = "% Change" in col # FU change column name
        is_metric_col = not is_change_col and any(char.isdigit() for char in col) # Basic check if year is in col name

        if is_change_col:
            base_metric_for_change = col.replace(" % Change", "").strip()
            # Format absolute change for percentage metrics, percentage change for others (FU Logic)
            if base_metric_for_change in percentage_metrics:
                 format_dict[col] = lambda x: f"{x:+.1f}%" if pd.notna(x) else 'N/A' # FU used % sign even for points diff
            else:
                 format_dict[col] = lambda x: f"{x:+.0f}%" if pd.notna(x) else 'N/A' # Standard percentage change
            highlight_change_cols.append(col)
        elif is_metric_col:
            # Apply standard metric formatting (FU Logic)
            if base_metric in ["Impressions", "Clicks", "Orders", "Units"]: format_dict[col] = "{:,.0f}"
            elif base_metric in ["Spend", "Sales", "CPC"]: format_dict[col] = "${:,.2f}"
            elif base_metric in ["ACOS", "CTR", "CVR", "Ad % Sale"]: format_dict[col] = '{:.1f}%'
            elif base_metric == "ROAS": format_dict[col] = '{:.2f}'
        elif df_copy[col].dtype == 'object' and col == df_copy.columns[0]: # Format the first (grouping) column as string
             format_dict[col] = "{}" # Ensure grouping column isn't formatted as number

    # --- Define Coloring Functions (FU Logic) ---
    def color_pos_neg_standard(val):
        """Standard coloring: positive is green, negative is red."""
        if isinstance(val, str) and val == "N/A": return 'color: grey'
        numeric_val = pd.to_numeric(val, errors='coerce')
        if pd.isna(numeric_val): return 'color: grey'
        elif numeric_val > 0.001: return 'color: green' # Added tolerance for floating point
        elif numeric_val < -0.001: return 'color: red'
        else: return 'color: inherit' # Or black/grey for zero change

    def color_pos_neg_inverted(val):
        """Inverted coloring (for ACOS): positive is red, negative is green."""
        if isinstance(val, str) and val == "N/A": return 'color: grey'
        numeric_val = pd.to_numeric(val, errors='coerce')
        if pd.isna(numeric_val): return 'color: grey'
        elif numeric_val > 0.001: return 'color: red'    # Positive change (ACOS increase) is red (bad)
        elif numeric_val < -0.001: return 'color: green' # Negative change (ACOS decrease) is green (good)
        else: return 'color: inherit' # Or black/grey for zero change

    # --- Apply Formatting ---
    try:
        styled_table = df_copy.style.format(format_dict, na_rep="N/A", precision=2) # Added precision
    except Exception as e:
        st.error(f"Error applying format to YOY table: {e}")
        return df_copy.style # Return basic styler on error

    # --- Apply Conditional Coloring ---
    for change_col in highlight_change_cols:
        if change_col in df_copy.columns:
            try:
                # Choose the appropriate coloring function based on the column name
                if change_col == "ACOS % Change": # FU column name
                     color_func_to_apply = color_pos_neg_inverted
                else:
                     color_func_to_apply = color_pos_neg_standard

                # Apply the chosen function element-wise using applymap (FU Logic)
                styled_table = styled_table.applymap(color_func_to_apply, subset=[change_col])

            except Exception as e:
                st.warning(f"Could not apply color style to YOY column '{change_col}': {e}") # FU Warning

    # --- Apply Alignment (Using set_properties like FU final version) ---
    text_align='right'
    try:
        first_col_name = df_copy.columns[0]
        styled_table = styled_table.set_properties(**{'text-align': text_align})
        styled_table = styled_table.set_properties(subset=[first_col_name], **{'text-align': 'left'})

    except Exception as e:
        st.warning(f"Could not apply specific alignment to YOY table: {e}. Using general alignment.") # FU Warning
        styled_table = styled_table.set_properties(**{'text-align': text_align}) # Fallback

    return styled_table


@st.cache_data
def calculate_yoy_summary_row(df, selected_metrics, years_to_process, id_col_name, id_col_value):
    """Calculates a single summary row with YoY comparison based on yearly totals. (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # This function's logic is identical to the provided FU version
    # Includes internal Ad % Sale denominator calculation
    if df is None or df.empty or not years_to_process: return pd.DataFrame()

    date_col = "WE Date" # Assume from FU version
    ad_sale_possible = ("Ad % Sale" in selected_metrics and {"Sales", "Total Sales", date_col}.issubset(df.columns)) # FU Check
    if "Ad % Sale" in selected_metrics and not ad_sale_possible:
        selected_metrics = [m for m in selected_metrics if m != "Ad % Sale"] # FU Logic

    summary_row_data = {id_col_name: id_col_value}
    yearly_totals = {yr: {} for yr in years_to_process}
    yearly_total_sales_denom = {yr: 0 for yr in years_to_process} # Denom calculated here

    # Determine base metrics needed (FU Logic)
    base_metrics_needed = set()
    for m in selected_metrics:
        if m in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]: base_metrics_needed.add(m)
        elif m == "CTR": base_metrics_needed.update(["Clicks", "Impressions"])
        elif m == "CVR": base_metrics_needed.update(["Orders", "Clicks"])
        elif m == "CPC": base_metrics_needed.update(["Spend", "Clicks"])
        elif m == "ACOS": base_metrics_needed.update(["Spend", "Sales"])
        elif m == "ROAS": base_metrics_needed.update(["Sales", "Spend"])
        elif m == "Ad % Sale": base_metrics_needed.add("Sales")

    # Calculate base totals and Ad % Sale Denom per year (FU Logic)
    for yr in years_to_process:
        df_year = df[df["Year"] == yr]
        if df_year.empty: continue
        for base_m in base_metrics_needed:
            if base_m in df_year.columns: yearly_totals[yr][base_m] = pd.to_numeric(df_year[base_m], errors='coerce').fillna(0).sum()
            else: yearly_totals[yr][base_m] = 0 # Default 0 if missing

        # Calculate denom for the year internally (FU Logic)
        if ad_sale_possible and "Ad % Sale" in selected_metrics:
            try:
                df_year_valid_dates = df_year.dropna(subset=[date_col, 'Total Sales'])
                df_year_valid_dates['Total Sales'] = pd.to_numeric(df_year_valid_dates['Total Sales'], errors='coerce')
                df_year_valid_dates.dropna(subset=['Total Sales'], inplace=True)
                if not df_year_valid_dates.empty:
                     unique_subset = [date_col] # FU logic for uniqueness
                     if "Marketplace" in df_year_valid_dates.columns: unique_subset.append("Marketplace")
                     unique_subset = [col for col in unique_subset if col in df_year_valid_dates.columns] # Check exist
                     if unique_subset:
                         unique_totals = df_year_valid_dates.drop_duplicates(subset=unique_subset)
                         yearly_total_sales_denom[yr] = unique_totals['Total Sales'].sum()
                     else: # Fallback sum
                         yearly_total_sales_denom[yr] = df_year_valid_dates['Total Sales'].sum()

            except Exception as e: yearly_total_sales_denom[yr] = 0 # Default 0 on error

    # Calculate derived metrics and populate row data (FU Logic)
    for metric in selected_metrics:
        for yr in years_to_process:
            totals_yr = yearly_totals.get(yr, {})
            calc_val = np.nan
            try:
                 if metric == "CTR": calc_val = (totals_yr.get("Clicks", 0) / totals_yr.get("Impressions", 0) * 100) if totals_yr.get("Impressions", 0) > 0 else 0
                 elif metric == "CVR": calc_val = (totals_yr.get("Orders", 0) / totals_yr.get("Clicks", 0) * 100) if totals_yr.get("Clicks", 0) > 0 else 0
                 elif metric == "CPC": calc_val = (totals_yr.get("Spend", 0) / totals_yr.get("Clicks", 0)) if totals_yr.get("Clicks", 0) > 0 else np.nan
                 elif metric == "ACOS": calc_val = (totals_yr.get("Spend", 0) / totals_yr.get("Sales", 0) * 100) if totals_yr.get("Sales", 0) > 0 else np.nan
                 elif metric == "ROAS": calc_val = (totals_yr.get("Sales", 0) / totals_yr.get("Spend", 0)) if totals_yr.get("Spend", 0) > 0 else np.nan
                 elif metric == "Ad % Sale":
                      denom_yr = yearly_total_sales_denom.get(yr, 0)
                      calc_val = (totals_yr.get("Sales", 0) / denom_yr * 100) if denom_yr > 0 else np.nan
                 elif metric in totals_yr: calc_val = totals_yr.get(metric) # Use aggregated base value

                 if isinstance(calc_val, (int, float)): calc_val = np.nan if calc_val in [np.inf, -np.inf] else calc_val
            except Exception as e: calc_val = np.nan # Set NaN on calculation error

            # Store calculated value for metric/year (used for change calc below)
            if yr in yearly_totals: yearly_totals[yr][metric] = calc_val
            # Add to output dict
            summary_row_data[f"{metric} {yr}"] = calc_val

    # Calculate % Change (FU Logic - single change column)
    actual_years_in_row = sorted([yr for yr in years_to_process if yr in yearly_totals and yearly_totals[yr]]) # Check dict exists
    if len(actual_years_in_row) >= 2:
        curr_yr, prev_yr = actual_years_in_row[-1], actual_years_in_row[-2]
        percentage_metrics = {"CTR", "CVR", "ACOS", "Ad % Sale"}
        for metric in selected_metrics:
            val_curr = yearly_totals.get(curr_yr, {}).get(metric, np.nan)
            val_prev = yearly_totals.get(prev_yr, {}).get(metric, np.nan)
            change_val = np.nan # Use different name than FU's pct_change for clarity
            # FU calculation logic
            if pd.notna(val_curr) and pd.notna(val_prev):
                 if metric in percentage_metrics: change_val = val_curr - val_prev # Absolute diff
                 else: # Percentage diff
                      if val_prev != 0: change_val = ((val_curr - val_prev) / abs(val_prev)) * 100
                      elif val_curr == 0: change_val = 0.0 # Handle 0 to 0
            # Handle potential Inf/-Inf before adding to dict
            change_val = np.nan if change_val in [np.inf, -np.inf] else change_val
            summary_row_data[f"{metric} % Change"] = change_val # FU column name

    # Create DataFrame and order columns (FU Logic)
    summary_df = pd.DataFrame([summary_row_data])
    ordered_summary_cols = [id_col_name]
    if len(actual_years_in_row) >= 2:
        curr_yr_o, prev_yr_o = actual_years_in_row[-1], actual_years_in_row[-2]
        for metric in selected_metrics:
            if f"{metric} {prev_yr_o}" in summary_df.columns: ordered_summary_cols.append(f"{metric} {prev_yr_o}")
            if f"{metric} {curr_yr_o}" in summary_df.columns: ordered_summary_cols.append(f"{metric} {curr_yr_o}")
            if f"{metric} % Change" in summary_df.columns: ordered_summary_cols.append(f"{metric} % Change")
    elif len(actual_years_in_row) == 1:
        yr_o = actual_years_in_row[0]
        ordered_summary_cols.extend([f"{metric} {yr_o}" for metric in selected_metrics if f"{metric} {yr_o}" in summary_df.columns])

    final_summary_cols = [col for col in ordered_summary_cols if col in summary_df.columns] # Ensure existence
    return summary_df[final_summary_cols]


# --- End of Chunk 2 (Unchanged) ---

# CONVERTED GSHEET INPUT CODE - CHUNK 3/3 - MODIFIED FOR SECRETS
# Logic based identically on File Uploader Version (April 4, 2025)
# Data input mechanism changed to load GSheet URL/Name from secrets.toml
# Contains main app logic, automatic loading, processing trigger, and tabs.
# Modified on: Thursday, April 10, 2025 at 1:57:54 PM BST


# [Imports, load_data_from_gsheet, and ALL helper functions from Chunk 1 & 2
#  (preprocess_ad_data, filter_data_by_timeframe, create_metric_comparison_chart,
#   create_performance_metrics_table, create_metric_over_time_chart, style_dataframe,
#   style_total_summary, style_metrics_table, generate_insights, create_yoy_grouped_table,
#   style_yoy_comparison_table, calculate_yoy_summary_row)
#  should be defined above this point]
# =============================================================================


# =============================================================================
# --- Title and Logo (IDENTICAL TO FILE UPLOADER VERSION) ---
# (UNCHANGED)
# =============================================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Advertising Dashboard 📊")
with col2:
    try: st.image("logo.png", width=250) # FU version used width=250
    except Exception as e: st.warning(f"Could not load logo.png: {e}")


# =============================================================================
# --- Automatic Data Loading using Secrets (MODIFIED SECTION) ---
# =============================================================================

# --- Get Config from Secrets ---
try:
    # Reads the values you added to secrets.toml
    GSHEET_URL_OR_ID = st.secrets["gsheet_config"]["url_or_id"]
    WORKSHEET_NAME = st.secrets["gsheet_config"]["worksheet_name"]
    secrets_loaded = True
    # Display which sheet is configured (optional, good for debugging)
    st.sidebar.caption(f"Configured GSheet: '{WORKSHEET_NAME}'")
    st.sidebar.info("Data source configured via secrets. Ensure sheet is shared with the service account.")

except KeyError as e:
    st.error(f"Missing configuration in secrets.toml: {e}")
    st.error("Ensure '[gsheet_config]' section with 'url_or_id' and 'worksheet_name' keys exists.")
    secrets_loaded = False
    st.stop() # Stop the app if config is missing
except Exception as e:
    st.error(f"Error loading GSheet config from secrets: {e}")
    secrets_loaded = False
    st.stop() # Stop the app on other secret loading errors

# --- Automatic Loading Trigger Logic ---
raw_data_available = False # Initialize
if secrets_loaded:
    # Initialize session state flags if they don't exist
    if "data_loaded_from_gsheet" not in st.session_state:
        st.session_state.data_loaded_from_gsheet = False
    if "processed_gsheet_url" not in st.session_state: # Tracks which sheet config was last processed
        st.session_state.processed_gsheet_url = None
    if "processed_gsheet_name" not in st.session_state:
         st.session_state.processed_gsheet_name = None
    if "current_gsheet_url" not in st.session_state: # Tracks currently loaded sheet config
         st.session_state.current_gsheet_url = None
    if "current_gsheet_name" not in st.session_state:
         st.session_state.current_gsheet_name = None


    # Determine if loading is needed
    needs_loading = False
    if not st.session_state.data_loaded_from_gsheet:
        needs_loading = True # Load if never loaded before in this session
        st.info("No data loaded yet. Attempting initial load from secrets configuration...")
    elif st.session_state.current_gsheet_url != GSHEET_URL_OR_ID or \
         st.session_state.current_gsheet_name != WORKSHEET_NAME:
        # This condition handles if secrets were updated *after* a successful load
        # It ensures the app tries to load the *new* sheet defined in secrets.
        needs_loading = True
        st.info("GSheet configuration in secrets may have changed. Reloading data...")
        st.session_state.data_loaded_from_gsheet = False # Force reload status


    # Perform loading if needed
    if needs_loading:
        with st.spinner(f"Loading data from Google Sheet '{WORKSHEET_NAME}'..."):
            # Call the existing function using secrets values
            raw_data = load_data_from_gsheet(GSHEET_URL_OR_ID, WORKSHEET_NAME)

        if not raw_data.empty:
            st.session_state["ad_data_raw"] = raw_data
            # Update session state to reflect the currently loaded data source config
            st.session_state.current_gsheet_url = GSHEET_URL_OR_ID
            st.session_state.current_gsheet_name = WORKSHEET_NAME
            st.session_state.data_loaded_from_gsheet = True
            st.sidebar.success(f"Loaded {len(raw_data)} rows.") # Success message in sidebar

            # Clear downstream state to force reprocessing with potentially new data
            keys_to_delete_on_reload = ['ad_data_filtered', 'ad_data_processed',
                                        'processed_marketplace', 'processed_gsheet_url',
                                        'processed_gsheet_name', 'marketplace_selector_value']
            for key in keys_to_delete_on_reload:
                if key in st.session_state: del st.session_state[key]
            st.rerun() # Rerun the script to trigger the processing logic below

        else:
            # Errors should be shown by load_data_from_gsheet function (likely in sidebar)
            st.session_state.data_loaded_from_gsheet = False
            if "ad_data_raw" in st.session_state: del st.session_state["ad_data_raw"]
            st.error(f"Failed to load data from GSheet: '{WORKSHEET_NAME}'. Check sharing, names, and secrets.")
            # Do NOT rerun on failure, wait for user action or fix

    # Update raw_data_available check based on successful load
    raw_data_available = "ad_data_raw" in st.session_state and not st.session_state["ad_data_raw"].empty and st.session_state.data_loaded_from_gsheet


# --- Process Loaded Data (Marketplace Selection & Preprocessing Trigger) ---
# This logic closely follows the File Uploader version's structure for marketplace selection
# and deciding when to re-process. It now runs AFTER the automatic loading attempt.
# (UNCHANGED LOGIC, operates on auto-loaded data)

selected_mp_widget = "All Marketplaces" # Default value if data not loaded

if raw_data_available:
    # --- Marketplace Selector (Logic from FU Version) ---
    marketplace_options = ["All Marketplaces"]
    default_marketplace = "All Marketplaces"
    default_mp_index = 0
    marketplace_col_name = "Marketplace" # Assume from FU context

    # Populate options from raw data
    if marketplace_col_name in st.session_state["ad_data_raw"].columns:
        raw_df_for_options = st.session_state["ad_data_raw"]
        if not raw_df_for_options.empty:
            available_marketplaces = sorted([str(mp) for mp in raw_df_for_options[marketplace_col_name].dropna().unique() if str(mp).strip()]) # Handle potential empty strings
            if available_marketplaces:
                marketplace_options.extend(available_marketplaces)
                # Try to set 'US' as default if available (FU Logic)
                target_default = "US"
                if target_default in marketplace_options:
                    default_marketplace = target_default
                    default_mp_index = marketplace_options.index(target_default)
    else:
        st.sidebar.warning(f"'{marketplace_col_name}' column not found. Cannot filter by Marketplace.")

    # Ensure session state value exists before rendering (FU Logic adaptation)
    if 'marketplace_selector_value' not in st.session_state:
        st.session_state.marketplace_selector_value = default_marketplace

    # Get current index based on state, ensure value is valid (FU Logic adaptation)
    try:
        current_value = st.session_state.marketplace_selector_value
        if current_value not in marketplace_options:
            current_value = default_marketplace # Reset
            st.session_state.marketplace_selector_value = current_value # Update state
        # current_mp_index = marketplace_options.index(current_value) # Index not strictly needed if using key
    except ValueError:
        # Fallback if state value is somehow invalid
        st.session_state.marketplace_selector_value = default_marketplace

    # Display the widget using key for state persistence (FU Logic adaptation)
    selected_mp_widget = st.sidebar.selectbox(
        "Select Marketplace",
        options=marketplace_options,
        key="marketplace_selector_value" # Links widget value to this session state key
    )

    # --- Check if Reprocessing is Needed (Logic adapted from FU version) ---
    needs_processing = False
    if "ad_data_processed" not in st.session_state:
        needs_processing = True
    # Check if marketplace selection changed
    elif st.session_state.get("processed_marketplace") != selected_mp_widget:
         needs_processing = True
    # Check if the GSheet source itself changed since last processing
    # Compares the *processed* sheet details with the *currently loaded* sheet details
    elif st.session_state.get("processed_gsheet_url") != st.session_state.get("current_gsheet_url") or \
         st.session_state.get("processed_gsheet_name") != st.session_state.get("current_gsheet_name"):
         needs_processing = True

    # --- Perform Filtering and Preprocessing ONLY if needed ---
    if needs_processing:
        with st.spinner("Processing data..."): # Use spinner for feedback
            current_selection_for_processing = selected_mp_widget # Use current selection

            try:
                if "ad_data_raw" not in st.session_state or st.session_state["ad_data_raw"].empty: # Added empty check here too
                    st.error("Raw data not available or empty for processing.")
                    # Skip processing block if raw data is missing
                else:
                    ad_data_to_filter = st.session_state["ad_data_raw"]

                    # --- Filter by Marketplace (FU Logic) ---
                    temp_filtered_data = pd.DataFrame() # Initialize
                    if current_selection_for_processing != "All Marketplaces":
                        if marketplace_col_name in ad_data_to_filter.columns:
                            temp_filtered_data = ad_data_to_filter[
                                ad_data_to_filter[marketplace_col_name].astype(str) == current_selection_for_processing
                            ].copy()
                        else:
                            # Warning issued above if column missing, use all data
                            temp_filtered_data = ad_data_to_filter.copy()
                    else:
                        temp_filtered_data = ad_data_to_filter.copy()

                    # --- PREPROCESSING (Using FU version's preprocess_ad_data) ---
                    if not temp_filtered_data.empty:
                        # Store intermediate filtered before passing to preprocess
                        st.session_state["ad_data_filtered"] = temp_filtered_data # Store potentially filtered data if needed elsewhere
                        # CALLING THE FILE UPLOADER VERSION'S PREPROCESSOR
                        st.session_state["ad_data_processed"] = preprocess_ad_data(temp_filtered_data)

                        if "ad_data_processed" not in st.session_state or st.session_state["ad_data_processed"].empty:
                             st.error("Preprocessing resulted in empty data. Please check data quality and formats for the selected marketplace.")
                             # Clear potentially invalid processed data but keep context
                             if "ad_data_processed" in st.session_state: del st.session_state["ad_data_processed"]
                             # Record the context that led to the empty processed data
                             st.session_state.processed_marketplace = current_selection_for_processing
                             st.session_state.processed_gsheet_url = st.session_state.current_gsheet_url
                             st.session_state.processed_gsheet_name = st.session_state.current_gsheet_name
                             # No rerun here, let user see the error
                        else:
                             # Store context associated with the successfully processed data
                             st.session_state.processed_marketplace = current_selection_for_processing
                             st.session_state.processed_gsheet_url = st.session_state.current_gsheet_url
                             st.session_state.processed_gsheet_name = st.session_state.current_gsheet_name
                             st.success(f"Data processed for '{current_selection_for_processing}'.") # Give success feedback
                             st.rerun() # Rerun to ensure UI updates with processed data

                    else:
                        st.warning(f"Data is empty after filtering for Marketplace: '{current_selection_for_processing}'. Cannot preprocess.")
                        # Clear potentially stale processed data
                        if "ad_data_processed" in st.session_state: del st.session_state["ad_data_processed"]
                        st.session_state.processed_marketplace = current_selection_for_processing # Store the MP that resulted in empty data
                        st.session_state.processed_gsheet_url = st.session_state.current_gsheet_url
                        st.session_state.processed_gsheet_name = st.session_state.current_gsheet_name
                        # st.rerun() # Optional rerun to clear potential old views

            except Exception as e:
                st.error(f"Error during data processing: {e}")
                # Clean up potentially inconsistent state on error
                keys_to_del = ['ad_data_filtered', 'ad_data_processed', 'processed_marketplace', 'processed_gsheet_url', 'processed_gsheet_name']
                for key in keys_to_del:
                    if key in st.session_state: del st.session_state[key]


# =============================================================================
# Display Dashboard Tabs Only When Data is Loaded and Processed
# (LOGIC IDENTICAL TO FILE UPLOADER VERSION, using processed data from GSheet now)
# (UNCHANGED LOGIC)
# =============================================================================
# Check if processed data is available and valid for the *current* GSheet source/marketplace config
processed_data_valid_and_current = False
if "ad_data_processed" in st.session_state and not st.session_state["ad_data_processed"].empty:
    # Check if the processed data corresponds to the GSheet source defined in secrets AND the current marketplace selection
    if 'GSHEET_URL_OR_ID' in locals() and 'WORKSHEET_NAME' in locals(): # Check if secrets were loaded
      if st.session_state.get("processed_gsheet_url") == GSHEET_URL_OR_ID and \
         st.session_state.get("processed_gsheet_name") == WORKSHEET_NAME and \
         st.session_state.get("processed_marketplace") == st.session_state.get("marketplace_selector_value"):
            processed_data_valid_and_current = True


if processed_data_valid_and_current: # Use the flag

    ad_data_processed_current = st.session_state["ad_data_processed"] # Use the validated processed data

    tabs_adv = st.tabs([
        "YOY Comparison",
        "Sponsored Products",
        "Sponsored Brands",
        "Sponsored Display"
    ])

    # -------------------------------
    # Tab 0: YOY Comparison (IDENTICAL TO FILE UPLOADER VERSION LOGIC)
    # (UNCHANGED LOGIC)
    # -------------------------------
    with tabs_adv[0]:
        st.markdown("### YOY Comparison")
        # Use the processed data from session state
        ad_data_overview = ad_data_processed_current.copy()

        st.markdown("#### Select Comparison Criteria")
        col1_yoy, col2_yoy, col3_yoy, col4_yoy = st.columns(4)

        with col1_yoy:
            if "Year" not in ad_data_overview.columns:
                 st.error("'Year' column missing. Cannot create YOY filters.")
                 available_years_yoy = []
                 selected_years_yoy = []
            else:
                 available_years_yoy = sorted(ad_data_overview["Year"].dropna().unique()) # Handle potential NaNs just in case
                 default_years_yoy = available_years_yoy[-2:] if len(available_years_yoy) >= 2 else available_years_yoy
                 selected_years_yoy = st.multiselect("Select Year(s):", available_years_yoy, default=default_years_yoy, key="yoy_years")

        with col2_yoy:
            timeframe_options_yoy = ["Specific Week", "Last 4 Weeks", "Last 8 Weeks", "Last 12 Weeks"]
            default_tf_index_yoy = timeframe_options_yoy.index("Last 4 Weeks") if "Last 4 Weeks" in timeframe_options_yoy else 0
            selected_timeframe_yoy = st.selectbox("Select Timeframe:", timeframe_options_yoy, index=default_tf_index_yoy, key="yoy_timeframe")

        with col3_yoy:
            available_weeks_str_yoy = ["Select..."]
            selected_week_yoy = None # Initialize
            if selected_years_yoy and "Week" in ad_data_overview.columns:
                try:
                    # Ensure selected_years_yoy contains integers if needed for filtering
                    selected_years_yoy_int = [int(y) for y in selected_years_yoy]
                    weeks_in_selected_years = ad_data_overview[ad_data_overview["Year"].isin(selected_years_yoy_int)]["Week"].unique()
                    available_weeks_yoy = sorted([int(w) for w in weeks_in_selected_years if pd.notna(w)])
                    available_weeks_str_yoy.extend([str(w) for w in available_weeks_yoy])
                except Exception as e: st.warning(f"Could not retrieve weeks: {e}") # FU Warning
            is_specific_week_yoy = (selected_timeframe_yoy == "Specific Week")
            selected_week_option_yoy = st.selectbox("Select Week:", available_weeks_str_yoy, index=0, key="yoy_week", disabled=(not is_specific_week_yoy))
            # FU logic for getting selected week
            selected_week_yoy = int(selected_week_option_yoy) if is_specific_week_yoy and selected_week_option_yoy != "Select..." else None

        with col4_yoy:
            # FU Logic for determining available metrics
            all_metrics_yoy = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "CPC", "ACOS", "ROAS", "Ad % Sale"]
            calculable_metrics_yoy = {"CTR", "CVR", "CPC", "ACOS", "ROAS", "Ad % Sale"}
            original_cols_yoy = set(ad_data_processed_current.columns) # Check processed data
            available_display_metrics_yoy = []
            for m in all_metrics_yoy:
                 if m in original_cols_yoy: available_display_metrics_yoy.append(m)
                 elif m in calculable_metrics_yoy:
                     can_calc_m = False
                     if m == "CTR" and {"Clicks", "Impressions"}.issubset(original_cols_yoy): can_calc_m = True
                     elif m == "CVR" and {"Orders", "Clicks"}.issubset(original_cols_yoy): can_calc_m = True
                     elif m == "CPC" and {"Spend", "Clicks"}.issubset(original_cols_yoy): can_calc_m = True
                     elif m == "ACOS" and {"Spend", "Sales"}.issubset(original_cols_yoy): can_calc_m = True
                     elif m == "ROAS" and {"Sales", "Spend"}.issubset(original_cols_yoy): can_calc_m = True
                     elif m == "Ad % Sale" and {"Sales", "Total Sales", "WE Date"}.issubset(original_cols_yoy): can_calc_m = True
                     if can_calc_m: available_display_metrics_yoy.append(m)
            default_metrics_list_yoy = ["Spend", "Sales", "Ad % Sale", "ACOS"]
            default_metrics_yoy = [m for m in default_metrics_list_yoy if m in available_display_metrics_yoy]
            selected_metrics_yoy = st.multiselect("Select Metrics:", available_display_metrics_yoy, default=default_metrics_yoy, key="yoy_metrics")
            if not selected_metrics_yoy:
                 selected_metrics_yoy = default_metrics_yoy[:1] if default_metrics_yoy else available_display_metrics_yoy[:1] if available_display_metrics_yoy else []
                 if not selected_metrics_yoy: st.warning("No metrics available for selection.") # FU Warning adjusted

        # --- YOY Tab Table Display Logic (IDENTICAL TO FILE UPLOADER VERSION) ---
        if not selected_years_yoy: st.warning("Please select at least one year.") # FU Warning
        elif not selected_metrics_yoy: st.warning("Please select at least one metric.") # FU Warning
        else:
            # Use the FU version's filter function
            filtered_data_yoy = filter_data_by_timeframe(ad_data_overview, selected_years_yoy, selected_timeframe_yoy, selected_week_yoy)
            if filtered_data_yoy.empty: st.info("No data available for the selected YOY criteria (Years/Timeframe/Week).") # FU Info
            else:
                 years_to_process_yoy = sorted(filtered_data_yoy['Year'].unique())
                 if not years_to_process_yoy: # Added check
                      st.info("Filtered data contains no valid years for comparison.")
                 else:
                      # --- Product Overview Table ---
                      st.markdown("---"); st.markdown("#### Overview by Product Type")
                      st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics.*") # FU Caption
                      # Use the FU version's table function
                      product_overview_yoy_table = create_yoy_grouped_table(df_filtered_period=filtered_data_yoy, group_by_col="Product", selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Product")
                      if not product_overview_yoy_table.empty:
                          styled_product_overview_yoy = style_yoy_comparison_table(product_overview_yoy_table) # Use FU styling
                          if styled_product_overview_yoy: st.dataframe(styled_product_overview_yoy, use_container_width=True)
                      else: st.info("No product overview data available.") # FU Info

                      # --- Portfolio Table ---
                      portfolio_col_yoy = next((col for col in ["Portfolio Name", "Portfolio"] if col in filtered_data_yoy.columns), None) # FU Check
                      if portfolio_col_yoy:
                          st.markdown("---"); st.markdown("#### Portfolio Performance")
                          st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics. Optionally filter by Product Type below.*") # FU Caption
                          portfolio_table_data_yoy = filtered_data_yoy.copy()
                          selected_product_portfolio_yoy = "All"
                          if "Product" in filtered_data_yoy.columns:
                              product_types_portfolio_yoy = ["All"] + sorted(filtered_data_yoy["Product"].unique().tolist())
                              selected_product_portfolio_yoy = st.selectbox("Filter Portfolio Table by Product Type:", product_types_portfolio_yoy, index=0, key="portfolio_product_filter_yoy")
                              if selected_product_portfolio_yoy != "All": portfolio_table_data_yoy = portfolio_table_data_yoy[portfolio_table_data_yoy["Product"] == selected_product_portfolio_yoy]

                          if portfolio_table_data_yoy.empty: st.info(f"No Portfolio data available for Product Type '{selected_product_portfolio_yoy}'.") # FU Info
                          else:
                              # Use the FU version's table and summary functions
                              portfolio_yoy_table = create_yoy_grouped_table(df_filtered_period=portfolio_table_data_yoy, group_by_col=portfolio_col_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Portfolio")
                              if not portfolio_yoy_table.empty:
                                  styled_portfolio_yoy = style_yoy_comparison_table(portfolio_yoy_table) # FU Style
                                  if styled_portfolio_yoy: st.dataframe(styled_portfolio_yoy, use_container_width=True)
                                  # Use FU summary function
                                  portfolio_summary_row_yoy = calculate_yoy_summary_row(df=portfolio_table_data_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="Portfolio", id_col_value=f"TOTAL - {selected_product_portfolio_yoy}")
                                  if not portfolio_summary_row_yoy.empty:
                                       st.markdown("###### YoY Total (Selected Period & Product Filter)") # FU Header
                                       styled_portfolio_summary_yoy = style_yoy_comparison_table(portfolio_summary_row_yoy) # FU Style
                                       if styled_portfolio_summary_yoy: st.dataframe(styled_portfolio_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                              else: st.info(f"No displayable portfolio data for Product Type '{selected_product_portfolio_yoy}'.") # FU Info

                      # --- Match Type Table ---
                      if {"Product", "Match Type"}.issubset(filtered_data_yoy.columns): # FU Check
                          st.markdown("---"); st.markdown("#### Match Type Performance")
                          st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics, broken down by Product Type.*") # FU Caption
                          product_types_match_yoy = ["Sponsored Products", "Sponsored Brands", "Sponsored Display"]
                          for product_type_m in product_types_match_yoy:
                              product_data_match_yoy = filtered_data_yoy[filtered_data_yoy["Product"] == product_type_m].copy()
                              if product_data_match_yoy.empty: continue
                              st.subheader(product_type_m) # FU used subheader
                              # Use FU table/summary/style functions
                              match_type_yoy_table = create_yoy_grouped_table(df_filtered_period=product_data_match_yoy, group_by_col="Match Type", selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Match Type")
                              if not match_type_yoy_table.empty:
                                  styled_match_type_yoy = style_yoy_comparison_table(match_type_yoy_table)
                                  if styled_match_type_yoy: st.dataframe(styled_match_type_yoy, use_container_width=True)
                                  match_type_summary_row_yoy = calculate_yoy_summary_row(df=product_data_match_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="Match Type", id_col_value=f"TOTAL - {product_type_m}")
                                  if not match_type_summary_row_yoy.empty:
                                       st.markdown("###### YoY Total (Selected Period)") # FU Header
                                       styled_match_type_summary_yoy = style_yoy_comparison_table(match_type_summary_row_yoy)
                                       if styled_match_type_summary_yoy: st.dataframe(styled_match_type_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                              else: st.info(f"No Match Type data available for {product_type_m}.") # FU Info

                      # --- RTW/Prospecting Table ---
                      rtw_col_name = "RTW/Prospecting" # Assume from FU context
                      if {"Product", rtw_col_name}.issubset(filtered_data_yoy.columns): # FU Check
                          st.markdown("---"); st.markdown(f"#### {rtw_col_name} Performance")
                          st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics. Choose a Product Type below.*") # FU Caption
                          rtw_product_types_yoy = ["Sponsored Products", "Sponsored Brands", "Sponsored Display"] # FU List
                          available_rtw_products_yoy = sorted([pt for pt in filtered_data_yoy["Product"].unique() if pt in rtw_product_types_yoy])
                          if available_rtw_products_yoy:
                              selected_rtw_product_yoy = st.selectbox(f"Select Product Type for {rtw_col_name} Analysis:", available_rtw_products_yoy, key="rtw_product_selector_yoy")
                              rtw_filtered_product_data_yoy = filtered_data_yoy[filtered_data_yoy["Product"] == selected_rtw_product_yoy].copy()
                              if not rtw_filtered_product_data_yoy.empty:
                                  # Use FU table/summary/style functions
                                  rtw_yoy_table = create_yoy_grouped_table(df_filtered_period=rtw_filtered_product_data_yoy, group_by_col=rtw_col_name, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name=rtw_col_name)
                                  if not rtw_yoy_table.empty:
                                       styled_rtw_yoy = style_yoy_comparison_table(rtw_yoy_table)
                                       if styled_rtw_yoy: st.dataframe(styled_rtw_yoy, use_container_width=True)
                                       rtw_summary_row_yoy = calculate_yoy_summary_row(df=rtw_filtered_product_data_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name=rtw_col_name, id_col_value=f"TOTAL - {selected_rtw_product_yoy}")
                                       if not rtw_summary_row_yoy.empty:
                                            st.markdown("###### YoY Total (Selected Period)") # FU Header
                                            styled_rtw_summary_yoy = style_yoy_comparison_table(rtw_summary_row_yoy)
                                            if styled_rtw_summary_yoy: st.dataframe(styled_rtw_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                                  else: st.info(f"No {rtw_col_name} data available for {selected_rtw_product_yoy}.") # FU Info
                              else: st.info(f"No {selected_rtw_product_yoy} data in selected period.") # FU Info
                          else: st.info("No relevant Product Types found for RTW/Prospecting analysis.") # FU Info

                      # --- Campaign Name Table ---
                      campaign_col_yoy = "Campaign Name" # Assume from FU context
                      if campaign_col_yoy in filtered_data_yoy.columns: # FU Check
                          st.markdown("---"); st.markdown(f"#### {campaign_col_yoy} Performance")
                          st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics.*") # FU Caption
                          # Use FU table/summary/style functions
                          campaign_yoy_table = create_yoy_grouped_table(df_filtered_period=filtered_data_yoy, group_by_col=campaign_col_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Campaign")
                          if not campaign_yoy_table.empty:
                              styled_campaign_yoy = style_yoy_comparison_table(campaign_yoy_table)
                              if styled_campaign_yoy: st.dataframe(styled_campaign_yoy, use_container_width=True, height=600) # FU Height
                              campaign_summary_row_yoy = calculate_yoy_summary_row(df=filtered_data_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="Campaign", id_col_value="TOTAL - All Campaigns")
                              if not campaign_summary_row_yoy.empty:
                                   st.markdown("###### YoY Total (Selected Period)") # FU Header
                                   styled_campaign_summary_yoy = style_yoy_comparison_table(campaign_summary_row_yoy)
                                   if styled_campaign_summary_yoy: st.dataframe(styled_campaign_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                          else: st.info(f"No displayable {campaign_col_yoy} data.") # FU Info

    # =========================================================================
    # == SP / SB / SD Tabs (LOGIC IDENTICAL TO FILE UPLOADER VERSION) ==
    # (UNCHANGED LOGIC)
    # =========================================================================
    for i, product_type_tab in enumerate(["Sponsored Products", "Sponsored Brands", "Sponsored Display"]):
        with tabs_adv[i+1]: # Start from tab index 1
            st.markdown(f"### {product_type_tab} Performance")
            st.caption("Charts use filters below. Tables show YoY comparison for the selected date range & metrics.") # FU Caption

            # Use unique keys for widgets in each tab
            tab_key_prefix = product_type_tab.lower().replace(" ", "_")

            ad_data_tab = ad_data_processed_current.copy() # Use validated data

            # Check if product type exists in the data (FU Check)
            product_col_name = "Product" # Assume from FU context
            if product_col_name not in ad_data_tab.columns or product_type_tab not in ad_data_tab[product_col_name].unique():
                 st.warning(f"No '{product_type_tab}' data found in the sheet for the selected Marketplace.") # FU Warning adjusted
                 continue # Skip to the next tab

            ad_data_tab_filtered_initial = ad_data_tab[ad_data_tab[product_col_name] == product_type_tab].copy()
            if ad_data_tab_filtered_initial.empty:
                 st.warning(f"No {product_type_tab} data available after initial filtering (check selected marketplace/dates).") # FU Warning
                 continue

            # --- Filters (FU Logic) ---
            with st.expander("Filters", expanded=True):
                 col1_tab, col2_tab, col3_tab = st.columns(3) # FU Layout
                 selected_metric_tab = None # Initialize for charts
                 selected_yoy_metrics_tab = [] # Initialize for tables
                 # Check column existence based on processed data (FU Check logic)
                 can_calc_ad_sale_tab = {"Sales", "Total Sales", "WE Date"}.issubset(ad_data_processed_current.columns)

                 # --- Determine Available Metrics (FU Logic) ---
                 all_possible_metrics = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "ACOS", "ROAS", "CPC", "Ad % Sale"]
                 original_cols_tab = set(ad_data_processed_current.columns) # Check processed data cols
                 available_metrics_tab = []
                 product_specific_cols = set(ad_data_tab_filtered_initial.columns) # Check product specific for existence
                 for m in all_possible_metrics:
                     can_display_m = False
                     if m in product_specific_cols: can_display_m = True
                     elif m == "CTR" and {"Clicks", "Impressions"}.issubset(original_cols_tab): can_display_m = True
                     elif m == "CVR" and {"Orders", "Clicks"}.issubset(original_cols_tab): can_display_m = True
                     elif m == "ACOS" and {"Spend", "Sales"}.issubset(original_cols_tab): can_display_m = True
                     elif m == "ROAS" and {"Sales", "Spend"}.issubset(original_cols_tab): can_display_m = True
                     elif m == "CPC" and {"Spend", "Clicks"}.issubset(original_cols_tab): can_display_m = True
                     elif m == "Ad % Sale" and can_calc_ad_sale_tab: can_display_m = True
                     if can_display_m: available_metrics_tab.append(m)
                 available_metrics_tab = sorted(list(set(available_metrics_tab)))

                 with col1_tab:
                     # Chart Metric Selector (Single Select - FU Logic)
                     default_metric_chart_tab = "Spend" if "Spend" in available_metrics_tab else available_metrics_tab[0] if available_metrics_tab else None
                     sel_metric_index_tab = available_metrics_tab.index(default_metric_chart_tab) if default_metric_chart_tab in available_metrics_tab else 0
                     if available_metrics_tab:
                          selected_metric_tab = st.selectbox("Select Metric for Charts", options=available_metrics_tab, index=sel_metric_index_tab, key=f"{tab_key_prefix}_metric")
                     else: st.warning(f"No metrics available for chart selection in {product_type_tab} tab.") # FU Warning

                 with col2_tab:
                     # Portfolio Selector (Single Select for Charts - FU Logic)
                     portfolio_col_tab = next((col for col in ["Portfolio Name", "Portfolio"] if col in ad_data_tab_filtered_initial.columns), None)
                     if not portfolio_col_tab:
                          selected_portfolio_tab = "All Portfolios"
                          st.info("Portfolio filtering (for charts) not available ('Portfolio Name' column missing).") # FU Info
                     else:
                          # FU filled NaNs here too
                          ad_data_tab_filtered_initial[portfolio_col_tab] = ad_data_tab_filtered_initial[portfolio_col_tab].fillna("Unknown Portfolio")
                          portfolio_options_tab = ["All Portfolios"] + sorted(ad_data_tab_filtered_initial[portfolio_col_tab].unique().tolist())
                          selected_portfolio_tab = st.selectbox("Select Portfolio (for Charts)", options=portfolio_options_tab, index=0, key=f"{tab_key_prefix}_portfolio")

                 with col3_tab:
                     # Table Metrics Selector (Multi Select - FU Logic)
                     default_metrics_table_list = ["Spend", "Sales", "Ad % Sale", "ACOS"] # FU Defaults
                     default_metrics_table_tab = [m for m in default_metrics_table_list if m in available_metrics_tab]
                     if not default_metrics_table_tab and available_metrics_tab:
                          default_metrics_table_tab = available_metrics_tab[:1] # FU Fallback

                     if available_metrics_tab:
                          selected_yoy_metrics_tab = st.multiselect(
                              "Select Metrics for YOY Tables",
                              options=available_metrics_tab,
                              default=default_metrics_table_tab,
                              key=f"{tab_key_prefix}_yoy_metrics"
                          )
                     else: st.warning(f"No metrics available for YOY table selection in {product_type_tab} tab.") # FU Warning

                 show_yoy_tab = st.checkbox("Show Year-over-Year Comparison (Chart - Weekly Points)", value=True, key=f"{tab_key_prefix}_show_yoy") # FU Checkbox

                 # Date Range Selector (FU Logic)
                 date_range_tab = None
                 min_date_tab, max_date_tab = None, None
                 date_col_name = "WE Date" # Assume from FU
                 if date_col_name in ad_data_tab_filtered_initial.columns and not ad_data_tab_filtered_initial[date_col_name].dropna().empty:
                      try:
                           # Ensure dates are valid datetime objects first
                           valid_dates = pd.to_datetime(ad_data_tab_filtered_initial[date_col_name], errors='coerce').dropna()
                           if not valid_dates.empty:
                               min_date_tab = valid_dates.min().date()
                               max_date_tab = valid_dates.max().date()
                               if min_date_tab <= max_date_tab:
                                   date_range_tab = st.date_input("Select Date Range", value=(min_date_tab, max_date_tab), min_value=min_date_tab, max_value=max_date_tab, key=f"{tab_key_prefix}_date_range")
                               else: st.warning(f"Invalid date range found in {product_type_tab} data.") # FU Warning
                           else: st.warning(f"No valid dates found in {product_type_tab} data for date range.")
                      except Exception as e: st.warning(f"Error setting date range for {product_type_tab}: {e}") # FU Warning
                 else: st.warning(f"Cannot determine date range for {product_type_tab} tab ('{date_col_name}' missing or empty).") # FU Warning

            # Apply Date Range Filter (FU Logic)
            ad_data_tab_date_filtered = ad_data_tab_filtered_initial.copy()
            # Also filter the *original* processed data for Ad % Sale denom calc (FU Logic)
            original_data_date_filtered_tab = ad_data_processed_current.copy()
            if date_range_tab and len(date_range_tab) == 2 and min_date_tab and max_date_tab:
                 start_date_tab, end_date_tab = date_range_tab
                 # FU version used this logic
                 if isinstance(start_date_tab, datetime.date) and isinstance(end_date_tab, datetime.date): # Check types
                      if start_date_tab >= min_date_tab and end_date_tab <= max_date_tab and start_date_tab <= end_date_tab:
                           # Filter product-specific data
                           ad_data_tab_date_filtered = ad_data_tab_date_filtered[ (ad_data_tab_date_filtered[date_col_name].dt.date >= start_date_tab) & (ad_data_tab_date_filtered[date_col_name].dt.date <= end_date_tab) ]
                           # Filter original data used for denominator
                           original_data_date_filtered_tab = original_data_date_filtered_tab[ (original_data_date_filtered_tab[date_col_name].dt.date >= start_date_tab) & (original_data_date_filtered_tab[date_col_name].dt.date <= end_date_tab) ]
                      else:
                           st.warning("Selected date range is invalid or outside data bounds. Using full data range.") # FU Warning
                           # Don't filter if range is bad - use data filtered only by product type
                           ad_data_tab_date_filtered = ad_data_tab_filtered_initial.copy()
                           original_data_date_filtered_tab = ad_data_processed_current.copy() # Reset denom base too
                 else:
                      st.warning("Invalid date objects received from date_input. Using full data range.")
                      ad_data_tab_date_filtered = ad_data_tab_filtered_initial.copy()
                      original_data_date_filtered_tab = ad_data_processed_current.copy()


            # --- Prepare Data for Ad % Sale Chart (Denominator - FU Logic) ---
            weekly_denominator_df_tab = pd.DataFrame()
            # Check based on *selected chart metric* and column availability
            if selected_metric_tab == "Ad % Sale" and can_calc_ad_sale_tab:
                 if not original_data_date_filtered_tab.empty:
                      try:
                           temp_denom_df = original_data_date_filtered_tab.copy()
                           temp_denom_df[date_col_name] = pd.to_datetime(temp_denom_df[date_col_name], errors='coerce')
                           temp_denom_df['Total Sales'] = pd.to_numeric(temp_denom_df['Total Sales'], errors='coerce')
                           # Ensure Year/Week exist (might not if preprocess failed earlier, though unlikely if we got here)
                           if 'Year' not in temp_denom_df.columns: temp_denom_df["Year"] = temp_denom_df[date_col_name].dt.year
                           if 'Week' not in temp_denom_df.columns: temp_denom_df["Week"] = temp_denom_df[date_col_name].dt.isocalendar().week
                           temp_denom_df['Year'] = pd.to_numeric(temp_denom_df['Year'], errors='coerce')
                           temp_denom_df['Week'] = pd.to_numeric(temp_denom_df['Week'], errors='coerce')
                           # Drop NaNs required for grouping and value
                           temp_denom_df.dropna(subset=[date_col_name, "Total Sales", "Year", "Week"], inplace=True)
                           if not temp_denom_df.empty:
                               temp_denom_df['Year'] = temp_denom_df['Year'].astype(int)
                               temp_denom_df['Week'] = temp_denom_df['Week'].astype(int)
                               # FU Logic for denominator uniqueness
                               unique_subset_denom = ['Year', 'Week']
                               if "Marketplace" in temp_denom_df.columns: unique_subset_denom.append("Marketplace")
                               unique_subset_denom = [col for col in unique_subset_denom if col in temp_denom_df.columns] # Check exist
                               if unique_subset_denom:
                                   unique_totals = temp_denom_df.drop_duplicates(subset=unique_subset_denom)
                                   weekly_denominator_df_tab = unique_totals.groupby(['Year', 'Week'], as_index=False)['Total Sales'].sum()
                               else: # Fallback if uniqueness columns missing
                                    weekly_denominator_df_tab = temp_denom_df.groupby(['Year', 'Week'], as_index=False)['Total Sales'].sum()

                               weekly_denominator_df_tab = weekly_denominator_df_tab.rename(columns={'Total Sales': 'Weekly_Total_Sales'})
                      except Exception as e: st.warning(f"Could not calculate weekly total sales denominator for Ad % Sale chart ({product_type_tab}): {e}") # FU Warning
                 else: st.warning(f"Cannot calculate Ad % Sale denominator: No original data in selected date range for {product_type_tab}.") # FU Warning

            # --- Display Charts (FU Logic) ---
            if ad_data_tab_date_filtered.empty:
                 # Message handled below before tables
                 pass
            elif selected_metric_tab is None:
                 st.warning("Please select a metric to visualize the charts.") # FU Warning
            else:
                 # Time Chart (Use FU Function)
                 st.subheader(f"{selected_metric_tab} Over Time")
                 fig1_tab = create_metric_over_time_chart(data=ad_data_tab_date_filtered, metric=selected_metric_tab, portfolio=selected_portfolio_tab, product_type=product_type_tab, show_yoy=show_yoy_tab, weekly_total_sales_data=weekly_denominator_df_tab)
                 st.plotly_chart(fig1_tab, use_container_width=True, key=f"{tab_key_prefix}_time_chart")

                 # Comparison Chart (Use FU Function)
                 # Need portfolio column name identified earlier
                 if selected_portfolio_tab == "All Portfolios" and portfolio_col_tab:
                      st.subheader(f"{selected_metric_tab} by Portfolio")
                      # FU logic check for Ad % Sale
                      if selected_metric_tab == "Ad % Sale":
                           st.info("'Ad % Sale' cannot be displayed in the Portfolio Comparison bar chart.") # FU Info
                      else:
                           # Call the FU comparison chart function
                           fig2_tab = create_metric_comparison_chart(ad_data_tab_date_filtered, selected_metric_tab, None, product_type_tab)
                           st.plotly_chart(fig2_tab, use_container_width=True, key=f"{tab_key_prefix}_portfolio_chart")


            # --- Display YOY Tables (FU Logic) ---
            st.markdown("---")
            st.subheader("Year-over-Year Portfolio Performance (Selected Period & Metrics)")

            if not portfolio_col_tab: # Use variable from portfolio selector logic
                 st.warning("Cannot generate YOY Portfolio table: 'Portfolio Name' column not found.") # FU Warning
            elif not selected_yoy_metrics_tab:
                 st.warning("Please select at least one metric in the 'Select Metrics for YOY Tables' filter to display the table.") # FU Warning
            elif ad_data_tab_date_filtered.empty:
                 st.info(f"No {product_type_tab} data available for the selected date range to build the YOY table.") # FU Info
            else:
                 years_in_tab_data = sorted(ad_data_tab_date_filtered['Year'].dropna().unique()) # Ensure handling potential NaNs before unique
                 if not years_in_tab_data:
                      st.info(f"No valid years found in the filtered {product_type_tab} data.")
                 else:
                      # Prepare data for table, potentially adding 'Total Sales' if needed (FU Logic for Ad % Sale)
                      data_for_yoy_table = ad_data_tab_date_filtered.copy()
                      if "Ad % Sale" in selected_yoy_metrics_tab and "Total Sales" in original_data_date_filtered_tab.columns:
                           merge_cols = ['WE Date', 'Year', 'Week'] # Base merge cols
                           if "Marketplace" in data_for_yoy_table.columns and "Marketplace" in original_data_date_filtered_tab.columns: merge_cols.append("Marketplace") # Add MP if available in both
                           # Ensure unique subset of original data with Total Sales
                           total_sales_data = original_data_date_filtered_tab[merge_cols + ['Total Sales']].drop_duplicates(subset=merge_cols)
                           # Drop existing Total Sales from target if it exists to avoid merge conflict
                           if 'Total Sales' in data_for_yoy_table.columns: data_for_yoy_table = data_for_yoy_table.drop(columns=['Total Sales'])
                           # Merge Total Sales onto the product-specific data
                           data_for_yoy_table = pd.merge(data_for_yoy_table, total_sales_data, on=merge_cols, how='left')

                      # Create Portfolio Breakdown Table (Use FU Function)
                      portfolio_yoy_table_tab = create_yoy_grouped_table(df_filtered_period=data_for_yoy_table, group_by_col=portfolio_col_tab, selected_metrics=selected_yoy_metrics_tab, years_to_process=years_in_tab_data, display_col_name="Portfolio")
                      # Create Summary Row (Use FU Function)
                      portfolio_yoy_summary_tab = calculate_yoy_summary_row(df=data_for_yoy_table, selected_metrics=selected_yoy_metrics_tab, years_to_process=years_in_tab_data, id_col_name="Portfolio", id_col_value="TOTAL")

                      # Display Tables (FU Logic)
                      if not portfolio_yoy_table_tab.empty:
                           st.markdown("###### YOY Portfolio Breakdown") # FU Header
                           styled_portfolio_yoy_tab = style_yoy_comparison_table(portfolio_yoy_table_tab) # FU Style
                           if styled_portfolio_yoy_tab: st.dataframe(styled_portfolio_yoy_tab, use_container_width=True)
                      else: st.info("No portfolio breakdown data available for the selected YOY metrics and period.") # FU Info

                      if not portfolio_yoy_summary_tab.empty:
                           st.markdown("###### YOY Total") # FU Header
                           styled_portfolio_summary_yoy_tab = style_yoy_comparison_table(portfolio_yoy_summary_tab) # FU Style
                           if styled_portfolio_summary_yoy_tab: st.dataframe(styled_portfolio_summary_yoy_tab.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                      else: st.info("No summary data available for the selected YOY metrics and period.") # FU Info

            # --- Insights Section (FU Logic) ---
            st.markdown("---")
            st.subheader("Key Insights (Latest Year in Selected Period)")

            # FU Logic: Check if date filtered data and years list exist
            if 'ad_data_tab_date_filtered' in locals() and not ad_data_tab_date_filtered.empty and 'years_in_tab_data' in locals() and years_in_tab_data:
                 latest_year_tab = years_in_tab_data[-1]
                 data_latest_year = ad_data_tab_date_filtered[ad_data_tab_date_filtered['Year'] == latest_year_tab].copy()

                 if not data_latest_year.empty:
                     # Calculate totals (FU Logic)
                     total_spend = pd.to_numeric(data_latest_year.get("Spend"), errors='coerce').fillna(0).sum()
                     total_sales = pd.to_numeric(data_latest_year.get("Sales"), errors='coerce').fillna(0).sum()
                     total_clicks = pd.to_numeric(data_latest_year.get("Clicks"), errors='coerce').fillna(0).sum()
                     total_impressions = pd.to_numeric(data_latest_year.get("Impressions"), errors='coerce').fillna(0).sum()
                     total_orders = pd.to_numeric(data_latest_year.get("Orders"), errors='coerce').fillna(0).sum()

                     # Calculate derived metrics (FU Logic)
                     insight_acos = (total_spend / total_sales * 100) if total_sales > 0 else np.nan
                     insight_roas = (total_sales / total_spend) if total_spend > 0 else np.nan
                     insight_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
                     insight_cvr = (total_orders / total_clicks * 100) if total_clicks > 0 else 0
                     insight_acos = np.nan if insight_acos in [np.inf, -np.inf] else insight_acos
                     insight_roas = np.nan if insight_roas in [np.inf, -np.inf] else insight_roas

                     # Prepare Series (FU Logic)
                     summary_series_for_insights = pd.Series({
                         "ACOS": insight_acos, "ROAS": insight_roas, "CTR": insight_ctr,
                         "CVR": insight_cvr, "Sales": total_sales, "Spend": total_spend
                     })

                     # Generate and display insights (Use FU Function)
                     insights_tab = generate_insights(summary_series_for_insights, product_type_tab)
                     for insight in insights_tab:
                          st.markdown(f"- {insight}")

                 else:
                     st.info(f"No data found for the latest year ({latest_year_tab}) in the selected period to generate insights.") # FU Info
            else:
                 st.info("No summary data available to generate insights (check date range and filters).") # FU Info


# =============================================================================
# Final Fallback Messages (MODIFIED SECTION for Secrets Input)
# =============================================================================
# Use 'elif' to chain conditions after the main 'if processed_data_valid_and_current:' block
elif not secrets_loaded:
     # This case is handled by st.stop() earlier, but good as a fallback
     st.error("App configuration failed. Could not load GSheet details from secrets.toml.")
elif not raw_data_available and secrets_loaded:
     # This means secrets were loaded, but data loading failed or returned empty
     st.warning(f"Failed to load data from the Google Sheet specified in secrets.")
     st.info("Troubleshooting Tips:")
     st.info(f"- Verify the `url_or_id` and `worksheet_name` in `.streamlit/secrets.toml` are correct.")
     st.info(f"- Ensure the service account email (`client_email` in secrets) has at least 'Viewer' access to the Google Sheet.")
     st.info(f"- Check the Google Sheet exists and the worksheet name is exact (case-sensitive).")
     st.info(f"- Check the app's logs for specific errors from the `load_data_from_gsheet` function (usually shown in the sidebar).")
elif raw_data_available and not processed_data_valid_and_current:
     # This means raw data is loaded, but processing failed or is stale (e.g., marketplace changed)
     if "ad_data_processed" not in st.session_state or st.session_state["ad_data_processed"].empty:
          st.warning("Data loaded, but processing failed or resulted in empty data. Check sheet content, required columns (e.g., 'WE Date'), and data formats.")
     else:
          # Data is processed but might be for different marketplace or outdated GSheet config
          st.info("Marketplace selection or underlying data may require reprocessing. Please wait or check status.")
          # The processing logic should trigger automatically if needed via st.rerun
elif not processed_data_valid_and_current and secrets_loaded:
     # General fallback if tabs aren't showing but secrets seemed okay and loading didn't explicitly fail
      st.warning("Could not display dashboard content. Waiting for data processing or check for errors.")

# --- End of Script ---