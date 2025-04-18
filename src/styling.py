# src/styling.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
from src.config import (PORTFOLIO_COL, MATCH_TYPE_COL, RTW_COL, CAMPAIGN_COL, # Grouping cols
                        IMPRESSIONS_COL, CLICKS_COL, ORDERS_COL, UNITS_COL, # Base metrics
                        SPEND_COL, SALES_COL, CTR, CVR, ACOS, ROAS, CPC, AD_PERC_SALE, # Derived metrics
                        PERCENTAGE_POINT_CHANGE_METRICS) # For YoY styling

# warnings.filterwarnings("ignore")


def style_dataframe(df, format_dict, highlight_cols=None, color_map_func=None, text_align='right', na_rep='N/A'):
    """Generic styling function for dataframes with alignment and NaN handling."""
    if df is None or df.empty: return None
    df_copy = df.copy().replace([np.inf, -np.inf], np.nan)
    valid_format_dict = {k: v for k, v in format_dict.items() if k in df_copy.columns}

    try:
        styled = df_copy.style.format(valid_format_dict, na_rep=na_rep, precision=2) # Added precision here too
    except Exception as e:
        st.error(f"Error applying format: {e}. Formatting dictionary: {valid_format_dict}")
        return df_copy.style # Basic styler on error

    if highlight_cols and color_map_func:
        if len(highlight_cols) == len(color_map_func):
            for col, func in zip(highlight_cols, color_map_func):
                if col in df_copy.columns:
                    try: styled = styled.applymap(func, subset=[col]) # Use applymap for elementwise coloring
                    except Exception as e: st.warning(f"Styling failed for column '{col}': {e}")
        else:
            st.error("Mismatch between highlight_cols and color_map_func in style_dataframe.")

    # Alignment using set_properties (generally more robust)
    try:
        first_col_name = df_copy.columns[0]
        styled = styled.set_properties(**{'text-align': text_align}) # Apply default alignment
        styled = styled.set_properties(subset=[first_col_name], **{'text-align': 'left'}) # Left-align first column
    except Exception as e:
        st.warning(f"Could not apply alignment: {e}")
        # Fallback just in case
        styled = styled.set_properties(**{'text-align': text_align})


    return styled

# Note: style_total_summary and style_metrics_table are kept as they were in the FU code
# but marked as potentially unused in the main app flow.
def style_total_summary(df):
    """Styles the single-row total summary table (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # THIS FUNCTION IS NO LONGER DIRECTLY USED FOR SP/SB/SD TABS but kept
    format_dict = {
        IMPRESSIONS_COL: "{:,.0f}", CLICKS_COL: "{:,.0f}", ORDERS_COL: "{:,.0f}",
        SPEND_COL: "${:,.2f}", SALES_COL: "${:,.2f}",
        CTR: "{:.1f}%", CVR: "{:.1f}%",
        ACOS: lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        ROAS: lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    }
    def color_acos(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f <= 15 else ("color: orange" if val_f <= 20 else "color: red")
        except (ValueError, TypeError): return "color: grey"
    def color_roas(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f > 3 else "color: red"
        except (ValueError, TypeError): return "color: grey"

    styled = style_dataframe(df, format_dict, highlight_cols=[ACOS, ROAS], color_map_func=[color_acos, color_roas], na_rep="N/A")
    if styled: return styled.set_properties(**{"font-weight": "bold"})
    return None

def style_metrics_table(df):
    """Styles the multi-row performance metrics table (by Portfolio) (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # THIS FUNCTION IS NO LONGER DIRECTLY USED FOR SP/SB/SD TABS but kept
    format_dict = {
        IMPRESSIONS_COL: "{:,.0f}", CLICKS_COL: "{:,.0f}", ORDERS_COL: "{:,.0f}",
        SPEND_COL: "${:,.2f}", SALES_COL: "${:,.2f}",
        CTR: "{:.1f}%", CVR: "{:.1f}%",
        ACOS: lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        ROAS: lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    }
    if UNITS_COL in df.columns: format_dict[UNITS_COL] = "{:,.0f}"
    if CPC in df.columns: format_dict[CPC] = "${:,.2f}"

    # Identify the grouping column dynamically (better than hardcoding 'Portfolio')
    grouping_cols = ["Portfolio", PORTFOLIO_COL, MATCH_TYPE_COL, RTW_COL, CAMPAIGN_COL, "Campaign"] # Potential names
    id_col_name = next((col for col in df.columns if col in grouping_cols), None)
    if id_col_name: format_dict[id_col_name] = "{}" # Format as string

    def color_acos(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f <= 15 else ("color: orange" if val_f <= 20 else "color: red")
        except (ValueError, TypeError): return "color: grey"
    def color_roas(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f > 3 else "color: red"
        except (ValueError, TypeError): return "color: grey"

    styled = style_dataframe(df, format_dict, highlight_cols=[ACOS, ROAS], color_map_func=[color_acos, color_roas], na_rep="N/A")
    return styled


def style_yoy_comparison_table(df):
    """Styles the YoY comparison table with formats and % change coloring."""
    if df is None or df.empty: return None
    df_copy = df.copy().replace([np.inf, -np.inf], np.nan)

    format_dict = {}
    highlight_change_cols = []

    # --- Determine Formats and Identify Change Columns ---
    for col in df_copy.columns:
        base_metric_match = re.match(r"([a-zA-Z\s%/]+)", col) # Include / for RTW/Prospecting
        base_metric = base_metric_match.group(1).strip() if base_metric_match else ""
        is_change_col = "% Change" in col
        is_metric_col = not is_change_col and any(char.isdigit() for char in col) # Basic check if year is in col name

        if is_change_col:
            base_metric_for_change = col.replace(" % Change", "").strip()
            # Format absolute change for percentage metrics, percentage change for others
            # Use config constant
            if base_metric_for_change in PERCENTAGE_POINT_CHANGE_METRICS:
                format_dict[col] = lambda x: f"{x:+.1f}pp" if pd.notna(x) else 'N/A' # Use 'pp' for percentage points
            else:
                format_dict[col] = lambda x: f"{x:+.0f}%" if pd.notna(x) else 'N/A' # Standard percentage change
            highlight_change_cols.append(col)
        elif is_metric_col:
            # Apply standard metric formatting (use constants)
            if base_metric in [IMPRESSIONS_COL, CLICKS_COL, ORDERS_COL, UNITS_COL]: format_dict[col] = "{:,.0f}"
            elif base_metric in [SPEND_COL, SALES_COL, CPC]: format_dict[col] = "${:,.2f}"
            elif base_metric in [ACOS, CTR, CVR, AD_PERC_SALE]: format_dict[col] = '{:.1f}%'
            elif base_metric == ROAS: format_dict[col] = '{:.2f}'
        elif df_copy[col].dtype == 'object' and col == df_copy.columns[0]: # Format the first (grouping) column as string
            format_dict[col] = "{}"

    # --- Define Coloring Functions ---
    def color_pos_neg_standard(val):
        """Standard coloring: positive is green, negative is red."""
        if isinstance(val, str) and val == "N/A": return 'color: grey'
        numeric_val = pd.to_numeric(val, errors='coerce')
        if pd.isna(numeric_val): return 'color: grey'
        elif numeric_val > 0.001: return 'color: green' # Tolerance for float
        elif numeric_val < -0.001: return 'color: red'
        else: return 'color: inherit' # Black/default for zero

    def color_pos_neg_inverted(val):
        """Inverted coloring (for ACOS): positive is red, negative is green."""
        if isinstance(val, str) and val == "N/A": return 'color: grey'
        numeric_val = pd.to_numeric(val, errors='coerce')
        if pd.isna(numeric_val): return 'color: grey'
        elif numeric_val > 0.001: return 'color: red'   # Positive change (ACOS increase) is red (bad)
        elif numeric_val < -0.001: return 'color: green' # Negative change (ACOS decrease) is green (good)
        else: return 'color: inherit' # Black/default for zero

    # --- Apply Styling using the generic function ---
    # Create the list of coloring functions aligned with highlight_change_cols
    color_funcs = []
    for change_col in highlight_change_cols:
        if change_col == f"{ACOS} % Change": # Use constant
            color_funcs.append(color_pos_neg_inverted)
        else:
            color_funcs.append(color_pos_neg_standard)

    styled_table = style_dataframe(
        df_copy,
        format_dict,
        highlight_cols=highlight_change_cols,
        color_map_func=color_funcs,
        na_rep="N/A"
    )

    return styled_table