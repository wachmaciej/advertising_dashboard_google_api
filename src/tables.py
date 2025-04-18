# src/tables.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from functools import reduce
import warnings
from src.config import (DATE_COL, YEAR_COL, WEEK_COL, PRODUCT_COL, PORTFOLIO_COL,
                        MARKETPLACE_COL, MATCH_TYPE_COL, RTW_COL, CAMPAIGN_COL,
                        CLICKS_COL, IMPRESSIONS_COL, ORDERS_COL, SPEND_COL, SALES_COL,
                        UNITS_COL, TOTAL_SALES_COL, # Needed for Ad % Sale calculation
                        CTR, CVR, ACOS, ROAS, CPC, AD_PERC_SALE, BASE_METRIC_COLS,
                        METRIC_COMPONENTS, PERCENTAGE_POINT_CHANGE_METRICS, ALL_POSSIBLE_METRICS)

# warnings.filterwarnings("ignore")

# Note: create_performance_metrics_table is intentionally kept as it was in the FU code
# but marked as potentially unused in the main app flow.
@st.cache_data
def create_performance_metrics_table(df, portfolio_name=None, campaign_type="Sponsored Products"):
    """Creates portfolio breakdown and total summary tables (LOGIC IDENTICAL TO FILE UPLOADER VERSION)"""
    # THIS FUNCTION IS NO LONGER DIRECTLY USED FOR SP/SB/SD TABS but kept for potential future use / other parts
    required_cols = {PRODUCT_COL, PORTFOLIO_COL, IMPRESSIONS_COL, CLICKS_COL, SPEND_COL, SALES_COL, ORDERS_COL}
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.warning(f"Performance table missing required columns: {missing}")
        return pd.DataFrame(), pd.DataFrame()

    filtered_df = df[df[PRODUCT_COL] == campaign_type].copy()
    filtered_df[PORTFOLIO_COL] = filtered_df[PORTFOLIO_COL].fillna("Unknown Portfolio")

    if portfolio_name and portfolio_name != "All Portfolios":
        if portfolio_name in filtered_df[PORTFOLIO_COL].unique():
            filtered_df = filtered_df[filtered_df[PORTFOLIO_COL] == portfolio_name]
        else:
            st.warning(f"Portfolio '{portfolio_name}' not found for {campaign_type} in performance table.")
            return pd.DataFrame(), pd.DataFrame()

    if filtered_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    try:
        metrics_by_portfolio = filtered_df.groupby(PORTFOLIO_COL).agg(
            Impressions=(IMPRESSIONS_COL, "sum"),
            Clicks=(CLICKS_COL, "sum"),
            Spend=(SPEND_COL, "sum"),
            Sales=(SALES_COL, "sum"),
            Orders=(ORDERS_COL, "sum")
        ).reset_index()
    except Exception as e:
        st.warning(f"Error aggregating performance table: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Calculate derived metrics after aggregation
    metrics_by_portfolio[CTR] = metrics_by_portfolio.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r.get("Impressions") else 0, axis=1)
    metrics_by_portfolio[CVR] = metrics_by_portfolio.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r.get("Clicks") else 0, axis=1)
    metrics_by_portfolio[ACOS] = metrics_by_portfolio.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r.get("Sales") else np.nan, axis=1)
    metrics_by_portfolio[ROAS] = metrics_by_portfolio.apply(lambda r: (r["Sales"] / r["Spend"]) if r.get("Spend") else np.nan, axis=1)
    metrics_by_portfolio = metrics_by_portfolio.replace([np.inf, -np.inf], np.nan)

    # Rounding
    for col in [CTR, CVR, ACOS]:
        if col in metrics_by_portfolio.columns: metrics_by_portfolio[col] = metrics_by_portfolio[col].round(1)
    for col in [SPEND_COL, SALES_COL, ROAS]:
        if col in metrics_by_portfolio.columns: metrics_by_portfolio[col] = metrics_by_portfolio[col].round(2)

    # Calculate Total Summary Row
    total_summary = pd.DataFrame()
    if not filtered_df.empty:
        sum_cols = [IMPRESSIONS_COL, CLICKS_COL, SPEND_COL, SALES_COL, ORDERS_COL]
        numeric_summary_data = filtered_df.copy()
        for col in sum_cols:
            if col in numeric_summary_data.columns:
                 numeric_summary_data[col] = pd.to_numeric(numeric_summary_data[col], errors='coerce').fillna(0)
            else: numeric_summary_data[col] = 0

        total_impressions = numeric_summary_data[IMPRESSIONS_COL].sum()
        total_clicks = numeric_summary_data[CLICKS_COL].sum()
        total_spend = numeric_summary_data[SPEND_COL].sum()
        total_sales = numeric_summary_data[SALES_COL].sum()
        total_orders = numeric_summary_data[ORDERS_COL].sum()
        total_ctr = (total_clicks / total_impressions * 100) if total_impressions else 0
        total_cvr = (total_orders / total_clicks * 100) if total_clicks else 0
        total_acos = (total_spend / total_sales * 100) if total_sales else np.nan
        total_roas = (total_sales / total_spend) if total_spend else np.nan
        total_acos = np.nan if total_acos in [np.inf, -np.inf] else total_acos
        total_roas = np.nan if total_roas in [np.inf, -np.inf] else total_roas
        total_summary_data = {
            "Metric": ["Total"],
            IMPRESSIONS_COL: [total_impressions], CLICKS_COL: [total_clicks], ORDERS_COL: [total_orders],
            SPEND_COL: [round(total_spend, 2)], SALES_COL: [round(total_sales, 2)],
            CTR: [round(total_ctr, 1)], CVR: [round(total_cvr, 1)],
            ACOS: [total_acos], ROAS: [total_roas]
        }
        total_summary = pd.DataFrame(total_summary_data)

    # Rename portfolio column for consistency before returning
    metrics_by_portfolio = metrics_by_portfolio.rename(columns={PORTFOLIO_COL: "Portfolio"}) # Use "Portfolio" for display consistency maybe?
    return metrics_by_portfolio, total_summary


@st.cache_data
def create_yoy_grouped_table(df_filtered_period, group_by_col, selected_metrics, years_to_process, display_col_name=None):
    """Creates a merged YoY comparison table grouped by a specific column."""
    # Includes internal Ad % Sale denominator calculation
    if df_filtered_period is None or df_filtered_period.empty: return pd.DataFrame()
    if group_by_col not in df_filtered_period.columns: st.warning(f"Grouping column '{group_by_col}' not found."); return pd.DataFrame()
    if not isinstance(selected_metrics, list) or not selected_metrics: st.warning("No metrics selected."); return pd.DataFrame()

    # Check required columns for Ad % Sale based on config constants
    ad_sale_possible = (AD_PERC_SALE in selected_metrics and {SALES_COL, TOTAL_SALES_COL, DATE_COL}.issubset(df_filtered_period.columns))

    if AD_PERC_SALE in selected_metrics and not ad_sale_possible:
        st.warning(f"Cannot calculate '{AD_PERC_SALE}'. Requires '{SALES_COL}', '{TOTAL_SALES_COL}', and '{DATE_COL}' columns.")
        selected_metrics = [m for m in selected_metrics if m != AD_PERC_SALE]
        if not selected_metrics: return pd.DataFrame()

    # Fill NaNs in grouping column
    df_filtered_period[group_by_col] = df_filtered_period[group_by_col].fillna(f"Unknown {group_by_col}")
    yearly_tables = []

    for yr in years_to_process:
        df_year = df_filtered_period[df_filtered_period[YEAR_COL] == yr].copy()
        if df_year.empty: continue

        # Determine base metrics needed (using config constants)
        base_metrics_to_sum_needed = set()
        for metric in selected_metrics:
            if metric in BASE_METRIC_COLS: base_metrics_to_sum_needed.add(metric)
            elif metric in METRIC_COMPONENTS: base_metrics_to_sum_needed.update(METRIC_COMPONENTS[metric])

        actual_base_present = {m for m in base_metrics_to_sum_needed if m in df_year.columns}
        # Check if any base metrics needed are present OR if any non-derived selected metrics are present
        non_derived_selected = [m for m in selected_metrics if m not in METRIC_COMPONENTS]
        if not actual_base_present and not any(m in df_year.columns for m in non_derived_selected): continue

        # Determine which of the selected metrics can actually be calculated/displayed for this year
        calculable_metrics_for_year = []
        original_cols_year = set(df_year.columns)
        for metric in selected_metrics:
            can_calc_yr = False
            # Check if directly present and numeric
            if metric in original_cols_year and pd.api.types.is_numeric_dtype(df_year[metric]): can_calc_yr = True
            # Check if derivable based on *present* base metrics
            elif metric in METRIC_COMPONENTS and METRIC_COMPONENTS[metric].issubset(actual_base_present): can_calc_yr = True
             # Special check for Ad % Sale possibility (requires check on original df cols, not just base sum cols)
            elif metric == AD_PERC_SALE and ad_sale_possible and SALES_COL in actual_base_present: can_calc_yr = True

            if can_calc_yr: calculable_metrics_for_year.append(metric)

        if not calculable_metrics_for_year: continue # Skip year if none of the selected metrics can be handled

        # Calculate Ad % Sale Denominator for the year (internally)
        total_sales_for_period = 0
        if AD_PERC_SALE in calculable_metrics_for_year:
            try:
                # Use only necessary columns for denominator calculation
                denom_cols = [DATE_COL, TOTAL_SALES_COL, YEAR_COL, WEEK_COL]
                if MARKETPLACE_COL in df_year.columns: denom_cols.append(MARKETPLACE_COL)
                df_year_valid_dates_total = df_year[denom_cols].copy()

                df_year_valid_dates_total[DATE_COL] = pd.to_datetime(df_year_valid_dates_total[DATE_COL], errors='coerce')
                df_year_valid_dates_total[TOTAL_SALES_COL] = pd.to_numeric(df_year_valid_dates_total[TOTAL_SALES_COL], errors='coerce')
                df_year_valid_dates_total[YEAR_COL] = pd.to_numeric(df_year_valid_dates_total[YEAR_COL], errors='coerce')
                df_year_valid_dates_total[WEEK_COL] = pd.to_numeric(df_year_valid_dates_total[WEEK_COL], errors='coerce')

                df_year_valid_dates_total.dropna(subset=[DATE_COL, TOTAL_SALES_COL, YEAR_COL, WEEK_COL], inplace=True)

                if not df_year_valid_dates_total.empty:
                    # Define unique subset based on available columns
                    unique_subset = [YEAR_COL, WEEK_COL] # Base uniqueness
                    if MARKETPLACE_COL in df_year_valid_dates_total.columns: unique_subset.append(MARKETPLACE_COL)

                    # Drop duplicates based on the unique time/market period
                    unique_weekly_totals = df_year_valid_dates_total.drop_duplicates(subset=unique_subset)
                    total_sales_for_period = unique_weekly_totals[TOTAL_SALES_COL].sum()

            except Exception as e: st.warning(f"Could not calculate total sales denominator for year {yr}: {e}")

        # Aggregate necessary base metrics
        # Ensure base columns are numeric before aggregation
        agg_dict_final = {}
        for m in actual_base_present:
             if m in df_year.columns: # Check again if column exists
                 # Coerce to numeric *before* deciding to aggregate
                 df_year[m] = pd.to_numeric(df_year[m], errors='coerce')
                 # Only aggregate if it's numeric *after* coercion attempt
                 if pd.api.types.is_numeric_dtype(df_year[m]):
                     agg_dict_final[m] = 'sum'

        if not agg_dict_final:
            # If no numeric base metrics to aggregate, create empty pivot with group names
            df_pivot = pd.DataFrame({group_by_col: df_year[group_by_col].unique()})
        else:
            try:
                # Now aggregate the numeric columns identified
                 df_pivot = df_year.groupby(group_by_col).agg(agg_dict_final).reset_index()
            except Exception as e: st.warning(f"Error aggregating data for {group_by_col} in year {yr}: {e}"); continue

        # Calculate derived metrics post-aggregation (use constants)
        if CTR in calculable_metrics_for_year: df_pivot[CTR] = df_pivot.apply(lambda r: (r.get(CLICKS_COL,0) / r.get(IMPRESSIONS_COL,0) * 100) if r.get(IMPRESSIONS_COL) else 0, axis=1)
        if CVR in calculable_metrics_for_year: df_pivot[CVR] = df_pivot.apply(lambda r: (r.get(ORDERS_COL,0) / r.get(CLICKS_COL,0) * 100) if r.get(CLICKS_COL) else 0, axis=1)
        if CPC in calculable_metrics_for_year: df_pivot[CPC] = df_pivot.apply(lambda r: (r.get(SPEND_COL,0) / r.get(CLICKS_COL,0)) if r.get(CLICKS_COL) else np.nan, axis=1)
        if ACOS in calculable_metrics_for_year: df_pivot[ACOS] = df_pivot.apply(lambda r: (r.get(SPEND_COL,0) / r.get(SALES_COL,0) * 100) if r.get(SALES_COL) else np.nan, axis=1)
        if ROAS in calculable_metrics_for_year: df_pivot[ROAS] = df_pivot.apply(lambda r: (r.get(SALES_COL,0) / r.get(SPEND_COL,0)) if r.get(SPEND_COL) else np.nan, axis=1)
        if AD_PERC_SALE in calculable_metrics_for_year: df_pivot[AD_PERC_SALE] = df_pivot.apply( lambda r: (r.get(SALES_COL, 0) / total_sales_for_period * 100) if total_sales_for_period > 0 else np.nan, axis=1 )

        # Clean up calculated metrics
        df_pivot = df_pivot.replace([np.inf, -np.inf], np.nan)
        # Select only columns needed and rename for the specific year
        final_cols_for_year = [group_by_col] + [m for m in calculable_metrics_for_year if m in df_pivot.columns]
        df_pivot_final = df_pivot[final_cols_for_year].rename(columns={m: f"{m} {yr}" for m in calculable_metrics_for_year})
        yearly_tables.append(df_pivot_final)

    # Merge yearly tables
    if not yearly_tables: return pd.DataFrame()
    try: merged_table = reduce(lambda left, right: pd.merge(left, right, on=group_by_col, how="outer"), yearly_tables)
    except Exception as e: st.error(f"Error merging yearly {group_by_col} tables: {e}"); return pd.DataFrame()

    # Fill NaNs for base metrics that were summed (use constants)
    # Identify base metric columns that exist in the merged table
    cols_to_fill_zero = []
    for yr in years_to_process:
        for m in BASE_METRIC_COLS: # Iterate through defined base metrics
            col_name_year = f"{m} {yr}"
            if col_name_year in merged_table.columns:
                cols_to_fill_zero.append(col_name_year)

    if cols_to_fill_zero: merged_table[cols_to_fill_zero] = merged_table[cols_to_fill_zero].fillna(0)

    # Order columns and calculate % change (single change column between last two years)
    ordered_cols = [group_by_col]
    # Get actual years present in merged table columns
    actual_years_in_data = sorted(list(set(int(y.group(1)) for col in merged_table.columns if (y := re.search(r'(\d{4})$', col)) is not None)))


    if len(actual_years_in_data) >= 2:
        current_year_sel, prev_year_sel = actual_years_in_data[-1], actual_years_in_data[-2]
        for metric in selected_metrics: # Iterate through originally selected metrics
            col_current, col_prev = f"{metric} {current_year_sel}", f"{metric} {prev_year_sel}"
            change_col_name = f"{metric} % Change"

            # Append year columns if they exist
            if col_prev in merged_table.columns: ordered_cols.append(col_prev)
            if col_current in merged_table.columns: ordered_cols.append(col_current)

            # Calculate change only if both columns exist
            if col_current in merged_table.columns and col_prev in merged_table.columns:
                # Convert to numeric safely before calculation
                val_curr = pd.to_numeric(merged_table[col_current], errors='coerce')
                val_prev = pd.to_numeric(merged_table[col_prev], errors='coerce')

                # Use config constant for checking metric type
                if metric in PERCENTAGE_POINT_CHANGE_METRICS: # Absolute change for % metrics
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

                merged_table[change_col_name] = merged_table[change_col_name].replace([np.inf, -np.inf], np.nan) # Handle potential Inf/-Inf
                ordered_cols.append(change_col_name) # Append the change column

    elif actual_years_in_data: # Only one year of data
        yr_single = actual_years_in_data[0]
        ordered_cols.extend([f"{m} {yr_single}" for m in selected_metrics if f"{m} {yr_single}" in merged_table.columns])

    # Final column selection and renaming
    ordered_cols = [col for col in ordered_cols if col in merged_table.columns] # Ensure columns exist
    merged_table_display = merged_table[ordered_cols].copy()
    final_display_col = display_col_name or group_by_col # Use display name if provided
    if group_by_col in merged_table_display.columns:
        merged_table_display = merged_table_display.rename(columns={group_by_col: final_display_col})

    # Sorting
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
            except Exception as e: st.warning(f"Could not sort table by column '{sort_col}': {e}")

    return merged_table_display


@st.cache_data
def calculate_yoy_summary_row(df, selected_metrics, years_to_process, id_col_name, id_col_value):
    """Calculates a single summary row with YoY comparison based on yearly totals."""
    # Includes internal Ad % Sale denominator calculation
    if df is None or df.empty or not years_to_process: return pd.DataFrame()

    # Check required columns for Ad % Sale based on config constants
    ad_sale_possible = (AD_PERC_SALE in selected_metrics and {SALES_COL, TOTAL_SALES_COL, DATE_COL}.issubset(df.columns))
    if AD_PERC_SALE in selected_metrics and not ad_sale_possible:
        st.warning(f"Cannot calculate '{AD_PERC_SALE}' summary. Requires '{SALES_COL}', '{TOTAL_SALES_COL}', and '{DATE_COL}' columns.")
        selected_metrics = [m for m in selected_metrics if m != AD_PERC_SALE]
        if not selected_metrics: return pd.DataFrame()

    summary_row_data = {id_col_name: id_col_value}
    yearly_totals = {yr: {} for yr in years_to_process}
    yearly_total_sales_denom = {yr: 0 for yr in years_to_process} # Denom calculated here

    # Determine base metrics needed (using config constants)
    base_metrics_needed = set()
    for m in selected_metrics:
        if m in BASE_METRIC_COLS: base_metrics_needed.add(m)
        elif m in METRIC_COMPONENTS: base_metrics_needed.update(METRIC_COMPONENTS[m])


    # Calculate base totals and Ad % Sale Denom per year
    for yr in years_to_process:
        df_year = df[df[YEAR_COL] == yr]
        if df_year.empty: continue
        for base_m in base_metrics_needed:
            if base_m in df_year.columns:
                yearly_totals[yr][base_m] = pd.to_numeric(df_year[base_m], errors='coerce').fillna(0).sum()
            else: yearly_totals[yr][base_m] = 0 # Default 0 if missing

        # Calculate denom for the year internally
        if ad_sale_possible and AD_PERC_SALE in selected_metrics:
            try:
                # Simplified denom calc - assumes Total Sales is consistent per week/market within the year df slice
                denom_cols = [DATE_COL, TOTAL_SALES_COL, YEAR_COL, WEEK_COL]
                if MARKETPLACE_COL in df_year.columns: denom_cols.append(MARKETPLACE_COL)
                df_year_valid_dates = df_year[denom_cols].copy()

                df_year_valid_dates[DATE_COL] = pd.to_datetime(df_year_valid_dates[DATE_COL], errors='coerce')
                df_year_valid_dates[TOTAL_SALES_COL] = pd.to_numeric(df_year_valid_dates[TOTAL_SALES_COL], errors='coerce')
                df_year_valid_dates[YEAR_COL] = pd.to_numeric(df_year_valid_dates[YEAR_COL], errors='coerce')
                df_year_valid_dates[WEEK_COL] = pd.to_numeric(df_year_valid_dates[WEEK_COL], errors='coerce')
                df_year_valid_dates.dropna(subset=[DATE_COL, TOTAL_SALES_COL, YEAR_COL, WEEK_COL], inplace=True)

                if not df_year_valid_dates.empty:
                    unique_subset = [YEAR_COL, WEEK_COL]
                    if MARKETPLACE_COL in df_year_valid_dates.columns: unique_subset.append(MARKETPLACE_COL)
                    unique_totals = df_year_valid_dates.drop_duplicates(subset=unique_subset)
                    yearly_total_sales_denom[yr] = unique_totals[TOTAL_SALES_COL].sum()

            except Exception as e: yearly_total_sales_denom[yr] = 0 # Default 0 on error

    # Calculate derived metrics and populate row data (use constants)
    for metric in selected_metrics:
        for yr in years_to_process:
            totals_yr = yearly_totals.get(yr, {})
            calc_val = np.nan
            try:
                clicks = totals_yr.get(CLICKS_COL, 0)
                impressions = totals_yr.get(IMPRESSIONS_COL, 0)
                orders = totals_yr.get(ORDERS_COL, 0)
                spend = totals_yr.get(SPEND_COL, 0)
                sales = totals_yr.get(SALES_COL, 0)

                if metric == CTR: calc_val = (clicks / impressions * 100) if impressions > 0 else 0
                elif metric == CVR: calc_val = (orders / clicks * 100) if clicks > 0 else 0
                elif metric == CPC: calc_val = (spend / clicks) if clicks > 0 else np.nan
                elif metric == ACOS: calc_val = (spend / sales * 100) if sales > 0 else np.nan
                elif metric == ROAS: calc_val = (sales / spend) if spend > 0 else np.nan
                elif metric == AD_PERC_SALE:
                    denom_yr = yearly_total_sales_denom.get(yr, 0)
                    calc_val = (sales / denom_yr * 100) if denom_yr > 0 else np.nan
                elif metric in totals_yr: calc_val = totals_yr.get(metric) # Use aggregated base value

                if isinstance(calc_val, (int, float)): calc_val = np.nan if calc_val in [np.inf, -np.inf] else calc_val
            except Exception as e: calc_val = np.nan # Set NaN on calculation error

            # Store calculated value for metric/year (used for change calc below)
            if yr in yearly_totals: yearly_totals[yr][metric] = calc_val
            # Add to output dict
            summary_row_data[f"{metric} {yr}"] = calc_val

    # Calculate % Change (single change column between last two years)
    actual_years_in_row = sorted([yr for yr in years_to_process if yr in yearly_totals and yearly_totals[yr]])
    if len(actual_years_in_row) >= 2:
        curr_yr, prev_yr = actual_years_in_row[-1], actual_years_in_row[-2]
        for metric in selected_metrics:
            val_curr = yearly_totals.get(curr_yr, {}).get(metric, np.nan)
            val_prev = yearly_totals.get(prev_yr, {}).get(metric, np.nan)
            change_val = np.nan
            if pd.notna(val_curr) and pd.notna(val_prev):
                # Use config constant
                if metric in PERCENTAGE_POINT_CHANGE_METRICS: change_val = val_curr - val_prev # Absolute diff
                else: # Percentage diff
                    if val_prev != 0: change_val = ((val_curr - val_prev) / abs(val_prev)) * 100
                    elif val_curr == 0: change_val = 0.0 # Handle 0 to 0
            change_val = np.nan if change_val in [np.inf, -np.inf] else change_val
            summary_row_data[f"{metric} % Change"] = change_val

    # Create DataFrame and order columns
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