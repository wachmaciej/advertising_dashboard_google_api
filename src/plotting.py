# src/plotting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
from src.config import (DATE_COL, YEAR_COL, WEEK_COL, PRODUCT_COL, PORTFOLIO_COL,
                        CLICKS_COL, IMPRESSIONS_COL, ORDERS_COL, SPEND_COL, SALES_COL,
                        TOTAL_SALES_COL, # Needed for Ad % Sale chart denominator merge check
                        CTR, CVR, ACOS, ROAS, CPC, AD_PERC_SALE, METRIC_COMPONENTS, DERIVED_METRICS)

# Suppress warnings within this module if needed, though better at app level
# warnings.filterwarnings("ignore")

@st.cache_data
def create_metric_comparison_chart(df, metric, portfolio_name=None, campaign_type="Sponsored Products"):
    """Creates a bar chart comparing a metric by Portfolio Name. Calculates derived metrics."""
    # required_cols_base = {PRODUCT_COL, PORTFOLIO_COL} # Only required if filtering by product type

    if df is None or df.empty:
        return go.Figure()

    filtered_df = df.copy() # Start with copy of passed data

    # <<< CHANGE: Conditional filtering by product type >>>
    if campaign_type != "All Product Types":
        # Only apply product type filter if not 'All' AND the column exists
        if PRODUCT_COL in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[PRODUCT_COL] == campaign_type].copy() # Apply filter
        else:
            # If PRODUCT_COL is missing, can't filter by specific product type
            # This case should ideally be handled before calling this function, but defensive check
            st.warning(f"'{PRODUCT_COL}' column not found for filtering in comparison chart, but a specific product type was selected.")
            return go.Figure()
    # else: if campaign_type is "All Product Types", use the entire dataframe passed


    if filtered_df.empty: # Check after potential product filter
        return go.Figure()

    # Check if PORTFOLIO_COL is available for grouping, otherwise return empty figure
    group_col = PORTFOLIO_COL
    if group_col not in filtered_df.columns:
        st.warning(f"Grouping column '{group_col}' not found for comparison chart.")
        return go.Figure()


    # Check if metric exists OR if base components for calculation exist
    metric_col_exists = metric in filtered_df.columns
    # Check base components exist *in the filtered data* for derived metrics
    base_components = METRIC_COMPONENTS.get(metric, set())
    can_calculate_metric = bool(base_components) and base_components.issubset(filtered_df.columns)

    if not metric_col_exists and not can_calculate_metric:
        missing = {metric} if not base_components else base_components - set(filtered_df.columns)
        st.warning(f"Comparison chart cannot display '{metric}'. Missing required columns: {missing} in {campaign_type} data.")
        return go.Figure()

    # Handle portfolio filtering (applied after product type filter)
    filtered_df[PORTFOLIO_COL] = filtered_df[PORTFOLIO_COL].fillna("Unknown Portfolio")
    if portfolio_name and portfolio_name != "All Portfolios":
        if portfolio_name in filtered_df[PORTFOLIO_COL].unique():
            filtered_df = filtered_df[filtered_df[PORTFOLIO_COL] == portfolio_name]
        else:
            # This warning is already handled in app.py, but kept for robustness
            # st.warning(f"Portfolio '{portfolio_name}' not found for {campaign_type}. Showing all.")
            portfolio_name = "All Portfolios" # Reset variable to reflect change

    if filtered_df.empty: # Check again after potential portfolio filter
        return go.Figure()

    # Aggregation and Calculation logic
    grouped_df = pd.DataFrame() # Initialize
    try:
        # Handle metrics calculated from aggregated base components
        if metric in DERIVED_METRICS: # Check against derived metrics list
            # Ensure base components are numeric before aggregation
            valid_base = True
            for base_col in base_components:
                 if base_col in filtered_df.columns and not pd.api.types.is_numeric_dtype(filtered_df[base_col]):
                     st.warning(f"Base column '{base_col}' for metric '{metric}' is not numeric. Attempting coercion.")
                     try: filtered_df[base_col] = pd.to_numeric(filtered_df[base_col], errors='coerce').fillna(0) # Coerce and fill 0 for sum
                     except Exception as e: st.warning(f"Failed to coerce '{base_col}': {e}"); valid_base = False; break # Exit inner loop
                 elif base_col not in filtered_df.columns: # Should be caught earlier, but defensive
                     st.warning(f"Base column '{base_col}' missing for metric '{metric}'.")
                     valid_base = False; break

            if not valid_base: return go.Figure()

            # Perform aggregation only for the required base columns that are numeric
            agg_cols = list(base_components & set(filtered_df.columns))
            if not agg_cols: # No numeric base columns found after checks
                 st.warning(f"No valid numeric base columns found for aggregating metric '{metric}'.")
                 return go.Figure()
            agg_dict = {col: "sum" for col in agg_cols}
            agg_df = filtered_df.groupby(group_col).agg(agg_dict).reset_index()


            # Recalculate derived metrics after aggregation
            if metric == CTR:
                 agg_df[metric] = agg_df.apply(lambda r: (r.get(CLICKS_COL,0) / r.get(IMPRESSIONS_COL,0) * 100) if r.get(IMPRESSIONS_COL) else 0, axis=1).round(2)
            elif metric == CVR:
                 agg_df[metric] = agg_df.apply(lambda r: (r.get(ORDERS_COL,0) / r.get(CLICKS_COL,0) * 100) if r.get(CLICKS_COL) else 0, axis=1).round(2)
            elif metric == ACOS:
                 agg_df[metric] = agg_df.apply(lambda r: (r.get(SPEND_COL,0) / r.get(SALES_COL,0) * 100) if r.get(SALES_COL) else np.nan, axis=1).round(2)
            elif metric == ROAS:
                 agg_df[metric] = agg_df.apply(lambda r: (r.get(SALES_COL,0) / r.get(SPEND_COL,0)) if r.get(SPEND_COL) else np.nan, axis=1).round(2)
            elif metric == CPC:
                 agg_df[metric] = agg_df.apply(lambda r: (r.get(SPEND_COL,0) / r.get(CLICKS_COL,0)) if r.get(CLICKS_COL) else np.nan, axis=1).round(2)
            elif metric == AD_PERC_SALE:
                 # Ad % Sale requires total sales denominator, does not aggregate correctly by portfolio here
                 st.info(f"Metric '{AD_PERC_SALE}' cannot be displayed in the Portfolio Comparison bar chart.")
                 return go.Figure()

            # Ensure the calculated metric column exists, even if all values are NaN
            if metric not in agg_df.columns:
                 agg_df[metric] = np.nan

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
        return go.Figure()

    grouped_df = grouped_df.sort_values(metric, ascending=False)

    title_suffix = f" - {portfolio_name}" if portfolio_name and portfolio_name != "All Portfolios" else ""
    # <<< CHANGE: Use campaign_type directly in title >>>
    chart_title = f"{metric} by Portfolio ({campaign_type}){title_suffix}"

    fig = px.bar(grouped_df, x=group_col, y=metric, title=chart_title, text_auto=True)

    # Apply formatting
    # Use constants for metric names in format conditions
    if metric in [SPEND_COL, SALES_COL]:
        fig.update_traces(texttemplate='%{y:$,.0f}')
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f")
    elif metric in [CTR, CVR, ACOS]: # Use constants
        fig.update_traces(texttemplate='%{y:.1f}%')
        fig.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".1f")
    elif metric == ROAS: # Use constant
        fig.update_traces(texttemplate='%{y:.2f}')
        fig.update_layout(yaxis_tickformat=".2f")
    elif metric == CPC: # Use constant
        fig.update_traces(texttemplate='%{y:$,.2f}') # Currency format for text on bars
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f") # Currency format for y-axis
    else: # Default formatting for Impressions, Clicks, Orders, Units (summed metrics)
        fig.update_traces(texttemplate='%{y:,.0f}')
        fig.update_layout(yaxis_tickformat=",.0f")

    fig.update_layout(margin=dict(t=50, b=50, l=20, r=20), height=400)
    return fig


@st.cache_data
def create_metric_over_time_chart(data, metric, portfolio, product_type, show_yoy=True, weekly_total_sales_data=None):
    """Create a chart showing metric over time with optional YoY comparison."""
    if data is None or data.empty:
        return go.Figure()

    base_required = {DATE_COL, YEAR_COL, WEEK_COL}
    if not base_required.issubset(data.columns):
        missing = base_required - set(data.columns)
        st.warning(f"Metric over time chart missing required date/time columns: {missing}")
        return go.Figure()
    if not pd.api.types.is_datetime64_any_dtype(data[DATE_COL]):
        st.warning(f"'{DATE_COL}' column is not datetime type for time chart. Attempting conversion.")
        # Attempt conversion
        try:
            data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors='coerce')
            if data[DATE_COL].isnull().all():
                st.error(f"Failed to convert '{DATE_COL}' to datetime for time chart.")
                return go.Figure()
        except Exception:
            st.error(f"Failed to convert '{DATE_COL}' to datetime for time chart.")
            return go.Figure()


    data_copy = data.copy() # Work on a copy

    filtered_data = data_copy.copy() # Start with copy of passed data

    # <<< CHANGE: Conditional filtering by product type >>>
    if product_type != "All Product Types":
        # Only apply product type filter if not 'All' AND the column exists
        if PRODUCT_COL in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[PRODUCT_COL] == product_type].copy()
        else:
             # If PRODUCT_COL is missing, cannot filter by specific product type
             st.warning(f"'{PRODUCT_COL}' column not found for filtering in time chart, but a specific product type was selected.")
             return go.Figure()
    # else: if product_type is "All Product Types", use the entire dataframe passed


    # Handle portfolio filtering (applied after product type filter)
    if PORTFOLIO_COL in filtered_data.columns:
         filtered_data[PORTFOLIO_COL] = filtered_data[PORTFOLIO_COL].fillna("Unknown Portfolio")
         if portfolio != "All Portfolios":
             if portfolio in filtered_data[PORTFOLIO_COL].unique():
                 filtered_data = filtered_data[filtered_data[PORTFOLIO_COL] == portfolio].copy()
             else:
                 # This warning is already handled in app.py, but kept for robustness
                 # st.warning(f"Portfolio '{portfolio}' not found for {product_type}. Showing all.")
                 portfolio = "All Portfolios" # Update variable to reflect change
    elif portfolio != "All Portfolios":
         # If PORTFOLIO_COL is missing, but a specific portfolio was selected, this is an issue
         st.warning(f"'{PORTFOLIO_COL}' column not found, cannot filter by Portfolio '{portfolio}'. Showing all data for the selected product type.")
         portfolio = "All Portfolios" # Reset to show all


    if filtered_data.empty: # Check after potential product and portfolio filters
        return go.Figure()

    # --- Define required base components for derived metrics ---
    base_needed_for_metric = METRIC_COMPONENTS.get(metric, set())
    is_derived_metric = metric in DERIVED_METRICS

    # --- Check if necessary columns exist for the selected metric ---
    metric_exists_in_input = metric in filtered_data.columns
    # Check base components exist *in the filtered data* for derived metrics
    base_components_exist = base_needed_for_metric.issubset(filtered_data.columns)

    ad_sale_check_passed = True # Assume pass unless specific checks fail
    if metric == AD_PERC_SALE:
        # Need SALES in the filtered data AND the weekly_total_sales_data (denominator)
        if SALES_COL not in filtered_data.columns: # Check 'Sales' column exists in the filtered data
            st.warning(f"Metric chart requires '{SALES_COL}' column for '{AD_PERC_SALE}'.")
            ad_sale_check_passed = False
        if weekly_total_sales_data is None or weekly_total_sales_data.empty:
            ad_sale_check_passed = False
        # Check required columns *in the passed denominator dataframe* if it's not empty
        elif not {YEAR_COL, WEEK_COL, "Weekly_Total_Sales"}.issubset(weekly_total_sales_data.columns):
            st.warning(f"Passed 'weekly_total_sales_data' is missing required columns ({YEAR_COL}, {WEEK_COL}, Weekly_Total_Sales).")
            ad_sale_check_passed = False
        # Ensure the denominator data contains relevant years/weeks for the filtered data period
        # This requires YEAR_COL and WEEK_COL in the filtered data
        elif YEAR_COL in filtered_data.columns and WEEK_COL in filtered_data.columns:
             if not weekly_total_sales_data[(weekly_total_sales_data[YEAR_COL].isin(filtered_data[YEAR_COL].unique())) & (weekly_total_sales_data[WEEK_COL].isin(filtered_data[WEEK_COL].unique()))].empty:
                 pass # Denom data has overlap with filtered data's time period - OK
             else:
                 st.warning(f"Passed 'weekly_total_sales_data' does not contain data for the selected time period.")
                 ad_sale_check_passed = False
        else: # Missing Year/Week in filtered data for this check
             st.warning(f"Missing '{YEAR_COL}' or '{WEEK_COL}' in data for checking Ad % Sale denominator coverage.")
             ad_sale_check_passed = False


    # If it's a derived metric, we MUST have its base components (or Sales for Ad % Sale if denom is separate)
    if is_derived_metric:
        if metric == AD_PERC_SALE:
            if not ({SALES_COL}.issubset(filtered_data.columns) and ad_sale_check_passed):
                 # This handles case where 'Sales' exists but denom check failed, or 'Sales' is missing
                 missing_reason = f"Missing '{SALES_COL}' column." if SALES_COL not in filtered_data.columns else "Denominator data is missing or invalid."
                 st.warning(f"Cannot calculate derived metric '{metric}'. {missing_reason}")
                 return go.Figure()
        elif not base_components_exist: # Other derived metrics
            missing_bases = base_needed_for_metric - set(filtered_data.columns)
            st.warning(f"Cannot calculate derived metric '{metric}'. Missing required base columns: {missing_bases}")
            return go.Figure()

    # If it's NOT a derived metric, it MUST exist in the input data and be numeric
    if not is_derived_metric:
         if metric not in filtered_data.columns:
              st.warning(f"Metric chart requires column '{metric}' in the data.")
              return go.Figure()
         if not pd.api.types.is_numeric_dtype(filtered_data[metric]):
              st.warning(f"Metric column '{metric}' is not numeric. Attempting coercion.")
              try: filtered_data[metric] = pd.to_numeric(filtered_data[metric], errors='coerce')
              except Exception as e: st.warning(f"Failed to coerce '{metric}': {e}"); return go.Figure()


    # --- Start Plotting ---
    # Get years from the *filtered* data
    years = sorted(filtered_data[YEAR_COL].dropna().unique().astype(int))
    fig = go.Figure()

    # Define hover formats based on metric type (use constants)
    if metric in [CTR, CVR, ACOS, AD_PERC_SALE]: hover_suffix = "%"; hover_format = ".1f"
    elif metric in [SPEND_COL, SALES_COL, CPC]: hover_suffix = ""; hover_format = "$,.2f" # Added CPC here
    elif metric == ROAS: hover_suffix = ""; hover_format = ".2f"
    else: hover_suffix = ""; hover_format = ",.0f" # Impressions, Clicks, Orders, Units
    # Updated hover template to explicitly show Year and Week from customdata
    base_hover_template = f"Year: %{{customdata[2]}}<br>Week: %{{customdata[0]}}<br>Date: %{{customdata[1]|%Y-%m-%d}}<br>{metric}: %{{y:{hover_format}}}{hover_suffix}<extra></extra>"


    processed_years = []
    colors = px.colors.qualitative.Plotly

    # ========================
    # YoY Plotting Logic
    # ========================
    # Only show YoY lines if explicitly selected AND there are at least 2 years of data in the filtered data
    # AND required columns for aggregation are present
    required_yoy_agg_cols = {DATE_COL, YEAR_COL, WEEK_COL}
    if show_yoy and len(years) >= 2 and required_yoy_agg_cols.issubset(filtered_data.columns):

        # Define columns needed for aggregation: base components + DATE_COL, YEAR_COL, WEEK_COL
        cols_to_agg_yoy_base = base_needed_for_metric | {DATE_COL, YEAR_COL, WEEK_COL}
        # Also include the original metric IF it's not derived and exists
        if not is_derived_metric and metric_exists_in_input:
             cols_to_agg_yoy_base.add(metric)

        actual_cols_to_agg_yoy = list(cols_to_agg_yoy_base & set(filtered_data.columns))

        # Check if any numeric column (base component or the metric itself if not derived) was found
        numeric_cols_for_agg_yoy = [col for col in actual_cols_to_agg_yoy if col not in [DATE_COL, YEAR_COL, WEEK_COL] and pd.api.types.is_numeric_dtype(filtered_data[col])]

        if not numeric_cols_for_agg_yoy or DATE_COL not in actual_cols_to_agg_yoy:
            st.warning(f"Not enough relevant numeric data or '{DATE_COL}' column found to aggregate for metric '{metric}' for the YoY chart.")
            show_yoy = False # Fallback to non-YoY
        else:
            try:
                agg_dict_yoy = {col: "sum" for col in numeric_cols_for_agg_yoy}
                agg_dict_yoy[DATE_COL] = 'min' # Get earliest date within the week for hover

                # Aggregate: Sum up base components (and original metric if not derived) by Year/Week
                grouped = filtered_data.groupby([YEAR_COL, WEEK_COL], as_index=False).agg(agg_dict_yoy)
                grouped[DATE_COL] = pd.to_datetime(grouped[DATE_COL])

            except Exception as e:
                st.warning(f"Could not group data by week for YoY chart: {e}")
                show_yoy = False # Fallback to non-YoY


        # --- *** ALWAYS RECALCULATE DERIVED METRICS POST-AGGREGATION *** ---
        # This block only runs if grouped was successfully created in the try block above
        if show_yoy: # Check again if we are still doing YoY
             metric_calculated_successfully = False # Flag
             if is_derived_metric:
                # Check if base components are available in the *aggregated* df
                agg_base_components_exist = base_needed_for_metric.issubset(grouped.columns)

                if metric == AD_PERC_SALE:
                    if agg_base_components_exist and ad_sale_check_passed: # Check SALES in grouped AND denom data check
                         try:
                            temp_denom = weekly_total_sales_data.copy()
                            # Ensure data types match for merge (Year and Week from grouped df)
                            if YEAR_COL in grouped.columns and YEAR_COL in temp_denom.columns: temp_denom[YEAR_COL] = temp_denom[YEAR_COL].astype(grouped[YEAR_COL].dtype)
                            if WEEK_COL in grouped.columns and WEEK_COL in temp_denom.columns: temp_denom[WEEK_COL] = temp_denom[WEEK_COL].astype(grouped[WEEK_COL].dtype)

                            # Ensure merge keys exist in both dataframes
                            merge_keys_denom = [k for k in [YEAR_COL, WEEK_COL] if k in grouped.columns and k in temp_denom.columns]
                            if not merge_keys_denom:
                                 st.warning("Missing Year or Week column for Ad % Sale denominator merge (YoY).")
                                 grouped[metric] = np.nan # Ensure metric column exists
                            else:
                                grouped_merged = pd.merge(grouped, temp_denom[merge_keys_denom + ['Weekly_Total_Sales']], on=merge_keys_denom, how='left')
                                # Ensure 'Sales' column exists in grouped_merged before calculating
                                if SALES_COL in grouped_merged.columns:
                                    # Handle potential division by zero or NaN in denominator
                                    grouped_merged[metric] = grouped_merged.apply(lambda r: (r.get(SALES_COL, 0) / r.get('Weekly_Total_Sales', 0) * 100) if pd.notna(r.get('Weekly_Total_Sales')) and r.get('Weekly_Total_Sales', 0) > 0 else np.nan, axis=1).round(1)
                                    grouped = grouped_merged.drop(columns=['Weekly_Total_Sales'], errors='ignore') # Drop temp col
                                    metric_calculated_successfully = True
                                else:
                                    st.warning(f"'{SALES_COL}' column not found after merging denominator data for Ad % Sale calculation (YoY).")
                                    grouped[metric] = np.nan # Ensure column exists

                         except Exception as e:
                             st.warning(f"Failed to merge/calculate Ad % Sale for YoY chart: {e}")
                             grouped[metric] = np.nan # Ensure column exists even if calculation fails

                    else:
                        # This handles cases where base components are missing in aggregated data or denom check failed
                        missing_reason = "Base components missing in aggregated data." if not agg_base_components_exist else "Denominator data is invalid."
                        st.warning(f"Cannot calculate '{AD_PERC_SALE}' YoY chart: {missing_reason}")
                        grouped[metric] = np.nan # Ensure column exists if calculation wasn't possible

                # Other derived metrics - require base components in the aggregated df
                elif agg_base_components_exist:
                     if metric == CTR:
                         grouped[metric] = grouped.apply(lambda r: (r.get(CLICKS_COL,0) / r.get(IMPRESSIONS_COL,0) * 100) if r.get(IMPRESSIONS_COL) else 0, axis=1).round(1)
                     elif metric == CVR:
                         grouped[metric] = grouped.apply(lambda r: (r.get(ORDERS_COL,0) / r.get(CLICKS_COL,0) * 100) if r.get(CLICKS_COL) else 0, axis=1).round(1)
                     elif metric == ACOS:
                         grouped[metric] = grouped.apply(lambda r: (r.get(SPEND_COL,0) / r.get(SALES_COL,0) * 100) if r.get(SALES_COL) else np.nan, axis=1).round(1)
                     elif metric == ROAS:
                         grouped[metric] = grouped.apply(lambda r: (r.get(SALES_COL,0) / r.get(SPEND_COL,0)) if r.get(SPEND_COL) else np.nan, axis=1).round(2)
                     elif metric == CPC:
                          grouped[metric] = grouped.apply(lambda r: (r.get(SPEND_COL,0) / r.get(CLICKS_COL,0)) if r.get(CLICKS_COL) else np.nan, axis=1).round(2)
                     metric_calculated_successfully = True

                if not metric_calculated_successfully and metric not in grouped.columns:
                     # If it's a derived metric and calculation failed, ensure the column exists with NaNs
                     grouped[metric] = np.nan
                     st.warning(f"Failed to calculate derived metric '{metric}' for YoY chart.")
                     show_yoy = False # Fallback to non-YoY if derived metric calculation fails

             else: # Metric was not derived, should be directly aggregated
                 # Check if the metric column exists in the aggregated DataFrame
                 if metric not in grouped.columns:
                      st.warning(f"Aggregated metric column '{metric}' not found (YoY).")
                      show_yoy = False # Fallback to non-YoY
                 # Ensure it's numeric
                 elif not pd.api.types.is_numeric_dtype(grouped[metric]):
                      st.warning(f"Aggregated metric column '{metric}' is not numeric (YoY).")
                      show_yoy = False # Fallback to non-YoY
                 else:
                     metric_calculated_successfully = True # Treat direct aggregation as success


        # Handle Inf/-Inf values if the metric column exists after calculation/aggregation
        if show_yoy and metric in grouped.columns:
            grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)

        # --- Plotting YoY data ---
        # Only plot if show_yoy is still True AND the metric column exists and has valid data
        if show_yoy and metric in grouped.columns and not grouped[metric].isnull().all():
             min_week_data, max_week_data = 53, 0
             # Filter out years with no data points for the specific metric
             valid_years_for_plotting = sorted(grouped.dropna(subset=[metric])[YEAR_COL].unique())

             if len(valid_years_for_plotting) < 2: # Double check minimum years after dropping NaNs
                 st.info(f"Not enough years with valid data points for '{metric}' to show meaningful YoY comparison ({len(valid_years_for_plotting)} years with valid data).")
                 show_yoy = False # Fallback to non-YoY if not enough years with *data* for the metric
             else:
                for i, year in enumerate(valid_years_for_plotting): # Iterate over years that actually have data for the metric
                    year_data = grouped[grouped[YEAR_COL] == year].sort_values(WEEK_COL).dropna(subset=[metric]) # Drop NaNs for this metric for this line

                    # Should not be empty due to valid_years_for_plotting check, but safety
                    if year_data.empty: continue

                    processed_years.append(year)
                    min_week_data = min(min_week_data, year_data[WEEK_COL].min())
                    max_week_data = max(max_week_data, year_data[WEEK_COL].max())
                    # customdata for hover: [Week, Date, Year]
                    custom_data_hover = year_data[[WEEK_COL, DATE_COL, YEAR_COL]]

                    fig.add_trace(
                        go.Scatter(x=year_data[WEEK_COL], y=year_data[metric], mode="lines+markers", name=f"{year}",
                                 line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=6),
                                 customdata=custom_data_hover, hovertemplate=base_hover_template,
                                 connectgaps=True) # <<< Added connectgaps=True here
                    )

                # Add month annotations if data was plotted
                if processed_years:
                     month_approx_weeks = { 1: 2.5, 2: 6.5, 3: 10.5, 4: 15, 5: 19.5, 6: 24, 7: 28, 8: 32.5, 9: 37, 10: 41.5, 11: 46, 12: 50.5 }
                     month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                     fig.update_layout(
                         xaxis=dict(
                             title="Week of Year",
                             tickmode = 'array',
                             # Generate ticks for all possible weeks + padding
                             tickvals = list(range(0, 55, 5)), # Show ticks every 5 weeks
                             # Include annotation positions in range calculation
                             range=[min(0, min_week_data - 2), max(54, max_week_data + 2)] # Ensure full range plus padding
                         ),
                         xaxis_showticklabels=True,
                         legend_title="Year",
                         margin=dict(b=70) # Adjusted bottom margin to make space for annotations if needed
                     )

                     # Add month annotations relative to the week of year axis
                     annotations = []
                     for month_num, week_val in month_approx_weeks.items():
                          # Only add annotation if it falls within the actual data range + some buffer
                          if week_val >= min_week_data - 4 and week_val <= max_week_data + 4:
                               annotations.append(dict(x=week_val, y=-0.1, xref="x", yref="paper", text=month_names[month_num-1], showarrow=False, font=dict(size=10, color="grey")))
                     fig.update_layout(annotations=annotations)

                # If YoY was successfully plotted, show_yoy remains True, the next block is skipped


    # ========================
    # Non-YoY Plotting Logic (Execute ONLY if show_yoy is False after all checks)
    # ========================
    if not show_yoy: # This block runs if show_yoy was initially False or fell back to False
        # Define columns needed for aggregation: base components + DATE_COL, YEAR_COL, WEEK_COL
        cols_to_agg_noyoy_base = base_needed_for_metric | {DATE_COL, YEAR_COL, WEEK_COL}
        if not is_derived_metric and metric_exists_in_input:
            cols_to_agg_noyoy_base.add(metric)

        actual_cols_to_agg_noyoy = list(cols_to_agg_noyoy_base & set(filtered_data.columns))

        if not {DATE_COL, YEAR_COL, WEEK_COL}.issubset(actual_cols_to_agg_noyoy):
            st.warning(f"Missing '{DATE_COL}', '{YEAR_COL}', or '{WEEK_COL}' for aggregation (non-YoY).")
            return go.Figure()

        # Check if any numeric column (base component or the metric itself if not derived) was found
        numeric_cols_for_agg_noyoy = [col for col in actual_cols_to_agg_noyoy if col not in [DATE_COL, YEAR_COL, WEEK_COL] and pd.api.types.is_numeric_dtype(filtered_data[col])]

        if not numeric_cols_for_agg_noyoy:
             st.warning(f"No valid numeric column found to aggregate for metric '{metric}' for the time chart (non-YoY).")
             return go.Figure()

        try:
            agg_dict_noyoy = {col: "sum" for col in numeric_cols_for_agg_noyoy}

            # Aggregate: Sum up base components (and original metric if not derived) by Date/Year/Week
            # Grouping by Date/Year/Week is crucial to ensure weekly points aligned correctly
            grouped = filtered_data.groupby([DATE_COL, YEAR_COL, WEEK_COL], as_index=False).agg(agg_dict_noyoy)
            grouped[DATE_COL] = pd.to_datetime(grouped[DATE_COL]) # Ensure datetime type

        except Exception as e:
            st.warning(f"Could not group data for time chart (non-YoY): {e}")
            return go.Figure()


        # --- *** ALWAYS RECALCULATE DERIVED METRICS POST-AGGREGATION *** ---
        metric_calculated_successfully = False # Flag
        if is_derived_metric:
             # Check if base components are available in the *aggregated* df
             agg_base_components_exist = base_needed_for_metric.issubset(grouped.columns)

             if metric == AD_PERC_SALE:
                 # For non-YoY Ad % Sale chart, we still need the weekly total sales denominator
                 if agg_base_components_exist and ad_sale_check_passed:
                     try:
                         temp_denom = weekly_total_sales_data.copy()
                         if YEAR_COL in grouped.columns and YEAR_COL in temp_denom.columns: temp_denom[YEAR_COL] = temp_denom[YEAR_COL].astype(grouped[YEAR_COL].dtype)
                         if WEEK_COL in grouped.columns and WEEK_COL in temp_denom.columns: temp_denom[WEEK_COL] = temp_denom[WEEK_COL].astype(grouped[WEEK_COL].dtype)

                         merge_keys_denom = [k for k in [YEAR_COL, WEEK_COL] if k in grouped.columns and k in temp_denom.columns]
                         if not merge_keys_denom:
                              st.warning("Missing Year or Week column for Ad % Sale denominator merge (non-YoY).")
                              grouped[metric] = np.nan # Ensure metric column exists
                         else:
                            grouped_merged = pd.merge(grouped, temp_denom[merge_keys_denom + ['Weekly_Total_Sales']], on=merge_keys_denom, how='left')
                            if SALES_COL in grouped_merged.columns:
                                grouped_merged[metric] = grouped_merged.apply(lambda r: (r.get(SALES_COL, 0) / r.get('Weekly_Total_Sales', 0) * 100) if pd.notna(r.get('Weekly_Total_Sales')) and r.get('Weekly_Total_Sales', 0) > 0 else np.nan, axis=1).round(1)
                                grouped = grouped_merged.drop(columns=['Weekly_Total_Sales'], errors='ignore')
                                metric_calculated_successfully = True
                            else:
                                 st.warning(f"'{SALES_COL}' column not found after merging denominator data for Ad % Sale calculation (non-YoY).")
                                 grouped[metric] = np.nan

                     except Exception as e:
                         st.warning(f"Failed to merge/calculate Ad % Sale for non-YoY chart: {e}")
                         grouped[metric] = np.nan
                 else:
                     st.warning(f"Cannot calculate '{AD_PERC_SALE}' non-YoY chart: Base components missing in aggregated data or denominator invalid.")
                     grouped[metric] = np.nan


             # Other derived metrics - require base components in the aggregated df
             elif agg_base_components_exist:
                  if metric == CTR:
                      grouped[metric] = grouped.apply(lambda r: (r.get(CLICKS_COL,0) / r.get(IMPRESSIONS_COL,0) * 100) if r.get(IMPRESSIONS_COL) else 0, axis=1).round(1)
                  elif metric == CVR:
                      grouped[metric] = grouped.apply(lambda r: (r.get(ORDERS_COL,0) / r.get(CLICKS_COL,0) * 100) if r.get(CLICKS_COL) else 0, axis=1).round(1)
                  elif metric == ACOS:
                      grouped[metric] = grouped.apply(lambda r: (r.get(SPEND_COL,0) / r.get(SALES_COL,0) * 100) if r.get(SALES_COL) else np.nan, axis=1).round(1)
                  elif metric == ROAS:
                      grouped[metric] = grouped.apply(lambda r: (r.get(SALES_COL,0) / r.get(SPEND_COL,0)) if r.get(SPEND_COL) else np.nan, axis=1).round(2)
                  elif metric == CPC:
                       grouped[metric] = grouped.apply(lambda r: (r.get(SPEND_COL,0) / r.get(CLICKS_COL,0)) if r.get(CLICKS_COL) else np.nan, axis=1).round(2)
                  metric_calculated_successfully = True

             if not metric_calculated_successfully and metric not in grouped.columns:
                  grouped[metric] = np.nan
                  st.warning(f"Failed to calculate derived metric '{metric}' for non-YoY chart.")
             # else: metric calculated successfully, or it's a base metric handled below

        else:
            # If metric wasn't derived, it should exist and be numeric after aggregation
            if metric not in grouped.columns:
                st.warning(f"Metric column '{metric}' not found after aggregation (non-YoY).")
                return go.Figure()
            if not pd.api.types.is_numeric_dtype(grouped[metric]):
                 st.warning(f"Aggregated metric column '{metric}' is not numeric (non-YoY).")
                 # Attempt coercion as a last resort before giving up
                 try: grouped[metric] = pd.to_numeric(grouped[metric], errors='coerce')
                 except Exception as e: st.warning(f"Failed to coerce aggregated '{metric}': {e}"); return go.Figure()
            metric_calculated_successfully = True


        # Handle Inf/-Inf values if the metric column exists
        if metric in grouped.columns:
             grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)

        # --- Plotting Non-YoY data ---
        if metric not in grouped.columns or grouped[metric].isnull().all():
            st.info(f"No valid data points for metric '{metric}' over time.")
            return go.Figure() # Return empty figure if all values are NaN or column missing

        grouped = grouped.sort_values(DATE_COL)
        # customdata for hover: [Week, Date, Year]
        custom_data_hover_noyoy = grouped[[WEEK_COL, DATE_COL, YEAR_COL]]
        fig.add_trace(
            go.Scatter(x=grouped[DATE_COL], y=grouped[metric], mode="lines+markers", name=metric,
                       line=dict(color="#1f77b4", width=2), marker=dict(size=6),
                       customdata=custom_data_hover_noyoy, hovertemplate=base_hover_template)
        )
        fig.update_layout(xaxis_title="Date", showlegend=False) # Hide legend if only one line


    # --- Final Chart Layout ---
    portfolio_title = f" for {portfolio}" if portfolio != "All Portfolios" else " for All Portfolios"
    # Use the selected product type string directly in the title
    final_chart_title = f"{metric} "

    # Determine the title based on whether YoY was successfully plotted or if we defaulted to non-YoY
    # Check if show_yoy is still True AND there are valid years for plotting AND the metric column exists and has data
    if show_yoy and ('valid_years_for_plotting' in locals() and len(valid_years_for_plotting) >= 2) and (metric in grouped.columns and not grouped[metric].isnull().all()):
        final_chart_title += f"Weekly Comparison {portfolio_title} ({product_type})"
        final_xaxis_title = "Week of Year"
    else: # If show_yoy became False, display non-YoY
        final_chart_title += f"Over Time (Weekly) {portfolio_title} ({product_type})"
        final_xaxis_title = "Week Ending Date" # More descriptive for time series


    final_margin = dict(t=80, b=70, l=70, r=30)
    fig.update_layout(
        title=final_chart_title, xaxis_title=final_xaxis_title, yaxis_title=metric,
        hovermode="x unified", template="plotly_white", yaxis=dict(rangemode="tozero"), margin=final_margin
    )

    # Apply Y-axis formatting based on the metric (use constants)
    if metric in [SPEND_COL, SALES_COL, CPC]: fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f")
    elif metric in [CTR, CVR, ACOS, AD_PERC_SALE]: fig.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".1f")
    elif metric == ROAS: fig.update_layout(yaxis_tickformat=".2f")
    else: fig.update_layout(yaxis_tickformat=",.0f") # Impressions, Clicks, Orders, Units

    return fig
