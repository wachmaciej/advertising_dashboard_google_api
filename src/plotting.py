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
    required_cols_base = {PRODUCT_COL, PORTFOLIO_COL}

    # Define base components needed if metric needs calculation
    base_components = METRIC_COMPONENTS.get(metric, set())

    if df is None or df.empty:
        return go.Figure()

    # Check base columns first
    if not required_cols_base.issubset(df.columns):
        missing = required_cols_base - set(df.columns)
        st.warning(f"Comparison chart missing base columns: {missing}")
        return go.Figure()

    filtered_df = df[df[PRODUCT_COL] == campaign_type].copy()
    if filtered_df.empty:
        return go.Figure()

    # Check if metric exists OR if base components for calculation exist
    metric_col_exists = metric in filtered_df.columns
    can_calculate_metric = bool(base_components) and base_components.issubset(filtered_df.columns)

    if not metric_col_exists and not can_calculate_metric:
        missing = {metric} if not base_components else base_components - set(filtered_df.columns)
        st.warning(f"Comparison chart cannot display '{metric}'. Missing required columns: {missing} in {campaign_type} data.")
        return go.Figure()

    # Handle portfolio filtering
    filtered_df[PORTFOLIO_COL] = filtered_df[PORTFOLIO_COL].fillna("Unknown Portfolio")
    if portfolio_name and portfolio_name != "All Portfolios":
        if portfolio_name in filtered_df[PORTFOLIO_COL].unique():
            filtered_df = filtered_df[filtered_df[PORTFOLIO_COL] == portfolio_name]
        else:
            st.warning(f"Portfolio '{portfolio_name}' not found for {campaign_type}. Showing all.")
            portfolio_name = "All Portfolios" # Reset variable to reflect change

    if filtered_df.empty: # Check again after potential portfolio filter
        return go.Figure()

    # Aggregation and Calculation logic
    grouped_df = pd.DataFrame() # Initialize
    group_col = PORTFOLIO_COL
    try:
        # Handle metrics calculated from aggregated base components
        if metric in DERIVED_METRICS: # Check against derived metrics list
            # Ensure base components are numeric before aggregation
            valid_base = True
            for base_col in base_components:
                if not pd.api.types.is_numeric_dtype(filtered_df[base_col]):
                     st.warning(f"Base column '{base_col}' for metric '{metric}' is not numeric.")
                     valid_base = False
                     break # Exit inner loop
            if not valid_base: return go.Figure()

            if metric == CTR:
                agg_df = filtered_df.groupby(group_col).agg(Nominator=(CLICKS_COL, "sum"), Denominator=(IMPRESSIONS_COL, "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"] * 100) if r["Denominator"] else 0, axis=1).round(2)
            elif metric == CVR:
                agg_df = filtered_df.groupby(group_col).agg(Nominator=(ORDERS_COL, "sum"), Denominator=(CLICKS_COL, "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"] * 100) if r["Denominator"] else 0, axis=1).round(2)
            elif metric == ACOS:
                agg_df = filtered_df.groupby(group_col).agg(Nominator=(SPEND_COL, "sum"), Denominator=(SALES_COL, "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"] * 100) if r["Denominator"] else np.nan, axis=1).round(2)
            elif metric == ROAS:
                agg_df = filtered_df.groupby(group_col).agg(Nominator=(SALES_COL, "sum"), Denominator=(SPEND_COL, "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"]) if r["Denominator"] else np.nan, axis=1).round(2)
            elif metric == CPC:
                agg_df = filtered_df.groupby(group_col).agg(Nominator=(SPEND_COL, "sum"), Denominator=(CLICKS_COL, "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"]) if r["Denominator"] else np.nan, axis=1).round(2)
            elif metric == AD_PERC_SALE:
                # Ad % Sale doesn't make sense grouped by portfolio in this chart context
                # It requires a total sales denominator, not a portfolio-specific one.
                st.info(f"Metric '{AD_PERC_SALE}' cannot be displayed in the Portfolio Comparison bar chart.")
                return go.Figure()


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

    base_required = {PRODUCT_COL, PORTFOLIO_COL, DATE_COL, YEAR_COL, WEEK_COL}
    if not base_required.issubset(data.columns):
        missing = base_required - set(data.columns)
        st.warning(f"Metric over time chart missing required columns: {missing}")
        return go.Figure()
    if not pd.api.types.is_datetime64_any_dtype(data[DATE_COL]):
        st.warning(f"'{DATE_COL}' column is not datetime type for time chart.")
        # Attempt conversion maybe? Or just return figure.
        try:
            data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors='coerce')
            if data[DATE_COL].isnull().all():
                st.error(f"Failed to convert '{DATE_COL}' to datetime for time chart.")
                return go.Figure()
        except Exception:
             st.error(f"Failed to convert '{DATE_COL}' to datetime for time chart.")
             return go.Figure()


    data_copy = data.copy() # Work on a copy

    filtered_data = data_copy[data_copy[PRODUCT_COL] == product_type].copy()
    filtered_data[PORTFOLIO_COL] = filtered_data[PORTFOLIO_COL].fillna("Unknown Portfolio")
    if portfolio != "All Portfolios":
        if portfolio in filtered_data[PORTFOLIO_COL].unique():
            filtered_data = filtered_data[filtered_data[PORTFOLIO_COL] == portfolio]
        else:
            st.warning(f"Portfolio '{portfolio}' not found for {product_type}. Showing all.")
            portfolio = "All Portfolios" # Update variable to reflect change

    if filtered_data.empty:
        return go.Figure()

    # --- Define required base components for derived metrics ---
    base_needed_for_metric = METRIC_COMPONENTS.get(metric, set())
    is_derived_metric = metric in DERIVED_METRICS

    # --- Check if necessary columns exist for the selected metric ---
    metric_exists_in_input = metric in filtered_data.columns
    base_components_exist = base_needed_for_metric.issubset(filtered_data.columns)

    ad_sale_check_passed = True # Assume pass unless specific checks fail
    if metric == AD_PERC_SALE:
        if SALES_COL not in filtered_data.columns: # Check 'Sales' column exists
            st.warning(f"Metric chart requires '{SALES_COL}' column for '{AD_PERC_SALE}'.")
            ad_sale_check_passed = False
        if weekly_total_sales_data is None or weekly_total_sales_data.empty:
            # This info message is okay if Ad % Sale is just one option the user *might* select
            # If it IS selected, the error will be caught below.
            # st.info(f"Denominator data (weekly total sales) not available for '{AD_PERC_SALE}' calculation.")
            ad_sale_check_passed = False
        # Check required columns *in the passed denominator dataframe*
        elif not {YEAR_COL, WEEK_COL, "Weekly_Total_Sales"}.issubset(weekly_total_sales_data.columns):
            st.warning(f"Passed 'weekly_total_sales_data' is missing required columns ({YEAR_COL}, {WEEK_COL}, Weekly_Total_Sales).")
            ad_sale_check_passed = False

    # If it's a derived metric, we MUST have its base components
    if is_derived_metric and not base_components_exist:
         # If Ad % Sale, 'Sales' is the base component needed in *this* dataframe
        if metric == AD_PERC_SALE:
            if SALES_COL not in filtered_data.columns:
                st.warning(f"Cannot calculate '{AD_PERC_SALE}'. Missing required base column: {SALES_COL}")
                return go.Figure()
            # If base exists but denom check failed:
            elif not ad_sale_check_passed:
                 st.warning(f"Cannot calculate '{AD_PERC_SALE}'. Denominator data source is missing or invalid.")
                 return go.Figure()
        else: # Other derived metrics
            missing_bases = base_needed_for_metric - set(filtered_data.columns)
            st.warning(f"Cannot calculate derived metric '{metric}'. Missing required base columns: {missing_bases}")
            return go.Figure()

    # If it's NOT a derived metric, it MUST exist in the input data
    if not is_derived_metric and not metric_exists_in_input:
        st.warning(f"Metric chart requires column '{metric}' in the data.")
        return go.Figure()

    # --- Start Plotting ---
    years = sorted(filtered_data[YEAR_COL].dropna().unique().astype(int))
    fig = go.Figure()

    # Define hover formats based on metric type (use constants)
    if metric in [CTR, CVR, ACOS, AD_PERC_SALE]: hover_suffix = "%"; hover_format = ".1f"
    elif metric in [SPEND_COL, SALES_COL, CPC]: hover_suffix = ""; hover_format = "$,.2f" # Added CPC here
    elif metric == ROAS: hover_suffix = ""; hover_format = ".2f"
    else: hover_suffix = ""; hover_format = ",.0f" # Impressions, Clicks, Orders, Units
    base_hover_template = f"Date: %{{customdata[1]|%Y-%m-%d}}<br>Week: %{{customdata[0]}}<br>{metric}: %{{y:{hover_format}}}{hover_suffix}<extra></extra>"

    processed_years = []
    colors = px.colors.qualitative.Plotly

    # ========================
    # YoY Plotting Logic
    # ========================
    if show_yoy and len(years) > 1:
        # Define columns needed for aggregation: base components + DATE_COL
        # Also include the original metric IF it's not derived and exists
        cols_to_agg_yoy_base = base_needed_for_metric | {DATE_COL}
        if not is_derived_metric and metric_exists_in_input:
            cols_to_agg_yoy_base.add(metric)

        actual_cols_to_agg_yoy = list(cols_to_agg_yoy_base & set(filtered_data.columns))

        if DATE_COL not in actual_cols_to_agg_yoy:
            st.warning(f"Missing '{DATE_COL}' for aggregation (YoY).")
            return go.Figure()

        try:
            agg_dict_yoy = {}
            numeric_aggregated = False
            for col in actual_cols_to_agg_yoy:
                if col == DATE_COL:
                    agg_dict_yoy[col] = 'min' # Get earliest date within the week for hover
                elif pd.api.types.is_numeric_dtype(filtered_data[col]):
                    agg_dict_yoy[col] = "sum"
                    numeric_aggregated = True

            # Check if any numeric column (base component or the metric itself if not derived) was found
            if not numeric_aggregated:
                 st.warning(f"No numeric column found to aggregate for metric '{metric}' for the YoY chart.")
                 return go.Figure()

            # Aggregate: Sum up base components (and original metric if not derived) by Year/Week
            grouped = filtered_data.groupby([YEAR_COL, WEEK_COL], as_index=False).agg(agg_dict_yoy)
            grouped[DATE_COL] = pd.to_datetime(grouped[DATE_COL])

        except Exception as e:
            st.warning(f"Could not group data by week for YoY chart: {e}")
            return go.Figure()

        # --- *** ALWAYS RECALCULATE DERIVED METRICS POST-AGGREGATION *** ---
        metric_calculated_successfully = False # Flag
        if is_derived_metric:
            if metric == CTR:
                if {CLICKS_COL, IMPRESSIONS_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[CLICKS_COL] / r[IMPRESSIONS_COL] * 100) if r.get(IMPRESSIONS_COL) else 0, axis=1).round(1)
                    metric_calculated_successfully = True
            elif metric == CVR:
                if {ORDERS_COL, CLICKS_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[ORDERS_COL] / r[CLICKS_COL] * 100) if r.get(CLICKS_COL) else 0, axis=1).round(1)
                    metric_calculated_successfully = True
            elif metric == ACOS:
                if {SPEND_COL, SALES_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[SPEND_COL] / r[SALES_COL] * 100) if r.get(SALES_COL) else np.nan, axis=1).round(1)
                    metric_calculated_successfully = True
            elif metric == ROAS:
                if {SALES_COL, SPEND_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[SALES_COL] / r[SPEND_COL]) if r.get(SPEND_COL) else np.nan, axis=1).round(2)
                    metric_calculated_successfully = True
            elif metric == CPC:
                 if {SPEND_COL, CLICKS_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[SPEND_COL] / r[CLICKS_COL]) if r.get(CLICKS_COL) else np.nan, axis=1).round(2)
                    metric_calculated_successfully = True
            elif metric == AD_PERC_SALE:
                if {SALES_COL}.issubset(grouped.columns) and ad_sale_check_passed: # Use flag
                    try:
                        temp_denom = weekly_total_sales_data.copy()
                        # Ensure data types match for merge
                        if YEAR_COL in grouped.columns and YEAR_COL in temp_denom.columns: temp_denom[YEAR_COL] = temp_denom[YEAR_COL].astype(grouped[YEAR_COL].dtype)
                        if WEEK_COL in grouped.columns and WEEK_COL in temp_denom.columns: temp_denom[WEEK_COL] = temp_denom[WEEK_COL].astype(grouped[WEEK_COL].dtype)
                        # Perform merge safely
                        grouped_merged = pd.merge(grouped, temp_denom[[YEAR_COL, WEEK_COL, 'Weekly_Total_Sales']], on=[YEAR_COL, WEEK_COL], how='left')
                        grouped_merged[metric] = grouped_merged.apply(lambda r: (r[SALES_COL] / r['Weekly_Total_Sales'] * 100) if pd.notna(r['Weekly_Total_Sales']) and r['Weekly_Total_Sales'] > 0 else np.nan, axis=1).round(1)
                        grouped = grouped_merged.drop(columns=['Weekly_Total_Sales'], errors='ignore') # Drop temp col
                        metric_calculated_successfully = True
                    except Exception as e:
                        st.warning(f"Failed to merge/calculate Ad % Sale for YoY chart: {e}")
                        grouped[metric] = np.nan # Ensure column exists even if calculation fails
                else:
                    # This handles case where 'Sales' exists but denom check failed
                    grouped[metric] = np.nan # Ensure column exists if calculation wasn't possible

            # Check flag after attempting all derived metric calculations
            if not metric_calculated_successfully:
                st.error(f"Internal Error: Failed to recalculate derived metric '{metric}' (YoY). Check base columns post-aggregation.")
                return go.Figure()
        else:
            # If metric wasn't derived, it should have been aggregated directly
            if metric not in grouped.columns:
                st.warning(f"Metric column '{metric}' not found after aggregation (YoY).")
                return go.Figure()
            metric_calculated_successfully = True # Treat direct aggregation as success

        # --- End Recalculation Block ---

        if metric_calculated_successfully: # Only attempt replace if metric should exist
            grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)
        else: # Should have returned earlier if calc failed, but as safety
            return go.Figure()

        # --- Plotting YoY data ---
        min_week_data, max_week_data = 53, 0
        for i, year in enumerate(years):
            year_data = grouped[grouped[YEAR_COL] == year].sort_values(WEEK_COL)
            if year_data.empty or year_data[metric].isnull().all(): continue

            processed_years.append(year)
            min_week_data = min(min_week_data, year_data[WEEK_COL].min())
            max_week_data = max(max_week_data, year_data[WEEK_COL].max())
            custom_data_hover = year_data[[WEEK_COL, DATE_COL]] # DATE_COL from 'min' aggregation

            fig.add_trace(
                go.Scatter(x=year_data[WEEK_COL], y=year_data[metric], mode="lines+markers", name=f"{year}",
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
    # Non-YoY Plotting Logic
    # ========================
    else:
        # Define columns needed for aggregation: base components + DATE_COL, YEAR_COL, WEEK_COL
        cols_to_agg_noyoy_base = base_needed_for_metric | {DATE_COL, YEAR_COL, WEEK_COL}
        if not is_derived_metric and metric_exists_in_input:
            cols_to_agg_noyoy_base.add(metric)

        actual_cols_to_agg_noyoy = list(cols_to_agg_noyoy_base & set(filtered_data.columns))

        if not {DATE_COL, YEAR_COL, WEEK_COL}.issubset(actual_cols_to_agg_noyoy):
            st.warning(f"Missing '{DATE_COL}', '{YEAR_COL}', or '{WEEK_COL}' for aggregation (non-YoY).")
            return go.Figure()

        try:
            agg_dict_noyoy = {}
            numeric_aggregated = False
            grouping_keys_noyoy = [DATE_COL, YEAR_COL, WEEK_COL] # Group by specific date point
            for col in actual_cols_to_agg_noyoy:
                if col not in grouping_keys_noyoy and pd.api.types.is_numeric_dtype(filtered_data[col]):
                    agg_dict_noyoy[col] = "sum"
                    numeric_aggregated = True

            if not numeric_aggregated:
                st.warning(f"No numeric column found to aggregate for metric '{metric}' for the time chart (non-YoY).")
                return go.Figure()

            # Aggregate: Sum up base components (and original metric if not derived) by Date/Year/Week
            grouped = filtered_data.groupby(grouping_keys_noyoy, as_index=False).agg(agg_dict_noyoy)
            grouped[DATE_COL] = pd.to_datetime(grouped[DATE_COL]) # Ensure datetime type

        except Exception as e:
            st.warning(f"Could not group data for time chart (non-YoY): {e}")
            return go.Figure()

        # --- *** ALWAYS RECALCULATE DERIVED METRICS POST-AGGREGATION *** ---
        metric_calculated_successfully = False # Flag
        if is_derived_metric:
            if metric == CTR:
                if {CLICKS_COL, IMPRESSIONS_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[CLICKS_COL] / r[IMPRESSIONS_COL] * 100) if r.get(IMPRESSIONS_COL) else 0, axis=1).round(1)
                    metric_calculated_successfully = True
            elif metric == CVR:
                 if {ORDERS_COL, CLICKS_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[ORDERS_COL] / r[CLICKS_COL] * 100) if r.get(CLICKS_COL) else 0, axis=1).round(1)
                    metric_calculated_successfully = True
            elif metric == ACOS:
                 if {SPEND_COL, SALES_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[SPEND_COL] / r[SALES_COL] * 100) if r.get(SALES_COL) else np.nan, axis=1).round(1)
                    metric_calculated_successfully = True
            elif metric == ROAS:
                 if {SALES_COL, SPEND_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[SALES_COL] / r[SPEND_COL]) if r.get(SPEND_COL) else np.nan, axis=1).round(2)
                    metric_calculated_successfully = True
            elif metric == CPC:
                  if {SPEND_COL, CLICKS_COL}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r[SPEND_COL] / r[CLICKS_COL]) if r.get(CLICKS_COL) else np.nan, axis=1).round(2)
                    metric_calculated_successfully = True
            elif metric == AD_PERC_SALE:
                if {SALES_COL}.issubset(grouped.columns) and ad_sale_check_passed: # Use flag
                    try:
                        temp_denom = weekly_total_sales_data.copy()
                        if YEAR_COL in grouped.columns and YEAR_COL in temp_denom.columns: temp_denom[YEAR_COL] = temp_denom[YEAR_COL].astype(grouped[YEAR_COL].dtype)
                        if WEEK_COL in grouped.columns and WEEK_COL in temp_denom.columns: temp_denom[WEEK_COL] = temp_denom[WEEK_COL].astype(grouped[WEEK_COL].dtype)
                        grouped_merged = pd.merge(grouped, temp_denom[[YEAR_COL, WEEK_COL, 'Weekly_Total_Sales']], on=[YEAR_COL, WEEK_COL], how='left')
                        grouped_merged[metric] = grouped_merged.apply(lambda r: (r[SALES_COL] / r['Weekly_Total_Sales'] * 100) if pd.notna(r['Weekly_Total_Sales']) and r['Weekly_Total_Sales'] > 0 else np.nan, axis=1).round(1)
                        grouped = grouped_merged.drop(columns=['Weekly_Total_Sales'], errors='ignore')
                        metric_calculated_successfully = True
                    except Exception as e:
                        st.warning(f"Failed to merge/calculate Ad % Sale for non-YoY chart: {e}")
                        grouped[metric] = np.nan
                else:
                    grouped[metric] = np.nan # Ensure column exists if calculation wasn't possible

            if not metric_calculated_successfully:
                st.error(f"Internal Error: Failed to recalculate derived metric '{metric}' (non-YoY). Check base columns post-aggregation.")
                return go.Figure()
        else:
            # If metric wasn't derived, it should exist
            if metric not in grouped.columns:
                st.warning(f"Metric column '{metric}' not found after aggregation (non-YoY).")
                return go.Figure()
            metric_calculated_successfully = True # Treat direct aggregation as success


        # Handle Inf/-Inf values
        if metric_calculated_successfully:
            grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)
        else: # Should have returned if calculation failed
            return go.Figure()

        # --- Plotting Non-YoY data ---
        if grouped[metric].isnull().all():
            st.info(f"No valid data points for metric '{metric}' over time (non-YoY).")
            return go.Figure() # Return empty figure if all values are NaN

        grouped = grouped.sort_values(DATE_COL)
        custom_data_hover_noyoy = grouped[[WEEK_COL, DATE_COL]]
        fig.add_trace(
            go.Scatter(x=grouped[DATE_COL], y=grouped[metric], mode="lines+markers", name=metric,
                       line=dict(color="#1f77b4", width=2), marker=dict(size=6),
                       customdata=custom_data_hover_noyoy, hovertemplate=base_hover_template)
        )
        fig.update_layout(xaxis_title="Date", showlegend=False)

    # --- Final Chart Layout ---
    portfolio_title = f" for {portfolio}" if portfolio != "All Portfolios" else " for All Portfolios"
    years_in_plot = processed_years if (show_yoy and len(years) > 1 and processed_years) else years # Get years actually plotted
    final_chart_title = f"{metric} "

    if show_yoy and len(years_in_plot) > 1:
        final_chart_title += f"Weekly Comparison {portfolio_title} ({product_type})"
        final_xaxis_title = "Week of Year"
    else:
        final_chart_title += f"Over Time (Weekly) {portfolio_title} ({product_type})"
        final_xaxis_title = "Week Ending Date" # More descriptive

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