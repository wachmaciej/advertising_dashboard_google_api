# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import warnings
import re # Keep re needed for widget logic possibly
import plotly.express as px # Needed for charts in new tab

# Import functions and constants from your modules
# <<< Ensure needed constants like BASE_METRIC_COLS, DERIVED_METRICS, METRIC_COMPONENTS etc. are available >>>
from src.config import * # Import all constants
from src.data_loader import load_data_from_gsheet
from src.data_processing import preprocess_ad_data, filter_data_by_timeframe
from src.plotting import create_metric_over_time_chart, create_metric_comparison_chart
from src.tables import create_yoy_grouped_table, calculate_yoy_summary_row
from src.styling import style_yoy_comparison_table
from src.insights import generate_insights
# <<< Import for campaign_analysis REMOVED >>>


# Filter warnings for a clean output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="YOY Dashboard - Advertising Data (GSheet Input)",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# --- Title and Logo ---
col1_title, col2_title = st.columns([3, 1])
with col1_title:
    st.title("Advertising Dashboard 📊")
with col2_title:
    try:
        st.image(f"assets/{LOGO_FILENAME}", width=250)
    except FileNotFoundError:
        st.warning(f"Logo file '{LOGO_FILENAME}' not found in 'assets/' folder.")
    except Exception as e:
        st.warning(f"Could not load logo file '{LOGO_FILENAME}': {e}")


# =============================================================================
# --- Automatic Data Loading using Secrets ---
# =============================================================================
# (This section remains unchanged)
secrets_loaded = False
GSHEET_URL_OR_ID = None
WORKSHEET_NAME = None
try:
    gsheet_config = st.secrets[GSHEET_CONFIG_KEY]
    GSHEET_URL_OR_ID = gsheet_config[GSHEET_URL_KEY]
    WORKSHEET_NAME = gsheet_config[GSHEET_NAME_KEY]
    secrets_loaded = True
    st.sidebar.caption(f"Configured GSheet: '{WORKSHEET_NAME}'")
    st.sidebar.info("Data source configured via secrets. Ensure sheet is shared with the service account.")
except KeyError as e:
    st.error(f"Missing configuration in secrets.toml: {e}")
    if GSHEET_CONFIG_KEY in str(e): st.error(f"Ensure '[{GSHEET_CONFIG_KEY}]' section exists.")
    elif GSHEET_URL_KEY in str(e) or GSHEET_NAME_KEY in str(e): st.error(f"Ensure '{GSHEET_URL_KEY}' and '{GSHEET_NAME_KEY}' keys exist within '[{GSHEET_CONFIG_KEY}]'.")
    else: st.error(f"Check for section '[{GCP_SERVICE_ACCOUNT_KEY}]'.")
    st.stop()
except Exception as e:
    st.error(f"Error loading GSheet config from secrets: {e}")
    st.stop()

raw_data_available = False
if secrets_loaded:
    if "data_loaded_from_gsheet" not in st.session_state: st.session_state.data_loaded_from_gsheet = False
    if "processed_gsheet_url" not in st.session_state: st.session_state.processed_gsheet_url = None
    if "processed_gsheet_name" not in st.session_state: st.session_state.processed_gsheet_name = None
    if "current_gsheet_url" not in st.session_state: st.session_state.current_gsheet_url = None
    if "current_gsheet_name" not in st.session_state: st.session_state.current_gsheet_name = None

    needs_loading = False
    if not st.session_state.data_loaded_from_gsheet: needs_loading = True
    elif st.session_state.current_gsheet_url != GSHEET_URL_OR_ID or st.session_state.current_gsheet_name != WORKSHEET_NAME:
        needs_loading = True
        st.info("GSheet configuration in secrets may have changed. Reloading data...")
        st.session_state.data_loaded_from_gsheet = False

    if needs_loading:
        with st.spinner(f"Loading data from Google Sheet '{WORKSHEET_NAME}'..."):
            raw_data = load_data_from_gsheet(GSHEET_URL_OR_ID, WORKSHEET_NAME)
        if raw_data is not None and not raw_data.empty: # Check raw_data is not None
            st.session_state["ad_data_raw"] = raw_data
            st.session_state.current_gsheet_url = GSHEET_URL_OR_ID
            st.session_state.current_gsheet_name = WORKSHEET_NAME
            st.session_state.data_loaded_from_gsheet = True
            st.sidebar.success(f"Loaded {len(raw_data)} rows.")
            keys_to_delete_on_reload = ['ad_data_filtered', 'ad_data_processed', 'processed_marketplace', 'processed_gsheet_url', 'processed_gsheet_name', 'marketplace_selector_value', 'deep_dive_product_selector', 'deep_dive_date_range']
            for key in keys_to_delete_on_reload:
                if key in st.session_state: del st.session_state[key]
            # Consider adding st.rerun() here if state management requires it after reload
        else:
            st.session_state.data_loaded_from_gsheet = False
            if "ad_data_raw" in st.session_state: del st.session_state["ad_data_raw"]
            # Error message is displayed by load_data_from_gsheet in the sidebar

    raw_data_available = "ad_data_raw" in st.session_state and not st.session_state["ad_data_raw"].empty and st.session_state.data_loaded_from_gsheet

# =============================================================================
# --- Process Loaded Data (Marketplace Selection & Preprocessing Trigger) ---
# =============================================================================
# (This section remains unchanged)
selected_mp_widget = "All Marketplaces"
if raw_data_available:
    marketplace_options = ["All Marketplaces"]
    default_marketplace = "All Marketplaces"
    if MARKETPLACE_COL in st.session_state["ad_data_raw"].columns:
        raw_df_for_options = st.session_state["ad_data_raw"]
        if not raw_df_for_options.empty:
            available_marketplaces = sorted([str(mp) for mp in raw_df_for_options[MARKETPLACE_COL].dropna().unique() if str(mp).strip()])
            if available_marketplaces:
                marketplace_options.extend(available_marketplaces)
                target_default = "US"
                if target_default in marketplace_options: default_marketplace = target_default
    else:
        st.sidebar.warning(f"'{MARKETPLACE_COL}' column not found. Cannot filter by Marketplace.")

    if 'marketplace_selector_value' not in st.session_state: st.session_state.marketplace_selector_value = default_marketplace
    try:
        current_value = st.session_state.marketplace_selector_value
        if current_value not in marketplace_options:
            current_value = default_marketplace
            st.session_state.marketplace_selector_value = current_value
        current_index = marketplace_options.index(current_value) if current_value in marketplace_options else 0
    except ValueError:
        st.session_state.marketplace_selector_value = default_marketplace
        current_index = marketplace_options.index(default_marketplace) if default_marketplace in marketplace_options else 0

    selected_mp_widget = st.sidebar.selectbox("Select Marketplace", options=marketplace_options, index=current_index, key="marketplace_selector_value")

    needs_processing = False
    if "ad_data_processed" not in st.session_state: needs_processing = True
    elif st.session_state.get("processed_marketplace") != selected_mp_widget: needs_processing = True
    elif st.session_state.get("processed_gsheet_url") != st.session_state.get("current_gsheet_url") or st.session_state.get("processed_gsheet_name") != st.session_state.get("current_gsheet_name"): needs_processing = True

    if needs_processing:
        with st.spinner("Processing data..."):
            current_selection_for_processing = selected_mp_widget
            try:
                if "ad_data_raw" not in st.session_state or st.session_state["ad_data_raw"].empty:
                    st.error("Raw data not available or empty for processing.")
                else:
                    ad_data_to_filter = st.session_state["ad_data_raw"]
                    temp_filtered_data = pd.DataFrame()
                    if current_selection_for_processing != "All Marketplaces":
                        if MARKETPLACE_COL in ad_data_to_filter.columns:
                            temp_filtered_data = ad_data_to_filter[ad_data_to_filter[MARKETPLACE_COL].astype(str) == current_selection_for_processing].copy()
                        else:
                            st.warning(f"Cannot filter raw data by Marketplace: '{MARKETPLACE_COL}' column missing.")
                            temp_filtered_data = ad_data_to_filter.copy()
                    else:
                        temp_filtered_data = ad_data_to_filter.copy()

                    if not temp_filtered_data.empty:
                        st.session_state["ad_data_filtered"] = temp_filtered_data
                        processed_data = preprocess_ad_data(temp_filtered_data) # Get result first
                        if processed_data is None or processed_data.empty: # Check if None or empty
                            st.error("Preprocessing resulted in empty data. Please check data quality and formats for the selected marketplace.")
                            if "ad_data_processed" in st.session_state: del st.session_state["ad_data_processed"]
                        else: # Preprocessing was successful
                             st.session_state["ad_data_processed"] = processed_data
                             st.session_state.processed_marketplace = current_selection_for_processing
                             st.session_state.processed_gsheet_url = st.session_state.current_gsheet_url
                             st.session_state.processed_gsheet_name = st.session_state.current_gsheet_name
                             st.success(f"Data processed for '{current_selection_for_processing}'.")
                             # Clear downstream state on successful processing
                             if 'deep_dive_product_selector' in st.session_state: del st.session_state['deep_dive_product_selector']
                             if 'deep_dive_date_range' in st.session_state: del st.session_state['deep_dive_date_range']
                             st.rerun() # Rerun after successful processing

                    else:
                        st.warning(f"Data is empty after filtering for Marketplace: '{current_selection_for_processing}'. Cannot preprocess.")
                        if "ad_data_processed" in st.session_state: del st.session_state["ad_data_processed"]
                        # Still update processed markers to prevent reprocessing loop if filter yields empty
                        st.session_state.processed_marketplace = current_selection_for_processing
                        st.session_state.processed_gsheet_url = st.session_state.current_gsheet_url
                        st.session_state.processed_gsheet_name = st.session_state.current_gsheet_name


            except Exception as e:
                st.error(f"Error during data processing: {e}")
                st.exception(e) # Show full traceback for easier debugging
                keys_to_del = ['ad_data_filtered', 'ad_data_processed', 'processed_marketplace', 'processed_gsheet_url', 'processed_gsheet_name']
                for key in keys_to_del:
                    if key in st.session_state: del st.session_state[key]


# =============================================================================
# --- Display Dashboard Tabs ---
# =============================================================================
processed_data_valid_and_current = False
if "ad_data_processed" in st.session_state and not st.session_state["ad_data_processed"].empty:
    if 'GSHEET_URL_OR_ID' in locals() and 'WORKSHEET_NAME' in locals() and 'selected_mp_widget' in locals():
         if st.session_state.get("processed_gsheet_url") == st.session_state.get("current_gsheet_url") and \
            st.session_state.get("processed_gsheet_name") == st.session_state.get("current_gsheet_name") and \
            st.session_state.get("processed_marketplace") == selected_mp_widget:
             processed_data_valid_and_current = True

if processed_data_valid_and_current:

    ad_data_processed_current = st.session_state["ad_data_processed"]
    current_processed_columns = set(ad_data_processed_current.columns) # Get columns once

    # <<< --- REMOVED "Campaign Movers" from tab list --- >>>
    tab_names = ["YOY Comparison", "Charts"]
    tabs_adv = st.tabs(tab_names)

    # -------------------------------
    # Tab 0: YOY Comparison
    # -------------------------------
    with tabs_adv[0]:
        # (This section remains unchanged)
        st.markdown("### YOY Comparison")
        ad_data_overview = ad_data_processed_current.copy()

        st.markdown("#### Select Comparison Criteria")
        col1_yoy, col2_yoy, col3_yoy, col4_yoy = st.columns(4)
        selected_years_yoy = []
        selected_week_yoy = None
        selected_metrics_yoy = []
        with col1_yoy:
            if YEAR_COL not in ad_data_overview.columns:
                st.error(f"'{YEAR_COL}' column missing. Cannot create YOY filters.")
                available_years_yoy = []
            else:
                available_years_yoy = sorted(ad_data_overview[YEAR_COL].dropna().unique())
                default_years_yoy = available_years_yoy[-2:] if len(available_years_yoy) >= 2 else available_years_yoy
                selected_years_yoy = st.multiselect("Select Year(s):", available_years_yoy, default=default_years_yoy, key="yoy_years")
        with col2_yoy:
            timeframe_options_yoy = ["Specific Week", "Last 1 Week", "Last 4 Weeks", "Last 8 Weeks", "Last 12 Weeks"]
            default_tf_label = "Last 4 Weeks"
            default_tf_index_yoy = timeframe_options_yoy.index(default_tf_label) if default_tf_label in timeframe_options_yoy else 0
            selected_timeframe_yoy = st.selectbox("Select Timeframe:", timeframe_options_yoy, index=default_tf_index_yoy, key="yoy_timeframe")
        with col3_yoy:
            available_weeks_str_yoy = ["Select..."]
            is_specific_week_yoy = (selected_timeframe_yoy == "Specific Week")
            if selected_years_yoy and WEEK_COL in ad_data_overview.columns:
                try:
                    selected_years_yoy_int = [int(y) for y in selected_years_yoy]
                    weeks_in_selected_years = ad_data_overview[ad_data_overview[YEAR_COL].isin(selected_years_yoy_int)][WEEK_COL].unique()
                    available_weeks_yoy = sorted([int(w) for w in weeks_in_selected_years if pd.notna(w)])
                    available_weeks_str_yoy.extend([str(w) for w in available_weeks_yoy])
                except Exception as e: st.warning(f"Could not retrieve weeks: {e}")
            selected_week_option_yoy = st.selectbox("Select Week:", available_weeks_str_yoy, index=0, key="yoy_week", disabled=(not is_specific_week_yoy))
            selected_week_yoy = int(selected_week_option_yoy) if is_specific_week_yoy and selected_week_option_yoy != "Select..." else None
        with col4_yoy:
            available_display_metrics_yoy = []
            for m in ALL_POSSIBLE_METRICS:
                can_display = False
                if m in current_processed_columns: can_display = True
                elif m in METRIC_COMPONENTS:
                    if METRIC_COMPONENTS[m].issubset(current_processed_columns):
                        # Check if components are numeric
                        components_numeric = all(pd.api.types.is_numeric_dtype(ad_data_overview[comp]) for comp in METRIC_COMPONENTS[m] if comp in ad_data_overview)
                        if components_numeric:
                             if m == AD_PERC_SALE and TOTAL_SALES_COL not in current_processed_columns: can_display = False
                             else: can_display = True
                if can_display: available_display_metrics_yoy.append(m)
            default_metrics_yoy_filtered = [m for m in DEFAULT_YOY_TAB_METRICS if m in available_display_metrics_yoy]
            selected_metrics_yoy = st.multiselect("Select Metrics:", available_display_metrics_yoy, default=default_metrics_yoy_filtered, key="yoy_metrics")
            if not selected_metrics_yoy:
                selected_metrics_yoy = default_metrics_yoy_filtered[:1] if default_metrics_yoy_filtered else available_display_metrics_yoy[:1]
                if not selected_metrics_yoy: st.warning("No metrics available for selection.")

        if not selected_years_yoy: st.warning("Please select at least one year.")
        elif not selected_metrics_yoy: st.warning("Please select at least one metric.")
        else:
            filtered_data_yoy = filter_data_by_timeframe(ad_data_overview, selected_years_yoy, selected_timeframe_yoy, selected_week_yoy)
            if filtered_data_yoy.empty: st.info("No data available for the selected YOY criteria (Years/Timeframe/Week).")
            else:
                years_to_process_yoy = sorted(filtered_data_yoy[YEAR_COL].unique())
                if not years_to_process_yoy: st.info("Filtered data contains no valid years for comparison.")
                else:
                    st.markdown("---"); st.markdown("#### Overview by Product Type")
                    st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics.*")
                    if PRODUCT_COL in filtered_data_yoy.columns:
                        product_overview_yoy_table = create_yoy_grouped_table(df_filtered_period=filtered_data_yoy, group_by_col=PRODUCT_COL, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Product")
                        if not product_overview_yoy_table.empty:
                            styled_product_overview_yoy = style_yoy_comparison_table(product_overview_yoy_table)
                            if styled_product_overview_yoy: st.dataframe(styled_product_overview_yoy, use_container_width=True)
                        else: st.info("No product overview data available.")
                    else: st.warning(f"'{PRODUCT_COL}' column not found, cannot display Overview by Product Type.")

                    if PORTFOLIO_COL in filtered_data_yoy.columns:
                        st.markdown("---"); st.markdown("#### Portfolio Performance")
                        st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics. Optionally filter by Product Type below.*")
                        portfolio_table_data_yoy = filtered_data_yoy.copy()
                        selected_product_portfolio_yoy = "All"
                        if PRODUCT_COL in filtered_data_yoy.columns:
                            product_types_portfolio_yoy = ["All"] + sorted(filtered_data_yoy[PRODUCT_COL].dropna().unique().tolist())
                            selected_product_portfolio_yoy = st.selectbox("Filter Portfolio Table by Product Type:", product_types_portfolio_yoy, index=0, key="portfolio_product_filter_yoy")
                            if selected_product_portfolio_yoy != "All":
                                portfolio_table_data_yoy = portfolio_table_data_yoy[portfolio_table_data_yoy[PRODUCT_COL] == selected_product_portfolio_yoy]
                        if portfolio_table_data_yoy.empty:
                            st.info(f"No Portfolio data available for Product Type '{selected_product_portfolio_yoy}' in the selected period.")
                        else:
                            portfolio_yoy_table = create_yoy_grouped_table(df_filtered_period=portfolio_table_data_yoy, group_by_col=PORTFOLIO_COL, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Portfolio")
                            if not portfolio_yoy_table.empty:
                                styled_portfolio_yoy = style_yoy_comparison_table(portfolio_yoy_table)
                                if styled_portfolio_yoy: st.dataframe(styled_portfolio_yoy, use_container_width=True)
                                portfolio_summary_row_yoy = calculate_yoy_summary_row(df=portfolio_table_data_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="Portfolio", id_col_value=f"TOTAL - {selected_product_portfolio_yoy}")
                                if not portfolio_summary_row_yoy.empty:
                                    st.markdown("###### YoY Total (Selected Period & Product Filter)")
                                    styled_portfolio_summary_yoy = style_yoy_comparison_table(portfolio_summary_row_yoy)
                                    if styled_portfolio_summary_yoy: st.dataframe(styled_portfolio_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                            else: st.info(f"No displayable portfolio data for Product Type '{selected_product_portfolio_yoy}'.")
                    else: st.info(f"'{PORTFOLIO_COL}' column not found, cannot display Portfolio Performance table.")

                    if PRODUCT_COL in filtered_data_yoy.columns:
                        present_product_types_for_tabs = [pt for pt in PRODUCT_TYPES_FOR_TABS if pt in filtered_data_yoy[PRODUCT_COL].unique()]
                        if {MATCH_TYPE_COL}.issubset(filtered_data_yoy.columns) and present_product_types_for_tabs:
                            st.markdown("---"); st.markdown("#### Match Type Performance")
                            st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics, broken down by Product Type.*")
                            for product_type_m in present_product_types_for_tabs:
                                product_data_match_yoy = filtered_data_yoy[filtered_data_yoy[PRODUCT_COL] == product_type_m].copy()
                                if product_data_match_yoy.empty: continue
                                st.subheader(product_type_m)
                                match_type_yoy_table = create_yoy_grouped_table(df_filtered_period=product_data_match_yoy, group_by_col=MATCH_TYPE_COL, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Match Type")
                                if not match_type_yoy_table.empty:
                                    styled_match_type_yoy = style_yoy_comparison_table(match_type_yoy_table)
                                    if styled_match_type_yoy: st.dataframe(styled_match_type_yoy, use_container_width=True)
                                    match_type_summary_row_yoy = calculate_yoy_summary_row(df=product_data_match_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="Match Type", id_col_value=f"TOTAL - {product_type_m}")
                                    if not match_type_summary_row_yoy.empty:
                                        st.markdown("###### YoY Total (Selected Period)")
                                        styled_match_type_summary_yoy = style_yoy_comparison_table(match_type_summary_row_yoy)
                                        if styled_match_type_summary_yoy: st.dataframe(styled_match_type_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                                else: st.info(f"No Match Type data available for {product_type_m}.")
                        elif {MATCH_TYPE_COL}.issubset(filtered_data_yoy.columns) and not present_product_types_for_tabs: st.info("No relevant Product Types found for Match Type analysis in the selected period.")
                        elif not {MATCH_TYPE_COL}.issubset(filtered_data_yoy.columns): st.info(f"'{MATCH_TYPE_COL}' column not found, cannot display Match Type Performance table.")

                        if {RTW_COL}.issubset(filtered_data_yoy.columns) and present_product_types_for_tabs:
                            st.markdown("---"); st.markdown(f"#### {RTW_COL} Performance")
                            st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics. Choose a Product Type below.*")
                            available_rtw_products_yoy = sorted([pt for pt in filtered_data_yoy[PRODUCT_COL].unique() if pt in PRODUCT_TYPES_FOR_TABS])
                            if available_rtw_products_yoy:
                                current_rtw_value = st.session_state.get("rtw_product_selector_yoy", available_rtw_products_yoy[0])
                                if current_rtw_value not in available_rtw_products_yoy: current_rtw_value = available_rtw_products_yoy[0]
                                selected_rtw_product_yoy = st.selectbox(f"Select Product Type for {RTW_COL} Analysis:", available_rtw_products_yoy, index=available_rtw_products_yoy.index(current_rtw_value), key="rtw_product_selector_yoy")
                                rtw_filtered_product_data_yoy = filtered_data_yoy[filtered_data_yoy[PRODUCT_COL] == selected_rtw_product_yoy].copy()
                                if not rtw_filtered_product_data_yoy.empty:
                                    rtw_yoy_table = create_yoy_grouped_table(df_filtered_period=rtw_filtered_product_data_yoy, group_by_col=RTW_COL, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name=RTW_COL)
                                    if not rtw_yoy_table.empty:
                                        styled_rtw_yoy = style_yoy_comparison_table(rtw_yoy_table)
                                        if styled_rtw_yoy: st.dataframe(styled_rtw_yoy, use_container_width=True)
                                        rtw_summary_row_yoy = calculate_yoy_summary_row(df=rtw_filtered_product_data_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name=RTW_COL, id_col_value=f"TOTAL - {selected_rtw_product_yoy}")
                                        if not rtw_summary_row_yoy.empty:
                                            st.markdown("###### YoY Total (Selected Period)")
                                            styled_rtw_summary_yoy = style_yoy_comparison_table(rtw_summary_row_yoy)
                                            if styled_rtw_summary_yoy: st.dataframe(styled_rtw_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                                    else: st.info(f"No {RTW_COL} data available for {selected_rtw_product_yoy}.")
                                else: st.info(f"No {selected_rtw_product_yoy} data in selected period.")
                            else: st.info(f"No relevant Product Types found for {RTW_COL} analysis in the selected period.")
                        elif {RTW_COL}.issubset(filtered_data_yoy.columns) and not present_product_types_for_tabs: st.info(f"No relevant Product Types found for {RTW_COL} analysis in the selected period.")
                        elif not {RTW_COL}.issubset(filtered_data_yoy.columns): st.info(f"'{RTW_COL}' column not found, cannot display {RTW_COL} Performance table.")
                    else: st.info(f"'{PRODUCT_COL}' column not found. Cannot display Match Type or {RTW_COL} Performance tables.")

                    if CAMPAIGN_COL in filtered_data_yoy.columns:
                        st.markdown("---"); st.markdown(f"#### Campaign Performance")
                        st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics.*")
                        campaign_yoy_table = create_yoy_grouped_table(df_filtered_period=filtered_data_yoy, group_by_col=CAMPAIGN_COL, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Campaign")
                        if not campaign_yoy_table.empty:
                            styled_campaign_yoy = style_yoy_comparison_table(campaign_yoy_table)
                            if styled_campaign_yoy: st.dataframe(styled_campaign_yoy, use_container_width=True, height=600)
                            campaign_summary_row_yoy = calculate_yoy_summary_row(df=filtered_data_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="Campaign", id_col_value="TOTAL - All Campaigns")
                            if not campaign_summary_row_yoy.empty:
                                st.markdown("###### YoY Total (Selected Period)")
                                styled_campaign_summary_yoy = style_yoy_comparison_table(campaign_summary_row_yoy)
                                if styled_campaign_summary_yoy: st.dataframe(styled_campaign_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                        else: st.info(f"No displayable '{CAMPAIGN_COL}' data.")
                    else: st.info(f"'{CAMPAIGN_COL}' column not found, cannot display Campaign Performance table.")


    # -------------------------------
    # Tab 1: Charts
    # -------------------------------
    with tabs_adv[1]:
        # (This section remains unchanged)
        st.markdown("")
        available_product_types_raw = []
        if PRODUCT_COL in ad_data_processed_current.columns:
            available_product_types_raw = sorted([ str(pt) for pt in ad_data_processed_current[PRODUCT_COL].dropna().unique() if str(pt).strip() ])
        available_product_types = ["All Product Types"]
        if PRODUCT_COL in ad_data_processed_current.columns:
            available_product_types_filtered_config = [pt for pt in available_product_types_raw if pt in PRODUCT_TYPES_FOR_TABS]
            if available_product_types_filtered_config: available_product_types.extend(available_product_types_filtered_config)
        else: st.warning(f"'{PRODUCT_COL}' column not found. Cannot filter or analyze by Product Type.")
        no_specific_options = (len(available_product_types) == 1 and PRODUCT_COL in ad_data_processed_current.columns)
        if no_specific_options: st.warning(f"Only 'All Product Types' option available. No data found for configured types: {PRODUCT_TYPES_FOR_TABS}.")
        default_product_index = 0
        selected_product_deep_dive = st.selectbox( "Select Product Type:", options=available_product_types, index=default_product_index, key="deep_dive_product_selector")
        st.markdown(f"#### {selected_product_deep_dive} Performance")
        st.caption("Charts use filters below.") # <<< REMOVED "Tables show YoY..." text >>>
        ad_data_product_filtered = pd.DataFrame()
        if selected_product_deep_dive == "All Product Types":
            if PRODUCT_COL in ad_data_processed_current.columns: ad_data_product_filtered = ad_data_processed_current[ad_data_processed_current[PRODUCT_COL].isin(PRODUCT_TYPES_FOR_TABS)].copy()
            else: ad_data_product_filtered = ad_data_processed_current.copy()
        else:
            if PRODUCT_COL in ad_data_processed_current.columns: ad_data_product_filtered = ad_data_processed_current[ad_data_processed_current[PRODUCT_COL] == selected_product_deep_dive].copy()
            else: st.error(f"Cannot filter by '{PRODUCT_COL}' as it is missing."); ad_data_product_filtered = pd.DataFrame()
        if ad_data_product_filtered.empty: st.warning(f"No data available for '{selected_product_deep_dive}' within the selected Marketplace/processing criteria.")
        else:
            tab_key_prefix = f"deep_dive_{selected_product_deep_dive.replace(' ', '_')}"
            with st.expander("Filters", expanded=True):
                col1_tab, col2_tab = st.columns(2) # <<< CHANGED TO 2 COLUMNS >>>
                selected_metric_tab = None
                # selected_yoy_metrics_tab = [] # <<< REMOVED YOY Metrics Selector Variable >>>
                selected_portfolio_tab = "All Portfolios"
                can_calc_ad_sale_tab = {SALES_COL, TOTAL_SALES_COL, DATE_COL}.issubset(current_processed_columns); original_cols_overall = current_processed_columns; product_specific_cols_tab = set(ad_data_product_filtered.columns)
                available_metrics_tab = []
                for m in ALL_POSSIBLE_METRICS:
                    can_display_m = False
                    if m in product_specific_cols_tab and pd.api.types.is_numeric_dtype(ad_data_product_filtered[m]): can_display_m = True
                    elif m in METRIC_COMPONENTS:
                         if METRIC_COMPONENTS[m].issubset(product_specific_cols_tab):
                              components_numeric = all(pd.api.types.is_numeric_dtype(ad_data_product_filtered[comp]) for comp in METRIC_COMPONENTS[m] if comp in ad_data_product_filtered)
                              if components_numeric:
                                  if m == AD_PERC_SALE and not can_calc_ad_sale_tab: can_display_m = False
                                  else: can_display_m = True
                    if can_display_m: available_metrics_tab.append(m)
                available_metrics_tab = sorted(list(set(available_metrics_tab)))
                if not available_metrics_tab: st.warning(f"No metrics available for selection for '{selected_product_deep_dive}'."); st.stop()

                with col1_tab: # Metric Selector
                    default_metric_chart_tab = DEFAULT_PRODUCT_TAB_CHART_METRIC if DEFAULT_PRODUCT_TAB_CHART_METRIC in available_metrics_tab else available_metrics_tab[0]
                    sel_metric_index_tab = available_metrics_tab.index(default_metric_chart_tab) if default_metric_chart_tab in available_metrics_tab else 0
                    selected_metric_tab = st.selectbox("Select Metric for Charts", options=available_metrics_tab, index=sel_metric_index_tab, key=f"{tab_key_prefix}_metric")
                with col2_tab: # Portfolio Selector
                    if PORTFOLIO_COL not in ad_data_product_filtered.columns: st.info(f"Portfolio filtering not available ('{PORTFOLIO_COL}' missing)."); selected_portfolio_tab = "All Portfolios"
                    else:
                        ad_data_product_filtered[PORTFOLIO_COL] = ad_data_product_filtered[PORTFOLIO_COL].fillna("Unknown Portfolio"); portfolio_options_tab = ["All Portfolios"] + sorted(ad_data_product_filtered[PORTFOLIO_COL].unique().tolist())
                        current_portfolio_sel = st.session_state.get(f"{tab_key_prefix}_portfolio", "All Portfolios");
                        if current_portfolio_sel not in portfolio_options_tab: current_portfolio_sel = "All Portfolios"
                        portfolio_sel_index = portfolio_options_tab.index(current_portfolio_sel); selected_portfolio_tab = st.selectbox("Select Portfolio (for Charts)", options=portfolio_options_tab, index=portfolio_sel_index, key=f"{tab_key_prefix}_portfolio")

                # <<< REMOVED YOY METRICS MULTISELECT (col3_tab) >>>

                show_yoy_tab = st.checkbox("Show Year-over-Year Comparison (Chart - Weekly Points)", value=True, key=f"{tab_key_prefix}_show_yoy")
                date_range_tab = None; min_date_tab, max_date_tab = None, None; fixed_start_date = datetime.date(2024, 1, 5)
                if DATE_COL in ad_data_product_filtered.columns and pd.api.types.is_datetime64_any_dtype(ad_data_product_filtered[DATE_COL]) and not ad_data_product_filtered[DATE_COL].dropna().empty:
                    try:
                        valid_dates = ad_data_product_filtered[DATE_COL].dropna()
                        if not valid_dates.empty:
                            min_date_tab = valid_dates.min().date(); max_date_tab = valid_dates.max().date()
                            if min_date_tab <= max_date_tab:
                                default_start_value = max(min_date_tab, fixed_start_date); default_end_value = max_date_tab
                                if default_start_value > default_end_value: default_start_value = default_end_value
                                date_range_key = f"{tab_key_prefix}_date_range";
                                if date_range_key not in st.session_state: st.session_state[date_range_key] = (default_start_value, default_end_value)
                                current_range_val = st.session_state[date_range_key]; current_start = max(min_date_tab, current_range_val[0]); current_end = min(max_date_tab, current_range_val[1])
                                if current_start > current_end: current_start, current_end = default_start_value, default_end_value
                                widget_default_value = (current_start, current_end)
                                date_range_tab = st.date_input( "Select Date Range", value=widget_default_value, min_value=min_date_tab, max_value=max_date_tab, key=date_range_key )
                            else: st.warning(f"Invalid date range in data (min > max).")
                        else: st.warning(f"No valid dates found for date range selection.")
                    except Exception as e: st.warning(f"Error setting date range: {e}")
                else: st.warning(f"Cannot determine date range ('{DATE_COL}' missing/invalid).")

            ad_data_tab_date_filtered = ad_data_product_filtered.copy(); original_data_date_filtered_tab = ad_data_processed_current.copy(); start_date_tab, end_date_tab = None, None
            if date_range_tab and len(date_range_tab) == 2 and min_date_tab and max_date_tab:
                start_date_tab, end_date_tab = date_range_tab; ad_data_tab_date_filtered[DATE_COL] = pd.to_datetime(ad_data_tab_date_filtered[DATE_COL], errors='coerce'); original_data_date_filtered_tab[DATE_COL] = pd.to_datetime(original_data_date_filtered_tab[DATE_COL], errors='coerce')
                ad_data_tab_date_filtered = ad_data_tab_date_filtered[ (ad_data_tab_date_filtered[DATE_COL].dt.date >= start_date_tab) & (ad_data_tab_date_filtered[DATE_COL].dt.date <= end_date_tab) ].copy()
                original_data_date_filtered_tab = original_data_date_filtered_tab[ (original_data_date_filtered_tab[DATE_COL].dt.date >= start_date_tab) & (original_data_date_filtered_tab[DATE_COL].dt.date <= end_date_tab) ].copy()
            elif DATE_COL in ad_data_tab_date_filtered.columns: st.warning("Date range could not be applied. Showing all available data for selected product.")

            weekly_denominator_df_tab = pd.DataFrame()
            if selected_metric_tab == AD_PERC_SALE and TOTAL_SALES_COL in original_data_date_filtered_tab.columns and DATE_COL in original_data_date_filtered_tab.columns:
                if not original_data_date_filtered_tab.empty:
                    try:
                        temp_denom_df = original_data_date_filtered_tab.copy(); temp_denom_df[DATE_COL] = pd.to_datetime(temp_denom_df[DATE_COL], errors='coerce'); temp_denom_df[TOTAL_SALES_COL] = pd.to_numeric(temp_denom_df[TOTAL_SALES_COL], errors='coerce')
                        if YEAR_COL not in temp_denom_df.columns: temp_denom_df[YEAR_COL] = temp_denom_df[DATE_COL].dt.year
                        if WEEK_COL not in temp_denom_df.columns: temp_denom_df[WEEK_COL] = temp_denom_df[DATE_COL].dt.isocalendar().week
                        temp_denom_df[YEAR_COL] = pd.to_numeric(temp_denom_df[YEAR_COL], errors='coerce'); temp_denom_df[WEEK_COL] = pd.to_numeric(temp_denom_df[WEEK_COL], errors='coerce'); temp_denom_df.dropna(subset=[DATE_COL, TOTAL_SALES_COL, YEAR_COL, WEEK_COL], inplace=True)
                        if not temp_denom_df.empty:
                            temp_denom_df[YEAR_COL] = temp_denom_df[YEAR_COL].astype(int); temp_denom_df[WEEK_COL] = temp_denom_df[WEEK_COL].astype(int)
                            unique_subset_denom = [YEAR_COL, WEEK_COL];
                            if MARKETPLACE_COL in temp_denom_df.columns: unique_subset_denom.append(MARKETPLACE_COL)
                            unique_totals = temp_denom_df.drop_duplicates(subset=unique_subset_denom)
                            weekly_denominator_df_tab = unique_totals.groupby([YEAR_COL, WEEK_COL], as_index=False)[TOTAL_SALES_COL].sum(); weekly_denominator_df_tab = weekly_denominator_df_tab.rename(columns={TOTAL_SALES_COL: 'Weekly_Total_Sales'})
                    except Exception as e: st.warning(f"Could not calculate weekly total sales denominator for chart: {e}")

            if ad_data_tab_date_filtered.empty: st.info(f"No '{selected_product_deep_dive}' data available for the selected date range and filters.")
            elif selected_metric_tab is None: st.warning("Please select a metric to visualize the charts.")
            else:
                st.subheader(f"{selected_metric_tab} Over Time")
                fig1_tab = create_metric_over_time_chart(data=ad_data_tab_date_filtered, metric=selected_metric_tab, portfolio=selected_portfolio_tab, product_type=selected_product_deep_dive, show_yoy=show_yoy_tab, weekly_total_sales_data=weekly_denominator_df_tab)
                st.plotly_chart(fig1_tab, use_container_width=True, key=f"{tab_key_prefix}_time_chart")
                if selected_portfolio_tab == "All Portfolios" and PORTFOLIO_COL in ad_data_tab_date_filtered.columns:
                    if selected_metric_tab != AD_PERC_SALE:
                        st.subheader(f"{selected_metric_tab} by Portfolio"); fig2_tab = create_metric_comparison_chart(ad_data_tab_date_filtered, selected_metric_tab, None, selected_product_deep_dive)
                        st.plotly_chart(fig2_tab, use_container_width=True, key=f"{tab_key_prefix}_portfolio_chart")
                elif PORTFOLIO_COL not in ad_data_tab_date_filtered.columns: st.info(f"'{PORTFOLIO_COL}' column not found, cannot display Portfolio Comparison chart.")

            # <<< --- ENTIRE BLOCK FOR YOY TABLES IN CHARTS TAB DELETED --- >>>

            # --- Insights Section ---
            st.markdown("---")
            st.subheader("Key Insights (Latest Year in Selected Period)")
            if 'ad_data_tab_date_filtered' in locals() and not ad_data_tab_date_filtered.empty and start_date_tab and end_date_tab:
                 years_in_filtered_tab_data = sorted(ad_data_tab_date_filtered[YEAR_COL].dropna().unique())
                 if years_in_filtered_tab_data:
                     latest_year_tab = years_in_filtered_tab_data[-1]; st.caption(f"Insights based on aggregated data for {latest_year_tab} ({start_date_tab.strftime('%Y-%m-%d')} to {end_date_tab.strftime('%Y-%m-%d')})")
                     data_latest_year = ad_data_tab_date_filtered[ad_data_tab_date_filtered[YEAR_COL] == latest_year_tab].copy()
                     if not data_latest_year.empty:
                         insight_base_cols = [SPEND_COL, SALES_COL, CLICKS_COL, IMPRESSIONS_COL, ORDERS_COL]; numeric_insight_data = data_latest_year.copy()
                         for col in insight_base_cols:
                              if col in numeric_insight_data.columns: numeric_insight_data[col] = pd.to_numeric(numeric_insight_data[col], errors='coerce').fillna(0)
                              else: numeric_insight_data[col] = 0
                         total_spend = numeric_insight_data[SPEND_COL].sum(); total_sales = numeric_insight_data[SALES_COL].sum(); total_clicks = numeric_insight_data[CLICKS_COL].sum(); total_impressions = numeric_insight_data[IMPRESSIONS_COL].sum(); total_orders = numeric_insight_data[ORDERS_COL].sum()
                         insight_acos = (total_spend / total_sales * 100) if total_sales > 0 else np.nan; insight_roas = (total_sales / total_spend) if total_spend > 0 else np.nan
                         insight_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0; insight_cvr = (total_orders / total_clicks * 100) if total_clicks > 0 else 0
                         insight_acos = np.nan if insight_acos in [np.inf, -np.inf] else insight_acos; insight_roas = np.nan if insight_roas in [np.inf, -np.inf] else insight_roas
                         summary_series_for_insights = pd.Series({ ACOS: insight_acos, ROAS: insight_roas, CTR: insight_ctr, CVR: insight_cvr, SALES_COL: total_sales, SPEND_COL: total_spend, IMPRESSIONS_COL: total_impressions, CLICKS_COL: total_clicks, ORDERS_COL: total_orders })
                         insights_tab = generate_insights(summary_series_for_insights, selected_product_deep_dive)
                         if insights_tab:
                             for insight in insights_tab: st.markdown(f"- {insight}")
                         else: st.info("No insights generated for the selected period.")
                     else: st.info(f"No data for latest year ({latest_year_tab}) in selected date range to generate insights.")
                 else: st.info(f"No valid years in filtered data ({start_date_tab.strftime('%Y-%m-%d')} to {end_date_tab.strftime('%Y-%m-%d')}) to generate insights.")
            else: st.info(f"No summary data available (check date range and filters) to generate insights.")


# =============================================================================
# Final Fallback Messages
# =============================================================================
# (This section remains unchanged)
elif not secrets_loaded:
    st.error("App configuration failed. Could not load GSheet details from secrets.toml.")
elif not raw_data_available and secrets_loaded:
    st.warning(f"Failed to load data from the Google Sheet specified in secrets.")
    st.info("Troubleshooting Tips:")
    st.info(f"- Verify the `{GSHEET_URL_KEY}` and `{GSHEET_NAME_KEY}` in `.streamlit/secrets.toml` are correct.")
    st.info(f"- Ensure the service account email (`client_email` in secrets) has at least 'Viewer' access to the Sheet.")
    st.info(f"- Check the GSheet exists and the worksheet name is exact (case-sensitive).")
    st.info(f"- Check app logs (if deployed) for specific errors.")
elif raw_data_available and not processed_data_valid_and_current:
     if "ad_data_processed" not in st.session_state or st.session_state["ad_data_processed"].empty:
         st.warning(f"Data loaded, but processing failed or resulted in empty data for '{selected_mp_widget}'. Check source data quality.")
     else: # Data is processed but for different criteria
         st.info("Data needs reprocessing due to changes in GSheet source or Marketplace selection. Please wait or check for errors.")
elif not processed_data_valid_and_current and secrets_loaded:
     st.warning("Waiting for data processing or check for errors.")

# --- End of Script ---
