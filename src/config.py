# src/config.py

# --- Column Names ---
DATE_COL = "WE Date"
PORTFOLIO_COL = "Portfolio Name" # Adjust if your sheet uses "Portfolio" sometimes
MARKETPLACE_COL = "Marketplace"
PRODUCT_COL = "Product"
CAMPAIGN_COL = "Campaign Name"
MATCH_TYPE_COL = "Match Type"
RTW_COL = "RTW/Prospecting" # Adjust if name varies

IMPRESSIONS_COL = "Impressions"
CLICKS_COL = "Clicks"
SPEND_COL = "Spend"
SALES_COL = "Sales"
ORDERS_COL = "Orders"
UNITS_COL = "Units"
TOTAL_SALES_COL = "Total Sales" # Used for Ad % Sale denominator

YEAR_COL = "Year"
WEEK_COL = "Week"

# --- Product Type Names (as they appear in the 'Product' column) ---
SP_PRODUCT = "Sponsored Products"
SB_PRODUCT = "Sponsored Brands"
SD_PRODUCT = "Sponsored Display"
PRODUCT_TYPES_FOR_TABS = [SP_PRODUCT, SB_PRODUCT, SD_PRODUCT]

# --- Metric Names & Definitions ---
# Keep metric names consistent with column names where they exist directly
CTR = "CTR"
CVR = "CVR"
CPC = "CPC"
ACOS = "ACOS"
ROAS = "ROAS"
AD_PERC_SALE = "Ad % Sale"

# Base metrics expected to be potentially present or summed
BASE_METRIC_COLS = [IMPRESSIONS_COL, CLICKS_COL, SPEND_COL, SALES_COL, ORDERS_COL, UNITS_COL]
# Derived metrics calculated from base components
DERIVED_METRICS = [CTR, CVR, CPC, ACOS, ROAS, AD_PERC_SALE]
# All metrics potentially displayed or used in calculations
ALL_POSSIBLE_METRICS = BASE_METRIC_COLS + DERIVED_METRICS

# Raw numeric columns to be cleaned and converted in preprocessing
# Add any other specific raw numeric columns from your sheet here
NUMERIC_COLS_TO_PREPROCESS = [
    IMPRESSIONS_COL, CLICKS_COL, SPEND_COL, SALES_COL, ORDERS_COL, UNITS_COL, TOTAL_SALES_COL, # Base data
    CTR, CVR, ACOS, ROAS, CPC # Include if these exist pre-calculated in raw data, otherwise they are derived later
    # Add "Orders %", "Spend %", "Sales %" if they are used from the raw data
]

# Categorical columns to ensure exist (filled with 'Unknown...' if missing)
CATEGORICAL_COLS_TO_ENSURE = [PRODUCT_COL, PORTFOLIO_COL, MARKETPLACE_COL, MATCH_TYPE_COL, RTW_COL, CAMPAIGN_COL]


# Components needed for derived metrics (use column name constants)
METRIC_COMPONENTS = {
    CTR: {CLICKS_COL, IMPRESSIONS_COL},
    CVR: {ORDERS_COL, CLICKS_COL},
    CPC: {SPEND_COL, CLICKS_COL},
    ACOS: {SPEND_COL, SALES_COL},
    ROAS: {SALES_COL, SPEND_COL},
    AD_PERC_SALE: {SALES_COL} # Denominator needs TOTAL_SALES_COL handled separately
}

# For YoY change calculation logic: Metrics where change is absolute difference (percentage points)
PERCENTAGE_POINT_CHANGE_METRICS = {CTR, CVR, ACOS, AD_PERC_SALE}

# --- Insight Thresholds ---
ACOS_TARGET = 15.0
ROAS_TARGET = 5.0
CTR_TARGET = 0.35
CVR_TARGET = 10.0

# --- Default Selections ---
DEFAULT_YOY_TAB_METRICS = [SPEND_COL, SALES_COL, AD_PERC_SALE, ACOS]
DEFAULT_PRODUCT_TAB_YOY_METRICS = [SPEND_COL, SALES_COL, AD_PERC_SALE, ACOS]
DEFAULT_PRODUCT_TAB_CHART_METRIC = SPEND_COL

# --- Asset Filenames ---
LOGO_FILENAME = "logo.png"

# --- Google Sheet Secrets Keys ---
GCP_SERVICE_ACCOUNT_KEY = "gcp_service_account"
GSHEET_CONFIG_KEY = "gsheet_config"
GSHEET_URL_KEY = "url_or_id"
GSHEET_NAME_KEY = "worksheet_name"