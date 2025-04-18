# src/data_loader.py
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
# Use constants for secrets keys for maintainability
from src.config import (GCP_SERVICE_ACCOUNT_KEY, GSHEET_CONFIG_KEY,
                       GSHEET_URL_KEY, GSHEET_NAME_KEY)

# Cache data for 10 minutes
@st.cache_data(ttl=600)
def load_data_from_gsheet(sheet_url, worksheet_name):
    """Loads data from a Google Sheet using service account credentials stored in Streamlit secrets."""
    # (Logic copied directly from original, uses standard gspread exceptions)
    # Removed original check/warning here as URL/Name now come from secrets/constants.

    try:
        creds_dict = st.secrets[GCP_SERVICE_ACCOUNT_KEY]
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
        # Do not print success message here, handle in main app logic
        return df

    except gspread.exceptions.SpreadsheetNotFound:
        st.sidebar.error(f"Error: Google Sheet not found using URL/ID from secrets.")
        st.sidebar.info(f"Check '{GSHEET_URL_KEY}' in secrets and sheet sharing permissions.")
        return pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        st.sidebar.error(f"Error: Worksheet '{worksheet_name}' (from secrets) not found. Check name (it's case-sensitive).")
        return pd.DataFrame()
    except KeyError as e:
        if GCP_SERVICE_ACCOUNT_KEY in str(e):
             st.sidebar.error(f"Error: Missing `[{GCP_SERVICE_ACCOUNT_KEY}]` secrets. Please configure Streamlit secrets.")
        elif GSHEET_CONFIG_KEY in str(e): # Check for gsheet_config section
             st.sidebar.error(f"Error: Missing `[{GSHEET_CONFIG_KEY}]` section in secrets.toml. Cannot load sheet details.")
        elif GSHEET_URL_KEY in str(e) or GSHEET_NAME_KEY in str(e):
             st.sidebar.error(f"Error: Missing '{GSHEET_URL_KEY}' or '{GSHEET_NAME_KEY}' within `[{GSHEET_CONFIG_KEY}]` in secrets.toml.")
        else:
             st.sidebar.error(f"A configuration error occurred accessing secrets: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.sidebar.error(f"An error occurred accessing Google Sheet: {e}")
        st.sidebar.info("Tips: Ensure service account email has 'Viewer' access. Verify URL/Name in secrets.")
        return pd.DataFrame()