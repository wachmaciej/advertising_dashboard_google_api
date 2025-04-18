# src/insights.py
import pandas as pd
import numpy as np
# Import constants for thresholds and metric names used in logic
from src.config import (ACOS, ROAS, CTR, CVR, SALES_COL, SPEND_COL,
                        ACOS_TARGET, ROAS_TARGET, CTR_TARGET, CVR_TARGET)

# Note: Caching generate_insights might be less useful if the input series changes frequently
# Consider removing cache if performance is not an issue here.
# @st.cache_data # Removed cache - input is likely unique per run section
def generate_insights(total_metrics_series, campaign_type):
    """Generates text insights based on a summary row (Pandas Series)."""
    # Use constants for targets and metric names
    insights = []
    acos = total_metrics_series.get(ACOS, np.nan)
    roas = total_metrics_series.get(ROAS, np.nan)
    ctr = total_metrics_series.get(CTR, np.nan)
    cvr = total_metrics_series.get(CVR, np.nan)
    sales = total_metrics_series.get(SALES_COL, 0)
    spend = total_metrics_series.get(SPEND_COL, 0)

    if pd.isna(sales): sales = 0
    if pd.isna(spend): spend = 0

    # --- Insight Logic using defined thresholds ---
    if spend > 0 and sales == 0:
        insights.append("⚠️ **Immediate Attention:** Spend occurred with zero attributed sales. Review targeting, keywords, and product pages urgently.")
        if pd.notna(ctr): insights.append(f"ℹ️ Click-through rate was {ctr:.2f}%.")
    else:
        # ACOS Insight
        if pd.isna(acos):
            if spend == 0 and sales == 0: insights.append("ℹ️ No spend or sales recorded for ACOS calculation.")
            elif sales == 0 and spend > 0: insights.append("ℹ️ ACOS is not applicable (No Sales from Spend).")
            elif spend == 0 and sales > 0: insights.append(f"✅ **ACOS:** ACOS is effectively 0% (Sales with no spend), which is below the target (≤{ACOS_TARGET}%).")
            elif spend == 0: insights.append("ℹ️ ACOS is not applicable (No Spend).") # Should this be infinite? Original said N/A.
        elif acos > ACOS_TARGET: # Compare with ACOS_TARGET
            insights.append(f"📈 **High ACOS:** Overall ACOS ({acos:.1f}%) is above the target ({ACOS_TARGET}%). Consider optimizing bids, keywords, or targeting.")
        else: # ACOS is <= ACOS_TARGET
            insights.append(f"✅ **ACOS:** Overall ACOS ({acos:.1f}%) is within the acceptable range (≤{ACOS_TARGET}%).")

        # ROAS Insight
        if pd.isna(roas):
            if spend == 0 and sales == 0: insights.append("ℹ️ No spend or sales recorded for ROAS calculation.")
            elif spend == 0 and sales > 0 : insights.append("✅ **ROAS:** ROAS is effectively infinite (Sales with No Spend).")
            elif spend > 0 and sales == 0: insights.append("ℹ️ ROAS is 0 (No Sales from Spend).")
        elif roas < ROAS_TARGET: # Compare with ROAS_TARGET
            insights.append(f"📉 **Low ROAS:** Overall ROAS ({roas:.2f}) is below the target of {ROAS_TARGET}. Review performance and strategy.")
        else: # ROAS is >= ROAS_TARGET
            insights.append(f"✅ **ROAS:** Overall ROAS ({roas:.2f}) is good (≥{ROAS_TARGET}).")

        # CTR Insight
        if pd.isna(ctr):
            insights.append("ℹ️ Click-Through Rate (CTR) could not be calculated (likely no impressions).")
        elif ctr < CTR_TARGET: # Compare with CTR_TARGET
            insights.append(f"📉 **Low CTR:** Click-through rate ({ctr:.2f}%) is low (<{CTR_TARGET}%). Review ad creative, relevance, or placement.")
        else: # CTR is >= CTR_TARGET
            insights.append(f"✅ **CTR:** Click-through rate ({ctr:.2f}%) is satisfactory (≥{CTR_TARGET}%).")

        # CVR Insight
        if pd.isna(cvr):
            insights.append("ℹ️ Conversion Rate (CVR) could not be calculated (likely no clicks).")
        elif cvr < CVR_TARGET: # Compare with CVR_TARGET
            insights.append(f"📉 **Low CVR:** Conversion rate ({cvr:.1f}%) is below the target ({CVR_TARGET}%). Review product listing pages and targeting.")
        else: # CVR is >= CVR_TARGET
            insights.append(f"✅ **CVR:** Conversion rate ({cvr:.1f}%) is good (≥{CVR_TARGET}%).")

    return insights