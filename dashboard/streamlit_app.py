"""
Interactive Dashboard: E-Commerce Customer Retention & Delivery Performance
=============================================================================
A Streamlit dashboard built on top of the outputs produced by
ecommerce_analysis.py. This is the piece meant to be deployed live (Streamlit
Community Cloud is free) so a recruiter can click a link and interact with
your analysis in 30 seconds, instead of reading a static README.

Run locally with:
    streamlit run dashboard/app.py

Deploy for free at https://streamlit.io/cloud - connect your GitHub repo,
point it at this file, and you get a shareable public URL for your resume.
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Retention & Delivery Analysis",
    page_icon="📦",
    layout="wide",
)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


# ---------------------------------------------------------------------------
# DATA LOADING (cached so the dashboard stays fast on repeat interactions)
# ---------------------------------------------------------------------------
@st.cache_data
def load_rfm_data():
    path = os.path.join(OUTPUTS_DIR, "rfm_customer_segments.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


rfm = load_rfm_data()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("📦 E-Commerce Customer Retention & Delivery Performance")
st.markdown(
    """
    **Business question:** Is customer retention actually a problem, and does
    delivery performance drive it? This dashboard explores customer value
    segments, purchase behavior, and the measured link between delivery
    delays and customer satisfaction, built on the
    [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
    """
)

if rfm is None:
    st.warning(
        "No data found yet. Run `python ecommerce_analysis.py` first to "
        "generate `outputs/rfm_customer_segments.csv`, then reload this page."
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# TOP-LINE KPI CARDS
# ---------------------------------------------------------------------------
total_customers = len(rfm)
repeat_customers = (rfm["frequency"] > 1).sum()
repeat_rate = repeat_customers / total_customers
total_revenue = rfm["monetary"].sum()
at_risk_revenue = rfm.loc[
    rfm["segment"].isin(["At Risk", "Can't Lose Them"]), "monetary"
].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Repeat Purchase Rate", f"{repeat_rate:.1%}")
col3.metric("Total Revenue", f"R${total_revenue:,.0f}")
col4.metric("Revenue At Risk", f"R${at_risk_revenue:,.0f}",
            help="Revenue from customers in the 'At Risk' and 'Can't Lose Them' segments")

st.divider()

# ---------------------------------------------------------------------------
# RFM SEGMENT EXPLORER
# ---------------------------------------------------------------------------
st.subheader("Customer Segments (RFM Analysis)")

left, right = st.columns([1, 1])

with left:
    segment_counts = rfm["segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "customers"]
    fig_count = px.bar(
        segment_counts.sort_values("customers"),
        x="customers", y="segment", orientation="h",
        title="Customers per Segment",
        color="customers", color_continuous_scale="Blues",
    )
    fig_count.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_count, use_container_width=True)

with right:
    segment_revenue = rfm.groupby("segment")["monetary"].sum().reset_index()
    fig_revenue = px.bar(
        segment_revenue.sort_values("monetary"),
        x="monetary", y="segment", orientation="h",
        title="Revenue per Segment (R$)",
        color="monetary", color_continuous_scale="Greens",
    )
    fig_revenue.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_revenue, use_container_width=True)

# --- Interactive filter: let the viewer drill into a specific segment ---
st.markdown("#### Explore a Segment")
selected_segment = st.selectbox(
    "Choose a customer segment to inspect:",
    options=sorted(rfm["segment"].unique()),
)
segment_df = rfm[rfm["segment"] == selected_segment].copy()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Customers in Segment", f"{len(segment_df):,}")
col_b.metric("Avg. Recency (days)", f"{segment_df['recency'].mean():.0f}")
col_c.metric("Avg. Lifetime Spend", f"R${segment_df['monetary'].mean():.2f}")

display_cols: list[str] = ["customer_unique_id", "recency", "frequency", "monetary", "rfm_score"]
display_df = pd.DataFrame(segment_df[display_cols]).sort_values(by="monetary", ascending=False).head(50)
st.dataframe(display_df, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# STATIC CHARTS FROM THE PYTHON ANALYSIS (cohort retention, delivery impact)
# ---------------------------------------------------------------------------
st.subheader("Cohort Retention & Delivery Performance")
st.caption(
    "These come directly from ecommerce_analysis.py. Re-run that script "
    "any time the underlying data changes, and this dashboard picks up the "
    "latest charts automatically."
)

chart_col1, chart_col2 = st.columns(2)

cohort_chart_path = os.path.join(OUTPUTS_DIR, "cohort_retention_heatmap.png")
if os.path.exists(cohort_chart_path):
    chart_col1.image(cohort_chart_path, caption="Cohort Retention Rate by Month", use_container_width=True)

delivery_chart_path = os.path.join(OUTPUTS_DIR, "delivery_vs_review_score.png")
if os.path.exists(delivery_chart_path):
    chart_col2.image(delivery_chart_path, caption="Review Score by Delivery Status", use_container_width=True)

st.divider()
st.caption(
    "Built by Raj Kumar Sunar· Data: Olist Brazilian E-Commerce Public Dataset · "
    "[View the code on GitHub](https://github.com/rajkumar789/Olist-Customer-Retention-Delivery-Performance-Analysis)"
)
