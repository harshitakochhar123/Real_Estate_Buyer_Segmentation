import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    text-align: center;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

.card-title {
    font-size: 16px;
    color: #666666;
}

.card-value {
    font-size: 30px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "final_customer_segmentation.csv"

try:
    df = pd.read_csv(DATA_FILE)

except Exception as e:
    st.error("Unable to load the customer segmentation CSV.")
    st.write("Make sure `final_customer_segmentation.csv` is in the same GitHub repository as `app.py`.")
    st.exception(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🏠 Real Estate Buyer Segmentation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Customer Segmentation & Investment Profiling</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

st.sidebar.write(
    "Use the filters below to explore the customer segments."
)


# ============================================================
# BASIC DATA CLEANING
# ============================================================

df.columns = df.columns.str.strip()

# Detect segment column
segment_column = None

for col in ["segment", "Segment", "customer_segment"]:
    if col in df.columns:
        segment_column = col
        break

# Detect cluster column
cluster_column = None

for col in ["behavior_cluster", "Behavior_Cluster", "cluster"]:
    if col in df.columns:
        cluster_column = col
        break


# ============================================================
# SIDEBAR FILTER
# ============================================================

if segment_column:

    segments = sorted(
        df[segment_column].dropna().astype(str).unique().tolist()
    )

    selected_segments = st.sidebar.multiselect(
        "Select Customer Segment",
        segments,
        default=segments
    )

    filtered_df = df[
        df[segment_column].astype(str).isin(selected_segments)
    ].copy()

else:

    filtered_df = df.copy()


# ============================================================
# KPI CALCULATIONS
# ============================================================

if "customers" in filtered_df.columns:

    total_customers = int(
        pd.to_numeric(
            filtered_df["customers"],
            errors="coerce"
        ).fillna(0).sum()
    )

else:

    total_customers = len(filtered_df)


if cluster_column:

    number_segments = filtered_df[cluster_column].nunique()

elif segment_column:

    number_segments = filtered_df[segment_column].nunique()

else:

    number_segments = 0


if "average_total_value" in filtered_df.columns:

    avg_total_value = pd.to_numeric(
        filtered_df["average_total_value"],
        errors="coerce"
    ).mean()

else:

    avg_total_value = 0


if "average_satisfaction" in filtered_df.columns:

    avg_satisfaction = pd.to_numeric(
        filtered_df["average_satisfaction"],
        errors="coerce"
    ).mean()

else:

    avg_satisfaction = 0


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "👥 Total Customers",
        f"{total_customers:,}"
    )

with col2:

    st.metric(
        "🎯 Customer Segments",
        number_segments
    )

with col3:

    st.metric(
        "💰 Avg Total Property Value",
        f"{avg_total_value:,.0f}"
    )

with col4:

    st.metric(
        "⭐ Avg Satisfaction",
        f"{avg_satisfaction:.2f}"
    )


st.divider()


# ============================================================
# CUSTOMER SEGMENT TABLE
# ============================================================

st.subheader("📊 Customer Segment Overview")

display_columns = [
    "behavior_cluster",
    "segment",
    "customers",
    "customer_percentage",
    "average_age",
    "average_satisfaction",
    "average_property_count",
    "average_total_value",
    "average_property_value",
    "average_max_property_value"
]

available_columns = [
    col for col in display_columns
    if col in filtered_df.columns
]

if available_columns:

    st.dataframe(
        filtered_df[available_columns],
        use_container_width=True,
        hide_index=True
    )

else:

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CHART 1 — CUSTOMER DISTRIBUTION
# ============================================================

st.subheader("👥 Customer Distribution by Segment")

if segment_column and "customers" in filtered_df.columns:

    chart_data = filtered_df[
        [segment_column, "customers"]
    ].copy()

    chart_data["customers"] = pd.to_numeric(
        chart_data["customers"],
        errors="coerce"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        chart_data[segment_column].astype(str),
        chart_data["customers"]
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Distribution by Behavioral Segment")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

else:

    st.info(
        "Customer segment information is not available in the CSV."
    )


# ============================================================
# CHART 2 — PROPERTY VALUE
# ============================================================

st.subheader("💰 Average Property Value by Segment")

if segment_column and "average_total_value" in filtered_df.columns:

    value_data = filtered_df[
        [segment_column, "average_total_value"]
    ].copy()

    value_data["average_total_value"] = pd.to_numeric(
        value_data["average_total_value"],
        errors="coerce"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        value_data[segment_column].astype(str),
        value_data["average_total_value"]
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Average Total Property Value")
    ax.set_title("Average Property Value by Customer Segment")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

else:

    st.info(
        "Property value information is not available."
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.subheader("💡 Key Business Insights")

if segment_column and not filtered_df.empty:

    largest_segment = filtered_df.loc[
        filtered_df["customers"].idxmax(),
        segment_column
    ]

    highest_value_segment = filtered_df.loc[
        filtered_df["average_total_value"].idxmax(),
        segment_column
    ]

    highest_property_segment = filtered_df.loc[
        filtered_df["average_property_count"].idxmax(),
        segment_column
    ]

    st.markdown(
        f"""
        **Largest Customer Segment:** {largest_segment}

        **Highest Average Property Value:** {highest_value_segment}

        **Highest Property Ownership:** {highest_property_segment}
        """
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.subheader("🎯 Recommended Business Strategies")

recommendations = {

    "Core Mainstream Buyers":
        "Cross-sell premium properties, home loans and additional investment opportunities.",

    "Younger Moderate-Value Buyers":
        "Use digital campaigns, affordable properties and entry-level investment products.",

    "Value-Oriented Buyers":
        "Promote value-for-money properties, competitive pricing and financing options.",

    "High-Value Portfolio Investors":
        "Provide personalized relationship management, premium properties and portfolio expansion opportunities."
}

for segment_name, strategy in recommendations.items():

    st.markdown(
        f"**{segment_name}:** {strategy}"
    )


# ============================================================
# RAW DATA
# ============================================================

with st.expander("🔎 View Full Segmentation Dataset"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Real Estate Buyer Segmentation | K-Means Machine Learning Project"
)
