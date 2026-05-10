import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(
    page_title="Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label { color: #8b949e; font-size: 13px; margin-bottom: 4px; }
.metric-value { color: #e6edf3; font-size: 28px; font-weight: 600; }
.metric-sub { color: #58a6ff; font-size: 12px; margin-top: 4px; }
.section-header {
    color: #e6edf3;
    font-size: 18px;
    font-weight: 600;
    margin: 1.5rem 0 1rem;
    padding-bottom: 8px;
    border-bottom: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "pig_output")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    font_color="#e6edf3",
)
PLOTLY_TEMPLATE = "plotly_dark"

@st.cache_data
def load_total():
    path = os.path.join(OUTPUT, "total_customers", "part-r-00000")
    df = pd.read_csv(path, header=None, names=["metric", "value"])
    return int(df["value"].iloc[0])

@st.cache_data
def load_date_range():
    path = os.path.join(OUTPUT, "date_range", "part-r-00000")
    df = pd.read_csv(path, header=None, names=["earliest", "latest"])
    return df.iloc[0]["earliest"], df.iloc[0]["latest"]

@st.cache_data
def load_top5():
    path = os.path.join(OUTPUT, "top5_countries", "part-r-00000")
    df = pd.read_csv(path, header=None, names=["country", "count"])
    return df.sort_values("count", ascending=False)

@st.cache_data
def load_country_counts():
    path = os.path.join(OUTPUT, "country_counts", "part-r-00000")
    df = pd.read_csv(path, header=None, names=["country", "count"])
    return df.sort_values("count", ascending=False)

@st.cache_data
def load_company_counts():
    path = os.path.join(OUTPUT, "company_counts", "part-r-00000")
    df = pd.read_csv(path, header=None, names=["company", "count"])
    return df.sort_values("count", ascending=False)

@st.cache_data
def load_egypt():
    path = os.path.join(OUTPUT, "egypt_customers", "part-r-00000")
    df = pd.read_csv(path, header=None,
                     names=["customer_id", "first_name", "last_name",
                             "city", "email", "subscription_date"])
    return df

with st.sidebar:
    st.markdown("## 📊 Customer Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🌍 Geography", "🏢 Companies", "🇪🇬 Egypt Customers"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<div style='color:#8b949e;font-size:12px'>Data processed via Apache Pig on Hadoop<br>1M customer records</div>",
                unsafe_allow_html=True)

if page == "🏠 Overview":
    st.markdown("# Customer Analytics Dashboard")
    st.markdown("Insights from 1,000,000 customer records processed with Apache Pig on Hadoop.")

    total     = load_total()
    earliest, latest = load_date_range()
    top5      = load_top5()
    countries = load_country_counts()
    egypt     = load_egypt()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-sub">deduplicated</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Countries</div>
            <div class="metric-value">{len(countries):,}</div>
            <div class="metric-sub">unique</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Earliest Subscription</div>
            <div class="metric-value" style="font-size:20px">{earliest}</div>
            <div class="metric-sub">first record</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Egypt Customers</div>
            <div class="metric-value">{len(egypt):,}</div>
            <div class="metric-sub">filtered subset</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Top 5 Countries by Customer Count</div>',
                unsafe_allow_html=True)
    fig = px.bar(
        top5, x="country", y="count",
        color="count", color_continuous_scale="Blues",
        labels={"count": "Customers", "country": "Country"},
        template="plotly_dark"
    )
    fig.update_layout(coloraxis_showscale=False, height=380,
                      margin=dict(l=0, r=0, t=10, b=0), **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Customer Distribution — Top 20 Countries</div>',
                unsafe_allow_html=True)
    top20 = countries.head(20)
    fig2 = px.pie(
        top20, names="country", values="count",
        hole=0.45, color_discrete_sequence=px.colors.sequential.Blues_r,
        template="plotly_dark"
    )
    fig2.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                       legend=dict(font=dict(size=11)), **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

elif page == "🌍 Geography":
    st.markdown("# Geographic Analysis")
    countries = load_country_counts()

    st.markdown('<div class="section-header">World Map — Customer Density</div>',
                unsafe_allow_html=True)
    fig = px.choropleth(
        countries, locations="country", locationmode="country names",
        color="count", color_continuous_scale="Blues",
        labels={"count": "Customers"},
        template="plotly_dark"
    )
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0),
                      geo=dict(bgcolor="#161b22", lakecolor="#0e1117",
                               landcolor="#1c2128", showframe=False), **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Top 15 Countries</div>',
                    unsafe_allow_html=True)
        top15 = countries.head(15)
        fig2 = px.bar(
            top15, x="count", y="country", orientation="h",
            color="count", color_continuous_scale="Blues",
            labels={"count": "Customers", "country": ""},
            template="plotly_dark"
        )
        fig2.update_layout(height=420, coloraxis_showscale=False,
                           margin=dict(l=0, r=0, t=10, b=0),
                           yaxis=dict(autorange="reversed"), **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Bottom 15 Countries</div>',
                    unsafe_allow_html=True)
        bottom15 = countries.tail(15).sort_values("count")
        fig3 = px.bar(
            bottom15, x="count", y="country", orientation="h",
            color="count", color_continuous_scale="Reds",
            labels={"count": "Customers", "country": ""},
            template="plotly_dark"
        )
        fig3.update_layout(height=420, coloraxis_showscale=False,
                           margin=dict(l=0, r=0, t=10, b=0),
                           yaxis=dict(autorange="reversed"), **PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Search Country</div>',
                unsafe_allow_html=True)
    search = st.text_input("Type a country name", placeholder="e.g. FRANCE")
    if search:
        result = countries[countries["country"].str.contains(search.upper())]
        if not result.empty:
            st.dataframe(result.reset_index(drop=True), use_container_width=True)
        else:
            st.info("No matching country found.")

elif page == "🏢 Companies":
    st.markdown("# Company Analysis")
    companies = load_company_counts()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Unique Companies</div>
            <div class="metric-value">{len(companies):,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Customers / Company</div>
            <div class="metric-value">{companies['count'].mean():.1f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Top 20 Companies by Customer Count</div>',
                unsafe_allow_html=True)
    top20 = companies.head(20)
    fig = px.bar(
        top20, x="company", y="count",
        color="count", color_continuous_scale="Teal",
        labels={"count": "Customers", "company": "Company"},
        template="plotly_dark"
    )
    fig.update_layout(height=400, coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_tickangle=-35, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Customer Count Distribution</div>',
                unsafe_allow_html=True)
    fig2 = px.histogram(
        companies, x="count", nbins=50,
        labels={"count": "Customers per Company", "y": "Number of Companies"},
        color_discrete_sequence=["#58a6ff"],
        template="plotly_dark"
    )
    fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

elif page == "🇪🇬 Egypt Customers":
    st.markdown("# Egypt Customer Subset")
    egypt = load_egypt()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Egypt Customers</div>
            <div class="metric-value">{len(egypt):,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Unique Cities</div>
            <div class="metric-value">{egypt['city'].nunique():,}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Date Range</div>
            <div class="metric-value" style="font-size:14px">
                {egypt['subscription_date'].min()}<br>→ {egypt['subscription_date'].max()}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Top Cities in Egypt</div>',
                unsafe_allow_html=True)
    city_counts = egypt["city"].value_counts().head(15).reset_index()
    city_counts.columns = ["city", "count"]
    fig = px.bar(
        city_counts, x="city", y="count",
        color="count", color_continuous_scale="Blues",
        labels={"count": "Customers", "city": "City"},
        template="plotly_dark"
    )
    fig.update_layout(height=380, coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_tickangle=-35, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Subscription Timeline</div>',
                unsafe_allow_html=True)
    egypt["subscription_date"] = pd.to_datetime(egypt["subscription_date"], errors="coerce")
    timeline = egypt.dropna(subset=["subscription_date"])
    timeline = timeline.groupby(timeline["subscription_date"].dt.to_period("M")
                                ).size().reset_index(name="count")
    timeline["subscription_date"] = timeline["subscription_date"].astype(str)
    fig2 = px.line(
        timeline, x="subscription_date", y="count",
        labels={"count": "New Subscriptions", "subscription_date": "Month"},
        color_discrete_sequence=["#58a6ff"],
        template="plotly_dark"
    )
    fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Customer Table</div>',
                unsafe_allow_html=True)
    search = st.text_input("Search by name or city", "")
    display = egypt.copy()
    if search:
        mask = (
            display["first_name"].str.contains(search, case=False, na=False) |
            display["last_name"].str.contains(search, case=False, na=False) |
            display["city"].str.contains(search, case=False, na=False)
        )
        display = display[mask]
    st.dataframe(display.reset_index(drop=True), use_container_width=True, height=400)