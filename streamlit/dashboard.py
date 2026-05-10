import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRM Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #378ADD;
    }
    .section-header {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #888;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.4rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ─── Color palette ───────────────────────────────────────────────────────────
COLORS = ["#378ADD", "#D85A30", "#1D9E75", "#BA7517", "#7F77DD",
          "#888780", "#D4537E", "#639922", "#E24B4A"]

# ─── Data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str):
    customers     = pd.read_excel(path, sheet_name="Customers")
    leads         = pd.read_excel(path, sheet_name="Leads")
    organizations = pd.read_excel(path, sheet_name="Organizations")
    people        = pd.read_excel(path, sheet_name="People")
    products      = pd.read_excel(path, sheet_name="Products")
    return customers, leads, organizations, people, products

@st.cache_data
def clean_and_engineer(customers, leads, organizations, people, products):
    # ── customers ──
    customers = customers.copy()
    customers["Subscription Date"] = pd.to_datetime(customers["Subscription Date"], errors="coerce")
    customers["sub_year"]  = customers["Subscription Date"].dt.year
    customers["sub_month"] = customers["Subscription Date"].dt.to_period("M")
    customers["days_since_subscription"] = (
        pd.Timestamp.today() - customers["Subscription Date"]
    ).dt.days

    # ── products ──
    products = products.copy()
    products["Price"] = pd.to_numeric(products["Price"], errors="coerce")
    products["Stock"] = pd.to_numeric(products["Stock"], errors="coerce")

    # ── organizations ──
    organizations = organizations.copy()
    organizations["Number of employees"] = pd.to_numeric(
        organizations["Number of employees"], errors="coerce"
    )
    organizations["Founded"] = pd.to_numeric(organizations["Founded"], errors="coerce")

    # ── people ──
    people = people.copy()
    people["Date of birth"] = pd.to_datetime(people["Date of birth"], errors="coerce")
    people["age"] = (pd.Timestamp.today() - people["Date of birth"]).dt.days // 365

    # ── leads ──
    leads = leads.copy()

    return customers, leads, organizations, people, products

@st.cache_data
def run_ml(customers):
    """Run K-Means clustering + churn classification."""
    df = customers.dropna(subset=["Country", "days_since_subscription"]).copy()

    le = LabelEncoder()
    df["country_encoded"] = le.fit_transform(df["Country"].astype(str))

    # Feature engineering
    df["sub_month_num"] = pd.to_datetime(
        df["Subscription Date"], errors="coerce"
    ).dt.month.fillna(0).astype(int)
    df["recency_group"] = pd.qcut(
        df["days_since_subscription"], q=4, labels=[0, 1, 2, 3]
    ).astype(int)
    df["quarter"] = (df["sub_month_num"] // 3)

    # ── Clustering ──
    cluster_features = df[["country_encoded", "days_since_subscription"]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_features)

    inertia = []
    for k in range(1, 10):
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        m.fit(X_scaled)
        inertia.append(m.inertia_)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # ── Churn label ──
    prob = df["days_since_subscription"] / df["days_since_subscription"].max()
    np.random.seed(42)
    df["churn"] = (np.random.rand(len(prob)) < prob).astype(int)

    X = df[["country_encoded", "recency_group", "quarter"]]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
    cm_rf     = confusion_matrix(y_test, y_pred_rf)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

    return df, inertia, report_rf, report_lr, cm_rf, importance

# ─── Sidebar — file upload ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 CRM Dashboard")
    st.caption("Upload your Excel file to get started")
    uploaded = st.file_uploader(
        "separated_data.xlsx",
        type=["xlsx"],
        help="Must contain sheets: Customers, Leads, Organizations, People, Products",
    )

    if uploaded:
        try:
            customers, leads, organizations, people, products = load_data(uploaded)
            customers, leads, organizations, people, products = clean_and_engineer(
                customers, leads, organizations, people, products
            )
            st.success(f"✅ Data loaded")
            st.markdown("---")
            st.markdown(f"**Customers** — {len(customers):,} rows")
            st.markdown(f"**Leads** — {len(leads):,} rows")
            st.markdown(f"**Organizations** — {len(organizations):,} rows")
            st.markdown(f"**People** — {len(people):,} rows")
            st.markdown(f"**Products** — {len(products):,} rows")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            uploaded = None

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📦 Products", "👥 Customers", "🔗 Leads", "🏢 Organizations", "🧑 People", "🤖 ML Analysis"],
    )

# ─── Guard: no file ──────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("## 👈 Upload your Excel file from the sidebar to start")
    st.info("The file should be the same `separated_data.xlsx` used in the notebook, "
            "with sheets: **Customers, Leads, Organizations, People, Products**.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCTS
# ════════════════════════════════════════════════════════════════════════════
if page == "📦 Products":
    st.title("📦 Products")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", f"{len(products):,}")
    col2.metric("Avg Price", f"${products['Price'].mean():.0f}")
    col3.metric("Avg Stock", f"{products['Stock'].mean():.0f}")
    col4.metric("Categories", f"{products['Category'].nunique()}")

    st.markdown('<div class="section-header">Category breakdown</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        cat_counts = products["Category"].value_counts().head(15)
        fig = px.bar(
            x=cat_counts.values, y=cat_counts.index, orientation="h",
            labels={"x": "Number of products", "y": ""},
            color_discrete_sequence=["#378ADD"],
            title="Top 15 product categories",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        avail = products["Availability"].value_counts()
        fig2 = px.pie(
            values=avail.values, names=avail.index,
            color_discrete_sequence=COLORS, title="Availability distribution",
            hole=0.38,
        )
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        fig2.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Pricing</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        fig3 = px.histogram(
            products, x="Price", nbins=40, color_discrete_sequence=["#7F77DD"],
            title="Price distribution",
        )
        fig3.add_vline(x=products["Price"].mean(), line_dash="dash",
                       line_color="#D85A30", annotation_text=f"Mean ${products['Price'].mean():.0f}")
        fig3.update_layout(height=360)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        avg_price = (
            products.groupby("Category")["Price"].mean()
            .sort_values(ascending=False).head(10)
        )
        fig4 = px.bar(
            x=avg_price.values, y=avg_price.index, orientation="h",
            labels={"x": "Avg price (USD)", "y": ""},
            color_discrete_sequence=["#BA7517"],
            title="Avg price — top 10 categories",
        )
        fig4.update_layout(yaxis=dict(autorange="reversed"), height=360)
        st.plotly_chart(fig4, use_container_width=True)

    # Brand filter
    st.markdown('<div class="section-header">Brand explorer</div>', unsafe_allow_html=True)
    brands = sorted(products["Brand"].dropna().unique())
    sel_brand = st.selectbox("Filter by brand", ["All"] + list(brands))
    df_b = products if sel_brand == "All" else products[products["Brand"] == sel_brand]
    st.dataframe(
        df_b[["Brand", "Category", "Price", "Stock", "Availability", "Color"]]
        .sort_values("Price", ascending=False)
        .reset_index(drop=True),
        use_container_width=True, height=300,
    )

# ════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMERS
# ════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customers":
    st.title("👥 Customers")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(customers):,}")
    col2.metric("Countries", f"{customers['Country'].nunique()}")
    col3.metric("Avg days subscribed",
                f"{customers['days_since_subscription'].mean():.0f}")

    st.markdown('<div class="section-header">Subscription trends</div>', unsafe_allow_html=True)

    year_counts = customers["sub_year"].value_counts().sort_index()
    fig = px.bar(
        x=year_counts.index.astype(str), y=year_counts.values,
        labels={"x": "Year", "y": "Customers"},
        color_discrete_sequence=["#378ADD"],
        title="Subscriptions by year",
    )
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)

    monthly = customers.groupby("sub_month").size().reset_index(name="count")
    monthly["label"] = monthly["sub_month"].astype(str)
    fig2 = px.area(
        monthly, x="label", y="count",
        labels={"label": "Month", "count": "New subscriptions"},
        color_discrete_sequence=["#1D9E75"],
        title="Monthly subscription trend",
    )
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Geography</div>', unsafe_allow_html=True)
    top_n = st.slider("Show top N countries", 5, 30, 15)
    top_countries = customers["Country"].value_counts().head(top_n)
    fig3 = px.bar(
        x=top_countries.values, y=top_countries.index, orientation="h",
        labels={"x": "Customers", "y": ""},
        color_discrete_sequence=["#7F77DD"],
        title=f"Top {top_n} countries by customer count",
    )
    fig3.update_layout(yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: LEADS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Leads":
    st.title("🔗 Leads")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Leads", f"{len(leads):,}")
    col2.metric("Sources", f"{leads['Source'].nunique()}")
    col3.metric("Deal Stages", f"{leads['Deal Stage'].nunique()}")

    st.markdown('<div class="section-header">Sources & stages</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        src = leads["Source"].value_counts().sort_values()
        fig = px.bar(
            x=src.values, y=src.index, orientation="h",
            labels={"x": "Leads", "y": ""},
            color_discrete_sequence=["#378ADD"],
            title="Lead sources",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        stage_colors = {
            "Closed Won": "#1D9E75", "Closed Lost": "#D85A30",
            "Disqualified": "#888780", "On Hold": "#BA7517",
            "Re-engagement": "#D4537E",
        }
        stage = leads["Deal Stage"].value_counts()
        colors = [stage_colors.get(s, "#378ADD") for s in stage.index]
        fig2 = px.bar(
            x=stage.values, y=stage.index, orientation="h",
            labels={"x": "Leads", "y": ""},
            title="Deal stage distribution",
            color=stage.index,
            color_discrete_map=stage_colors,
        )
        fig2.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Lead owner performance</div>', unsafe_allow_html=True)

    top_n_owners = st.slider("Show top N owners by total leads", 5, 30, 15)

    top_owners = (
        leads.groupby("Lead Owner").size()
        .sort_values(ascending=False)
        .head(top_n_owners)
        .index.tolist()
    )

    owner_stage = (
        leads[leads["Lead Owner"].isin(top_owners)]
        .groupby(["Lead Owner", "Deal Stage"])
        .size()
        .reset_index(name="count")
    )

    fig3 = px.bar(
        owner_stage,
        x="count",
        y="Lead Owner",
        color="Deal Stage",
        orientation="h",
        color_discrete_map=stage_colors,
        title=f"Top {top_n_owners} owners — deals by stage",
        labels={"count": "Leads", "Lead Owner": "Owner"},
        text="count",
    )
    fig3.update_traces(textposition="inside", textfont_size=11)
    fig3.update_layout(
        height=top_n_owners * 38 + 100,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Number of leads",
        legend=dict(orientation="v", x=1.01, y=1),
        barmode="stack",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: ORGANIZATIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Organizations":
    st.title("🏢 Organizations")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orgs", f"{len(organizations):,}")
    col2.metric("Industries", f"{organizations['Industry'].nunique()}")
    col3.metric("Avg employees",
                f"{organizations['Number of employees'].mean():.0f}")

    st.markdown('<div class="section-header">Industry & size</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        top_ind = organizations["Industry"].value_counts().head(15)
        fig = px.bar(
            x=top_ind.values, y=top_ind.index, orientation="h",
            color_discrete_sequence=["#1D9E75"],
            title="Top 15 industries",
            labels={"x": "Orgs", "y": ""},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            organizations, x="Number of employees", nbins=40,
            color_discrete_sequence=["#7F77DD"],
            title="Employee count distribution",
        )
        fig2.add_vline(
            x=organizations["Number of employees"].mean(), line_dash="dash",
            line_color="#D85A30",
            annotation_text=f"Mean {organizations['Number of employees'].mean():.0f}",
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(
            organizations, x="Founded", nbins=30,
            color_discrete_sequence=["#BA7517"],
            title="Founding year distribution",
        )
        fig3.update_layout(height=340)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        top_countries_org = organizations["Country"].value_counts().head(15)
        fig4 = px.bar(
            x=top_countries_org.values, y=top_countries_org.index, orientation="h",
            color_discrete_sequence=["#D85A30"],
            title="Top 15 countries — organizations",
            labels={"x": "Orgs", "y": ""},
        )
        fig4.update_layout(yaxis=dict(autorange="reversed"), height=340)
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: PEOPLE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧑 People":
    st.title("🧑 People")

    valid_ages = people["age"][(people["age"] >= 0) & (people["age"] <= 100)]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total People", f"{len(people):,}")
    col2.metric("Avg Age", f"{valid_ages.mean():.0f} yrs")
    col3.metric("Job Titles", f"{people['Job Title'].nunique()}")

    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sex = people["Sex"].value_counts()
        fig = px.pie(
            values=sex.values, names=sex.index,
            color_discrete_sequence=["#378ADD", "#D4537E"],
            title="Gender distribution", hole=0.4,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            x=valid_ages, nbins=30,
            color_discrete_sequence=["#7F77DD"],
            title="Age distribution",
            labels={"x": "Age (years)", "y": "Count"},
        )
        fig2.add_vline(x=valid_ages.mean(), line_dash="dash", line_color="#D85A30",
                       annotation_text=f"Mean {valid_ages.mean():.0f} yrs")
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Job titles</div>', unsafe_allow_html=True)
    top_n_jobs = st.slider("Top N job titles", 5, 30, 15)
    top_jobs = people["Job Title"].value_counts().head(top_n_jobs)
    fig3 = px.bar(
        x=top_jobs.values, y=top_jobs.index, orientation="h",
        color_discrete_sequence=["#1D9E75"],
        title=f"Top {top_n_jobs} job titles",
        labels={"x": "Count", "y": ""},
    )
    fig3.update_layout(yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: ML ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Analysis":
    st.title("🤖 ML Analysis")
    st.caption("K-Means clustering + Churn prediction (Random Forest vs Logistic Regression)")

    with st.spinner("Running ML pipeline…"):
        df_ml, inertia, report_rf, report_lr, cm_rf, importance = run_ml(customers)

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model performance</div>', unsafe_allow_html=True)

    acc_rf  = report_rf["accuracy"]
    acc_lr  = report_lr["accuracy"]
    rec_rf  = report_rf.get("1", {}).get("recall", 0)
    rec_lr  = report_lr.get("1", {}).get("recall", 0)
    churn_r = df_ml["churn"].mean()
    cluster_n = df_ml["cluster"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("RF Accuracy",     f"{acc_rf:.0%}")
    col2.metric("LR Accuracy",     f"{acc_lr:.0%}")
    col3.metric("RF Churn Recall", f"{rec_rf:.0%}")
    col4.metric("LR Churn Recall", f"{rec_lr:.0%}")
    col5.metric("Avg Churn Rate",  f"{churn_r:.0%}")

    # ── Elbow curve ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Clustering — elbow method</div>', unsafe_allow_html=True)
    fig_elbow = px.line(
        x=list(range(1, 10)), y=inertia, markers=True,
        labels={"x": "K (number of clusters)", "y": "Inertia"},
        title="Elbow method — optimal K = 4",
        color_discrete_sequence=["#378ADD"],
    )
    fig_elbow.add_vline(x=4, line_dash="dash", line_color="#D85A30",
                        annotation_text="K = 4")
    fig_elbow.update_layout(height=320)
    st.plotly_chart(fig_elbow, use_container_width=True)

    # ── Cluster scatter ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Customer segments</div>', unsafe_allow_html=True)
    fig_clust = px.scatter(
        df_ml, x="days_since_subscription", y="country_encoded",
        color=df_ml["cluster"].astype(str),
        color_discrete_sequence=COLORS,
        labels={
            "days_since_subscription": "Days since subscription",
            "country_encoded": "Country (encoded)",
            "color": "Cluster",
        },
        title="K-Means clusters (K=4)",
        opacity=0.7,
    )
    fig_clust.update_traces(marker=dict(size=4))
    fig_clust.update_layout(height=380, legend_title_text="Cluster")
    st.plotly_chart(fig_clust, use_container_width=True)

    # Cluster stats table
    cluster_stats = (
        df_ml.groupby("cluster")
        .agg(
            customers=("cluster", "count"),
            avg_days=("days_since_subscription", "mean"),
            churn_rate=("churn", "mean"),
        )
        .reset_index()
        .rename(columns={
            "cluster": "Cluster", "customers": "Customers",
            "avg_days": "Avg days subscribed", "churn_rate": "Churn rate",
        })
    )
    cluster_stats["Avg days subscribed"] = cluster_stats["Avg days subscribed"].round(0).astype(int)
    cluster_stats["Churn rate"] = cluster_stats["Churn rate"].map("{:.1%}".format)
    st.dataframe(cluster_stats, use_container_width=True, hide_index=True)

    # ── Churn distribution ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Churn distribution</div>', unsafe_allow_html=True)
    churn_counts = df_ml["churn"].value_counts().rename({0: "No churn", 1: "Churn"})
    fig_churn = px.pie(
        values=churn_counts.values, names=churn_counts.index,
        color_discrete_sequence=["#1D9E75", "#D85A30"],
        title="Churn label distribution", hole=0.4,
    )
    fig_churn.update_traces(textposition="outside", textinfo="percent+label")
    fig_churn.update_layout(showlegend=False, height=320)
    col_a, col_b = st.columns([1, 2])
    col_a.plotly_chart(fig_churn, use_container_width=True)

    # ── Confusion matrix ──────────────────────────────────────────────────
    with col_b:
        fig_cm = px.imshow(
            cm_rf,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No churn", "Churn"], y=["No churn", "Churn"],
            color_continuous_scale="Blues",
            text_auto=True,
            title="Confusion matrix — Random Forest",
        )
        fig_cm.update_layout(height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature importance</div>', unsafe_allow_html=True)
    fig_imp = px.bar(
        x=importance.values, y=importance.index, orientation="h",
        color_discrete_sequence=["#7F77DD"],
        title="Feature importance — Random Forest",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig_imp.update_layout(height=300)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Model comparison ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model comparison</div>', unsafe_allow_html=True)

    metrics_labels = ["Accuracy", "Precision (churn)", "Recall (churn)", "F1 (churn)"]
    rf_vals = [
        report_rf["accuracy"],
        report_rf.get("1", {}).get("precision", 0),
        report_rf.get("1", {}).get("recall", 0),
        report_rf.get("1", {}).get("f1-score", 0),
    ]
    lr_vals = [
        report_lr["accuracy"],
        report_lr.get("1", {}).get("precision", 0),
        report_lr.get("1", {}).get("recall", 0),
        report_lr.get("1", {}).get("f1-score", 0),
    ]

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name="Random Forest", x=metrics_labels, y=rf_vals,
        marker_color="#378ADD",
        text=[f"{v:.0%}" for v in rf_vals], textposition="outside",
    ))
    fig_compare.add_trace(go.Bar(
        name="Logistic Regression", x=metrics_labels, y=lr_vals,
        marker_color="#D85A30",
        text=[f"{v:.0%}" for v in lr_vals], textposition="outside",
    ))
    fig_compare.update_layout(
        barmode="group",
        title="Random Forest vs Logistic Regression",
        yaxis=dict(range=[0, 1.1], tickformat=".0%"),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    st.info(
        "**Key insight:** Random Forest is better at detecting churn (higher recall on class 1), "
        "while Logistic Regression balances both classes more evenly. "
        "Country encoding dominates feature importance due to synthetic churn label generation — "
        "in a real dataset, recency features would be more meaningful."
    )
