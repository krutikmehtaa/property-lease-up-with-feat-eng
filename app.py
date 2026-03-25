"""
Lease-Up Analysis Dashboard
────────────────────────────
Streamlit + Gemini AI  |  Markets: Austin-Round Rock, TX & Akron, OH
Period: April 2008 – September 2020
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lease-Up Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS  (mirrors the notebook exactly)
# ─────────────────────────────────────────────────────────────
FILES = {
    "Austin": "MSA1.xlsx",
    "Akron":  "MSA2.xlsx",
}
MONTH_COL_START = 30
N_MONTHS        = 150
META_MAP = {
    0:  "market_code",  1:  "market_name",  13: "proj_id",
    14: "submarket",    15: "name",          16: "address",
    17: "city",         18: "state",         21: "year_built",
    25: "units",        26: "area_per_unit", 29: "overall_status",
}
DELIVERY_FLAGS  = {"LU", "UC/LU"}
STAB_THRESHOLD  = 0.90
AC_ORDER = {
    "A+": 12, "A": 11, "A-": 10,
    "B+":  9, "B":  8, "B-":  7,
    "C+":  6, "C":  5, "C-":  4,
    "D+":  3, "D":  2, "D-":  1,
}
EMBED_FEATS = [
    "units", "area_per_unit", "eff_rent_delivery",
    "age_at_delivery", "delivery_month_of_year", "occ_ramp_rate",
    "ask_eff_spread_at_delivery", "ac_at_delivery", "conc_at_delivery",
]
PALETTE = {"Austin": "#2563EB", "Akron": "#DC2626"}
CLUSTER_COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]


def resolve_data_source(label: str, default_path: str) -> str:
    """Resolve Excel source from local file or Streamlit secrets.

    Priority:
    1) `st.secrets["DATA_SOURCES"][label]` (URL/path)
    2) local default file path (e.g., MSA1.xlsx)
    """
    try:
        if "DATA_SOURCES" in st.secrets and label in st.secrets["DATA_SOURCES"]:
            src = str(st.secrets["DATA_SOURCES"][label]).strip()
            if src:
                return src
    except Exception:
        # Fall back to local files when secrets are unavailable.
        pass
    return default_path


# ─────────────────────────────────────────────────────────────
# DATA PIPELINE  (identical logic to the notebook)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading market data…")
def load_all_data():
    """Load, process, and feature-engineer everything once.
    st.cache_data means this only runs on the first load — not on every
    user interaction, which keeps the app fast."""

    markets = {}
    for label, path in FILES.items():
        source = resolve_data_source(label, path)
        ps_raw   = pd.read_excel(source, sheet_name="Property Status",  skiprows=2, header=None)
        occ_raw  = pd.read_excel(source, sheet_name="Occ & Concession", skiprows=2, header=None)
        rent_raw = pd.read_excel(source, sheet_name="Rent",             skiprows=2, header=None)
        ac_raw   = pd.read_excel(source, sheet_name="Asset Class",      skiprows=2, header=None)

        # Read month labels from the first row after skiprows=2.
        # Handles values like 'Apr-08' / 'Sep-09' and any datetime-like cell values.
        months_raw = rent_raw.iloc[0, MONTH_COL_START:MONTH_COL_START + N_MONTHS].tolist()

        def _parse_month(v):
            if pd.isna(v):
                return pd.NaT
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return pd.NaT
                dt = pd.to_datetime(s, format="%b-%y", errors="coerce")
                if pd.isna(dt):
                    dt = pd.to_datetime(s, errors="coerce")
                return dt
            return pd.to_datetime(v, errors="coerce")

        months = pd.Index([_parse_month(m) for m in months_raw], dtype="datetime64[ns]")

        meta = (
            ps_raw.iloc[1:, list(META_MAP.keys())]
            .rename(columns=META_MAP)
            .reset_index(drop=True)
        )
        meta["market"] = label

        def extract_block(raw_df, block_num):
            s     = MONTH_COL_START + block_num * N_MONTHS
            block = raw_df.iloc[1:, s: s + N_MONTHS].copy()
            block.columns = months
            block.index   = meta.index
            return block.apply(pd.to_numeric, errors="coerce")

        status   = ps_raw.iloc[1:, MONTH_COL_START: MONTH_COL_START + N_MONTHS].copy()
        status.columns = months
        status.index   = meta.index

        ac_block = ac_raw.iloc[1:, MONTH_COL_START: MONTH_COL_START + N_MONTHS].copy()
        ac_block.columns = months
        ac_block.index   = meta.index

        markets[label] = {
            "meta":      meta,
            "status":    status,
            "occ":       extract_block(occ_raw,  0),
            "conc_amt":  extract_block(occ_raw,  1),
            "eff_rent":  extract_block(rent_raw, 1),
            "ask_rent":  extract_block(rent_raw, 0),
            "ac":        ac_block,
            "months":    months,
        }

    # ── Delivered properties ──────────────────────────────────
    all_delivered = []
    for label, data in markets.items():
        for idx, row in data["status"].iterrows():
            non_null = row.dropna()
            if non_null.empty:
                continue
            if str(non_null.iloc[0]).strip().upper() in DELIVERY_FLAGS:
                rec = data["meta"].loc[idx].to_dict()
                rec["delivery_month"]  = non_null.index[0]
                rec["delivery_status"] = str(non_null.iloc[0]).strip().upper()
                rec["delivery_year"]   = non_null.index[0].year
                all_delivered.append(rec)

    delivered_df = pd.DataFrame(all_delivered).reset_index(drop=True)

    # ── Lease-up time ─────────────────────────────────────────
    lu_rows = []
    for _, row in delivered_df.iterrows():
        data     = markets[row["market"]]
        match    = data["meta"][data["meta"]["name"] == row["name"]]
        if match.empty:
            continue
        orig_idx = match.index[0]
        delivery = row["delivery_month"]
        occ_post = data["occ"].loc[orig_idx, data["occ"].columns >= delivery]
        stable   = occ_post[occ_post >= STAB_THRESHOLD]
        occ_d    = occ_post.iloc[0] if not occ_post.empty else np.nan

        if stable.empty:
            lu_rows.append({**row.to_dict(),
                             "stabilization_month": pd.NaT, "lease_up_months": np.nan,
                             "occ_at_delivery": occ_d, "did_not_stabilize": True})
        else:
            stab_mo = stable.index[0]
            months_ = (stab_mo.year - delivery.year) * 12 + (stab_mo.month - delivery.month)
            lu_rows.append({**row.to_dict(),
                             "stabilization_month": stab_mo, "lease_up_months": months_,
                             "occ_at_delivery": occ_d, "did_not_stabilize": False})

    lu_df = pd.DataFrame(lu_rows).reset_index(drop=True)

    # ── Rent growth ───────────────────────────────────────────
    rg_rows = []
    for _, row in lu_df.iterrows():
        data     = markets[row["market"]]
        match    = data["meta"][data["meta"]["name"] == row["name"]]
        if match.empty:
            continue
        orig_idx = match.index[0]
        delivery = row["delivery_month"]
        post     = data["eff_rent"].loc[orig_idx, data["eff_rent"].columns >= delivery].dropna()

        if post.empty:
            rg_rows.append({**row.to_dict(), "eff_rent_delivery": np.nan,
                             "eff_rent_endpoint": np.nan, "rent_growth_pct": np.nan,
                             "neg_rent_growth": False})
            continue

        eff0 = post.iloc[0]
        if not row["did_not_stabilize"] and pd.notna(row.get("stabilization_month")):
            stab_mo   = row["stabilization_month"]
            eff_end   = data["eff_rent"].loc[orig_idx, stab_mo] if stab_mo in data["eff_rent"].columns else post.iloc[-1]
        else:
            eff_end = post.iloc[-1]

        growth = (eff_end - eff0) / eff0 * 100 if pd.notna(eff0) and eff0 != 0 else np.nan
        rg_rows.append({**row.to_dict(), "eff_rent_delivery": eff0, "eff_rent_endpoint": eff_end,
                         "rent_growth_pct": growth,
                         "neg_rent_growth": (growth < 0) if pd.notna(growth) else False})

    df = pd.DataFrame(rg_rows).reset_index(drop=True)
    if df.empty:
        embed_empty = pd.DataFrame(
            columns=EMBED_FEATS + [
                "name", "market", "submarket", "lease_up_months", "did_not_stabilize",
                "cluster", "emb_x", "emb_y", "cluster_label",
            ]
        )
        return df, embed_empty, "N/A"

    # ── Feature engineering ───────────────────────────────────
    def _ts(row, key):
        data  = markets[row["market"]]
        match = data["meta"][data["meta"]["name"] == row["name"]]
        if match.empty:
            return None
        return data[key].loc[match.index[0], data[key].columns >= row["delivery_month"]].dropna()

    df["age_at_delivery"]   = (df["delivery_year"] - df["year_built"].fillna(df["delivery_year"])).clip(lower=0).fillna(0).astype(int)
    df["delivery_month_of_year"] = df["delivery_month"].dt.month

    def occ_ramp(row):
        s = _ts(row, "occ")
        if s is None or len(s) < 2: return np.nan
        w = s.iloc[:6]; return (w.iloc[-1] - w.iloc[0]) / (len(w) - 1)

    def spread(row):
        e = _ts(row, "eff_rent"); a = _ts(row, "ask_rent")
        if e is None or a is None or e.empty or a.empty: return np.nan
        av = a.iloc[0]
        return (av - e.iloc[0]) / av if av and not pd.isna(av) else np.nan

    def ac_num(row):
        s = _ts(row, "ac")
        if s is None or s.empty: return np.nan
        return AC_ORDER.get(str(s.iloc[0]).strip(), np.nan)

    def conc(row):
        s = _ts(row, "conc_amt")
        if s is None or s.empty: return 0.0
        v = s.iloc[0]; return float(v) if pd.notna(v) else 0.0

    df["occ_ramp_rate"]              = df.apply(occ_ramp, axis=1)
    df["ask_eff_spread_at_delivery"] = df.apply(spread,   axis=1)
    df["ac_at_delivery"]             = df.apply(ac_num,   axis=1)
    df["conc_at_delivery"]           = df.apply(conc,     axis=1)
    df["size_tier"] = pd.cut(df["area_per_unit"], bins=[0, 700, 900, 1100, float("inf")],
                              labels=["Micro", "Small", "Mid", "Large"], right=True)

    # ── Clustering ────────────────────────────────────────────
    embed = df[EMBED_FEATS + ["name", "market", "submarket", "lease_up_months", "did_not_stabilize"]].copy()
    for col in EMBED_FEATS:
        median_val = embed[col].median()
        embed[col] = embed[col].fillna(median_val if pd.notna(median_val) else 0)

    X = StandardScaler().fit_transform(embed[EMBED_FEATS].values)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(X)

    if n == 1:
        embed["cluster"] = 0
        embed["emb_x"] = 0.0
        embed["emb_y"] = 0.0
        embed["cluster_label"] = "Cluster 1"
        return df, embed, "N/A (single sample)"

    # UMAP when enough data, PCA otherwise
    try:
        import umap
        if n < 15:
            raise ValueError("too few samples")
        X_2d   = umap.UMAP(n_components=2, n_neighbors=min(15, n - 1), min_dist=0.1, random_state=42).fit_transform(X)
        method = "UMAP"
    except Exception:
        pca    = PCA(n_components=2, random_state=42)
        X_2d   = pca.fit_transform(X)
        vr     = pca.explained_variance_ratio_
        method = f"PCA ({vr[0]*100:.0f}% + {vr[1]*100:.0f}% variance)"

    # Optimal k
    if n > 4:
        sil    = {k: silhouette_score(X, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X))
                  for k in range(2, min(6, n))}
        best_k = max(sil, key=sil.get)
    else:
        best_k = min(2, n)

    embed["cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
    embed["emb_x"]   = X_2d[:, 0]
    embed["emb_y"]   = X_2d[:, 1]
    embed["cluster_label"] = "Cluster " + (embed["cluster"] + 1).astype(str)

    return df, embed, method


# ─────────────────────────────────────────────────────────────
# GEMINI AI SETUP
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini():
    """Initialise Gemini once and reuse the client."""
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-2.5-flash")


def ask_gemini(model, prompt: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


def build_data_context(df: pd.DataFrame) -> str:
    """Summarise the dataset into a compact string for the AI prompt.
    We don't send the raw dataframe — just the key stats — to keep token
    usage low and responses focused."""
    stab = df[~df["did_not_stabilize"]]
    neg  = df[df["neg_rent_growth"] == True]

    lines = [
        "You are a real estate data analyst. Here is a summary of the lease-up dataset:",
        f"- Markets: Austin-Round Rock TX ({(df['market']=='Austin').sum()} delivered props) and Akron OH ({(df['market']=='Akron').sum()} delivered props)",
        f"- Period: April 2008 – September 2020",
        f"- Total delivered properties: {len(df)}",
        f"- Properties that stabilised (≥90% occ): {len(stab)}",
        f"- Properties that never stabilised in the window: {len(df) - len(stab)}",
        f"- Austin avg lease-up: {stab[stab['market']=='Austin']['lease_up_months'].mean():.1f} months",
        f"- Akron avg lease-up: {stab[stab['market']=='Akron']['lease_up_months'].mean():.1f} months",
        f"- Properties with negative effective rent growth: {len(neg)} ({len(neg)/len(df)*100:.0f}%)",
        f"- Delivery years range: {df['delivery_year'].min()} – {df['delivery_year'].max()}",
        "",
        "Top 10 slowest lease-ups (stabilised only):",
    ]
    top10 = stab.nlargest(10, "lease_up_months")[["name", "market", "lease_up_months", "units", "eff_rent_delivery"]].to_string(index=False)
    lines.append(top10)
    lines.append("")
    lines.append("Answer the user's question based only on this data. Be concise and insightful.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CHART BUILDERS  (all Plotly for interactivity)
# ─────────────────────────────────────────────────────────────
def chart_deliveries_by_year(df):
    counts = df.groupby(["delivery_year", "market"]).size().reset_index(name="count")
    fig = px.bar(
        counts, x="delivery_year", y="count", color="market",
        color_discrete_map=PALETTE, barmode="group",
        labels={"delivery_year": "Year", "count": "Properties Delivered", "market": "Market"},
        title="Properties Delivered by Year",
    )
    fig.update_layout(legend_title_text="Market", plot_bgcolor="white")
    return fig


def chart_lease_up_distribution(df):
    stab = df[~df["did_not_stabilize"]].dropna(subset=["lease_up_months"])
    fig  = px.histogram(
        stab, x="lease_up_months", color="market",
        color_discrete_map=PALETTE, nbins=20, barmode="overlay", opacity=0.75,
        labels={"lease_up_months": "Lease-Up Months", "market": "Market"},
        title="Lease-Up Time Distribution",
    )
    fig.update_layout(plot_bgcolor="white", legend_title_text="Market")
    return fig


def chart_stabilization_mix(df):
    mix = df.assign(stabilization_state=np.where(df["did_not_stabilize"], "Not Stabilized", "Stabilized"))
    counts = mix.groupby(["market", "stabilization_state"]).size().reset_index(name="count")
    fig = px.bar(
        counts,
        x="market",
        y="count",
        color="stabilization_state",
        barmode="stack",
        color_discrete_map={"Stabilized": "#16A34A", "Not Stabilized": "#DC2626"},
        labels={"market": "Market", "count": "Properties"},
        title="Stabilization Mix by Market",
    )
    fig.update_layout(plot_bgcolor="white", legend_title_text="")
    return fig


def chart_leaseup_trend(df):
    stab = df[~df["did_not_stabilize"]].dropna(subset=["lease_up_months", "delivery_year"])
    if stab.empty:
        return None
    trend = (
        stab.groupby(["delivery_year", "market"])["lease_up_months"]
        .median()
        .reset_index(name="median_lease_up_months")
    )
    fig = px.line(
        trend,
        x="delivery_year",
        y="median_lease_up_months",
        color="market",
        markers=True,
        color_discrete_map=PALETTE,
        labels={
            "delivery_year": "Delivery Year",
            "median_lease_up_months": "Median Lease-Up (Months)",
            "market": "Market",
        },
        title="Lease-Up Speed Trend by Delivery Cohort",
    )
    fig.update_layout(plot_bgcolor="white", legend_title_text="Market")
    return fig


def chart_rent_growth(df):
    sub = df.dropna(subset=["rent_growth_pct"]).copy()
    sub["color"] = sub["rent_growth_pct"].apply(lambda x: "#DC2626" if x < 0 else "#16A34A")
    sub_sorted   = sub.sort_values("rent_growth_pct")

    fig = px.scatter(
        sub_sorted, x="rent_growth_pct", y="name",
        color="market", color_discrete_map=PALETTE,
        hover_data=["units", "delivery_month", "lease_up_months"],
        labels={"rent_growth_pct": "Rent Growth (%)", "name": "Property"},
        title="Effective Rent Growth During Lease-Up",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
    fig.update_layout(plot_bgcolor="white", height=max(400, len(sub) * 18))
    return fig


def chart_pricing_pressure(df):
    stab = df[~df["did_not_stabilize"]].dropna(
        subset=["ask_eff_spread_at_delivery", "lease_up_months", "units"]
    )
    if stab.empty:
        return None
    fig = px.scatter(
        stab,
        x="ask_eff_spread_at_delivery",
        y="lease_up_months",
        size="units",
        color="market",
        color_discrete_map=PALETTE,
        hover_data=["name", "conc_at_delivery", "eff_rent_delivery"],
        labels={
            "ask_eff_spread_at_delivery": "Ask vs Effective Spread at Delivery",
            "lease_up_months": "Lease-Up Months",
            "units": "Units",
        },
        title="Pricing Pressure vs Lease-Up Time",
    )
    fig.add_vline(x=stab["ask_eff_spread_at_delivery"].median(), line_dash="dot", line_color="#94A3B8")
    fig.add_hline(y=stab["lease_up_months"].median(), line_dash="dot", line_color="#94A3B8")
    fig.update_layout(plot_bgcolor="white", legend_title_text="Market")
    return fig


def chart_clusters(embed_df, method):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Coloured by Cluster", "Coloured by Lease-Up Time"),
    )

    # Left: clusters
    for i, (cid, grp) in enumerate(embed_df.groupby("cluster_label")):
        fig.add_trace(go.Scatter(
            x=grp["emb_x"], y=grp["emb_y"], mode="markers",
            name=cid,
            marker=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], size=9,
                        line=dict(color="white", width=1)),
            text=grp["name"], hovertemplate="<b>%{text}</b><br>Market: " +
                 grp["market"].values[0] + "<extra></extra>",
        ), row=1, col=1)

    # Right: lease-up time
    stab_e   = embed_df[~embed_df["did_not_stabilize"] & embed_df["lease_up_months"].notna()]
    unstab_e = embed_df[embed_df["did_not_stabilize"]]

    if not stab_e.empty:
        fig.add_trace(go.Scatter(
            x=stab_e["emb_x"], y=stab_e["emb_y"], mode="markers",
            name="Stabilised",
            marker=dict(
                color=stab_e["lease_up_months"],
                colorscale="RdYlGn_r", showscale=True,
                colorbar=dict(title="Months", x=1.02),
                size=9, line=dict(color="white", width=1),
            ),
            text=stab_e["name"],
            hovertemplate="<b>%{text}</b><br>%{marker.color:.0f} months<extra></extra>",
        ), row=1, col=2)

    if not unstab_e.empty:
        fig.add_trace(go.Scatter(
            x=unstab_e["emb_x"], y=unstab_e["emb_y"], mode="markers",
            name="Not stabilised",
            marker=dict(color="#94A3B8", symbol="x", size=10),
            text=unstab_e["name"],
            hovertemplate="<b>%{text}</b><br>Not yet stabilised<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        title=f"Property Clusters — {method}",
        plot_bgcolor="white", height=480,
        legend=dict(orientation="v", x=1.08),
    )
    return fig


def chart_feature_importance(df):
    """Show how each engineered feature correlates with lease-up months."""
    stab = df[~df["did_not_stabilize"]].dropna(subset=["lease_up_months"])
    feat_cols = ["occ_ramp_rate", "ask_eff_spread_at_delivery", "ac_at_delivery",
                 "conc_at_delivery", "age_at_delivery", "delivery_month_of_year"]

    corrs = {}
    for col in feat_cols:
        sub = stab.dropna(subset=[col])
        if len(sub) > 2:
            corrs[col] = sub[col].corr(sub["lease_up_months"])

    if not corrs:
        return None

    corr_df = (pd.DataFrame.from_dict(corrs, orient="index", columns=["correlation"])
               .sort_values("correlation"))
    corr_df["color"] = corr_df["correlation"].apply(lambda x: "#DC2626" if x > 0 else "#16A34A")

    fig = px.bar(
        corr_df.reset_index(), x="correlation", y="index",
        orientation="h", color="color",
        color_discrete_map="identity",
        labels={"index": "Feature", "correlation": "Correlation with Lease-Up Time"},
        title="Feature Correlation with Lease-Up Time",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
    fig.update_layout(plot_bgcolor="white", showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    # ── Load data ─────────────────────────────────────────────
    df, embed_df, embed_method = load_all_data()
    gemini = get_gemini()

    if df.empty:
        st.error("No delivered-property records were parsed from input files. Check the Excel structure and month columns.")
        return

    # ── Sidebar filters ───────────────────────────────────────
    st.sidebar.image("https://img.icons8.com/color/96/building.png", width=60)
    st.sidebar.title("Filters")

    markets_sel = st.sidebar.multiselect(
        "Market", options=["Austin", "Akron"], default=["Austin", "Akron"]
    )
    years = df["delivery_year"].dropna()
    if years.empty:
        st.error("No valid delivery years were parsed. Please verify month/date values in the source Excel files.")
        return
    year_min, year_max = int(years.min()), int(years.max())
    year_range = st.sidebar.slider("Delivery Year", year_min, year_max, (year_min, year_max))

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: Apr 2008 – Sep 2020  |  MSA1 (Austin) & MSA2 (Akron)")

    # Apply filters
    mask = (
        df["market"].isin(markets_sel) &
        df["delivery_year"].between(year_range[0], year_range[1])
    )
    dff      = df[mask].copy()
    embed_f  = embed_df[embed_df["market"].isin(markets_sel)].copy()

    if dff.empty:
        st.warning("No properties match the current filters.")
        return

    # ── Header ────────────────────────────────────────────────
    st.title("🏢 Lease-Up Analysis Dashboard")
    st.caption("Austin-Round Rock, TX  ·  Akron, OH  ·  April 2008 – September 2020")

    # ── KPI strip ─────────────────────────────────────────────
    stab = dff[~dff["did_not_stabilize"]]
    neg  = dff[dff["neg_rent_growth"] == True]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delivered Properties", len(dff))
    c2.metric("Avg Lease-Up Time",
              f"{stab['lease_up_months'].mean():.1f} mo" if not stab.empty else "N/A")
    c3.metric("Never Stabilised", len(dff) - len(stab),
              help="Properties that hadn't reached 90% occ by Sep 2020")
    c4.metric("Negative Rent Growth", f"{len(neg)} ({len(neg)/len(dff)*100:.0f}%)")

    st.markdown("---")

    # ── Tab layout ────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 Deliveries",
        "⏱ Lease-Up",
        "💰 Rent Growth",
        "🔵 Clusters",
        "🤖 Ask AI",
    ])

    # ── TAB 1: Deliveries ─────────────────────────────────────
    with tab1:
        st.subheader("Properties Delivered Since April 2008")
        st.plotly_chart(chart_deliveries_by_year(dff), use_container_width=True)
        st.plotly_chart(chart_stabilization_mix(dff), use_container_width=True)

        # Market-level business scorecard
        score = (
            dff.groupby("market")
            .apply(lambda g: pd.Series({
                "Delivered": len(g),
                "Stabilization Rate": (~g["did_not_stabilize"]).mean(),
                "Median Lease-Up (mo)": g.loc[~g["did_not_stabilize"], "lease_up_months"].median(),
                "Negative Rent Growth Rate": g["neg_rent_growth"].mean(),
                "Avg Concession at Delivery ($)": g["conc_at_delivery"].mean(),
            }))
            .reset_index()
        )
        score["Stabilization Rate"] = score["Stabilization Rate"].map(lambda x: f"{x*100:.0f}%")
        score["Negative Rent Growth Rate"] = score["Negative Rent Growth Rate"].map(lambda x: f"{x*100:.0f}%")
        score["Median Lease-Up (mo)"] = score["Median Lease-Up (mo)"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        score["Avg Concession at Delivery ($)"] = score["Avg Concession at Delivery ($)"].map(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        st.markdown("**Market Scorecard**")
        st.dataframe(score, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**By Market**")
            by_mkt = dff.groupby("market").size().reset_index(name="count")
            st.dataframe(by_mkt, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**By Delivery Status**")
            by_status = dff.groupby("delivery_status").size().reset_index(name="count")
            st.dataframe(by_status, use_container_width=True, hide_index=True)

    # ── TAB 2: Lease-Up ───────────────────────────────────────
    with tab2:
        st.subheader("Lease-Up Time Analysis")
        st.plotly_chart(chart_lease_up_distribution(dff), use_container_width=True)
        trend_fig = chart_leaseup_trend(dff)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)

        # Stats per market
        for mkt in markets_sel:
            sub = stab[stab["market"] == mkt]
            if sub.empty:
                continue
            with st.expander(f"📊 {mkt} — detailed stats"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg",    f"{sub['lease_up_months'].mean():.1f} mo")
                c2.metric("Median", f"{sub['lease_up_months'].median():.1f} mo")
                c3.metric("Range",  f"{int(sub['lease_up_months'].min())} – {int(sub['lease_up_months'].max())} mo")

                st.dataframe(
                    sub[["name", "units", "delivery_month", "stabilization_month", "lease_up_months"]]
                    .sort_values("lease_up_months", ascending=False)
                    .reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

    # ── TAB 3: Rent Growth ────────────────────────────────────
    with tab3:
        st.subheader("Effective Rent Growth During Lease-Up")
        st.caption("Rent growth = (rent at stabilisation – rent at delivery) / rent at delivery")

        show_neg_only = st.checkbox("Show only negative rent growth properties", value=False)
        display_df   = dff[dff["neg_rent_growth"] == True] if show_neg_only else dff
        display_df   = display_df.dropna(subset=["rent_growth_pct"])

        if display_df.empty:
            st.info("No properties to display with current filters.")
        else:
            st.plotly_chart(chart_rent_growth(display_df), use_container_width=True)
            pressure_fig = chart_pricing_pressure(display_df)
            if pressure_fig:
                st.plotly_chart(pressure_fig, use_container_width=True)
            st.dataframe(
                display_df[["market", "name", "units", "eff_rent_delivery",
                             "eff_rent_endpoint", "rent_growth_pct"]]
                .sort_values("rent_growth_pct")
                .reset_index(drop=True)
                .assign(rent_growth_pct=lambda x: x["rent_growth_pct"].map("{:+.1f}%".format)),
                use_container_width=True,
                hide_index=True,
            )

    # ── TAB 4: Clusters ───────────────────────────────────────
    with tab4:
        st.subheader("Property Clustering — Embedding Space")
        st.caption(f"Dimensionality reduction: {embed_method}")

        if embed_f.empty:
            st.info("No cluster data for the selected market(s).")
        else:
            st.plotly_chart(chart_clusters(embed_f, embed_method), use_container_width=True)

            # Feature correlation chart
            corr_fig = chart_feature_importance(dff)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)

            # Cluster profiles table
            st.markdown("**Cluster Profiles (mean values)**")
            profile = (
                embed_f.groupby("cluster_label")[
                    ["units", "eff_rent_delivery", "occ_ramp_rate",
                     "ask_eff_spread_at_delivery", "conc_at_delivery", "lease_up_months"]
                ].mean().round(2)
            )
            profile.insert(0, "properties", embed_f.groupby("cluster_label").size())
            st.dataframe(profile, use_container_width=True)

            # AI-generated cluster insight
            st.markdown("---")
            st.markdown("**🤖 AI Cluster Interpretation**")
            if st.button("Explain the clusters with AI"):
                with st.spinner("Generating insight…"):
                    profile_str = profile.to_string()
                    prompt = f"""
You are a real estate analyst. Below are the mean feature profiles for property clusters
identified in a lease-up analysis of Austin TX and Akron OH apartment properties (2008–2020).

{profile_str}

Feature definitions:
- units: number of units in the property
- eff_rent_delivery: effective rent ($/month) at the time of delivery
- occ_ramp_rate: average monthly occupancy gain in the first 6 months (e.g. 0.05 = 5%/month)
- ask_eff_spread_at_delivery: (asking rent – effective rent) / asking rent — higher means heavier concessions
- conc_at_delivery: dollar value of concessions offered at delivery
- lease_up_months: average months to reach 90% occupancy

Write a brief (3–4 sentences per cluster) plain-English interpretation.
For each cluster, describe: what type of property it represents, why it leases up at the pace it does,
and one practical implication for a real estate investor. Be direct and professional.
"""
                    insight = ask_gemini(gemini, prompt)
                    st.markdown(insight)

    # ── TAB 5: Ask AI ─────────────────────────────────────────
    with tab5:
        st.subheader("🤖 Ask Anything About the Data")
        st.caption("Powered by Google Gemini 2.5 Flash")

        # Suggested questions as quick-launch buttons
        st.markdown("**Quick questions:**")
        q_cols = st.columns(3)
        suggested = [
            "Which market has a faster average lease-up and why?",
            "What characterises properties with the worst rent growth?",
            "How did the 2008 financial crisis affect lease-up times?",
        ]
        for i, q in enumerate(suggested):
            if q_cols[i].button(q, use_container_width=True):
                st.session_state["prefill_question"] = q

        st.markdown("---")

        # Text input (pre-filled if a quick button was clicked)
        prefill  = st.session_state.pop("prefill_question", "")
        question = st.text_input(
            "Or type your own question:",
            value=prefill,
            placeholder="e.g. Which properties took longer than 20 months to stabilize in Austin?",
        )

        if question:
            with st.spinner("Thinking…"):
                context = build_data_context(dff)
                full_prompt = f"{context}\n\nUser question: {question}"
                answer = ask_gemini(gemini, full_prompt)
            st.markdown("**Answer:**")
            st.markdown(answer)

            # Allow follow-up
            follow_up = st.text_input("Follow-up question:", key="follow_up")
            if follow_up:
                with st.spinner("Thinking…"):
                    fu_prompt = f"{context}\n\nPrevious question: {question}\nAnswer: {answer}\n\nFollow-up: {follow_up}"
                    fu_answer = ask_gemini(gemini, fu_prompt)
                st.markdown("**Answer:**")
                st.markdown(fu_answer)


if __name__ == "__main__":
    main()
