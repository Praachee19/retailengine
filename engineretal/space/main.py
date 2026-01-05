import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass

# -----------------------------
# Configuration and assumptions
# -----------------------------
@dataclass
class RetailAssumptions:
    shelf_width_cm: float = 800.0
    shelf_levels: int = 4
    min_facings: int = 1
    service_level_days: int = 7
    replenishment_cycle_days: int = 2
    target_osa: float = 0.97


# -----------------------------
# Synthetic data generation
# -----------------------------
def generate_synthetic_retail_data(
    n_stores: int = 40,
    n_skus: int = 120,
    n_weeks: int = 16,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)

    store_ids = [f"S{str(i).zfill(3)}" for i in range(1, n_stores + 1)]
    regions = rng.choice(["North", "South", "East", "West"], n_stores)
    formats = rng.choice(["Hyper", "Super", "Convenience"], n_stores)
    footfall_index = np.clip(rng.normal(1.0, 0.25, n_stores), 0.5, 1.8)
    affluence_index = np.clip(rng.normal(1.0, 0.20, n_stores), 0.6, 1.6)
    space_index = np.where(formats == "Hyper", 1.25, np.where(formats == "Super", 1.0, 0.7))

    stores_df = pd.DataFrame({
        "store_id": store_ids,
        "region": regions,
        "format": formats,
        "footfall_index": footfall_index,
        "affluence_index": affluence_index,
        "space_index": space_index
    })

    sku_ids = [f"SKU{str(i).zfill(4)}" for i in range(1, n_skus + 1)]
    categories = rng.choice(["Fresh", "Dairy", "Snacks", "Beverages", "Personal Care"], n_skus)
    segments = rng.choice(["Core", "Premium", "Value", "Impulse"], n_skus)
    unit_width_cm = np.clip(rng.normal(8, 2.5, n_skus), 4, 18)
    price = np.round(np.clip(rng.lognormal(3.2, 0.35, n_skus), 30, 600), 0)
    margin_rate = np.clip(rng.normal(0.28, 0.08, n_skus), 0.10, 0.55)
    base_weekly_demand = rng.lognormal(2.1, 0.8, n_skus)

    skus_df = pd.DataFrame({
        "sku_id": sku_ids,
        "category": categories,
        "segment": segments,
        "unit_width_cm": unit_width_cm,
        "price": price,
        "margin_rate": margin_rate,
        "base_weekly_demand": base_weekly_demand,
        "perishable": np.where(categories == "Fresh", 1, 0)
    })

    weeks = range(1, n_weeks + 1)
    rows = []

    for _, s in stores_df.iterrows():
        for _, k in skus_df.iterrows():
            for w in weeks:
                demand = k["base_weekly_demand"] * s["footfall_index"]
                oos = rng.random() < 0.08
                sales = demand * (0.55 if oos else 1.0)

                rows.append({
                    "store_id": s["store_id"],
                    "region": s["region"],
                    "format": s["format"],
                    "sku_id": k["sku_id"],
                    "category": k["category"],
                    "segment": k["segment"],
                    "week": w,
                    "price": k["price"],
                    "margin_rate": k["margin_rate"],
                    "unit_width_cm": k["unit_width_cm"],
                    "true_demand_units": demand,
                    "sales_units": sales,
                    "is_oos_event": int(oos),
                    "true_on_hand_units": rng.uniform(5, 40),
                    "system_on_hand_units": rng.uniform(5, 45),
                    "shelf_capacity_units": rng.uniform(10, 60)
                })

    sales_df = pd.DataFrame(rows)
    sales_df["sales_value"] = sales_df["sales_units"] * sales_df["price"]
    sales_df["gross_profit"] = sales_df["sales_value"] * sales_df["margin_rate"]

    return stores_df, skus_df, sales_df


# -----------------------------
# Core analytics
# -----------------------------
def build_store_features(sales_df):
    agg = sales_df.groupby(["store_id", "region", "format"], as_index=False).agg(
        sales_value=("sales_value", "sum"),
        units=("sales_units", "sum"),
        gross_profit=("gross_profit", "sum"),
        oos_rate=("is_oos_event", "mean")
    )

    sales_df["abs_record_error"] = abs(
        sales_df["system_on_hand_units"] - sales_df["true_on_hand_units"]
    )

    rec = sales_df.groupby("store_id", as_index=False).agg(
        avg_record_error_units=("abs_record_error", "mean"),
        avg_capacity_units=("shelf_capacity_units", "mean")
    )

    return agg.merge(rec, on="store_id", how="left")


def cluster_stores(store_feat, n_clusters=6, seed=42):
    X = store_feat.select_dtypes("number")
    Xs = StandardScaler().fit_transform(X)

    model = KMeans(
        n_clusters=int(np.clip(n_clusters, 2, len(store_feat))),
        random_state=seed,
        n_init="auto"
    )
    store_feat["store_cluster"] = model.fit_predict(Xs)
    return store_feat


def demand_forecast_simple(sales_df, lookback_weeks=6):
    recent = sales_df[sales_df["week"] > sales_df["week"].max() - lookback_weeks]
    return recent.groupby(
        ["store_id", "sku_id"],
        as_index=False
    )["sales_units"].mean().rename(
        columns={"sales_units": "forecast_units_weekly"}
    )


def recommend_assortment_and_facings(
    stores_df,
    skus_df,
    sales_df,
    store_clusters_df,
    assumptions,
    top_n_per_category=20
):

    df = sales_df.merge(
        store_clusters_df[["store_id", "store_cluster"]],
        on="store_id",
        how="left"
    )

    df = df.drop(
        columns=["category", "segment", "price", "margin_rate", "unit_width_cm"],
        errors="ignore"
    )

    df = df.merge(
        skus_df[["sku_id", "category", "segment", "unit_width_cm", "price", "margin_rate"]],
        on="sku_id",
        how="left",
        validate="m:1"
    )

    perf = df.groupby(
        ["store_cluster", "category", "sku_id"],
        as_index=False
    ).agg(
        sales_value=("sales_value", "sum"),
        gross_profit=("gross_profit", "sum"),
        oos_rate=("is_oos_event", "mean"),
        avg_width=("unit_width_cm", "mean")
    )

    perf["score"] = perf["gross_profit"] * (1 - 0.35 * perf["oos_rate"])
    perf["rank"] = perf.groupby(["store_cluster", "category"])["score"].rank(ascending=False)
    chosen = perf[perf["rank"] <= top_n_per_category].copy()

    fc = demand_forecast_simple(sales_df)
    fc = fc.merge(store_clusters_df[["store_id", "store_cluster"]], on="store_id")

    cluster_fc = fc.groupby(
        ["store_cluster", "sku_id"],
        as_index=False
    )["forecast_units_weekly"].mean()

    chosen = chosen.merge(
        cluster_fc,
        on=["store_cluster", "sku_id"],
        how="left"
    ).fillna({"forecast_units_weekly": 0})

    daily = chosen["forecast_units_weekly"] / 7
    target_units = daily * assumptions.service_level_days

    units_per_facing = np.where(
        chosen["avg_width"] <= 8, 6,
        np.where(chosen["avg_width"] <= 12, 4, 3)
    )

    chosen["recommended_facings"] = np.ceil(
        np.clip(target_units / units_per_facing, assumptions.min_facings, 40)
    ).astype(int)

    shelf_budget = assumptions.shelf_width_cm * assumptions.shelf_levels
    out = []

    for (cl, cat), grp in chosen.groupby(["store_cluster", "category"]):
        grp = grp.copy()
        grp["facing_width_cm"] = grp["recommended_facings"] * grp["avg_width"]

        if grp["facing_width_cm"].sum() > shelf_budget:
            scale = shelf_budget / grp["facing_width_cm"].sum()
            grp["recommended_facings"] = np.maximum(
                assumptions.min_facings,
                np.floor(grp["recommended_facings"] * scale)
            ).astype(int)

        grp["shelf_width_budget_cm"] = shelf_budget
        out.append(grp)

    rec_df = pd.concat(out, ignore_index=True)

    rec_df["shelf_level"] = (
        rec_df.groupby(["store_cluster", "category"]).cumcount()
        % assumptions.shelf_levels + 1
    )
    rec_df["position_index"] = (
        rec_df.groupby(["store_cluster", "category"]).cumcount() + 1
    )

    return rec_df


def compute_kpis(sales_df):
    df = sales_df.copy()
    df["record_error_units"] = abs(
        df["system_on_hand_units"] - df["true_on_hand_units"]
    )

    return {
        "total_sales_value": df["sales_value"].sum(),
        "total_gross_profit": df["gross_profit"].sum(),
        "inventory_record_accuracy_proxy": max(
            0, 1 - df["record_error_units"].mean() / df["true_on_hand_units"].mean()
        ),
        "oos_event_rate": df["is_oos_event"].mean(),
        "lost_sales_value_proxy": (
            (df["true_demand_units"] - df["sales_units"])
            .clip(lower=0)
            * df["price"]
        ).sum(),
        "understock_risk_rate": (
            df["true_on_hand_units"] < 0.25 * df["shelf_capacity_units"]
        ).mean(),
        "overstock_risk_rate": (
            df["true_on_hand_units"] > 1.25 * df["shelf_capacity_units"]
        ).mean()
    }


# -----------------------------
# PLANOGRAM VISUAL (ADDED)
# -----------------------------
def render_planogram_grid(rec_df, store_cluster, category):
    pg = rec_df[
        (rec_df["store_cluster"] == store_cluster) &
        (rec_df["category"] == category)
    ]

    if pg.empty:
        return None

    fig = px.scatter(
        pg,
        x="position_index",
        y="shelf_level",
        size="recommended_facings",
        color="score",
        text="sku_id",
        size_max=60,
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=520,
        showlegend=False
    )

    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Retail Space and Assortment Optimizer", layout="wide")
st.title("Retail Space and Assortment Optimizer")

assumptions = RetailAssumptions()
stores_df, skus_df, sales_df = generate_synthetic_retail_data()

store_feat = build_store_features(sales_df)
clustered = cluster_stores(store_feat)

rec_df = recommend_assortment_and_facings(
    stores_df,
    skus_df,
    sales_df,
    clustered,
    assumptions
)

kpis = compute_kpis(sales_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sales", f"{kpis['total_sales_value']:,.0f}")
c2.metric("Gross Profit", f"{kpis['total_gross_profit']:,.0f}")
c3.metric("Inventory Accuracy", f"{kpis['inventory_record_accuracy_proxy']*100:.1f}%")
c4.metric("OOS Rate", f"{kpis['oos_event_rate']*100:.1f}%")

st.divider()

left, right = st.columns([1.1, 0.9])

with left:
    fig = px.scatter(
        clustered,
        x="sales_value",
        y="gross_profit",
        color="store_cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Planogram visual grid")
    sel_cluster = st.selectbox("Cluster", sorted(rec_df["store_cluster"].unique()))
    sel_cat = st.selectbox("Category", sorted(rec_df["category"].unique()))

    fig_pg = render_planogram_grid(rec_df, sel_cluster, sel_cat)
    if fig_pg:
        st.plotly_chart(fig_pg, use_container_width=True)
