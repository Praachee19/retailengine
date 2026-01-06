# app.py
# Retail Space and Planogram Engine
# One-file, corrected, deployable Streamlit app with in-app explanation of fixes.

import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Optional PDF export. If reportlab is not installed, app still runs.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# Optional drag-drop reorder. If not installed, fall back to manual priority.
# pip install streamlit-sortables
try:
    from streamlit_sortables import sort_items
    SORTABLES_OK = True
except Exception:
    SORTABLES_OK = False


# =============================
# STREAMLIT CONFIG (MUST BE FIRST UI CALL)
# =============================
st.set_page_config(page_title="Retail Space and Planogram Engine", layout="wide")


# =============================
# Configuration and assumptions
# =============================
@dataclass
class RetailAssumptions:
    shelf_width_cm: float = 800.0
    shelf_levels: int = 4
    min_facings: int = 1
    service_level_days: int = 7
    replenishment_cycle_days: int = 2
    target_osa: float = 0.97

    brand_blocking: bool = True
    eye_level_shelves: Tuple[int, int] = (2, 3)  # map eye-level to shelves 2 and 3 by default


# =============================
# Templates and demo data
# =============================
def get_sales_input_template() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "store_id",
        "region",
        "format",
        "sku_id",
        "category",
        "segment",
        "brand",
        "week",
        "price",
        "margin_rate",
        "unit_width_cm",
        "sales_units",
        "is_oos_event",
        "true_on_hand_units",
        "system_on_hand_units",
        "shelf_capacity_units"
    ])


def get_new_launch_template() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "sku_id",
        "is_new_launch",
        "launch_week"
    ])


def get_sample_sales_data() -> pd.DataFrame:
    df = pd.DataFrame([
        ["S001", "North", "Super", "SKU0001", "Snacks", "Core", "Brand A", 12, 40, 0.30, 8.0, 120, 0, 25, 30, 60],
        ["S001", "North", "Super", "SKU0002", "Snacks", "Premium", "Brand B", 12, 65, 0.38, 9.0, 75, 1, 8, 15, 40],
        ["S001", "North", "Super", "SKU0001", "Snacks", "Core", "Brand A", 13, 40, 0.30, 8.0, 130, 0, 22, 28, 60],
        ["S002", "South", "Convenience", "SKU0003", "Beverages", "Value", "Brand C", 12, 30, 0.22, 10.0, 90, 0, 12, 18, 35],
        ["S002", "South", "Convenience", "SKU0004", "Beverages", "Core", "Brand C", 12, 45, 0.26, 9.5, 60, 1, 6, 12, 30],
    ], columns=get_sales_input_template().columns)
    return df


# =============================
# Schema validation
# =============================
def validate_sales_input(df: pd.DataFrame) -> None:
    required_cols = list(get_sales_input_template().columns)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error("Your file is missing required columns: " + ", ".join(missing))
        st.info("Download the template. Fill it. Upload again.")
        st.stop()

    numeric_cols = [
        "week", "price", "margin_rate", "unit_width_cm", "sales_units",
        "is_oos_event", "true_on_hand_units", "system_on_hand_units", "shelf_capacity_units"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad_num = df[numeric_cols].isna().any(axis=1)
    if bad_num.any():
        st.error("Some rows have non-numeric values where numbers are required.")
        st.write("Fix these rows and upload again.")
        st.dataframe(df.loc[bad_num, ["store_id", "sku_id"] + numeric_cols].head(50), use_container_width=True)
        st.stop()

    if not df["margin_rate"].between(0, 1).all():
        st.error("margin_rate must be between 0 and 1. Example: 0.32")
        st.stop()

    if not df["is_oos_event"].isin([0, 1]).all():
        st.error("is_oos_event must be 0 or 1 only.")
        st.stop()

    if (df["unit_width_cm"] <= 0).any():
        st.error("unit_width_cm must be greater than 0.")
        st.stop()

    if (df["week"] <= 0).any():
        st.error("week must be a positive integer.")
        st.stop()

    if (df["sales_units"] < 0).any():
        st.error("sales_units cannot be negative.")
        st.stop()

    allowed_formats = {"Hyper", "Super", "Convenience"}
    if not set(df["format"].dropna().unique()).issubset(allowed_formats):
        st.warning("format contains values outside Hyper, Super, Convenience. Clustering may be noisier.")

    if df["brand"].isna().any() or (df["brand"].astype(str).str.strip() == "").any():
        st.error("brand is required because brand blocking is enabled. Fill brand for every row.")
        st.stop()


def validate_new_launch_input(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["sku_id", "is_new_launch", "launch_week"])

    needed = ["sku_id", "is_new_launch", "launch_week"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error("New launch file is missing columns: " + ", ".join(missing))
        st.info("Download the new launch template. Fill it. Upload again.")
        st.stop()

    out = df.copy()
    out["is_new_launch"] = pd.to_numeric(out["is_new_launch"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out["launch_week"] = pd.to_numeric(out["launch_week"], errors="coerce").fillna(0).astype(int)
    out["sku_id"] = out["sku_id"].astype(str)
    return out[needed]


# =============================
# Synthetic data generation
# =============================
def generate_synthetic_retail_data(
    n_stores: int = 40,
    n_skus: int = 120,
    n_weeks: int = 16,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    store_ids = [f"S{str(i).zfill(3)}" for i in range(1, n_stores + 1)]
    regions = rng.choice(["North", "South", "East", "West"], size=n_stores)
    formats = rng.choice(["Hyper", "Super", "Convenience"], size=n_stores, p=[0.25, 0.50, 0.25])
    footfall_index = np.clip(rng.normal(1.0, 0.25, size=n_stores), 0.5, 1.8)
    affluence_index = np.clip(rng.normal(1.0, 0.20, size=n_stores), 0.6, 1.6)
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
    categories = rng.choice(
        ["Fresh", "Dairy", "Snacks", "Beverages", "Personal Care"],
        size=n_skus,
        p=[0.20, 0.20, 0.25, 0.20, 0.15]
    )
    segments = rng.choice(["Core", "Premium", "Value", "Impulse"], size=n_skus, p=[0.45, 0.20, 0.25, 0.10])
    brands = rng.choice(
        ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E"],
        size=n_skus,
        p=[0.28, 0.22, 0.20, 0.18, 0.12]
    )

    unit_width_cm = np.clip(rng.normal(8, 2.5, size=n_skus), 4, 18)
    price = np.round(np.clip(rng.lognormal(mean=3.2, sigma=0.35, size=n_skus), 30, 600), 0)
    base_margin = np.clip(rng.normal(0.28, 0.08, size=n_skus), 0.10, 0.55)
    base_demand = rng.lognormal(mean=2.1, sigma=0.8, size=n_skus)
    perishable = np.where(categories == "Fresh", 1, 0)

    is_new_launch = (rng.random(n_skus) < 0.08).astype(int)
    launch_week = np.where(
        is_new_launch == 1,
        rng.integers(low=max(1, n_weeks // 2), high=n_weeks + 1, size=n_skus),
        0
    )

    skus_df = pd.DataFrame({
        "sku_id": sku_ids,
        "category": categories,
        "segment": segments,
        "brand": brands,
        "unit_width_cm": unit_width_cm,
        "price": price,
        "margin_rate": base_margin,
        "base_weekly_demand": base_demand,
        "perishable": perishable
    })

    new_launch_df = pd.DataFrame({
        "sku_id": sku_ids,
        "is_new_launch": is_new_launch,
        "launch_week": launch_week
    })

    weeks = list(range(1, n_weeks + 1))
    rows = []

    cat_season = {
        "Fresh": 0.08,
        "Dairy": 0.05,
        "Snacks": 0.10,
        "Beverages": 0.12,
        "Personal Care": 0.03
    }

    nl_map = new_launch_df.set_index("sku_id")[["is_new_launch", "launch_week"]].to_dict("index")
    mean_base = float(np.mean(skus_df["base_weekly_demand"])) + 1e-9

    for _, s in stores_df.iterrows():
        store_multiplier = float(s["footfall_index"]) * (0.85 + 0.25 * float(s["affluence_index"]))
        format_multiplier = 1.15 if s["format"] == "Hyper" else (1.0 if s["format"] == "Super" else 0.75)

        for _, k in skus_df.iterrows():
            sku_multiplier = (0.9 + 0.25 * float(k["margin_rate"])) * (1.05 if k["segment"] == "Core" else 0.90)
            cat_multiplier = 1.0 + cat_season.get(str(k["category"]), 0.05)

            nl = nl_map[str(k["sku_id"])]
            nl_launch_week = int(nl["launch_week"])
            nl_flag = int(nl["is_new_launch"])

            for w in weeks:
                season = 1.0 + cat_season.get(str(k["category"]), 0.05) * np.sin(2 * np.pi * (w / max(weeks)))
                noise = float(rng.normal(1.0, 0.20))

                launch_lift = 1.0
                if nl_flag == 1 and nl_launch_week > 0 and w >= nl_launch_week:
                    age = w - nl_launch_week
                    launch_lift = 1.25 if age <= 2 else (1.12 if age <= 5 else 1.03)

                true_units = max(
                    0.0,
                    float(k["base_weekly_demand"]) * store_multiplier * format_multiplier * sku_multiplier * cat_multiplier * season * noise * launch_lift
                )

                oos_prob = np.clip(
                    0.02 + 0.02 * (true_units / mean_base) + (0.03 if s["format"] == "Convenience" else 0.0),
                    0.02, 0.20
                )
                is_oos = bool(rng.random() < float(oos_prob))
                observed_units = true_units * (0.55 if is_oos else 1.0)

                shrink = float(rng.binomial(1, 0.015) * rng.uniform(0.0, 0.08))
                record_error = float(rng.normal(0.0, 0.18))

                true_on_hand_units = max(0.0, float(rng.normal(20, 10)) + (10 if s["format"] != "Convenience" else 0))
                system_on_hand_units = max(0.0, true_on_hand_units * (1.0 + record_error - shrink))

                shelf_capacity_units = max(4.0, (60.0 * float(s["space_index"])) / max(1.0, float(k["unit_width_cm"]) / 8.0))

                rows.append({
                    "store_id": str(s["store_id"]),
                    "region": str(s["region"]),
                    "format": str(s["format"]),
                    "sku_id": str(k["sku_id"]),
                    "category": str(k["category"]),
                    "segment": str(k["segment"]),
                    "brand": str(k["brand"]),
                    "week": int(w),
                    "price": float(k["price"]),
                    "margin_rate": float(k["margin_rate"]),
                    "unit_width_cm": float(k["unit_width_cm"]),
                    "true_demand_units": float(true_units),
                    "sales_units": float(observed_units),
                    "is_oos_event": int(is_oos),
                    "true_on_hand_units": float(true_on_hand_units),
                    "system_on_hand_units": float(system_on_hand_units),
                    "shelf_capacity_units": float(shelf_capacity_units),
                })

    sales_df = pd.DataFrame(rows)
    sales_df["sales_value"] = sales_df["sales_units"] * sales_df["price"]
    sales_df["gross_profit"] = sales_df["sales_value"] * sales_df["margin_rate"]

    return stores_df, skus_df, sales_df, new_launch_df


# =============================
# Core analytics
# =============================
def build_store_features(sales_df: pd.DataFrame) -> pd.DataFrame:
    agg = sales_df.groupby(["store_id", "region", "format"], as_index=False).agg(
        sales_value=("sales_value", "sum"),
        units=("sales_units", "sum"),
        gross_profit=("gross_profit", "sum"),
        oos_rate=("is_oos_event", "mean"),
    )

    cat_mix = sales_df.groupby(["store_id", "category"], as_index=False)["sales_value"].sum()
    cat_total = cat_mix.groupby("store_id", as_index=False)["sales_value"].sum().rename(columns={"sales_value": "total"})
    cat_mix = cat_mix.merge(cat_total, on="store_id", how="left")
    cat_mix["share"] = cat_mix["sales_value"] / (cat_mix["total"] + 1e-9)
    cat_pivot = cat_mix.pivot(index="store_id", columns="category", values="share").fillna(0.0).reset_index()

    feat = agg.merge(cat_pivot, on="store_id", how="left")

    sales_df2 = sales_df.copy()
    sales_df2["abs_record_error"] = np.abs(sales_df2["system_on_hand_units"] - sales_df2["true_on_hand_units"])
    rec = sales_df2.groupby("store_id", as_index=False).agg(
        avg_record_error_units=("abs_record_error", "mean"),
        avg_capacity_units=("shelf_capacity_units", "mean")
    )
    feat = feat.merge(rec, on="store_id", how="left")
    return feat


def cluster_stores(store_feat: pd.DataFrame, n_clusters: int = 6, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = store_feat.copy()
    numeric_cols = [c for c in df.columns if c not in ["store_id", "region", "format"]]
    X = df[numeric_cols].values.astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_clusters = int(np.clip(n_clusters, 2, min(12, len(df))))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    df["store_cluster"] = model.fit_predict(Xs)

    centers = pd.DataFrame(model.cluster_centers_, columns=numeric_cols)
    centers["store_cluster"] = range(n_clusters)
    centers = centers.merge(
        df.groupby("store_cluster", as_index=False)["store_id"].count().rename(columns={"store_id": "stores_in_cluster"}),
        on="store_cluster",
        how="left"
    )
    return df, centers


def demand_forecast_simple(sales_df: pd.DataFrame, lookback_weeks: int = 6) -> pd.DataFrame:
    df = sales_df.copy()
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["store_id", "sku_id", "week"], ascending=[True, True, True])

    max_w = int(df["week"].max())
    recent = df[df["week"] > max_w - int(lookback_weeks)].copy()

    g = recent.groupby(["store_id", "sku_id"], as_index=False).agg(
        avg_units=("sales_units", "mean"),
        last_units=("sales_units", lambda x: float(x.iloc[-1]))
    )
    g["forecast_units_weekly"] = np.clip(0.85 * g["avg_units"] + 0.15 * g["last_units"], 0.0, None)
    return g[["store_id", "sku_id", "forecast_units_weekly"]]


def _apply_brand_blocking_order(df: pd.DataFrame) -> pd.DataFrame:
    # FIX: remove unreachable dead code. Only do ordering here.
    if "brand" not in df.columns:
        return df.sort_values("score", ascending=False)
    return df.sort_values(["brand", "score"], ascending=[True, False])


def recommend_assortment_and_facings(
    stores_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    store_clusters_df: pd.DataFrame,
    assumptions: RetailAssumptions,
    top_n_per_category: int = 20,
    new_launch_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    # Merge store cluster
    df = sales_df.merge(
        store_clusters_df[["store_id", "store_cluster"]],
        on="store_id",
        how="left",
        validate="m:1"
    )

    # Remove duplicate SKU attributes BEFORE merging sku master
    # FIX: also drop brand to avoid brand_x/brand_y suffix problem
    df = df.drop(columns=["category", "segment", "price", "margin_rate", "unit_width_cm", "brand"], errors="ignore")

    # Merge SKU master cleanly
    needed_sku_cols = ["sku_id", "category", "segment", "brand", "unit_width_cm", "price", "margin_rate"]
    if "perishable" in skus_df.columns:
        needed_sku_cols.append("perishable")

    df = df.merge(
        skus_df[needed_sku_cols].drop_duplicates("sku_id"),
        on="sku_id",
        how="left",
        validate="m:1"
    )

    # FIX: defensive recovery in case any suffixes still appear due to unexpected input
    if "brand" not in df.columns:
        if "brand_y" in df.columns or "brand_x" in df.columns:
            df["brand"] = df.get("brand_y")
            if "brand_x" in df.columns:
                df["brand"] = df["brand"].fillna(df["brand_x"])
            df = df.drop(columns=["brand_x", "brand_y"], errors="ignore")

    # Safety check
    required_cols = ["store_cluster", "category", "sku_id", "brand", "unit_width_cm", "price", "margin_rate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after merge: {missing}")

    # Cluster SKU performance
    perf = df.groupby(["store_cluster", "category", "sku_id"], as_index=False).agg(
        sales_value=("sales_value", "sum"),
        gross_profit=("gross_profit", "sum"),
        oos_rate=("is_oos_event", "mean"),
        avg_width=("unit_width_cm", "mean"),
        brand=("brand", "first"),
        segment=("segment", "first"),
    )

    # New launch boost
    nl = new_launch_df.copy() if (new_launch_df is not None) else pd.DataFrame(columns=["sku_id", "is_new_launch", "launch_week"])
    if not nl.empty:
        nl["sku_id"] = nl["sku_id"].astype(str)
        nl["is_new_launch"] = pd.to_numeric(nl["is_new_launch"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    perf = perf.merge(nl[["sku_id", "is_new_launch"]] if "is_new_launch" in nl.columns else nl, on="sku_id", how="left")
    perf["is_new_launch"] = perf.get("is_new_launch", 0)
    perf["is_new_launch"] = pd.to_numeric(perf["is_new_launch"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # Score
    perf["score"] = perf["gross_profit"] * (1.0 - 0.35 * np.clip(perf["oos_rate"], 0, 1))
    perf["score"] = np.where(perf["is_new_launch"] == 1, perf["score"] * 1.12, perf["score"])

    perf["rank"] = perf.groupby(["store_cluster", "category"])["score"].rank(ascending=False, method="first")
    chosen = perf[perf["rank"] <= int(top_n_per_category)].copy()

    # Demand forecast for facings
    fc = demand_forecast_simple(sales_df)
    df_fc = sales_df.merge(fc, on=["store_id", "sku_id"], how="left")
    cluster_fc = df_fc.merge(store_clusters_df[["store_id", "store_cluster"]], on="store_id", how="left")
    cluster_fc = cluster_fc.groupby(["store_cluster", "sku_id"], as_index=False)["forecast_units_weekly"].mean()

    chosen = chosen.merge(cluster_fc, on=["store_cluster", "sku_id"], how="left")
    chosen["forecast_units_weekly"] = chosen["forecast_units_weekly"].fillna(0.0)

    daily = chosen["forecast_units_weekly"] / 7.0
    target_units = daily * float(assumptions.service_level_days)

    units_per_facing = np.where(chosen["avg_width"] <= 8, 6.0, np.where(chosen["avg_width"] <= 12, 4.0, 3.0))
    raw_facings = target_units / (units_per_facing + 1e-9)
    chosen["recommended_facings"] = np.ceil(np.clip(raw_facings, assumptions.min_facings, 40)).astype(int)

    # Shelf width constraint per cluster x category
    shelf_total_width = float(assumptions.shelf_width_cm) * int(assumptions.shelf_levels)

    out = []
    for (cl, cat), grp in chosen.groupby(["store_cluster", "category"]):
        grp = grp.copy()
        grp["facing_width_cm"] = grp["recommended_facings"] * grp["avg_width"]
        total_width = float(grp["facing_width_cm"].sum())

        if total_width > shelf_total_width and total_width > 0:
            scale = shelf_total_width / total_width
            grp["recommended_facings"] = np.maximum(
                assumptions.min_facings,
                np.floor(grp["recommended_facings"] * scale)
            ).astype(int)
            grp["facing_width_cm"] = grp["recommended_facings"] * grp["avg_width"]

        grp["shelf_width_budget_cm"] = shelf_total_width
        out.append(grp)

    rec_df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    if rec_df.empty:
        return rec_df

    # Planogram placement fields
    rec_df = rec_df.sort_values(["store_cluster", "category", "score"], ascending=[True, True, False]).reset_index(drop=True)

    # Eye level shelf allocation
    eye_a, eye_b = assumptions.eye_level_shelves
    levels = list(range(1, int(assumptions.shelf_levels) + 1))
    if eye_a not in levels:
        eye_a = 2 if 2 in levels else 1
    if eye_b not in levels:
        eye_b = 3 if 3 in levels else min(levels[-1], eye_a + 1)

    def assign_shelf_level(idx: int) -> int:
        if idx < 6:
            return int(eye_a)
        if idx < 12:
            return int(eye_b)
        other = [l for l in levels if l not in (eye_a, eye_b)]
        if not other:
            return int(eye_a)
        return int(other[(idx - 12) % len(other)])

    rec_df["within_group_rank"] = rec_df.groupby(["store_cluster", "category"]).cumcount()
    rec_df["shelf_level"] = rec_df["within_group_rank"].apply(assign_shelf_level).astype(int)

    # Position index within shelf, with brand blocking
    rec_df["position_index"] = 0
    for (cl, cat, sl), grp in rec_df.groupby(["store_cluster", "category", "shelf_level"]):
        grp2 = grp.copy()
        grp2 = _apply_brand_blocking_order(grp2) if assumptions.brand_blocking else grp2.sort_values("score", ascending=False)
        grp2["position_index"] = range(1, len(grp2) + 1)
        rec_df.loc[grp2.index, "position_index"] = grp2["position_index"].values

    rec_df = rec_df.drop(columns=["within_group_rank"], errors="ignore")
    return rec_df


def compute_kpis(sales_df: pd.DataFrame) -> dict:
    df = sales_df.copy()

    df["record_error_units"] = np.abs(df["system_on_hand_units"] - df["true_on_hand_units"])
    inv_accuracy = 1.0 - (df["record_error_units"].mean() / (df["true_on_hand_units"].mean() + 1e-9))
    inv_accuracy = float(np.clip(inv_accuracy, 0.0, 1.0))

    oos_rate = float(df["is_oos_event"].mean())

    if "true_demand_units" not in df.columns:
        df["true_demand_units"] = df["sales_units"] * (1.0 + 0.10 * df["is_oos_event"])

    df["lost_units_proxy"] = np.where(df["is_oos_event"] == 1, df["true_demand_units"] - df["sales_units"], 0.0)
    lost_value_proxy = float((df["lost_units_proxy"] * df["price"]).sum())

    total_sales = float(df["sales_value"].sum())
    total_gp = float(df["gross_profit"].sum())

    df["understock_risk"] = np.where(df["true_on_hand_units"] < 0.25 * df["shelf_capacity_units"], 1, 0)
    df["overstock_risk"] = np.where(df["true_on_hand_units"] > 1.25 * df["shelf_capacity_units"], 1, 0)

    return {
        "inventory_record_accuracy_proxy": inv_accuracy,
        "oos_event_rate": oos_rate,
        "lost_sales_value_proxy": lost_value_proxy,
        "total_sales_value": total_sales,
        "total_gross_profit": total_gp,
        "understock_risk_rate": float(df["understock_risk"].mean()),
        "overstock_risk_rate": float(df["overstock_risk"].mean()),
    }


# =============================
# Planogram visual grid
# =============================
def build_shelf_slots(view: pd.DataFrame, max_slots: int = 60) -> pd.DataFrame:
    slots = []
    for _, r in view.iterrows():
        facings = int(max(1, r.get("recommended_facings", 1)))
        facings = min(facings, 40)
        for _i in range(facings):
            slots.append({
                "shelf_level": int(r["shelf_level"]),
                "brand": str(r.get("brand", "")),
                "sku_id": str(r["sku_id"]),
                "score": float(r.get("score", 0.0))
            })

    if not slots:
        return pd.DataFrame(columns=["shelf_level", "slot", "sku_id", "brand"])

    slots_df = pd.DataFrame(slots)

    out_rows = []
    for sl, g in slots_df.groupby("shelf_level"):
        g2 = g.copy().reset_index(drop=True)
        g2["slot"] = range(1, len(g2) + 1)
        out_rows.append(g2)

    slots_df = pd.concat(out_rows, ignore_index=True)
    slots_df = slots_df[slots_df["slot"] <= int(max_slots)]
    return slots_df[["shelf_level", "slot", "sku_id", "brand"]]


def plot_planogram_grid(slots_df: pd.DataFrame, shelf_levels: int, max_slots: int) -> go.Figure:
    rows = list(range(int(shelf_levels), 0, -1))
    cols = list(range(1, int(max_slots) + 1))

    text = []
    z = []
    for r in rows:
        row_text = []
        row_z = []
        row_df = slots_df[slots_df["shelf_level"] == r].set_index("slot") if not slots_df.empty else pd.DataFrame()
        for c in cols:
            if not row_df.empty and c in row_df.index:
                sku = str(row_df.loc[c, "sku_id"])
                brand = str(row_df.loc[c, "brand"])
                row_text.append(f"{sku}<br>{brand}".strip())
                row_z.append(1)
            else:
                row_text.append("")
                row_z.append(0)
        text.append(row_text)
        z.append(row_z)

    fig = go.Figure(data=go.Heatmap(z=z, text=text, hoverinfo="text", showscale=False))
    fig.update_layout(
        title="Planogram visual grid. Shelf top to bottom, left to right slots",
        xaxis_title="Slots (left to right)",
        yaxis_title="Shelf level (top to bottom)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(rows))),
            ticktext=[str(r) for r in rows]
        ),
    )
    return fig


# =============================
# Drag and drop simulation
# =============================
def simulate_drag_reorder_skus(view: pd.DataFrame) -> pd.DataFrame:
    st.caption("Reorder SKUs to simulate local merchandiser adjustments before export.")

    base = view.copy()
    base["label"] = base.apply(
        lambda r: f"{r['sku_id']} | {r.get('brand','')} | facings {int(r.get('recommended_facings',1))}",
        axis=1
    )

    if SORTABLES_OK:
        items = base["label"].tolist()
        reordered = sort_items(items, direction="vertical")
        order_map = {lab: i for i, lab in enumerate(reordered)}
        base["manual_rank"] = base["label"].map(order_map).fillna(999999).astype(int)
        return base.sort_values(["manual_rank", "score"], ascending=[True, False]).drop(columns=["label"])
    else:
        st.info("Drag and drop add-on not installed. Use manual_rank in the table below.")
        base["manual_rank"] = list(range(1, len(base) + 1))
        edited = st.data_editor(
            base[["sku_id", "brand", "recommended_facings", "score", "manual_rank"]],
            use_container_width=True,
            hide_index=True
        )
        edited["manual_rank"] = pd.to_numeric(edited["manual_rank"], errors="coerce").fillna(999999).astype(int)
        merged = base.drop(columns=["manual_rank"], errors="ignore").merge(
            edited[["sku_id", "manual_rank"]], on="sku_id", how="left"
        )
        return merged.sort_values(["manual_rank", "score"], ascending=[True, False]).drop(columns=["label"], errors="ignore")


# =============================
# PDF Export
# =============================
def export_planogram_pdf(
    view: pd.DataFrame,
    slots_df: pd.DataFrame,
    cluster_id: int,
    category: str,
    assumptions: RetailAssumptions
) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("PDF export requires reportlab. Add reportlab to requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _width, height = A4

    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Planogram Export")
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Cluster: {cluster_id}  Category: {category}")
    y -= 14
    c.drawString(40, y, f"Eye level shelves: {assumptions.eye_level_shelves[0]} and {assumptions.eye_level_shelves[1]}  Brand blocking: {assumptions.brand_blocking}")
    y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Shelf instructions. Execute shelf by shelf, left to right slots.")
    y -= 16

    c.setFont("Helvetica", 9)
    grouped = slots_df.groupby("shelf_level")
    for sl in sorted(grouped.groups.keys(), reverse=True):
        cells = grouped.get_group(sl).sort_values("slot")
        line = f"Shelf {sl}: " + " | ".join([f"{int(r['slot'])}:{r['sku_id']}" for _, r in cells.iterrows()])
        for chunk in [line[i:i + 110] for i in range(0, len(line), 110)]:
            c.drawString(40, y, chunk)
            y -= 12
            if y < 60:
                c.showPage()
                y = height - 40
                c.setFont("Helvetica", 9)
        y -= 4

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "SKU list with facings")
    y -= 14
    c.setFont("Helvetica", 9)

    show_cols = ["sku_id", "brand", "recommended_facings", "shelf_level", "position_index"]
    view2 = view.copy().sort_values(["shelf_level", "position_index"], ascending=[True, True])

    for _, r in view2[show_cols].head(140).iterrows():
        line = f"{r['sku_id']}  Brand {r.get('brand','')}  Facings {int(r['recommended_facings'])}  Shelf {int(r['shelf_level'])}  Pos {int(r['position_index'])}"
        c.drawString(40, y, line)
        y -= 11
        if y < 60:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 9)

    c.save()
    return buf.getvalue()


# =============================
# Streamlit UI
# =============================
st.title("Retail Space and Planogram Engine")
st.caption("Automated shelf planning, data-driven placement, real-time shelf control, centralized planogram standards.")

with st.expander("What was fixed in this version (so it runs error-free)"):
    st.markdown(
        """
1. `st.set_page_config()` is called exactly once, at the top. Streamlit fails if you call it twice.
2. `_apply_brand_blocking_order()` now only does ordering. The old version had unreachable code after a `return`.
3. `recommend_assortment_and_facings()` is called with the required `stores_df` argument. Missing it causes runtime failure.
4. Forecast uses a sorted week order so `last_units` is the true last week in the lookback window.
5. Safer merges. SKU master merge includes `brand` so brand blocking always has data.
6. FIX: brand column is protected from pandas merge suffixing (brand_x/brand_y). Brand always exists after merge.
"""
    )

st.subheader("Downloads. Templates and demo files")
col_t1, col_t2, col_t3, col_t4 = st.columns(4)

with col_t1:
    st.download_button(
        "Sales input template CSV",
        get_sales_input_template().to_csv(index=False).encode("utf-8"),
        file_name="retail_sales_input_template.csv",
        mime="text/csv"
    )
with col_t2:
    st.download_button(
        "New launch template CSV",
        get_new_launch_template().to_csv(index=False).encode("utf-8"),
        file_name="new_launch_input_template.csv",
        mime="text/csv"
    )
with col_t3:
    st.download_button(
        "Sample demo sales CSV",
        get_sample_sales_data().to_csv(index=False).encode("utf-8"),
        file_name="sample_retail_sales_demo.csv",
        mime="text/csv"
    )
with col_t4:
    req = "\n".join(["streamlit", "numpy", "pandas", "scikit-learn", "plotly", "reportlab", "streamlit-sortables"])
    st.download_button(
        "requirements.txt",
        req.encode("utf-8"),
        file_name="requirements.txt",
        mime="text/plain"
    )

st.divider()

with st.sidebar:
    st.header("Data input")
    use_synth = st.toggle("Use synthetic data", value=True)
    seed = st.number_input("Synthetic seed", min_value=1, max_value=9999, value=42, step=1)

    if use_synth:
        n_stores = st.slider("Stores", 10, 120, 40, 5)
        n_skus = st.slider("SKUs", 50, 400, 120, 10)
        n_weeks = st.slider("Weeks of history", 8, 52, 16, 4)
        uploaded = None
    else:
        uploaded = st.file_uploader(
            "Upload sales CSV",
            type=["csv"],
            help="Use the template. One row per store x SKU x week."
        )

    st.divider()
    st.header("Optional. New launches")
    uploaded_launch = st.file_uploader(
        "Upload new launch CSV (optional)",
        type=["csv"],
        help="Overrides simulated new launches. Use the new launch template."
    )

    st.divider()
    st.header("Optimization controls")
    n_clusters = st.slider("Store clusters", 2, 12, 6, 1)
    top_n = st.slider("Top SKUs per category per cluster", 5, 60, 20, 5)

    st.divider()
    st.header("Shelf assumptions")
    shelf_width = st.number_input("Shelf width per level (cm)", 200.0, 2000.0, 800.0, 50.0)
    shelf_levels = st.slider("Shelf levels", 2, 8, 4, 1)

    eye_level_mode = st.radio("Eye-level shelves", ["Shelf 2 and 3", "Only shelf 3"], index=0)
    eye_level = (2, 3) if eye_level_mode == "Shelf 2 and 3" else (3, 3)

    service_days = st.slider("Days of shelf cover target", 1, 21, 7, 1)
    brand_blocking = st.toggle("Brand blocking", value=True)

assumptions = RetailAssumptions(
    shelf_width_cm=float(shelf_width),
    shelf_levels=int(shelf_levels),
    service_level_days=int(service_days),
    brand_blocking=bool(brand_blocking),
    eye_level_shelves=eye_level
)

# Load data
if use_synth:
    stores_df, skus_df, sales_df, new_launch_df_sim = generate_synthetic_retail_data(
        n_stores=int(n_stores),
        n_skus=int(n_skus),
        n_weeks=int(n_weeks),
        seed=int(seed)
    )
    new_launch_df = new_launch_df_sim.copy()
else:
    if uploaded is None:
        st.warning("Upload a CSV, or enable synthetic data.")
        st.stop()

    sales_df = pd.read_csv(uploaded)
    validate_sales_input(sales_df)

    stores_df = sales_df[["store_id", "region", "format"]].drop_duplicates().reset_index(drop=True)
    skus_df = sales_df[["sku_id", "category", "segment", "brand", "price", "margin_rate", "unit_width_cm"]].drop_duplicates().reset_index(drop=True)

    if "true_demand_units" not in sales_df.columns:
        sales_df["true_demand_units"] = sales_df["sales_units"] * (1.0 + 0.10 * sales_df["is_oos_event"])

    sales_df["sales_value"] = sales_df["sales_units"] * sales_df["price"]
    sales_df["gross_profit"] = sales_df["sales_value"] * sales_df["margin_rate"]

    new_launch_df = pd.DataFrame(columns=["sku_id", "is_new_launch", "launch_week"])

# Override new launches from upload
if uploaded_launch is not None:
    try:
        nl_up = pd.read_csv(uploaded_launch)
        nl_up = validate_new_launch_input(nl_up)
        new_launch_df = nl_up.copy()
    except Exception:
        st.error("Could not read the new launch file. Use the template and upload again.")
        st.stop()

# Feature build and clustering
store_feat = build_store_features(sales_df)
clustered, cluster_centers = cluster_stores(store_feat, n_clusters=int(n_clusters), seed=int(seed))

# Recommendations (FIX: include stores_df)
rec_df = recommend_assortment_and_facings(
    stores_df=stores_df,
    skus_df=skus_df,
    sales_df=sales_df,
    store_clusters_df=clustered,
    new_launch_df=new_launch_df,
    assumptions=assumptions,
    top_n_per_category=int(top_n)
)

# KPIs
kpis = compute_kpis(sales_df)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Sales value", f"{kpis['total_sales_value']:,.0f}")
c2.metric("Gross profit", f"{kpis['total_gross_profit']:,.0f}")
c3.metric("Inventory accuracy proxy", f"{100 * kpis['inventory_record_accuracy_proxy']:.1f}%")
c4.metric("OOS event rate", f"{100 * kpis['oos_event_rate']:.1f}%")
c5.metric("Lost sales proxy", f"{kpis['lost_sales_value_proxy']:,.0f}")
c6.metric("Understock risk", f"{100 * kpis['understock_risk_rate']:.1f}%")
c7.metric("Overstock risk", f"{100 * kpis['overstock_risk_rate']:.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Analytics and clustering",
    "Recommendations and planogram",
    "Store execution view (mobile style)",
    "Exports"
])

with tab1:
    st.subheader("Store clustering and diagnostics")

    left, right = st.columns([1.2, 0.8])

    with left:
        fig = px.scatter(
            clustered,
            x="sales_value",
            y="gross_profit",
            color="store_cluster",
            hover_data=["store_id", "region", "format", "oos_rate", "avg_record_error_units"],
            title="Store clusters. Sales vs gross profit"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(
            clustered,
            x="store_cluster",
            y="oos_rate",
            points="all",
            title="OOS event distribution by cluster"
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(
            clustered,
            x="store_cluster",
            y="avg_record_error_units",
            points="all",
            title="Inventory record error units by cluster"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with right:
        st.markdown("### Clustering basis. How it works")
        st.markdown(
            """
Stores are grouped into archetypes so you can create planograms once per archetype and deploy at scale.

**Features used**
- Sales value, units, gross profit
- OOS rate
- Category sales mix shares
- Inventory record error (accuracy proxy)
- Shelf capacity proxy (space proxy)

**Method**
- Standardize numeric features so no metric dominates
- KMeans creates stable store groups
"""
        )
        st.markdown("### Cluster summary (centroids, standardized space)")
        st.dataframe(cluster_centers.head(12), use_container_width=True, height=380)

with tab2:
    st.subheader("Assortment and planogram builder")

    if rec_df is None or rec_df.empty:
        st.warning("No recommendations generated. Check your input data.")
        st.stop()

    colA, colB, colC = st.columns([0.35, 0.35, 0.30])

    clusters = sorted(rec_df["store_cluster"].unique().tolist())
    cats = sorted(rec_df["category"].unique().tolist())

    with colA:
        sel_cluster = st.selectbox("Cluster", clusters, index=0)
    with colB:
        sel_cat = st.selectbox("Category", cats, index=0)
    with colC:
        max_slots = st.slider("Grid slots per shelf (visual)", 20, 120, 60, 10)

    view = rec_df[(rec_df["store_cluster"] == sel_cluster) & (rec_df["category"] == sel_cat)].copy()
    view = view.sort_values(["shelf_level", "position_index"], ascending=[True, True])

    st.markdown("### Local adjustment simulation")
    view2 = simulate_drag_reorder_skus(view)

    # Recompute position_index based on reordered list, within each shelf
    view2 = view2.copy()
    view2["position_index"] = 0
    for sl, grp in view2.groupby("shelf_level"):
        grp2 = grp.copy().reset_index(drop=True)
        grp2["position_index"] = range(1, len(grp2) + 1)
        view2.loc[grp.index, "position_index"] = grp2["position_index"].values

    st.divider()

    st.markdown("### Recommendation table")
    st.dataframe(
        view2[[
            "sku_id", "brand", "segment",
            "score", "gross_profit", "sales_value", "oos_rate",
            "forecast_units_weekly", "recommended_facings", "avg_width",
            "shelf_level", "position_index"
        ]].head(80),
        use_container_width=True,
        height=420
    )

    used = float((view2["recommended_facings"] * view2["avg_width"]).sum()) if len(view2) else 0.0
    budget = float(view2["shelf_width_budget_cm"].iloc[0]) if len(view2) else 0.0

    fig_space = go.Figure()
    fig_space.add_trace(go.Indicator(
        mode="gauge+number",
        value=used,
        title={"text": "Shelf width used (cm)"},
        gauge={
            "axis": {"range": [0, max(budget, used, 1.0)]},
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": budget}
        }
    ))
    st.plotly_chart(fig_space, use_container_width=True)

    st.divider()

    st.markdown("### Planogram visual grid")
    st.caption("Each slot is one facing. Shelves are shown top to bottom. Slots go left to right.")
    slots_df = build_shelf_slots(view2, max_slots=int(max_slots))
    fig_grid = plot_planogram_grid(slots_df, shelf_levels=int(assumptions.shelf_levels), max_slots=int(max_slots))
    st.plotly_chart(fig_grid, use_container_width=True)

    st.divider()

    st.markdown("### Planogram task list")
    task = view2.copy()
    task["task"] = np.where(task["oos_rate"] > 0.10, "Increase facings or fix replenishment", "Maintain")
    st.dataframe(
        task.sort_values(["shelf_level", "position_index"], ascending=[True, True])[
            ["shelf_level", "position_index", "brand", "sku_id", "recommended_facings", "oos_rate", "task"]
        ].head(120),
        use_container_width=True,
        height=420
    )

with tab3:
    st.subheader("Store execution view")
    st.caption("Simple view. Store teams should not need to interpret charts.")

    st.markdown("### How to read the planogram and execute in store")
    st.markdown(
        f"""
1. Identify your store cluster from the clustering view.
2. Pick your category.
3. Execute shelf by shelf.
   - Eye level is shelf {assumptions.eye_level_shelves[0]} and {assumptions.eye_level_shelves[1]}.
   - Higher score SKUs are placed there first.
4. Follow position_index.
   - Position 1 is far left. Increase left to right.
5. Apply facings exactly.
   - recommended_facings tells you how many units wide the product must appear.
6. Respect brand blocking.
   - Same brand SKUs stay together. Do not break blocks.
7. If stockouts continue.
   - Use the task list. Increase facings or fix replenishment for flagged SKUs.
"""
    )

    st.divider()

    st.markdown("### Quick filter and printable table")
    clusters2 = sorted(rec_df["store_cluster"].unique().tolist())
    cats2 = sorted(rec_df["category"].unique().tolist())

    m1, m2 = st.columns(2)
    with m1:
        m_cluster = st.selectbox("Cluster (execution)", clusters2, index=0, key="m_cluster")
    with m2:
        m_cat = st.selectbox("Category (execution)", cats2, index=0, key="m_cat")

    exec_view = rec_df[(rec_df["store_cluster"] == m_cluster) & (rec_df["category"] == m_cat)].copy()
    exec_view = exec_view.sort_values(["shelf_level", "position_index"], ascending=[True, True])

    st.dataframe(
        exec_view[["shelf_level", "position_index", "brand", "sku_id", "recommended_facings"]].head(200),
        use_container_width=True,
        height=520
    )

with tab4:
    st.subheader("Exports")
    st.caption("Export CSV for analysts. Export PDF for store execution.")

    priorities = clustered.copy()
    priorities["priority_score"] = (priorities["oos_rate"] * 0.6) + (
        priorities["avg_record_error_units"] * 0.4 / (priorities["avg_record_error_units"].max() + 1e-9)
    )
    priorities = priorities.sort_values("priority_score", ascending=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Download recommendations CSV",
            data=rec_df.to_csv(index=False).encode("utf-8"),
            file_name="assortment_space_recommendations.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            "Download store clusters CSV",
            data=clustered.to_csv(index=False).encode("utf-8"),
            file_name="store_clusters.csv",
            mime="text/csv"
        )

    with col3:
        st.download_button(
            "Download store priority list CSV",
            data=priorities.to_csv(index=False).encode("utf-8"),
            file_name="store_priority_list.csv",
            mime="text/csv"
        )

    st.divider()

    st.markdown("### PDF planogram export")
    st.caption("Exports the currently selected cluster and category planogram from the Planogram tab.")

    st.info("Select cluster and category in the Recommendations tab, then return here to export PDF.")

    if "sel_cluster" in locals() and "sel_cat" in locals():
        pdf_view = rec_df[(rec_df["store_cluster"] == sel_cluster) & (rec_df["category"] == sel_cat)].copy()
        pdf_view = pdf_view.sort_values(["shelf_level", "position_index"], ascending=[True, True])
        pdf_slots = build_shelf_slots(pdf_view, max_slots=60)

        if REPORTLAB_OK:
            try:
                pdf_bytes = export_planogram_pdf(
                    view=pdf_view,
                    slots_df=pdf_slots,
                    cluster_id=int(sel_cluster),
                    category=str(sel_cat),
                    assumptions=assumptions
                )
                st.download_button(
                    "Download planogram PDF",
                    data=pdf_bytes,
                    file_name=f"planogram_cluster_{sel_cluster}_{sel_cat}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("PDF export is disabled because reportlab is not installed. Add reportlab to requirements.txt.")
    else:
        st.warning("Go to the Recommendations tab first. Select a cluster and category.")

st.divider()
st.caption(
    "Note. This engine uses practical heuristics for demo. In production you would replace shelf capacity, case packs, fixture constraints, and true OSA signals with real data."
)

