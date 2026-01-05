import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass

# =========================================================
# CONFIGURATION AND BUSINESS BASIS
# =========================================================
@dataclass
class RetailAssumptions:
    shelf_width_cm: float = 800.0          # shelf width per level in cm
    shelf_levels: int = 5                  # number of shelf levels
    fixture_height_inches: float = 78.0    # fixture height in inches. Example 6.5 ft
    min_facings: int = 1                   # minimum facings per SKU
    service_level_days: int = 7            # target days of cover on shelf
    replenishment_cycle_days: int = 2      # replenishment cycle
    eye_level_min_in: float = 48.0         # 4 feet
    eye_level_max_in: float = 60.0         # 5 feet


# =========================================================
# SYNTHETIC DATA GENERATION
# =========================================================
def generate_synthetic_retail_data(
    n_stores: int = 40,
    n_skus: int = 120,
    n_weeks: int = 16,
    seed: int = 42
):
    """
    Returns:
      stores_df: store master
      skus_df: sku master
      sales_df: weekly store x sku x week signals including OOS, inventory, sales
    """
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

    # Brand blocking support
    brands = rng.choice(
        ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E", "Brand F"],
        size=n_skus,
        p=[0.18, 0.18, 0.16, 0.16, 0.16, 0.16]
    )

    unit_width_cm = np.clip(rng.normal(8, 2.5, size=n_skus), 4, 18)
    price = np.round(np.clip(rng.lognormal(mean=3.2, sigma=0.35, size=n_skus), 30, 600), 0)
    margin_rate = np.clip(rng.normal(0.28, 0.08, size=n_skus), 0.10, 0.55)
    base_weekly_demand = rng.lognormal(mean=2.1, sigma=0.8, size=n_skus)

    # Simulated new launches
    is_new_launch = rng.random(n_skus) < 0.10
    launch_week = rng.integers(1, max(2, 1 + int(0.6 * n_weeks)), size=n_skus)
    launch_week = np.where(is_new_launch, launch_week, 0)

    skus_df = pd.DataFrame({
        "sku_id": sku_ids,
        "category": categories,
        "segment": segments,
        "brand": brands,
        "unit_width_cm": unit_width_cm,
        "price": price,
        "margin_rate": margin_rate,
        "base_weekly_demand": base_weekly_demand,
        "perishable": np.where(categories == "Fresh", 1, 0),
        "is_new_launch": is_new_launch.astype(int),
        "launch_week": launch_week.astype(int),
    })

    cat_season = {
        "Fresh": 0.08,
        "Dairy": 0.05,
        "Snacks": 0.10,
        "Beverages": 0.12,
        "Personal Care": 0.03
    }

    weeks = list(range(1, n_weeks + 1))
    rows = []

    for _, s in stores_df.iterrows():
        store_multiplier = s["footfall_index"] * (0.85 + 0.25 * s["affluence_index"])
        format_multiplier = 1.15 if s["format"] == "Hyper" else (1.0 if s["format"] == "Super" else 0.75)

        for _, k in skus_df.iterrows():
            sku_multiplier = (0.9 + 0.25 * k["margin_rate"]) * (1.05 if k["segment"] == "Core" else 0.90)
            cat_multiplier = 1.0 + cat_season.get(k["category"], 0.05)

            for w in weeks:
                season = 1.0 + cat_season.get(k["category"], 0.05) * np.sin(2 * np.pi * (w / max(weeks)))
                noise = rng.normal(1.0, 0.20)

                launch_boost = 1.0
                if k["is_new_launch"] == 1 and w >= max(1, k["launch_week"]):
                    launch_boost = 1.10

                true_units = max(
                    0.0,
                    k["base_weekly_demand"] * store_multiplier * format_multiplier * sku_multiplier * cat_multiplier * season * noise * launch_boost
                )

                oos_prob = np.clip(
                    0.02
                    + 0.02 * (true_units / (np.mean(skus_df["base_weekly_demand"]) + 1e-9))
                    + (0.03 if s["format"] == "Convenience" else 0.0),
                    0.02, 0.20
                )
                is_oos = rng.random() < oos_prob
                observed_units = true_units * (0.55 if is_oos else 1.0)

                shrink = rng.binomial(1, 0.015) * rng.uniform(0.0, 0.08)
                record_error = rng.normal(0.0, 0.18)

                true_on_hand_units = max(0.0, rng.normal(20, 10) + (10 if s["format"] != "Convenience" else 0))
                system_on_hand_units = max(0.0, true_on_hand_units * (1.0 + record_error - shrink))

                shelf_capacity_units = max(4.0, (60.0 * s["space_index"]) / max(1.0, k["unit_width_cm"] / 8.0))

                rows.append({
                    "store_id": s["store_id"],
                    "region": s["region"],
                    "format": s["format"],
                    "sku_id": k["sku_id"],
                    "category": k["category"],
                    "segment": k["segment"],
                    "brand": k["brand"],
                    "week": w,
                    "price": k["price"],
                    "margin_rate": k["margin_rate"],
                    "unit_width_cm": k["unit_width_cm"],
                    "is_new_launch": int(k["is_new_launch"]),
                    "launch_week": int(k["launch_week"]),
                    "true_demand_units": true_units,
                    "sales_units": observed_units,
                    "is_oos_event": int(is_oos),
                    "true_on_hand_units": true_on_hand_units,
                    "system_on_hand_units": system_on_hand_units,
                    "shelf_capacity_units": shelf_capacity_units
                })

    sales_df = pd.DataFrame(rows)
    sales_df["sales_value"] = sales_df["sales_units"] * sales_df["price"]
    sales_df["gross_profit"] = sales_df["sales_value"] * sales_df["margin_rate"]

    return stores_df, skus_df, sales_df


# =========================================================
# CLUSTERING LOGIC. BUSINESS BASIS AND IMPLEMENTATION
# =========================================================
def build_store_features(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stores are clustered by OPERATIONAL BEHAVIOR, not by geography.

    Feature intent:
    - sales_value: store commercial importance
    - gross_profit: profitability profile
    - oos_rate: execution stress and availability issues
    - avg_record_error_units: inventory accuracy stress
    - avg_capacity_units: physical capacity proxy
    - category mix shares: drives localized assortment needs
    """
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

    tmp = sales_df.copy()
    tmp["abs_record_error"] = np.abs(tmp["system_on_hand_units"] - tmp["true_on_hand_units"])
    rec = tmp.groupby("store_id", as_index=False).agg(
        avg_record_error_units=("abs_record_error", "mean"),
        avg_capacity_units=("shelf_capacity_units", "mean")
    )
    feat = feat.merge(rec, on="store_id", how="left")
    return feat


def cluster_stores(store_feat: pd.DataFrame, n_clusters: int = 6, seed: int = 42) -> pd.DataFrame:
    """
    Clustering method:
    - Standardize numeric features so sales does not dominate OOS or accuracy
    - KMeans gives stable, interpretable store archetypes for planogram scaling
    """
    df = store_feat.copy()
    numeric_cols = [c for c in df.columns if c not in ["store_id", "region", "format"]]
    X = df[numeric_cols].values
    Xs = StandardScaler().fit_transform(X)

    n_clusters = int(np.clip(n_clusters, 2, min(12, len(df))))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    df["store_cluster"] = model.fit_predict(Xs)
    return df


# =========================================================
# FORECAST AND SCORING
# =========================================================
def demand_forecast_simple(sales_df: pd.DataFrame, lookback_weeks: int = 6) -> pd.DataFrame:
    """
    Moving average baseline forecast for demo.
    """
    df = sales_df.copy()
    max_w = df["week"].max()
    recent = df[df["week"] > max_w - lookback_weeks].copy()

    g = recent.groupby(["store_id", "sku_id"], as_index=False).agg(
        avg_units=("sales_units", "mean"),
        last_units=("sales_units", lambda x: x.iloc[-1])
    )
    g["forecast_units_weekly"] = np.clip(0.85 * g["avg_units"] + 0.15 * g["last_units"], 0.0, None)
    return g[["store_id", "sku_id", "forecast_units_weekly"]]


def compute_sku_scoring(df: pd.DataFrame, w_profit: float, w_velocity: float, w_lost: float, w_launch: float) -> pd.DataFrame:
    """
    Multi factor placement score.
    - Profit. Velocity. Lost sales proxy. New launch boost
    """
    tmp = df.copy()

    tmp["lost_units_proxy"] = np.where(tmp["is_oos_event"] == 1, np.clip(tmp["true_demand_units"] - tmp["sales_units"], 0.0, None), 0.0)
    tmp["lost_value_proxy"] = tmp["lost_units_proxy"] * tmp["price"]

    sku_perf = tmp.groupby(["store_cluster", "category", "sku_id", "brand"], as_index=False).agg(
        gross_profit=("gross_profit", "sum"),
        velocity=("sales_units", "mean"),
        lost_value=("lost_value_proxy", "sum"),
        oos_rate=("is_oos_event", "mean"),
        avg_width=("unit_width_cm", "mean"),
        is_new_launch=("is_new_launch", "max"),
        launch_week=("launch_week", "max"),
    )

    # Normalise within cluster x category for fair scoring
    def zscore(s):
        s = s.astype(float)
        return (s - s.mean()) / (s.std() + 1e-9)

    sku_perf["z_gp"] = sku_perf.groupby(["store_cluster", "category"])["gross_profit"].transform(zscore)
    sku_perf["z_vel"] = sku_perf.groupby(["store_cluster", "category"])["velocity"].transform(zscore)
    sku_perf["z_lost"] = sku_perf.groupby(["store_cluster", "category"])["lost_value"].transform(zscore)

    # Launch boost only if flagged. Kept small so it does not override economics
    sku_perf["launch_boost"] = np.where(sku_perf["is_new_launch"] == 1, 1.0, 0.0)

    sku_perf["score"] = (
        w_profit * sku_perf["z_gp"]
        + w_velocity * sku_perf["z_vel"]
        + w_lost * sku_perf["z_lost"]
        + w_launch * sku_perf["launch_boost"]
    )

    # Penalize persistent OOS slightly because it indicates shelf stress
    sku_perf["score"] = sku_perf["score"] * (1.0 - 0.25 * np.clip(sku_perf["oos_rate"], 0, 1))
    return sku_perf


# =========================================================
# PLANOGRAM ENGINE. RULE BASED SHELF ZONES AND BRAND BLOCKING
# =========================================================
def shelf_level_midpoints(assumptions: RetailAssumptions) -> pd.DataFrame:
    level_h = assumptions.fixture_height_inches / max(1, assumptions.shelf_levels)
    levels = np.arange(1, assumptions.shelf_levels + 1)
    mid = (levels - 0.5) * level_h
    df = pd.DataFrame({"shelf_level": levels, "mid_height_in": mid})
    df["is_eye_level"] = ((df["mid_height_in"] >= assumptions.eye_level_min_in) & (df["mid_height_in"] <= assumptions.eye_level_max_in)).astype(int)
    return df


def recommend_assortment_and_facings(
    sales_df: pd.DataFrame,
    store_clusters_df: pd.DataFrame,
    assumptions: RetailAssumptions,
    top_n_per_category: int,
    w_profit: float,
    w_velocity: float,
    w_lost: float,
    w_launch: float
) -> pd.DataFrame:
    """
    Output includes:
    - recommended_facings
    - shelf_level based on physical height zones
    - position_index with brand blocking
    - planogram explanations for store execution
    """
    df = sales_df.merge(store_clusters_df[["store_id", "store_cluster"]], on="store_id", how="left")

    # Score at cluster x category x sku level
    scored = compute_sku_scoring(df, w_profit, w_velocity, w_lost, w_launch)

    # Select top N per cluster x category
    scored["rank"] = scored.groupby(["store_cluster", "category"])["score"].rank(ascending=False, method="first")
    chosen = scored[scored["rank"] <= top_n_per_category].copy()

    # Forecast to guide facings
    fc = demand_forecast_simple(sales_df)
    fc = fc.merge(store_clusters_df[["store_id", "store_cluster"]], on="store_id", how="left")
    cluster_fc = fc.groupby(["store_cluster", "sku_id"], as_index=False)["forecast_units_weekly"].mean()

    chosen = chosen.merge(cluster_fc, on=["store_cluster", "sku_id"], how="left")
    chosen["forecast_units_weekly"] = chosen["forecast_units_weekly"].fillna(0.0)

    # Facings
    daily = chosen["forecast_units_weekly"] / 7.0
    target_units = daily * assumptions.service_level_days
    units_per_facing = np.where(chosen["avg_width"] <= 8, 6.0, np.where(chosen["avg_width"] <= 12, 4.0, 3.0))
    raw_facings = target_units / (units_per_facing + 1e-9)
    chosen["recommended_facings"] = np.ceil(np.clip(raw_facings, assumptions.min_facings, 40)).astype(int)

    # Shelf width constraint per cluster x category
    shelf_total_width = assumptions.shelf_width_cm * assumptions.shelf_levels
    out = []

    for (cl, cat), grp in chosen.groupby(["store_cluster", "category"]):
        grp = grp.copy()
        grp["facing_width_cm"] = grp["recommended_facings"] * grp["avg_width"]
        total_width = grp["facing_width_cm"].sum()

        if total_width > shelf_total_width and total_width > 0:
            scale = shelf_total_width / total_width
            grp["recommended_facings"] = np.maximum(assumptions.min_facings, np.floor(grp["recommended_facings"] * scale)).astype(int)
            grp["facing_width_cm"] = grp["recommended_facings"] * grp["avg_width"]

        grp["shelf_width_budget_cm"] = shelf_total_width
        out.append(grp)

    rec_df = pd.concat(out, ignore_index=True)

    # Assign shelf zones using physical eye level definition
    levels = shelf_level_midpoints(assumptions)
    eye_levels = levels[levels["is_eye_level"] == 1]["shelf_level"].tolist()
    non_eye_levels = levels[levels["is_eye_level"] == 0]["shelf_level"].tolist()

    # Rule driven placement
    # 1. New launches and top score go to eye level
    # 2. Bulky items go to bottom. Proxy using width
    # 3. Remaining fill other levels
    rec_df["is_bulky"] = (rec_df["avg_width"] >= 13).astype(int)

    rec_df = rec_df.sort_values(["store_cluster", "category", "score"], ascending=[True, True, False]).reset_index(drop=True)
    rec_df["shelf_level"] = 1

    def assign_levels(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()

        # Priority buckets
        grp["priority_bucket"] = 3
        grp.loc[(grp["is_new_launch"] == 1), "priority_bucket"] = 1
        grp.loc[(grp["priority_bucket"] != 1) & (grp["rank"] <= max(3, int(0.25 * len(grp)))), "priority_bucket"] = 2
        grp.loc[(grp["is_bulky"] == 1), "priority_bucket"] = 4

        # Start empty assignment
        grp["shelf_level"] = 0

        # Assign bulky to bottom level 1 first
        bottom_level = 1
        bulky_idx = grp.index[grp["priority_bucket"] == 4].tolist()
        if len(bulky_idx) > 0:
            grp.loc[bulky_idx, "shelf_level"] = bottom_level

        # Assign priority 1 and 2 into eye levels
        eye_target_levels = eye_levels if len(eye_levels) > 0 else [min(assumptions.shelf_levels, max(2, assumptions.shelf_levels - 2))]
        eye_cycle = 0
        for idx in grp.index[grp["priority_bucket"].isin([1, 2])].tolist():
            lvl = eye_target_levels[eye_cycle % len(eye_target_levels)]
            grp.at[idx, "shelf_level"] = lvl
            eye_cycle += 1

        # Assign remaining to non eye levels excluding bottom already used
        fill_levels = [l for l in non_eye_levels if l != bottom_level]
        if len(fill_levels) == 0:
            fill_levels = [l for l in range(1, assumptions.shelf_levels + 1) if l != bottom_level]

        fill_cycle = 0
        for idx in grp.index[grp["shelf_level"] == 0].tolist():
            lvl = fill_levels[fill_cycle % len(fill_levels)]
            grp.at[idx, "shelf_level"] = lvl
            fill_cycle += 1

        return grp

    rec_df = rec_df.groupby(["store_cluster", "category"], group_keys=False).apply(assign_levels)

    # Brand blocking. Within each shelf_level, group by brand, then by score
    # This is a practical retail execution pattern
    rec_df["position_index"] = 0

    def assign_positions(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        grp = grp.sort_values(["shelf_level", "brand", "score"], ascending=[True, True, False])

        # Within each shelf_level keep brand blocks contiguous
        pos = 1
        for lvl in sorted(grp["shelf_level"].unique().tolist()):
            shelf = grp[grp["shelf_level"] == lvl].copy()

            # Brand block order. Highest total score brands first
            brand_order = (
                shelf.groupby("brand")["score"].sum()
                .sort_values(ascending=False)
                .index.tolist()
            )

            ordered_rows = []
            for b in brand_order:
                ordered_rows.append(shelf[shelf["brand"] == b].sort_values("score", ascending=False))
            shelf2 = pd.concat(ordered_rows, axis=0)

            shelf2["position_index"] = range(pos, pos + len(shelf2))
            pos += len(shelf2)

            grp.loc[shelf2.index, "position_index"] = shelf2["position_index"].values

        return grp

    rec_df = rec_df.groupby(["store_cluster", "category"], group_keys=False).apply(assign_positions)

    # Add store execution notes
    rec_df["execution_note"] = "Follow shelf_level and position_index. Keep brand blocks together. Use recommended_facings."
    rec_df.loc[rec_df["is_new_launch"] == 1, "execution_note"] = "New launch. Place on eye level if possible. Keep recommended_facings."
    rec_df.loc[rec_df["is_bulky"] == 1, "execution_note"] = "Bulky. Prefer bottom shelf. Keep aisle safety."

    # Attach shelf height info for explanation
    rec_df = rec_df.merge(levels, on="shelf_level", how="left")
    return rec_df


# =========================================================
# REAL TIME SHELF CONTROL. ALERTS AND ACTIONS
# =========================================================
def build_realtime_alerts(sales_df: pd.DataFrame, store_clusters_df: pd.DataFrame, rec_df: pd.DataFrame, assumptions: RetailAssumptions, selected_week: int) -> pd.DataFrame:
    """
    Real time shelf control logic.
    Proxy using current week as today. Works with synthetic data and uploads.

    days_of_cover = system_on_hand / (forecast_daily)
    If days_of_cover < replenishment_cycle_days. Stockout risk
    """
    df = sales_df[sales_df["week"] == selected_week].copy()
    df = df.merge(store_clusters_df[["store_id", "store_cluster"]], on="store_id", how="left")

    fc = demand_forecast_simple(sales_df)
    df = df.merge(fc, on=["store_id", "sku_id"], how="left")
    df["forecast_units_weekly"] = df["forecast_units_weekly"].fillna(0.0)

    df["forecast_daily"] = df["forecast_units_weekly"] / 7.0
    df["days_of_cover"] = df["system_on_hand_units"] / (df["forecast_daily"] + 1e-9)

    # Bring planogram score from rec_df at cluster x sku
    score_map = rec_df.groupby(["store_cluster", "sku_id"], as_index=False)["score"].max()
    df = df.merge(score_map, on=["store_cluster", "sku_id"], how="left")
    df["score"] = df["score"].fillna(0.0)

    df["stockout_risk"] = (df["days_of_cover"] < assumptions.replenishment_cycle_days).astype(int)
    df["overstock_risk"] = (df["system_on_hand_units"] > 1.25 * df["shelf_capacity_units"]).astype(int)

    # Action recommendations
    df["recommended_action"] = "Maintain"
    df.loc[df["stockout_risk"] == 1, "recommended_action"] = "Urgent. Replenish. Consider increasing facings for high score SKUs"
    df.loc[(df["overstock_risk"] == 1) & (df["stockout_risk"] == 0), "recommended_action"] = "Reduce facings or move to secondary shelf. Avoid overstock"

    # Prioritise
    df["priority"] = (0.65 * df["stockout_risk"] + 0.25 * df["overstock_risk"] + 0.10 * (df["score"] > df["score"].quantile(0.75)).astype(int))
    df = df.sort_values(["priority", "score"], ascending=[False, False])

    return df


# =========================================================
# KPIS
# =========================================================
def compute_kpis(sales_df: pd.DataFrame) -> dict:
    df = sales_df.copy()
    df["record_error_units"] = np.abs(df["system_on_hand_units"] - df["true_on_hand_units"])
    inv_accuracy = 1.0 - (df["record_error_units"].mean() / (df["true_on_hand_units"].mean() + 1e-9))
    inv_accuracy = float(np.clip(inv_accuracy, 0.0, 1.0))

    oos_rate = float(df["is_oos_event"].mean())
    df["lost_units_proxy"] = np.where(df["is_oos_event"] == 1, df["true_demand_units"] - df["sales_units"], 0.0)
    lost_value_proxy = float((np.clip(df["lost_units_proxy"], 0.0, None) * df["price"]).sum())

    df["understock_risk"] = np.where(df["true_on_hand_units"] < 0.25 * df["shelf_capacity_units"], 1, 0)
    df["overstock_risk"] = np.where(df["true_on_hand_units"] > 1.25 * df["shelf_capacity_units"], 1, 0)

    return {
        "inventory_record_accuracy_proxy": inv_accuracy,
        "oos_event_rate": oos_rate,
        "lost_sales_value_proxy": lost_value_proxy,
        "total_sales_value": float(df["sales_value"].sum()),
        "total_gross_profit": float(df["gross_profit"].sum()),
        "understock_risk_rate": float(df["understock_risk"].mean()),
        "overstock_risk_rate": float(df["overstock_risk"].mean()),
    }


# =========================================================
# PLANOGRAM VISUAL GRID
# =========================================================
def render_planogram_grid(rec_df: pd.DataFrame, store_cluster: int, category: str, assumptions: RetailAssumptions):
    """
    How to read the planogram.
    - Y axis shelf_level represents vertical shelf placement
    - Shelf level is derived from physical height, eye level is 4 to 5 feet
    - X axis position_index is the left to right order
    - Size is recommended_facings
    - Color is score. Higher means higher commercial priority
    - Brand blocking means same brand items are kept contiguous
    """
    pg = rec_df[(rec_df["store_cluster"] == store_cluster) & (rec_df["category"] == category)].copy()
    if pg.empty:
        st.info("No planogram for this selection.")
        return

    levels = shelf_level_midpoints(assumptions)
    eye_levels = levels[levels["is_eye_level"] == 1]["shelf_level"].tolist()

    fig = px.scatter(
        pg,
        x="position_index",
        y="shelf_level",
        size="recommended_facings",
        color="score",
        text="sku_id",
        hover_data=["brand", "recommended_facings", "score", "mid_height_in", "execution_note"],
        size_max=60
    )

    fig.update_traces(textposition="middle center", marker=dict(opacity=0.85))
    fig.update_layout(
        title="Planogram visual grid. Execute shelf_level then position_index. Keep brand blocks together",
        xaxis_title="Left to right position",
        yaxis_title="Shelf level",
        yaxis=dict(autorange="reversed"),
        height=560,
        showlegend=False
    )

    # Mark eye level bands if present
    if len(eye_levels) > 0:
        for lvl in eye_levels:
            fig.add_hline(y=lvl, line_width=1, line_dash="dot")

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# STREAMLIT APP
# =========================================================
st.set_page_config(page_title="Retail Space and Assortment Optimizer", layout="wide")
st.title("Retail Space and Assortment Optimizer")
st.caption("Automated shelf planning. Data driven placement. Real time shelf control. Centralized standards. Synthetic data by default.")

with st.sidebar:
    st.header("Data input")
    use_synth = st.toggle("Use synthetic data", value=True)
    seed = st.number_input("Seed", min_value=1, max_value=9999, value=42, step=1)
    n_stores = st.slider("Stores", 10, 120, 40, 5)
    n_skus = st.slider("SKUs", 50, 400, 120, 10)
    n_weeks = st.slider("Weeks", 8, 52, 16, 4)

    st.divider()
    st.header("Fixture and shelf heights")
    shelf_levels = st.slider("Shelf levels", 3, 8, 5, 1)
    fixture_height_inches = st.number_input("Fixture height inches", 60.0, 96.0, 78.0, 1.0)
    shelf_width = st.number_input("Shelf width per level cm", 200.0, 2000.0, 800.0, 50.0)

    st.divider()
    st.header("Optimization controls")
    n_clusters = st.slider("Store clusters", 2, 12, 6, 1)
    top_n = st.slider("Top SKUs per category per cluster", 5, 60, 20, 5)
    service_days = st.slider("Days of cover target", 1, 21, 7, 1)
    repl_days = st.slider("Replenishment cycle days", 1, 7, 2, 1)

    st.divider()
    st.header("Placement scoring weights")
    st.caption("These weights control prime placement and ranking.")
    w_profit = st.slider("Profit weight", 0.0, 1.0, 0.45, 0.05)
    w_velocity = st.slider("Velocity weight", 0.0, 1.0, 0.35, 0.05)
    w_lost = st.slider("Lost sales weight", 0.0, 1.0, 0.15, 0.05)
    w_launch = st.slider("New launch boost weight", 0.0, 1.0, 0.05, 0.05)

    # Normalise weights
    ssum = w_profit + w_velocity + w_lost + w_launch
    if ssum <= 0:
        w_profit, w_velocity, w_lost, w_launch = 0.45, 0.35, 0.15, 0.05
        ssum = 1.0
    w_profit, w_velocity, w_lost, w_launch = w_profit / ssum, w_velocity / ssum, w_lost / ssum, w_launch / ssum

    st.divider()
    st.header("Uploads")
    uploaded_sales = None
    uploaded_launches = None

    if not use_synth:
        uploaded_sales = st.file_uploader(
            "Upload sales CSV",
            type=["csv"],
            help="Expected columns include store_id, region, format, sku_id, category, segment, brand, week, price, margin_rate, unit_width_cm, sales_units, is_oos_event, true_on_hand_units, system_on_hand_units, shelf_capacity_units"
        )

    uploaded_launches = st.file_uploader(
        "Upload new launches CSV",
        type=["csv"],
        help="Expected columns: sku_id, is_new_launch. Optional: launch_week"
    )

assumptions = RetailAssumptions(
    shelf_width_cm=float(shelf_width),
    shelf_levels=int(shelf_levels),
    fixture_height_inches=float(fixture_height_inches),
    service_level_days=int(service_days),
    replenishment_cycle_days=int(repl_days),
)

if use_synth:
    stores_df, skus_df, sales_df = generate_synthetic_retail_data(
        n_stores=int(n_stores),
        n_skus=int(n_skus),
        n_weeks=int(n_weeks),
        seed=int(seed)
    )
else:
    if uploaded_sales is None:
        st.warning("Upload a sales CSV or enable synthetic data.")
        st.stop()
    sales_df = pd.read_csv(uploaded_sales)
    stores_df = sales_df[["store_id", "region", "format"]].drop_duplicates().reset_index(drop=True)

    # SKU master inferred from fact table if user provided it
    base_cols = ["sku_id", "category", "segment", "brand", "price", "margin_rate", "unit_width_cm"]
    missing = [c for c in base_cols if c not in sales_df.columns]
    if len(missing) > 0:
        st.error(f"Sales CSV missing columns: {missing}")
        st.stop()
    skus_df = sales_df[base_cols].drop_duplicates().reset_index(drop=True)

# Apply launch upload if provided
if uploaded_launches is not None:
    launches = pd.read_csv(uploaded_launches)
    if "sku_id" not in launches.columns or "is_new_launch" not in launches.columns:
        st.error("Launches CSV must include sku_id and is_new_launch.")
        st.stop()

    launches["is_new_launch"] = launches["is_new_launch"].astype(int)
    if "launch_week" in launches.columns:
        launches["launch_week"] = launches["launch_week"].fillna(0).astype(int)
    else:
        launches["launch_week"] = 0

    skus_df = skus_df.merge(launches[["sku_id", "is_new_launch", "launch_week"]], on="sku_id", how="left", suffixes=("", "_u"))
    skus_df["is_new_launch"] = np.where(skus_df["is_new_launch_u"].notna(), skus_df["is_new_launch_u"], skus_df.get("is_new_launch", 0)).astype(int)
    skus_df["launch_week"] = np.where(skus_df["launch_week_u"].notna(), skus_df["launch_week_u"], skus_df.get("launch_week", 0)).astype(int)
    skus_df = skus_df.drop(columns=["is_new_launch_u", "launch_week_u"], errors="ignore")

    # Push launch flags into sales_df for scoring
    sales_df = sales_df.drop(columns=["is_new_launch", "launch_week"], errors="ignore")
    sales_df = sales_df.merge(skus_df[["sku_id", "is_new_launch", "launch_week"]], on="sku_id", how="left")
    sales_df["is_new_launch"] = sales_df["is_new_launch"].fillna(0).astype(int)
    sales_df["launch_week"] = sales_df["launch_week"].fillna(0).astype(int)

# If synthetic, these columns already exist. If upload sales, ensure they exist
if "is_new_launch" not in sales_df.columns:
    sales_df = sales_df.merge(skus_df[["sku_id", "is_new_launch", "launch_week"]], on="sku_id", how="left")
    sales_df["is_new_launch"] = sales_df["is_new_launch"].fillna(0).astype(int)
    sales_df["launch_week"] = sales_df["launch_week"].fillna(0).astype(int)

# Build store features and clusters
store_feat = build_store_features(sales_df)
clustered = cluster_stores(store_feat, n_clusters=int(n_clusters), seed=int(seed))

# Attach store_cluster to sales for scoring
sales_with_cluster = sales_df.merge(clustered[["store_id", "store_cluster"]], on="store_id", how="left")

# Build planogram recommendations
rec_df = recommend_assortment_and_facings(
    sales_df=sales_df,
    store_clusters_df=clustered,
    assumptions=assumptions,
    top_n_per_category=int(top_n),
    w_profit=float(w_profit),
    w_velocity=float(w_velocity),
    w_lost=float(w_lost),
    w_launch=float(w_launch),
)

# KPIs
kpis = compute_kpis(sales_df)

# =========================================================
# TOP KPIS
# =========================================================
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("Sales value", f"{kpis['total_sales_value']:,.0f}")
k2.metric("Gross profit", f"{kpis['total_gross_profit']:,.0f}")
k3.metric("Inventory accuracy proxy", f"{100*kpis['inventory_record_accuracy_proxy']:.1f}%")
k4.metric("OOS event rate", f"{100*kpis['oos_event_rate']:.1f}%")
k5.metric("Lost sales proxy", f"{kpis['lost_sales_value_proxy']:,.0f}")
k6.metric("Understock risk", f"{100*kpis['understock_risk_rate']:.1f}%")
k7.metric("Overstock risk", f"{100*kpis['overstock_risk_rate']:.1f}%")

st.divider()

# =========================================================
# TABS FOR THE 4 SOLUTIONS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Automated shelf planning",
    "Data driven placement",
    "Real time shelf control",
    "Central standards"
])

with tab1:
    st.subheader("Automated shelf planning. Physical shelf zones with eye level at 4 to 5 feet")

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown(
            "Clustering basis. Stores are grouped by operational behavior using sales value, gross profit, OOS rate, inventory record error, shelf capacity, and category mix. "
            "Standardization is applied so large sales stores do not dominate the clustering."
        )

        fig = px.scatter(
            clustered,
            x="sales_value",
            y="gross_profit",
            color="store_cluster",
            hover_data=["store_id", "region", "format", "oos_rate", "avg_record_error_units"],
            title="Store clusters. Sales vs gross profit"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("Shelf height model. Eye level is 48 to 60 inches. This determines which shelf levels are prime.")
        levels = shelf_level_midpoints(assumptions)
        st.dataframe(levels, use_container_width=True, height=220)

    st.divider()
    st.subheader("Planogram visual grid. Brand blocking enabled")

    clusters = sorted(rec_df["store_cluster"].unique().tolist())
    cats = sorted(rec_df["category"].unique().tolist())

    sel_cluster = st.selectbox("Cluster", clusters, index=0, key="pg_cluster_1")
    sel_cat = st.selectbox("Category", cats, index=0, key="pg_cat_1")

    render_planogram_grid(rec_df, sel_cluster, sel_cat, assumptions)

    st.markdown(
        "How to execute in store. "
        "Start with shelf level. Level numbering is bottom to top. Then place SKUs left to right by position_index. "
        "Keep same brand SKUs together because brand blocking is enabled. "
        "Apply recommended_facings for each SKU. If space is short, remove lowest score SKU first."
    )

with tab2:
    st.subheader("Data driven placement. Best sellers and new launches get prime shelf space")

    st.markdown(
        "Score logic. Score is a weighted combination of profit, velocity, lost sales proxy, and new launch boost. "
        "You can tune weights in the sidebar."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("Top SKUs by score for a selected cluster and category.")
        sel_cluster2 = st.selectbox("Cluster", sorted(rec_df["store_cluster"].unique()), key="cluster2")
        sel_cat2 = st.selectbox("Category", sorted(rec_df["category"].unique()), key="cat2")
        view = rec_df[(rec_df["store_cluster"] == sel_cluster2) & (rec_df["category"] == sel_cat2)].copy()
        view = view.sort_values("score", ascending=False)
        st.dataframe(
            view[[
                "sku_id", "brand", "is_new_launch", "gross_profit", "velocity", "lost_value",
                "oos_rate", "score", "recommended_facings", "shelf_level", "position_index", "execution_note"
            ]].head(60),
            use_container_width=True,
            height=520
        )

    with c2:
        st.markdown("Shelf width usage for this category planogram.")
        used = float(view["facing_width_cm"].sum()) if len(view) else 0.0
        budget = float(view["shelf_width_budget_cm"].iloc[0]) if len(view) else 1.0

        fig_space = go.Figure()
        fig_space.add_trace(go.Indicator(
            mode="gauge+number",
            value=used,
            title={"text": "Shelf width used cm"},
            gauge={
                "axis": {"range": [0, max(budget, used, 1.0)]},
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": budget}
            }
        ))
        st.plotly_chart(fig_space, use_container_width=True)

with tab3:
    st.subheader("Real time shelf control. Stockout and overstock alerts with actions")

    st.markdown(
        "This uses days of cover as a practical control metric. "
        "days_of_cover = system_on_hand / forecast_daily. "
        "If days_of_cover is below replenishment cycle days, it flags stockout risk."
    )

    max_week = int(sales_df["week"].max())
    sel_week = st.slider("Select current week for real time view", 1, max_week, max_week, 1)

    alerts = build_realtime_alerts(sales_df, clustered, rec_df, assumptions, sel_week)

    st.markdown("Highest priority issues first.")
    st.dataframe(
        alerts[[
            "store_id", "store_cluster", "category", "sku_id", "brand",
            "system_on_hand_units", "forecast_units_weekly", "days_of_cover",
            "stockout_risk", "overstock_risk", "score", "recommended_action"
        ]].head(60),
        use_container_width=True,
        height=520
    )

with tab4:
    st.subheader("Centralized planogram standards. Governance and consistency")

    st.markdown(
        "Central standards mean one set of rules per format and category, with controlled local variations. "
        "This tab gives you an editable standards table, plus a simple version note so teams follow one approved plan."
    )

    if "standards" not in st.session_state:
        st.session_state["standards"] = pd.DataFrame({
            "format": ["Hyper", "Super", "Convenience"],
            "category": ["Snacks", "Snacks", "Snacks"],
            "min_facings": [1, 1, 1],
            "max_facings": [40, 35, 25],
            "brand_blocking": ["Yes", "Yes", "Yes"],
            "eye_level_rule": ["48-60 inches", "48-60 inches", "48-60 inches"],
            "notes": ["Default", "Default", "Default"]
        })

    st.markdown("Edit standards. These can be exported and used as a governance baseline.")
    edited = st.data_editor(st.session_state["standards"], use_container_width=True, num_rows="dynamic")
    st.session_state["standards"] = edited

    st.text_input("Planogram version note", value="v1.0. Cluster based planogram with brand blocking and eye level zones")

    st.download_button(
        "Download standards CSV",
        data=st.session_state["standards"].to_csv(index=False).encode("utf-8"),
        file_name="planogram_standards.csv",
        mime="text/csv"
    )

st.divider()

st.subheader("Downloads")
st.download_button(
    "Download recommendations CSV",
    data=rec_df.to_csv(index=False).encode("utf-8"),
    file_name="assortment_space_recommendations.csv",
    mime="text/csv"
)
st.download_button(
    "Download store clusters CSV",
    data=clustered.to_csv(index=False).encode("utf-8"),
    file_name="store_clusters.csv",
    mime="text/csv"
)

st.caption(
    "Deployment note. Use requirements.txt, not Poetry. Required packages. streamlit, numpy, pandas, scikit-learn, plotly."
)
