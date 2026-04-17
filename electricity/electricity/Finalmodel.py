import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from multiprocessing import freeze_support
from pathlib import Path
import matplotlib.patheffects as pe

def main():
    # ==============================
    # 0. OUTPUT FOLDERS
    # ==============================
    base = Path(__file__).resolve().parent
    output_dir = base / "output"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    output_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    # ==============================
    # 1. LOAD DATA
    # ==============================
    df = pd.read_csv(base / "final_dataset2.csv")

    # Expected columns:
    # ward_id, year, radiance, population_density, built_area, urban

    # ==============================
    # 2. CLEAN DATA
    # ==============================
    required_cols = ["ward_id", "year", "radiance", "population_density", "built_area", "urban"]
    df = df.dropna(subset=required_cols).copy()

    df["ward_id"] = df["ward_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df["urban"] = df["urban"].astype(int)

    df = df[(df["year"] >= 2015) & (df["year"] <= 2025)].copy()

    # ==============================
    # 3. FEATURE ENGINEERING
    # ==============================
    for c in ["radiance", "population_density", "built_area"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df[c] = df[c].clip(lower=0)
        df[f"log_{c}"] = np.log1p(df[c])

    scaler = StandardScaler()
    df[["rad_z", "pop_z", "built_z"]] = scaler.fit_transform(
        df[["log_radiance", "log_population_density", "log_built_area"]]
    )

    year_min = df["year"].min()
    year_max = df["year"].max()

    if year_max == year_min:
        raise ValueError("Your dataset has only one year. The model needs more than one year for a time trend.")

    df["year_scaled"] = (df["year"] - year_min) / (year_max - year_min)

    df["rad_x_urban"] = df["rad_z"] * df["urban"]
    df["built_x_urban"] = df["built_z"] * df["urban"]

    # ==============================
    # 4. PROXY ELECTRIFICATION LABEL
    # ==============================
    proxy = (
        0.45 * df["rad_z"]
        + 0.25 * df["built_z"]
        + 0.15 * df["pop_z"]
        + 0.30 * df["urban"]
        + 0.20 * df["year_scaled"]
    )

    proxy = (proxy - proxy.min()) / (proxy.max() - proxy.min())
    df["electricity"] = (proxy >= proxy.quantile(0.58)).astype(int)

    # ==============================
    # 5. ENCODE WARDS
    # ==============================
    ward_codes, ward_uniques = pd.factorize(df["ward_id"])
    df["ward_idx"] = ward_codes
    n_wards = len(ward_uniques)

    # ==============================
    # 6. BAYESIAN HIERARCHICAL MODEL
    # ==============================
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1.5)

        beta_rad = pm.Normal("beta_rad", mu=1.2, sigma=0.7)
        beta_pop = pm.Normal("beta_pop", mu=0.4, sigma=0.6)
        beta_built = pm.Normal("beta_built", mu=0.9, sigma=0.6)
        beta_urban = pm.Normal("beta_urban", mu=1.0, sigma=0.5)
        beta_year = pm.Normal("beta_year", mu=0.8, sigma=0.4)

        beta_rad_urban = pm.Normal("beta_rad_urban", mu=0.3, sigma=0.4)
        beta_built_urban = pm.Normal("beta_built_urban", mu=0.3, sigma=0.4)

        sigma_ward = pm.HalfNormal("sigma_ward", sigma=1.0)
        ward_effect = pm.Normal("ward_effect", mu=0, sigma=sigma_ward, shape=n_wards)

        logit_p = (
            alpha
            + beta_rad * df["rad_z"].values
            + beta_pop * df["pop_z"].values
            + beta_built * df["built_z"].values
            + beta_urban * df["urban"].values
            + beta_year * df["year_scaled"].values
            + beta_rad_urban * df["rad_x_urban"].values
            + beta_built_urban * df["built_x_urban"].values
            + ward_effect[df["ward_idx"].values]
        )

        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        y_obs = pm.Bernoulli("y_obs", p=p, observed=df["electricity"].values)

        trace = pm.sample(
            draws=1500,
            tune=1500,
            target_accept=0.95,
            chains=4,
            cores=1,
            return_inferencedata=True
        )

    # ==============================
    # 7. STANDARD DIAGNOSTIC PLOTS
    # ==============================

    # ---------- Posterior ----------
    az.plot_posterior(
        trace,
        var_names=[
            "alpha", "beta_rad", "beta_pop", "beta_built",
            "beta_urban", "beta_year", "beta_rad_urban", "beta_built_urban"
        ],
        hdi_prob=0.95
    )
    plt.suptitle("Posterior Distributions")
    plt.tight_layout()
    plt.savefig(figures_dir / "posterior_distributions.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------- Improved Trace Plot ----------
    trace_vars = [
        "alpha",
        "beta_rad",
        "beta_pop",
        "beta_built",
        "beta_urban",
        "beta_year",
        "sigma_ward"
    ]

    axes = az.plot_trace(
        trace,
        var_names=trace_vars,
        figsize=(14, 10)
    )

    fig = plt.gcf()

    # Improve spacing (fix clustering)
    plt.subplots_adjust(
        hspace=0.5,
        wspace=0.3,
        top=0.92,
        bottom=0.08
    )

    # Flatten axes for labeling
    axes = axes.ravel()

    for i, var in enumerate(trace_vars):
        # Trace plot (left column)
        axes[i * 2].set_title(f"{var} Trace", fontsize=10)
        axes[i * 2].set_ylabel("Value")
        axes[i * 2].set_xlabel("Draws")

        # Posterior (right column)
        axes[i * 2 + 1].set_title(f"{var} Posterior", fontsize=10)
        axes[i * 2 + 1].set_xlabel("Value")
        axes[i * 2 + 1].set_ylabel("Density")

    # Add explanation text
    fig.text(
        0.5, 0.01,
        "Each colored line represents an MCMC chain. Overlapping and stable chains indicate good convergence.",
        ha="center",
        fontsize=9
    )

    plt.savefig(figures_dir / "trace_plots.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==============================
    # 8. NUMERICAL MCMC DIAGNOSTICS
    # ==============================
    summary = az.summary(
        trace,
        var_names=[
            "alpha", "beta_rad", "beta_pop", "beta_built",
            "beta_urban", "beta_year", "beta_rad_urban",
            "beta_built_urban", "sigma_ward"
        ],
        round_to=3
    )

    print("\n===== MCMC SUMMARY =====")
    print(summary)
    summary.to_csv(tables_dir / "mcmc_summary.csv")

    bad_rhat = summary[summary["r_hat"] > 1.01]
    low_ess = summary[(summary["ess_bulk"] < 400) | (summary["ess_tail"] < 400)]

    print("\n===== CONVERGENCE FLAGS =====")
    if bad_rhat.empty:
        print("All monitored parameters have r_hat <= 1.01")
    else:
        print("Parameters with r_hat > 1.01:")
        print(bad_rhat[["mean", "sd", "r_hat"]])

    if low_ess.empty:
        print("All monitored parameters have acceptable ESS")
    else:
        print("Parameters with low ESS:")
        print(low_ess[["mean", "sd", "ess_bulk", "ess_tail"]])

    # ==============================
    # 9. PREDICTED PROBABILITIES
    # ==============================
    posterior_means = {
        v: trace.posterior[v].mean().item()
        for v in [
            "alpha", "beta_rad", "beta_pop", "beta_built",
            "beta_urban", "beta_year", "beta_rad_urban", "beta_built_urban"
        ]
    }

    ward_effect_mean = trace.posterior["ward_effect"].mean(dim=("chain", "draw")).values

    linear_pred = (
        posterior_means["alpha"]
        + posterior_means["beta_rad"] * df["rad_z"]
        + posterior_means["beta_pop"] * df["pop_z"]
        + posterior_means["beta_built"] * df["built_z"]
        + posterior_means["beta_urban"] * df["urban"]
        + posterior_means["beta_year"] * df["year_scaled"]
        + posterior_means["beta_rad_urban"] * df["rad_x_urban"]
        + posterior_means["beta_built_urban"] * df["built_x_urban"]
        + ward_effect_mean[df["ward_idx"]]
    )

    df["predicted_prob"] = 1 / (1 + np.exp(-linear_pred))
    df["predicted_prob"] = df["predicted_prob"].clip(0.001, 0.999)
    df["predicted_class"] = (df["predicted_prob"] >= 0.5).astype(int)

    df.to_csv(tables_dir / "model_predictions.csv", index=False)

    # ==============================
    # 10. POSTERIOR PREDICTIVE CHECK
    # ==============================
    with model:
        ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"], random_seed=42)

    try:
        idata_ppc = az.from_dict(
            posterior_predictive={"y_obs": ppc.posterior_predictive["y_obs"].values}
        )
        az.plot_ppc(idata_ppc)
        plt.title("Posterior Predictive Check")
        plt.tight_layout()
        plt.savefig(figures_dir / "posterior_predictive_check.png", dpi=300, bbox_inches="tight")
        plt.show()
    except Exception as e:
        print(f"Posterior predictive plot skipped: {e}")

    # ==============================
    # 11. CLASSIFICATION / PROBABILITY DIAGNOSTICS
    # ==============================
    auc = roc_auc_score(df["electricity"], df["predicted_prob"])
    brier = brier_score_loss(df["electricity"], df["predicted_prob"])

    print("\n===== PREDICTION QUALITY =====")
    print(f"AUC: {auc:.4f}")
    print(f"Brier score: {brier:.4f}")

    pd.DataFrame({
        "metric": ["AUC", "Brier score"],
        "value": [auc, brier]
    }).to_csv(tables_dir / "prediction_quality.csv", index=False)

    # ===============================
    # ADDITIONAL VALIDATION METRICS + PLOTS
    # ===============================
    from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        precision_recall_curve,
        average_precision_score
    )

    # --------------------------------
    # Choose probability column safely
    # --------------------------------
    prob_col = "predicted_prob" if "predicted_prob" in df.columns else "probability"

    # Binary class at 0.5 threshold
    df["predicted_class"] = (df[prob_col] >= 0.5).astype(int)

    # --------------------------------
    # 1. CLASSIFICATION METRICS TABLE
    # --------------------------------
    acc = accuracy_score(df["electricity"], df["predicted_class"])
    prec = precision_score(df["electricity"], df["predicted_class"], zero_division=0)
    rec = recall_score(df["electricity"], df["predicted_class"], zero_division=0)
    f1 = f1_score(df["electricity"], df["predicted_class"], zero_division=0)

    extra_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [acc, prec, rec, f1]
    })

    print("\n===== ADDITIONAL CLASSIFICATION METRICS =====")
    print(extra_metrics)

    extra_metrics.to_csv(tables_dir / "additional_classification_metrics.csv", index=False)

    # --------------------------------
    # 2. CONFUSION MATRIX
    # --------------------------------
    cm = confusion_matrix(df["electricity"], df["predicted_class"])

    cm_df = pd.DataFrame(
        cm,
        index=["Observed_0", "Observed_1"],
        columns=["Predicted_0", "Predicted_1"]
    )
    cm_df.to_csv(tables_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------
    # 3. PRECISION-RECALL CURVE
    # --------------------------------
    precision_vals, recall_vals, _ = precision_recall_curve(df["electricity"], df[prob_col])
    ap = average_precision_score(df["electricity"], df[prob_col])

    pr_df = pd.DataFrame({
        "precision": precision_vals[:-1] if len(precision_vals) > 1 else precision_vals,
        "recall": recall_vals[:-1] if len(recall_vals) > 1 else recall_vals
    })
    pr_df.to_csv(tables_dir / "precision_recall_curve.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.plot(recall_vals, precision_vals, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation: Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------
    # 4. PROBABILITY DISTRIBUTION BY OBSERVED CLASS
    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df.loc[df["electricity"] == 0, prob_col], bins=30, alpha=0.6, label="Observed = 0")
    plt.hist(df.loc[df["electricity"] == 1, prob_col], bins=30, alpha=0.6, label="Observed = 1")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Predicted Probability Distribution by Observed Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "probability_distribution_by_class.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------
    # 5. RESIDUAL ANALYSIS
    # --------------------------------
    df["residual"] = df["electricity"] - df[prob_col]

    residual_summary = df["residual"].describe().reset_index()
    residual_summary.columns = ["Statistic", "Value"]
    residual_summary.to_csv(tables_dir / "residual_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(df["residual"], bins=30)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "residual_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(df[prob_col], df["residual"], alpha=0.25)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted Probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "residuals_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------
    # 6. DECILE VALIDATION TABLE
    # --------------------------------
    df["prob_decile"] = pd.qcut(df[prob_col], 10, labels=False, duplicates="drop")

    decile_table = (
        df.groupby("prob_decile", as_index=False)
        .agg(
            min_prob=(prob_col, "min"),
            max_prob=(prob_col, "max"),
            mean_prob=(prob_col, "mean"),
            observed_rate=("electricity", "mean"),
            count=("electricity", "size")
        )
        .sort_values("prob_decile")
    )

    print("\n===== DECILE VALIDATION TABLE =====")
    print(decile_table)

    decile_table.to_csv(tables_dir / "decile_validation_table.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(decile_table["prob_decile"], decile_table["mean_prob"], marker="o", label="Mean predicted")
    plt.plot(decile_table["prob_decile"], decile_table["observed_rate"], marker="o", label="Observed rate")
    plt.xlabel("Probability Decile")
    plt.ylabel("Rate")
    plt.title("Decile Validation: Predicted vs Observed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "decile_validation_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------
    # 7. OBSERVED VS PREDICTED BY YEAR
    # --------------------------------
    yearly_validation = (
        df.groupby("year", as_index=False)
        .apply(lambda x: pd.Series({
            "predicted_mean": np.average(x[prob_col], weights=x["population_density"] + 1),
            "observed_mean": np.average(x["electricity"], weights=x["population_density"] + 1),
            "count": len(x)
        }))
        .reset_index(drop=True)
    )

    print("\n===== YEARLY VALIDATION =====")
    print(yearly_validation)

    yearly_validation.to_csv(tables_dir / "yearly_validation.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(yearly_validation["year"], yearly_validation["predicted_mean"], marker="o", label="Predicted")
    plt.plot(yearly_validation["year"], yearly_validation["observed_mean"], marker="o", label="Observed")
    plt.xlabel("Year")
    plt.ylabel("Population-weighted Mean")
    plt.title("Observed vs Predicted by Year")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "observed_vs_predicted_by_year.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==============================
    # 12. CALIBRATION CHECK
    # ==============================
    n_bins = 10
    df["prob_bin"] = pd.cut(
        df["predicted_prob"],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True
    )

    calibration = (
        df.groupby("prob_bin", observed=False)
        .agg(
            mean_pred=("predicted_prob", "mean"),
            obs_rate=("electricity", "mean"),
            count=("electricity", "size")
        )
        .dropna()
        .reset_index()
    )

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(calibration["mean_pred"], calibration["obs_rate"], marker="o")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Electrification Rate")
    plt.title("Calibration Plot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n===== CALIBRATION TABLE =====")
    print(calibration)
    calibration.to_csv(tables_dir / "calibration_table.csv", index=False)

    # ==============================
    # 14. TREND VALIDATION
    # ==============================
    trend_nat = (
        df.groupby("year")
        .apply(lambda x: np.average(x["predicted_prob"], weights=x["population_density"] + 1))
        .reset_index(name="predicted_access")
    )

    obs_nat = (
        df.groupby("year")
        .apply(lambda x: np.average(x["electricity"], weights=x["population_density"] + 1))
        .reset_index(name="observed_proxy_access")
    )

    trend_compare = trend_nat.merge(obs_nat, on="year", how="left")

    plt.figure(figsize=(9, 5))
    plt.plot(trend_compare["year"], trend_compare["predicted_access"], marker="o", label="Predicted")
    plt.plot(trend_compare["year"], trend_compare["observed_proxy_access"], marker="o", label="Observed proxy")
    plt.title("National Electricity Trend: Predicted vs Observed Proxy")
    plt.xlabel("Year")
    plt.ylabel("Population-weighted Access")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "national_trend.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n===== NATIONAL TREND CHECK =====")
    print(trend_compare)

    trend_compare.to_csv(tables_dir / "national_trend_check.csv", index=False)

    print("\n2015 predicted access:",
          trend_compare.loc[trend_compare["year"] == trend_compare["year"].min(), "predicted_access"].iloc[0])
    print(f"{year_max} predicted access:",
          trend_compare.loc[trend_compare["year"] == trend_compare["year"].max(), "predicted_access"].iloc[0])

    # ==============================
    # 15. URBAN VS RURAL TREND
    # ==============================
    df["urban"] = df["urban"].astype(int)

    trend_ur = (
        df.groupby(["year", "urban"])
        .apply(lambda x: pd.Series({
            "electricity_access": np.average(
                x["predicted_prob"],
                weights=x["population_density"] + 1
            ),
            "simple_mean": x["predicted_prob"].mean(),
            "n": len(x)
        }))
        .reset_index()
    )

    plt.figure(figsize=(8, 6))

    label_map = {0: "Rural", 1: "Urban"}
    line_width_map = {0: 2.0, 1: 3.0}
    zorder_map = {0: 2, 1: 3}

    for u in sorted(trend_ur["urban"].unique()):
        sub = trend_ur[trend_ur["urban"] == u].sort_values("year")
        plt.plot(
            sub["year"],
            sub["electricity_access"],
            marker="o",
            linewidth=line_width_map.get(u, 2.0),
            zorder=zorder_map.get(u, 2),
            label=label_map.get(u, f"Urban={u}")
        )

    plt.legend()
    plt.title("Urban vs Rural Electrification Trend")
    plt.xlabel("Year")
    plt.ylabel("Population-weighted Probability")

    # stretch the top so the urban line is clearly visible,
    # but do not show a y-axis tick of 2
    plt.ylim(0, 1.08)
    plt.yticks(np.arange(0, 1.01, 0.2))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "urban_rural.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n===== URBAN TREND CHECK =====")
    print(trend_ur)
    trend_ur.to_csv(tables_dir / "urban_rural_trend.csv", index=False)

    print("\nCounts by year and urban class:")
    counts_year_urban = df.groupby(["year", "urban"]).size().reset_index(name="count")
    print(counts_year_urban)
    counts_year_urban.to_csv(tables_dir / "counts_by_year_and_urban.csv", index=False)

    # ==============================
    # 16. MAP PREPARATION
    # ==============================
    wards = gpd.read_file(base / "wards" / "zwe_polbnda_adm3_250k_cso.shp")
    districts = gpd.read_file(base / "ZWE_adm" / "ZWE_adm2.shp")
    provinces = gpd.read_file(base / "ZWE_adm" / "ZWE_adm1.shp")

    if wards.crs is None:
        wards = wards.set_crs("EPSG:4326")
    if districts.crs is None:
        districts = districts.set_crs("EPSG:4326")
    if provinces.crs is None:
        provinces = provinces.set_crs("EPSG:4326")

    districts = districts.to_crs(wards.crs)
    provinces = provinces.to_crs(wards.crs)

    wards["ZIMWARDSID"] = wards["ZIMWARDSID"].astype(str)

    provinces_proj = provinces.to_crs(epsg=3857)
    provinces_proj["centroid"] = provinces_proj.geometry.centroid
    provinces["centroid"] = provinces_proj["centroid"].to_crs(provinces.crs)

    province_name_col = None
    for col in ["NAME_1", "name", "province", "PROVINCE"]:
        if col in provinces.columns:
            province_name_col = col
            break

    # latest-year map
    latest_year = df["year"].max()
    map_df_latest = df[df["year"] == latest_year].copy()

    map_df_latest = (
        map_df_latest.groupby("ward_id", as_index=False)
        .agg({
            "predicted_prob": "mean",
            "urban": "max",
            "population_density": "mean",
            "radiance": "mean",
            "built_area": "mean"
        })
    )

    map_df_latest["ward_id"] = map_df_latest["ward_id"].astype(str)

    wards_latest = wards.merge(
        map_df_latest,
        left_on="ZIMWARDSID",
        right_on="ward_id",
        how="left"
    )

    wards_latest["predicted_prob"] = wards_latest["predicted_prob"].fillna(
        wards_latest["predicted_prob"].median()
    )

    # 2015 map
    year_2015 = 2015
    map_df_2015 = df[df["year"] == year_2015].copy()

    map_df_2015 = (
        map_df_2015.groupby("ward_id", as_index=False)
        .agg({
            "predicted_prob": "mean",
            "urban": "max",
            "population_density": "mean",
            "radiance": "mean",
            "built_area": "mean"
        })
    )

    map_df_2015["ward_id"] = map_df_2015["ward_id"].astype(str)

    wards_2015 = wards.merge(
        map_df_2015,
        left_on="ZIMWARDSID",
        right_on="ward_id",
        how="left"
    )

    wards_2015["predicted_prob"] = wards_2015["predicted_prob"].fillna(
        wards_2015["predicted_prob"].median()
    )

    # ==============================
    # 17. MAP 1: ELECTRIFICATION PROBABILITY (LATEST YEAR)
    # ==============================
    fig, ax = plt.subplots(figsize=(10, 10))

    wards_latest.plot(
        column="predicted_prob",
        cmap="viridis",
        linewidth=0,
        ax=ax
    )

    wards_latest.boundary.plot(ax=ax, color="white", linewidth=0.18)
    districts.boundary.plot(ax=ax, color="black", linewidth=0.65)
    provinces.boundary.plot(ax=ax, color="black", linewidth=1.35)

    if province_name_col is not None:
        for _, row in provinces.iterrows():
            ax.text(
                row["centroid"].x,
                row["centroid"].y,
                row[province_name_col],
                fontsize=9,
                fontweight="bold",
                color="black",
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")]
            )

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(wards_latest["predicted_prob"])
    plt.colorbar(
        sm,
        ax=ax,
        fraction=0.03,
        pad=0.02,
        label="Electrification Probability"
    )

    ax.set_title(f"Electrification Probability Map ({latest_year})")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(figures_dir / f"electricification_probability_map_{latest_year}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==============================
    # 18. MAP 2: ELECTRIFICATION PROBABILITY (2015)
    # ==============================
    fig, ax = plt.subplots(figsize=(10, 10))

    wards_2015.plot(
        column="predicted_prob",
        cmap="viridis",
        linewidth=0,
        ax=ax
    )

    wards_2015.boundary.plot(ax=ax, color="white", linewidth=0.18)
    districts.boundary.plot(ax=ax, color="black", linewidth=0.65)
    provinces.boundary.plot(ax=ax, color="black", linewidth=1.35)

    if province_name_col is not None:
        for _, row in provinces.iterrows():
            ax.text(
                row["centroid"].x,
                row["centroid"].y,
                row[province_name_col],
                fontsize=9,
                fontweight="bold",
                color="black",
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")]
            )

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(wards_2015["predicted_prob"])
    plt.colorbar(
        sm,
        ax=ax,
        fraction=0.03,
        pad=0.02,
        label="Electrification Probability"
    )

    ax.set_title("Electrification Probability Map (2015)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(figures_dir / "electricification_probability_map_2015.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==============================
    # 19. MAP 3: WARD-LEVEL AGGREGATED
    # ==============================
    ward_prob = (
        df.groupby("ward_id", as_index=False)["predicted_prob"]
        .mean()
    )
    ward_prob["ward_id"] = ward_prob["ward_id"].astype(str)

    wards_clean = wards.merge(
        ward_prob,
        left_on="ZIMWARDSID",
        right_on="ward_id",
        how="left"
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    wards_clean.plot(
        column="predicted_prob",
        cmap="viridis",
        linewidth=0,
        ax=ax
    )

    wards_clean.boundary.plot(ax=ax, color="white", linewidth=0.18)
    provinces.boundary.plot(ax=ax, color="black", linewidth=1.35)

    if province_name_col is not None:
        for _, row in provinces.iterrows():
            ax.text(
                row["centroid"].x,
                row["centroid"].y,
                row[province_name_col],
                fontsize=9,
                fontweight="bold",
                color="black",
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")]
            )

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(wards_clean["predicted_prob"])
    plt.colorbar(
        sm,
        ax=ax,
        fraction=0.03,
        pad=0.02,
        label="Ward-Level Electrification Probability"
    )

    ax.set_title("Ward-Level Electrification Map (Aggregated)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(figures_dir / "ward_level_map.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==============================
    # 20. MAP 4: CLASSIFICATION MAP
    # ==============================
    def classify_access(p):
        if pd.isna(p):
            return "Moderate Access"
        elif p >= 0.75:
            return "High Access"
        elif p >= 0.45:
            return "Moderate Access"
        else:
            return "Low Access"

    wards_latest["access_class"] = wards_latest["predicted_prob"].apply(classify_access)

    fig, ax = plt.subplots(figsize=(10, 10))

    wards_latest.plot(
        column="access_class",
        categorical=True,
        legend=True,
        linewidth=0,
        ax=ax
    )

    wards_latest.boundary.plot(ax=ax, color="white", linewidth=0.18)
    provinces.boundary.plot(ax=ax, color="black", linewidth=1.35)

    if province_name_col is not None:
        for _, row in provinces.iterrows():
            ax.text(
                row["centroid"].x,
                row["centroid"].y,
                row[province_name_col],
                fontsize=9,
                fontweight="bold",
                color="black",
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")]
            )

    ax.set_title(f"Ward Classification of Electrification ({latest_year})")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(figures_dir / "ward_classification_map.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("ALL OUTPUTS GENERATED SUCCESSFULLY")
    print(f"Figures saved in: {figures_dir}")
    print(f"Tables saved in: {tables_dir}")


if __name__ == "__main__":
    freeze_support()
    main()