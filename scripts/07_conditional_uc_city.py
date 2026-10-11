# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Conditional uncertainties for cities (heuristic interaction check)
#
# The uncertainty decomposition in the main analysis averages each source
# (scenario, response, internal variability, downscaling, fit) over everything
# else. If there are strong interactions, these averages hide a lot, e.g. internal
# variability could depend on the SSP, the downscaling method, or the GCM.
#
# This is **not** a formal procedure. We use exactly the same
# estimates as `sa_city_utils.calculate_df_uc` (95% ranges, i.e. 2.5th to 97.5th percentile, of either the change in
# return level from the stationary or non-stationary GEV fits, or the linear trend in the bulk
# variables; set `analysis_type` in the settings), but rather than averaging them over all groups we
# keep the group-level values and average within levels of one factor:
#
# | uncertainty | one estimate per | can be conditioned on |
# |---|---|---|
# | scenario | ensemble (or ensemble-GCM) | ensemble, GCM |
# | response | ensemble-SSP | ensemble, SSP |
# | internal variability | ensemble-GCM-SSP (>= 5 members) | ensemble, GCM, SSP |
# | downscaling | GCM-SSP-member | SSP, GCM |
# | fit | simulation | ensemble, GCM, SSP |
#
# Bars are the mean across the group-level estimates within each level.
# "All" is the unconditional value used in the main figures.
#
# Caveats: the ensembles are unbalanced, so conditioning on one factor also changes the
# mix of the others. In particular, downscaling uncertainty by SSP is
# also a change of downscaling pair (ssp370: LOCA2 vs GARD-LENS; ssp245/ssp585:
# LOCA2 vs STAR-ESDM). Downscaling uncertainty by GCM is not shown (too many GCMs).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

import plotting_utils as pu
import sa_city_utils as sacu
from utils import city_list, gev_metric_ids, trend_metric_ids
from utils import roar_data_path as project_data_path
from utils import roar_code_path as project_code_path

# %%
# Set fonts
font_dirs = ["/storage/home/dcl5300/fonts"]
for font_file in font_manager.findSystemFonts(fontpaths=font_dirs):
    font_manager.fontManager.addfont(font_file)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 1.5

# %% [markdown]
# ## Settings

# %%
city = "chicago"  # any key of utils.city_list
analysis_type = "extremes"  # "extremes": change in GEV return level; "trends": linear trend in bulk variables
metric_ids = ["max_tasmax", "min_tasmin", "max_pr"]  # extremes (gev_metric_ids)
trend_metrics = ["avg_tas", "sum_pr"]  # trends (trend_metric_ids)
stationary = False  # False: non-stationary (trend) MLE fits
nonstationary_scale = True  # non-stationary only: trend in the scale as well as location
return_period = 100
n_boot = 100  # extremes: bootstrap samples in the fit files (100 or 1000)
n_boot_trends = 1000  # trends: bootstrap samples in the fit files (250 or 1000)
n_min_members = 5
if analysis_type == "trends":
    metric_ids = trend_metrics
    plot_col = "slope"  # per year; scaled to per decade when read
    data_dir = f"{project_data_path}/trends/cities/original_grid"
    file_str = f"2015-2100_nboot{n_boot_trends}"
    fit_str = "linear trend, 2015-2100"
    change_str = "trend"
    store_str = "trend"
    unit_labels = pu.trend_labels_abs
    value_scale = 10
elif stationary:
    plot_col = f"{return_period}yr_return_level_diff"
    fit_method = "lmom"
    hist_slice, proj_slice = "1950-2014", "2050-2100"
    fit_str = f"{proj_slice} vs {hist_slice}; stationary GEV"
    file_str = f"{hist_slice}_{proj_slice}_{fit_method}_stat_nbootproj{n_boot}_nboothist1"
    store_str = "stat"
else:
    plot_col = f"{return_period}yr_return_level_diff_2075-1975"
    fit_method = "mle"
    scale_str = "_scale" if nonstationary_scale else ""
    fit_str = f"2075 vs 1975; non-stationary GEV"
    file_str = f"1950-2100_{fit_method}_nonstat{scale_str}_nboot{n_boot}"
    store_str = f"nonstat{scale_str}"
if analysis_type == "extremes":
    data_dir = f"{project_data_path}/extreme_value/cities/original_grid/freq"
    change_str = f"change in {return_period}-year return level"
    unit_labels = pu.gev_labels
    value_scale = 1
store = True  # save figures to figs/si

assert analysis_type in ["extremes", "trends"]
valid_metrics = gev_metric_ids if analysis_type == "extremes" else trend_metric_ids
assert city in city_list and set(metric_ids) <= set(valid_metrics)

ensemble_colors = {"LOCA2": "#4477AA", "STAR-ESDM": "#CCBB44", "GARD-LENS": "#AA3377"}
all_color = "0.6"

# Uncertainty types shown, in the order and with the labels of the main figures
uc_types = ["ssp_uc", "gcm_uc", "iv_uc", "dsc_uc", "fit_uc"]
uc_ticklabels = [pu.uc_labels[u if u != "fit_uc" else "fit_uc_mean"].replace(" ", "\n") for u in uc_types]

# %% [markdown]
# ## Read and calculate group-level uncertainties

# %%
def read_city_uc_groups(city, metric_id):
    df = pd.read_csv(
        f"{data_dir}/{city}_{metric_id}_{file_str}_samples.csv"
    )
    df[plot_col] *= value_scale
    # Bootstrap samples of the non-stationary fits are stored negated for min_tasmin
    if analysis_type == "extremes" and metric_id == "min_tasmin" and not stationary:
        df.loc[df["n_boot"] != "main", plot_col] *= -1
    df["gcm"] = df["gcm"].replace(
        {"canesm5": "CanESM5", "cesm2": "CESM2-LENS", "ecearth3": "EC-Earth3"}
    )
    return sacu.calculate_uc_groups(df, plot_col, n_min_members=n_min_members)


groups = {m: read_city_uc_groups(city, m) for m in metric_ids}

# %% [markdown]
# ## Conditional uncertainty figure
#
# **a)** Each uncertainty conditional on SSP (top row) and on the downscaling method (bottom row). A
# source is missing from the row that conditions on a factor it is defined over (e.g. no
# scenario uncertainty by SSP), and where a method has no estimate (e.g. GARD-LENS only has one SSP).
#
# **b)** Internal variability by GCM and SSP, grouped by downscaling method. Only GCMs with at least
# `n_min_members` members in an ensemble-SSP combination have an estimate (STAR-ESDM has one
# member per GCM, so is absent).

# %%
def plot_conditional_bars(ax, df_groups, by, colors, labels=None):
    levels = list(colors.keys())
    df_all = sacu.summarise_uc_groups(df_groups, by=None)
    df_by = sacu.summarise_uc_groups(df_groups, by=by)

    n = len(levels) + 1
    width = 0.8 / n
    for i, level in enumerate([None] + levels):
        df_l = df_all if level is None else df_by[df_by["level"] == level]
        df_l = df_l.set_index("uc_type").reindex(uc_types)
        ax.bar(
            np.arange(len(uc_types)) + (i - (n - 1) / 2) * width,
            df_l["mean"],
            width=width,
            color=all_color if level is None else colors[level],
            label="All" if level is None else (labels or {}).get(level, level),
            zorder=3,
        )
    ax.set_xticks(np.arange(len(uc_types)), uc_ticklabels, fontsize=9)
    ax.grid(alpha=0.2, zorder=0)
    ax.set_ylim(bottom=0)


def plot_iv_by_gcm(ax, df_groups, ensembles=("LOCA2", "GARD-LENS")):
    """One bar per GCM-SSP; GCMs grouped by downscaling method."""
    df = df_groups[df_groups["uc_type"] == "iv_uc"]
    ssps = list(pu.ssp_colors.keys())
    width = 0.8 / len(ssps)

    xticks, xticklabels, x0 = [], [], 0
    for ensemble in ensembles:
        df_e = df[df["ensemble"] == ensemble]
        gcms = sorted(df_e["gcm"].unique())
        for i, ssp in enumerate(ssps):
            ax.bar(
                x0 + np.arange(len(gcms)) + (i - (len(ssps) - 1) / 2) * width,
                df_e[df_e["ssp"] == ssp].set_index("gcm")["value"].reindex(gcms),
                width=width,
                color=pu.ssp_colors[ssp],
                label=pu.ssp_labels[ssp] if ensemble == ensembles[0] else None,
                zorder=3,
            )
        xticks += list(x0 + np.arange(len(gcms)))
        xticklabels += gcms
        ax.text(
            x0 + (len(gcms) - 1) / 2, 1.01, ensemble,
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=12,
        )
        x0 += len(gcms) + 1
        if ensemble != ensembles[-1]:
            ax.axvline(x0 - 1, color="black", lw=1)
    ax.set_xticks(xticks, xticklabels, rotation=-45, ha="left", fontsize=8)
    ax.set_xlim(-0.6, x0 - 1.4)
    ax.grid(alpha=0.2, zorder=0)
    ax.set_ylim(bottom=0)


fig = plt.figure(figsize=(4.5 * len(metric_ids), 13), layout="constrained")
subfig_a, subfig_b = fig.subfigures(2, 1, height_ratios=[2, 1.3])

# a) Conditional on SSP and method
axs = subfig_a.subplots(2, len(metric_ids), sharex=True)
axs = axs.reshape(2, -1)
for idm, metric_id in enumerate(metric_ids):
    plot_conditional_bars(axs[0, idm], groups[metric_id], "ssp", pu.ssp_colors, pu.ssp_labels)
    plot_conditional_bars(axs[1, idm], groups[metric_id], "ensemble", ensemble_colors)
    axs[0, idm].set_title(pu.title_labels[metric_id])
    for ax in axs[:, idm]:
        ax.set_ylabel(f"95% range {unit_labels[metric_id]}")
axs[0, -1].legend(title="Conditional on SSP", fontsize=9, title_fontsize=9, loc="upper left")
axs[1, -1].legend(title="Conditional on method", fontsize=9, title_fontsize=9, loc="upper left")
subfig_a.suptitle("a) Conditional on SSP (top) and downscaling method (bottom)", x=0.03, ha="left", fontsize=16, fontstyle='italic')

# b) Internal variability by GCM
axs = subfig_b.subplots(1, len(metric_ids))
for idm, metric_id in enumerate(metric_ids):
    plot_iv_by_gcm(axs[idm], groups[metric_id])
    axs[idm].set_ylabel(f"95% range {unit_labels[metric_id]}")
handles, labels = axs[0].get_legend_handles_labels()
subfig_b.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=10)
subfig_b.suptitle("b) Internal variability by GCM and SSP", x=0.03, ha="left", fontsize=16, fontstyle='italic')

fig.suptitle(
    f"{pu.city_names[city]}: conditional uncertainty in {change_str} "
    f"({fit_str})",
    fontweight="bold",
    fontsize=18
)

if store:
    plt.savefig(f"{project_code_path}/figs/si/{city}_conditional_uc_{store_str}.pdf", bbox_inches="tight")
plt.show()

# %%
