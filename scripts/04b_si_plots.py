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

# %%
import matplotlib.pyplot as plt
import plotting_utils as pu
from utils import roar_code_path as project_code_path

# %%
# Set fonts
from matplotlib import font_manager
font_dirs = ["/storage/home/dcl5300/fonts"]

font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 1.5

# %%
# UC rank maps: non-stat diff
pu.plot_uc_rank_maps(
    plot_metric_ids=["max_tasmax", "min_tasmin", "max_pr"],
    proj_slice="1950-2100",
    hist_slice=None,
    stationary=False,
    title = "Uncertainty ranking for change in 100-year return level (1975 to 2075; non-stationary GEV)",
    store_path = f"{project_code_path}/figs/si/uc_rankings_100yr_nonstat-scale_diff.png"
)

# %%
# UC rank maps: non-stat abs
pu.plot_uc_rank_maps(
    plot_metric_ids=["max_tasmax", "min_tasmin", "max_pr"],
    proj_slice="1950-2100",
    hist_slice=None,
    stationary=False,
    time_str = '2075',
    title = "Uncertainty ranking for absolute 100-year return level (2075; non-stationary GEV)",
    store_path = f"{project_code_path}/figs/si/uc_rankings_100yr_nonstat-scale_2075.png"
)

# %%
import importlib
importlib.reload(pu)

# %%
# UC rank maps: stat diff
pu.plot_uc_rank_maps(
    plot_metric_ids=["max_tasmax", "min_tasmin", "max_pr"],
    proj_slice="2050-2100",
    hist_slice="1950-2014",
    stationary=True,
    fit_method='lmom',
    stat_str='stat',
    time_str = 'diff',
    title = "Uncertainty ranking for change in 100-year return level (1950-2014 to 2050-100; stationary GEV)",
    store_path = f"{project_code_path}/figs/si/uc_rankings_100yr_stat_diff.png"
)

# %%
# UC rank maps: trends
pu.plot_uc_rank_maps(
    plot_metric_ids=["avg_tas", "sum_pr"],
    proj_slice="2015-2100",
    hist_slice=None,
    stationary=None,
    analysis_type = "trends",
    plot_col="slope",
    plot_fit_uc = False,
    title = "Uncertainty ranking for trend in annual metrics (2015 to 2100)",
    figsize=(10,4.75),
    store_path = f"{project_code_path}/figs/si/uc_rankings_trends.png"
)

# %%
import importlib
importlib.reload(pu)

# %%
# Figure 3 equivalent but for trends
plt.rcParams["font.size"] = 12
pu.plot_ensemble_mean_uq(
    figsize=(10,12),
    plot_metric_ids=['max_tasmax', 'min_tasmin', 'max_pr'],
    plot_col='slope',
    analysis_type='trends',
    summary_title='a) Ensemble mean trend (2015-2100) and associated uncertainty',
    proj_slice = '2015-2100',
    height_ratios = [1,1.35],
    plot_fit_uc=False,
    x_title=0.01,
    a_y_title=1.12,
    a_y_titles=1.11,
    b_y_title=1.08,
    b_y_titles=1.08,
    hspace=0.12,
    store_path = f'{project_code_path}/figs/si/uq_trends_extremal_metrics.png'
)

# %%
# Trends in annual metrics
proj_slice='2015-2100'
plot_col='slope'

for metric_id in ['avg_tas', 'sum_pr']:
    pu.plot_ensemble_ssp_means(
        metric_id=metric_id,
        proj_slice=proj_slice,
        hist_slice=None,
        plot_col=plot_col,
        return_period=None,
        fit_method=None,
        stationary=None,
        title = f"Ensemble mean trend by SSP \n({pu.title_labels[metric_id]}; 2015-2100)",
        stat_str=None,
        time_str='diff',
        analysis_type='trends',
        store_path = f"{project_code_path}/figs/si/ensemble_ssp_{metric_id}_trend.png")

# %%
import importlib
importlib.reload(pu)

# %%
# Figure 2 equivalent but for levels
plt.rcParams["font.size"] = 12
for proj_slice in ['2040-2060', '2080-2100']:
    pu.plot_ensemble_mean_uq(
        figsize = (11,8),
        plot_metric_ids = ['avg_tas', 'sum_pr'],
        plot_col = 'average',
        analysis_type = 'averages',
        summary_title = f'Ensemble mean climatology ({proj_slice}) and associated uncertainty',
        proj_slice = proj_slice,
        plot_fit_uc = False,
        height_ratios = [1.2, 2],
        x_title=0.01,
        a_y_titles=0.98,
        a_y_title=1.08,
        b_y_title=1.1,
        b_y_titles=1.08,
        fs=10,
        store_path = f'{project_code_path}/figs/si/uq_averages_{proj_slice}.png'
    )

# %%
# Changes + uncertainty, non-stationary GEV
proj_slice='1950-2100'
plot_col='100yr_return_level'
return_period=100

for metric_id in ['max_tasmax', 'min_tasmin', 'max_pr']:
    pu.plot_ensemble_ssp_means(
        metric_id=metric_id,
        proj_slice=proj_slice,
        hist_slice=None,
        plot_col=plot_col,
        return_period=return_period,
        fit_method='mle',
        stationary=False,
        title = f"Ensemble mean change by SSP in 100-year return level \n({pu.title_labels[metric_id]}; 2075 from 1975; non-stationary GEV)",
        stat_str='nonstat_scale',
        time_str='diff_2075-1975',
        analysis_type='extreme_value',
        store_path = f"{project_code_path}/figs/si/ensemble_ssp_{metric_id}_nonstat-scale_diff.png")

# %%
# Changes + uncertainty, stationary GEV
proj_slice='2050-2100'
hist_slice='1950-2014'
plot_col='100yr_return_level'
return_period=100

for metric_id in ['max_tasmax', 'min_tasmin', 'max_pr']:
    pu.plot_ensemble_ssp_means(
        metric_id=metric_id,
        proj_slice=proj_slice,
        hist_slice=hist_slice,
        plot_col=plot_col,
        return_period=return_period,
        fit_method='lmom',
        stationary=True,
        title = f"Ensemble mean change by SSP in 100-year return level \n({pu.title_labels[metric_id]}; 2050-2100 from 1950-2014; stationary GEV)",
        stat_str='stat',
        time_str='diff',
        analysis_type='extreme_value',
        store_path = f"{project_code_path}/figs/si/ensemble_ssp_{metric_id}_stat_diff.png")

# %%
import importlib
importlib.reload(pu)

# %%
# Levels + uncertainty, stationary GEV
proj_slice='2050-2100'
hist_slice=None
plot_col='100yr_return_level'
return_period=100

for metric_id in ['max_tasmax', 'min_tasmin', 'max_pr']:
    pu.plot_ensemble_ssp_means(
        metric_id=metric_id,
        proj_slice=proj_slice,
        hist_slice=hist_slice,
        plot_col=plot_col,
        return_period=return_period,
        fit_method='lmom',
        stationary=True,
        title = f"Ensemble mean 100-year return level by SSP \n({pu.title_labels[metric_id]}; 2050-2100; stationary GEV)",
        stat_str='stat',
        time_str='proj',
        analysis_type='extreme_value',
        store_path = f"{project_code_path}/figs/si/ensemble_ssp_{metric_id}_stat_proj.png")

# %%
# Levels + uncertainty, non-stationary GEV
proj_slice='1950-2100'
hist_slice=None
plot_col='100yr_return_level'
return_period=100

for metric_id in ['max_tasmax', 'min_tasmin', 'max_pr']:
    pu.plot_ensemble_ssp_means(
        metric_id=metric_id,
        proj_slice=proj_slice,
        hist_slice=hist_slice,
        plot_col=plot_col,
        return_period=return_period,
        fit_method='mle',
        stationary=False,
        title = f"Ensemble mean 100-year return level by SSP \n({pu.title_labels[metric_id]}; 2075; non-stationary GEV)",
        stat_str='nonstat_scale',
        time_str='2075',
        analysis_type='extreme_value',
        store_path = f"{project_code_path}/figs/si/ensemble_ssp_{metric_id}_nonstat-scale_proj.png")

# %%
# UQ: changes, stationary GEV
pu.plot_ensemble_mean_uq(
    figsize=(10,10.5),
    plot_metric_ids=['max_tasmax', 'min_tasmin', 'max_pr'],
    plot_col='100yr_return_level',
    analysis_type='extreme_value',
    summary_title="a) Ensemble mean change in 100-year return level and associated uncertainty (1950-2014 to 2050-2100; stationary GEV)",
    time_str = 'diff',
    proj_slice='2050-2100',
    hist_slice='1950-2014',
    return_period=100,
    fit_method='lmom',
    vmax_uc=60,
    stationary=True,
    stat_str='stat',
    height_ratios = [1,1.35],
    x_title=0.01,
    a_y_title=1.12,
    a_y_titles=1.11,
    b_y_title=1.08,
    b_y_titles=1.08,
    hspace=0.12,
    store_path = f"{project_code_path}/figs/si/uq_extremes_stat_diff.png"
)

# %%
# UQ: levels, non-stationary GEV
pu.plot_ensemble_mean_uq(
    figsize=(10,10.5),
    plot_metric_ids=['max_tasmax', 'min_tasmin', 'max_pr'],
    plot_col='100yr_return_level',
    analysis_type='extreme_value',
    summary_title="a) Ensemble mean 100-year return level and associated uncertainty (2075; non-stationary GEV)",
    time_str = '2075',
    height_ratios = [1,1.35],
    x_title=0.01,
    a_y_title=1.12,
    a_y_titles=1.11,
    b_y_title=1.08,
    b_y_titles=1.08,
    hspace=0.12,
    store_path = f"{project_code_path}/figs/si/uq_extremes_nonstat-scale_2075.png"
)

# %%
# UQ: levels, stationary GEV
pu.plot_ensemble_mean_uq(
    figsize=(10,10.5),
    plot_metric_ids=['max_tasmax', 'min_tasmin', 'max_pr'],
    plot_col='100yr_return_level',
    analysis_type='extreme_value',
    summary_title="a) Ensemble mean 100-year return level and associated uncertainty (2050-2100; stationary GEV)",
    time_str = 'proj',
    proj_slice='2050-2100',
    hist_slice=None,
    return_period=100,
    fit_method='lmom',
    vmax_uc=60,
    stationary=True,
    stat_str='stat',
    height_ratios = [1,1.35],
    x_title=0.01,
    a_y_title=1.12,
    a_y_titles=1.11,
    b_y_title=1.08,
    b_y_titles=1.08,
    hspace=0.12,
    store_path = f"{project_code_path}/figs/si/uq_extremes_stat_2050-2100.png"
)

# %%
# UQ: changes, non-stationary GEV, relative precip
pu.plot_ensemble_mean_uq(
    figsize=(12,5),
    plot_metric_ids=['max_pr'],
    rel_metric_ids=['max_pr'],
    plot_col='100yr_return_level',
    analysis_type='extreme_value',
    summary_title="a) Ensemble mean change in 100-year return level and associated uncertainty (1975 to 2075; non-stationary GEV)",
    stat_str='nonstat_scale',
    time_str='chfc_2075-1975',
    proj_slice='1950-2100',
    hist_slice=None,
    return_period=100,
    fit_method='mle',
    vmax_uc=80,
    stationary=False,
    height_ratios = [1,1],
    x_title=0.01,
    a_y_title=1.25,
    a_y_titles=1.12,
    b_y_title=1.15,
    b_y_titles=1.05,
    hspace=0.18,
    cbar_ax = [0.4, 0.01, 0.2, 0.05],
    narrow_subfigs=True,
    store_path = f"{project_code_path}/figs/si/uq_extremes_nonstat-scale_chfc.png"
)

# %%
# UQ: changes, trend, relative precip
pu.plot_ensemble_mean_uq(
    figsize=(12,5.5),
    plot_metric_ids=['sum_pr'],
    rel_metric_ids=['sum_pr'],
    plot_col = 'slope',
    analysis_type = 'trends',
    summary_title = 'a) Ensemble mean trend (2015-2100) and associated uncertainty',
    proj_slice = '2015-2100',
    plot_fit_uc=False,
    height_ratios = [1,1],
    x_title=0.01,
    a_y_title=1.25,
    a_y_titles=1.12,
    b_y_title=1.15,
    b_y_titles=1.05,
    hspace=0.18,
    cbar_ax = [0.4, 0.01, 0.2, 0.05],
    narrow_subfigs=True,
    store_path = f"{project_code_path}/figs/si/uq_trend_chfc.png"
)

# %%
