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
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import plotting_utils as pu
import overview_plot_utils as opu

from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path

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
## Get correlations reported in main text
# avg tas
ds_summary = xr.open_dataset(f'{project_data_path}/results/summary_avg_tas_2015-2100_None_slope_2015-2100_None_None_LOCA2grid_nearest.nc')
ds_uq = xr.open_dataset(f'{project_data_path}/results/avg_tas_2015-2100_slope_2015-2100_None_None_LOCA2grid_nearest.nc')
corr = xr.corr(ds_uq['uc_95w_main'], ds_summary.sel(quantile='mean').mean(dim=["ensemble", "ssp"])['slope']).item()
print(f'avg tas corr: {corr:.2f}')

# sum pr
ds_summary = xr.open_dataset(f'{project_data_path}/results/summary_sum_pr_2015-2100_slope_2015-2100_None_None_LOCA2grid_nearest.nc')
ds_uq = xr.open_dataset(f'{project_data_path}/results/sum_pr_2015-2100_slope_2015-2100_None_None_LOCA2grid_nearest.nc')
corr = xr.corr(ds_uq['uc_95w_main'], ds_summary.sel(quantile='mean').mean(dim=["ensemble", "ssp"])['slope']).item()
print(f'sum pr corr: {corr:.2f}')

# max pr
ds_summary = xr.open_dataset(f'{project_data_path}/results/summary_max_pr_1950-2100_100yr_return_level_diff_2075-1975_mle_nonstat_scale_LOCA2grid_nearest.nc')
ds_uq = xr.open_dataset(f'{project_data_path}/results/max_pr_1950-2100_100yr_return_level_diff_2075-1975_mle_nonstat_scale_LOCA2grid_nearest.nc')
corr = xr.corr(ds_uq['uc_95w_main'], ds_summary.sel(quantile='mean').mean(dim=["ensemble", "ssp"])['100yr_return_level']).item()
print(f'max pr corr: {corr:.2f}')

# %%
##########################
# FIGURE 1
##########################
opu.make_figure(city='nyc', metric_id='max_tasmax', save_fig=True, ylims=(-7.5, 15))

# %%
import importlib
importlib.reload(pu)

# %%
##########################
# FIGURE 2
##########################
plt.rcParams["font.size"] = 12
pu.plot_ensemble_mean_uq(
    figsize=(11,8),
    plot_metric_ids=['avg_tas', 'sum_pr'],
    plot_col = 'slope',
    analysis_type = 'trends',
    summary_title = 'a) Ensemble mean trend (2015-2100) and associated uncertainty',
    proj_slice = '2015-2100',
    norm = 'relative',
    vmax_uc = 60,
    plot_fit_uc=False,
    height_ratios=[1.2, 2],
    x_title=0.01,
    a_y_titles=0.98,
    a_y_title=1.08,
    b_y_title=1.1,
    b_y_titles=1.08,
    fs=10,
    store_path = f'{project_code_path}/figs/main/figure2.png'
)

# %%
######################
# FIGURE 3
######################
plt.rcParams["font.size"] = 12
pu.plot_ensemble_mean_uq(
    figsize=(10,10.5),
    plot_metric_ids=['max_tasmax', 'min_tasmin', 'max_pr'],
    plot_col='100yr_return_level',
    analysis_type='extreme_value',
    summary_title="a) Ensemble mean change in 100-year return level (1975 to 2075) and associated uncertainty",
    time_str = 'diff_2075-1975',
    norm = 'relative',
    vmax_uc = 60,
    height_ratios = [1,1.35],
    x_title=0.01,
    a_y_title=1.12,
    a_y_titles=1.11,
    b_y_title=1.08,
    b_y_titles=1.08,
    hspace=0.12,
    store_path = f'{project_code_path}/figs/main/figure3.png'
)

# %%
import importlib
importlib.reload(pu)

# %%
##################
# FIGURE 4
##################
plt.rcParams["font.size"] = 10
fs = 14

fig = plt.figure(figsize=(12, 11), layout='constrained')
subfigs = fig.subfigures(4, 1, hspace=0.05, height_ratios=[1, 1, 1, 0.1])
metric_ids = ['max_tasmax', 'min_tasmin', 'max_pr']

######### Nonstat change
subfigs[0].suptitle('a) Non-stationary GEV: change from 1975 to 2075', fontweight='bold', fontsize=fs)
subfigs_row0 = subfigs[0].subfigures(1, 3, wspace=0.05)

response_ylims = [(1.4, 9),(-1, 15.5),(-52,110)]
uc_ylims = [(0,35),(0,35),(0,45)]

for idm, metric_id in enumerate(metric_ids):
    subfigs_row0[idm].suptitle(f"{pu.subplot_labels[idm]} {pu.title_labels[metric_id]}", fontstyle="italic", fontsize=fs)
    axs0 = subfigs_row0[idm].subplots(1, 2, sharex=True)
    
    # Response
    pu.plot_response_rls(
        coord_or_mean="mean",
        proj_slice="1950-2100",
        hist_slice=None,
        fit_method="mle",
        stat_str="nonstat_scale",
        metric_ids=[metric_id],
        axs=axs0[0],
        idm_start=idm,
        legend=False,
        time_str='diff_2075-1975',
        ax_title=False,
        xlabel=True,
        xticklabels=True,
        ylims=response_ylims[idm], 
    )
    
    # UC
    pu.plot_uc_rls(
        coord_or_mean="mean",
        proj_slice="1950-2100",
        hist_slice=None,
        fit_method="mle",
        stat_str="nonstat_scale",
        metric_ids=[metric_id],
        axs=axs0[1],
        legend=False,
        ylim=uc_ylims[idm],
        time_str='diff_2075-1975',
        ax_title=False,
        xlabel=True,
        xticklabels=True,
        norm = "relative"
    )

######### Stat change
subfigs[1].suptitle('b) Stationary GEV: change from 1950-2014 to 2050-2100', fontweight='bold', fontsize=fs)
subfigs_row2 = subfigs[1].subfigures(1, 3, wspace=0.05)
for idm, metric_id in enumerate(metric_ids):
    subfigs_row2[idm].suptitle(f"{pu.subplot_labels[idm]} {pu.title_labels[metric_id]}", fontstyle="italic", fontsize=fs)
    axs2 = subfigs_row2[idm].subplots(1, 2, sharex=True)
    
    # Response
    pu.plot_response_rls(
        coord_or_mean="mean",
        proj_slice="2050-2100",
        hist_slice="1950-2014",
        fit_method="lmom",
        stat_str="stat",
        metric_ids=[metric_id],
        axs=axs2[0],
        idm_start=idm,
        legend=False,
        time_str='diff',
        ax_title=False,
        ylims=response_ylims[idm],
    )
    
    # UC
    legend_elements = pu.plot_uc_rls(
        coord_or_mean="mean",
        proj_slice="2050-2100",
        hist_slice="1950-2014",
        fit_method="lmom",
        stat_str="stat",
        metric_ids=[metric_id],
        axs=axs2[1],
        legend=False,
        ylim=uc_ylims[idm],
        time_str='diff',
        ax_title=False,
        return_legend=True,
        norm = "relative"
    )

######### Nonstat abs
subfigs[2].suptitle('c) Non-stationary GEV: absolute values in 2075', fontweight='bold', fontsize=fs)
subfigs_row1 = subfigs[2].subfigures(1, 3, wspace=0.05)
for idm, metric_id in enumerate(metric_ids):
    subfigs_row1[idm].suptitle(f"{pu.subplot_labels[idm]} {pu.title_labels[metric_id]}", fontstyle="italic", fontsize=fs)
    axs1 = subfigs_row1[idm].subplots(1, 2, sharex=True)
    
    # Response
    pu.plot_response_rls(
        coord_or_mean="mean",
        proj_slice="1950-2100",
        hist_slice=None,
        fit_method="mle",
        stat_str="nonstat_scale",
        metric_ids=[metric_id],
        axs=axs1[0],
        idm_start=idm,
        legend=False,
        time_str='2075',
        ax_title=False,
        xlabel=True,
        xticklabels=True
    )
    
    # UC
    pu.plot_uc_rls(
        coord_or_mean="mean",
        proj_slice="1950-2100",
        hist_slice=None,
        fit_method="mle",
        stat_str="nonstat_scale",
        metric_ids=[metric_id],
        axs=axs1[1],
        ylim=uc_ylims[idm],
        legend=False,
        time_str='2075',
        ax_title=False,
        xlabel=True,
        xticklabels=True,
        norm = "relative"
    )

######### Legend
ax_legend = subfigs[3].subplots(1, 1)
ax_legend.axis('off')
ax_legend.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=fs-2)

plt.savefig(f'{project_code_path}/figs/main/figure4.pdf', bbox_inches='tight')

# %%
