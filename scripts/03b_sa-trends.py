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
import os
from glob import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs

import sa_utils as sau
import plotting_utils as pu

from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path
from utils import trend_metric_ids as metric_ids

# %%
# Set fonts
import matplotlib.pyplot as plt
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
# For summary stats
encoding={'quantile': {'dtype': 'U6'}, 'ensemble': {'dtype': 'U9'}, 'ssp': {'dtype': 'U6'}}

# %%
metric_ids = [
    'avg_tas',
    # 'avg_tasmax',
    # 'avg_tasmin',
    'sum_pr',
    'max_tasmax',
    'min_tasmin',
    'max_pr'
]

# %% [markdown]
# ## Analysis

# %%
import importlib
importlib.reload(sau)

# %%
# Calculate all and store
grids = ["LOCA2"]
regrid_method = "nearest"
col_name = "slope"
proj_slice = "2015-2100"
hist_slice = None
fit_method = None
stationary = None
n_boot_proj = 250
stat_name = None

# Loop through metrics
for metric_id in metric_ids:
    # Do for each grid
    for grid in grids:
        # Do for abs, norm space
        for rel in [True, False]:
            if rel and metric_id not in ['max_pr', 'sum_pr']:
                continue
            # UC
            sau.uc_all(metric_id=metric_id,
                       grid=grid,
                       fit_method=fit_method,
                       stationary=stationary,
                       regrid_method=regrid_method,
                       proj_slice=proj_slice,
                       hist_slice=hist_slice,
                       rel = rel,
                       time_name = proj_slice,
                       col_name_main=col_name,
                       stat_name = stat_name,
                       col_name_boot=col_name,
                       n_boot_proj=n_boot_proj,
                       analysis_type="trends")
            # Summary stats
            sau.summary_stats_main(metric_id=metric_id,
                                   grid=grid,
                                   fit_method=fit_method,
                                   stationary=stationary,
                                   stat_name=stat_name,
                                   regrid_method=regrid_method,
                                   proj_slice=proj_slice,
                                   hist_slice=hist_slice,
                                   rel=rel,
                                   col_name=col_name,
                                   col_name_boot=col_name,
                                   time_name=proj_slice,
                                   analysis_type="trends")

# %%
