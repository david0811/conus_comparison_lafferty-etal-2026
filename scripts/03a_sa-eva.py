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
import numpy as np
import pandas as pd
import xarray as xr
import dask
import sa_utils as sau
import plotting_utils as pu

from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path
from utils import gev_metric_ids, gard_gcms

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%
# For summary stats
encoding={'quantile': {'dtype': 'U6'}, 'ensemble': {'dtype': 'U9'}, 'ssp': {'dtype': 'U6'}}

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Non-stationary

# %%
##################################################################
# Non-stationary scale bootstrap, absolute projections
##################################################################
grid = "LOCA2"
regrid_method = "nearest"
proj_slice = "1950-2100"
hist_slice = None
return_levels = [10,25,50,100]
fit_method = "mle"
stationary = False
stat_name = "nonstat_scale"

_preprocess_func = lambda x: x.sel(time=2075)
time_name = '2075'

# Loop through metrics
for metric_id in gev_metric_ids[:3]:
    # Loop through return levels
    for return_level in return_levels:
        # UC
        sau.uc_all(metric_id=metric_id,
                   grid=grid,
                   fit_method=fit_method,
                   stationary=stationary,
                   stat_name=stat_name,
                   regrid_method=regrid_method,
                   proj_slice=proj_slice,
                   hist_slice=hist_slice,
                   col_name_main=f"{return_level}yr_return_level",
                   col_name_boot=f"{return_level}yr_return_level",
                   time_name=time_name,
                   _preprocess_func_boot=_preprocess_func,
                   _preprocess_func_main=_preprocess_func)
        # Summary
        sau.summary_stats_main(metric_id=metric_id,
                               grid=grid,
                               fit_method=fit_method,
                               stationary=stationary,
                               stat_name=stat_name,
                               regrid_method=regrid_method,
                               proj_slice=proj_slice,
                               hist_slice=hist_slice,
                               time_name=time_name,
                               col_name=f"{return_level}yr_return_level",
                               col_name_boot=f"{return_level}yr_return_level",
                                _preprocess_func=_preprocess_func)

# %%
##################################################################
# Non-stationary scale bootstrap, change from hist, abs
##################################################################
grid = "LOCA2"
regrid_method = "nearest"
proj_slice = "1950-2100"
hist_slice = None
return_levels = [10,25,50,100]
fit_method = "mle"
stationary = False
stat_name = "nonstat_scale"

_preprocess_func_main = lambda x: x.sel(time=2075) - x.sel(time=1975)
time_name = '2075-1975'
_preprocess_func_boot = lambda x: x.sel(time_diff = time_name)

# Loop through metrics
for metric_id in gev_metric_ids[:3]:
    # Loop through return levels
    for return_level in return_levels:
        # Calculate UC
        sau.uc_all(metric_id=metric_id,
                   grid=grid,
                   fit_method=fit_method,
                   stationary=stationary,
                   stat_name=stat_name,
                   regrid_method=regrid_method,
                   proj_slice=proj_slice,
                   hist_slice=hist_slice,
                   col_name_main=f"{return_level}yr_return_level",
                   col_name_boot=f"{return_level}yr_return_level_diff",
                   time_name=time_name,
                   _preprocess_func_boot=_preprocess_func_boot,
                   _preprocess_func_main=_preprocess_func_main,)
        # Summary
        sau.summary_stats_main(metric_id=metric_id,
                               grid=grid,
                               fit_method=fit_method,
                               stationary=stationary,
                               stat_name=stat_name,
                               regrid_method=regrid_method,
                               proj_slice=proj_slice,
                               hist_slice=hist_slice,
                               time_name=time_name,
                               col_name=f"{return_level}yr_return_level",
                               col_name_boot=f"{return_level}yr_return_level_diff",
                               _preprocess_func=_preprocess_func_main)

# %%
##################################################################
# Non-stationary scale bootstrap, change from hist, change factor
##################################################################
grid = "LOCA2"
regrid_method = "nearest"
proj_slice = "1950-2100"
hist_slice = None
return_levels = [10,25,50,100]
fit_method = "mle"
stationary = False
stat_name = "nonstat_scale"

_preprocess_func_main = lambda x: x.sel(time=2075) / x.sel(time=1975)
time_name = '2075-1975'
_preprocess_func_boot = lambda x: x.sel(time_diff = time_name)

metric_id  ='max_pr'

# Loop through return levels
for return_level in return_levels:
    # UC
    sau.uc_all(metric_id=metric_id,
               grid=grid,
               fit_method=fit_method,
               stationary=stationary,
               stat_name=stat_name,
               regrid_method=regrid_method,
               proj_slice=proj_slice,
               hist_slice=hist_slice,
               col_name_main=f"{return_level}yr_return_level",
               col_name_boot=f"{return_level}yr_return_level_chfc",
               time_name=time_name,
               _preprocess_func_boot=_preprocess_func_boot,
               _preprocess_func_main=_preprocess_func_main)
    # Summary
    sau.summary_stats_main(metric_id=metric_id,
                           grid=grid,
                           fit_method=fit_method,
                           stationary=stationary,
                           stat_name=stat_name,
                           regrid_method=regrid_method,
                           proj_slice=proj_slice,
                           hist_slice=hist_slice,
                           time_name=time_name,
                           col_name=f"{return_level}yr_return_level",
                           col_name_boot=f"{return_level}yr_return_level_chfc",
                           _preprocess_func=_preprocess_func_main)

# %% [markdown]
# ### Stationary

# %%
# Stationary bootstrap
grid = "LOCA2"
regrid_method = "nearest"
proj_slice = "2050-2100"
return_levels = [10,25,50,100]
fit_method = "lmom"
stationary = True
stat_str = "stat"
filter_vals = None
filter_str = ""

preprocess_funcs = {
    'diff': lambda x: x.sel(time='diff'), # diff
    'chfc': lambda x: x.sel(time='chfc'), # change factor
    'proj': lambda x: x.sel(time='proj'), # absolute projection
}


# Loop through metrics
for metric_id in gev_metric_ids[:3]:
    # Do for projection, change
    for time_name in list(preprocess_funcs.keys()):
        _preprocess_func = preprocess_funcs[time_name]
        hist_slice = None if time_name == 'proj' else "1950-2014"
        # Loop through return levels
        for return_level in return_levels:
            # UC
            sau.uc_all(metric_id=metric_id,
                       grid=grid,
                       fit_method=fit_method,
                       stationary=stationary,
                       regrid_method=regrid_method,
                       proj_slice=proj_slice,
                       stat_name=stat_str,
                       time_name=time_name,
                       hist_slice=hist_slice,
                       col_name_main=f"{return_level}yr_return_level",
                       col_name_boot=f"{return_level}yr_return_level",
                       _preprocess_func_boot=_preprocess_func)
            # Summary
            sau.summary_stats_main(metric_id=metric_id,
                                   grid=grid,
                                   fit_method=fit_method,
                                   stationary=stationary,
                                   stat_name=stat_str,
                                   regrid_method=regrid_method,
                                   proj_slice=proj_slice,
                                   hist_slice=hist_slice,
                                   time_name=time_name,
                                   col_name=f"{return_level}yr_return_level",
                                   col_name_boot=f"{return_level}yr_return_level",)

# %%
