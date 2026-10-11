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
# For summary stats
encoding={'quantile': {'dtype': 'U6'}, 'ensemble': {'dtype': 'U9'}, 'ssp': {'dtype': 'U6'}}

# %% [markdown]
# ## Analysis

# %%
# Calculate all and store
# grids = ["LOCA2", "STAR-ESDM"]
grids = ["LOCA2"]
regrid_method = "nearest"
proj_slices = ["2040-2060", "2080-2100"]
hist_slices = [None, "1950-2014"]
fit_method = None
stationary = None
stat_name = None

# Loop through metrics
for metric_id in ['avg_tas', 'sum_pr']:
    # Get variable name
    col_name = metric_id.split("_")[1]
    # Do for each grid
    for grid in grids:
        # Do for each slice
        for proj_slice in proj_slices:
            for hist_slice in hist_slices:
                # Do for abs, norm space
                for rel in [True, False]:
                    # skip normalization for no change metrics
                    if rel == True and hist_slice is None:
                        continue
                    
                    with warnings.catch_warnings():
                        #### UC
                        sau.uc_all(metric_id=metric_id,
                                   grid=grid,
                                   fit_method=fit_method,
                                   stationary=stationary,
                                   regrid_method=regrid_method,
                                   proj_slice=proj_slice,
                                   hist_slice=hist_slice,
                                   stat_name=stat_name,
                                   rel=rel,
                                   time_name=None,
                                   col_name_main=col_name,
                                   col_name_boot=col_name,
                                   include_fit_uc = False,
                                   analysis_type="averages")
                    
                        ## Summary stats
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
                                               time_name=None,
                                               analysis_type="averages")

# %%
