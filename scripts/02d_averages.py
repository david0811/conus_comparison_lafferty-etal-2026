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
import avg_utils as au
from regridding import regrid
from utils import loca_gard_mapping, city_list, trend_metric_ids
from utils import roar_code_path as project_code_path, roar_data_path as project_data_path

# %%
# ############
# ### Dask ###
# ############
# from dask_jobqueue import SLURMCluster

# cluster = SLURMCluster(
#     # account="pches",
#     account="open",
#     cores=1,
#     memory="10GiB",
#     walltime="00:30:00",
# )

# cluster.scale(jobs=20)  # ask for jobs

# from dask.distributed import Client

# client = Client(cluster)

# client

# %% [markdown]
# ## Calculate averages

# %%
import importlib
importlib.reload(au)

# %%
# Fit all
proj_year_groups = [None, [2040,2060], [2080,2100]]
hist_year_groups = [None, [1950,2014]]

for metric_id in trend_metric_ids:
    for proj_years in proj_year_groups:
        for hist_years in hist_year_groups:
            if not (proj_year_groups is None and hist_year_groups is None):
                au.avg_calc_all(metric_id = metric_id,
                                proj_years = proj_years,
                                hist_years = hist_years)

# %% [markdown]
# ## Regridding

# %%
# %%time
# Regrid to LOCA using NN
method = "nearest"
target = "LOCA2"

for metric_id in trend_metric_ids:
    # Save path
    store_path = f"{project_data_path}/averages/loca_grid/{metric_id}"
    # Get all files
    files = glob(f"{project_data_path}/averages/original_grid/{metric_id}/STAR-ESDM*") + \
            glob(f"{project_data_path}/averages/original_grid/{metric_id}/GARD-LENS*")

    # Loop through
    for file in files:
        # Check if done
        file_name = file.split('/')[-1]
        store_name = file_name.replace(".nc", f"_{method}.nc")
        if not os.path.exists(f"{store_path}/{store_name}"):
            try:
                # Regrid and store
                ds_tmp = xr.open_dataset(file)
                ds_out = regrid(ds_tmp, method=method, target=target)
                ds_out.to_netcdf(f"{store_path}/{store_name}")
                del ds_out, ds_tmp
            except Exception as e:
                print(file)
                print(e)

# %%
