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
import trend_utils as tu
from regridding import regrid
from utils import loca_gard_mapping, city_list, trend_metric_ids

# %%
################
#### Paths #####
################
# Update these for reproduction

project_data_path = "/storage/group/pches/default/users/dcl5300/conus_comparison_lafferty-etal-2024"
project_code_path = "/storage/home/dcl5300/work/current_projects/conus_comparison_lafferty-etal-2024"

# %%
#######################
### Dask
#######################
from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    # account="open",
    account="pches_cr_default",
    queue='basic',
    cores=1,
    processes=1,
    memory="5GiB",
    walltime="00:30:00",
)

cluster.scale(5)

from dask.distributed import Client
client = Client(cluster)
client

# %% [markdown]
# ## Fit trend

# %%
# Fit all: spatial, main
for metric_id in trend_metric_ids:
    tu.trend_fit_all(metric_id)

# %%
# Fit all: spatial, bootstrap
for metric_id in trend_metric_ids:
    tu.trend_fit_all(metric_id, n_boot=250)

# %% [markdown]
# ## Regridding

# %%
# %%time
# Regrid to LOCA using NN
method = "nearest"
target = "LOCA2"

for metric_id in trend_metric_ids:
    # Save path
    store_path = f"{project_data_path}/trends/loca_grid/{metric_id}"
    # Get all files
    files = glob(f"{project_data_path}/trends/original_grid/{metric_id}/STAR-ESDM*") + \
            glob(f"{project_data_path}/trends/original_grid/{metric_id}/GARD-LENS*")

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
            except Exception as e:
                print(file)
                print(e)

# %%
