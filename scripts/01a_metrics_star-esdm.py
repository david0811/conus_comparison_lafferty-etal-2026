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

import dask
import numpy as np
import pandas as pd
import xarray as xr

from utils import city_list, gev_metric_ids
import metric_funcs as mf

# %% [markdown]
# ### Preliminaries

# %%
################
#### Paths #####
################
# Update these for reproduction
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path

star_path = "/storage/group/pches/default/users/dcl5300/STAR-ESDM/"  # raw STAR-ESDM outputs

# %%
##############
### Models ###
##############
ssp245_gcms = np.unique(
    [
        file.split("/")[-1].split(".")[1]
        for file in glob(f"{star_path}/ssp245/*.nc")
    ]
)
ssp585_gcms = np.unique(
    [
        file.split("/")[-1].split(".")[1]
        for file in glob(f"{star_path}/ssp585/*.nc")
    ]
)

if (ssp245_gcms == ssp585_gcms).all():
    gcms = ssp245_gcms
else:
    print("Model mismatch")

# %%
############
### Dask ###
############
from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    # account="pches",
    account="open",
    cores=1,
    memory="20GiB",
    walltime="00:20:00",
)

cluster.scale(jobs=10)  # ask for jobs

from dask.distributed import Client

client = Client(cluster)

client


# %% [markdown]
# ## Calculate metrics

# %%
###############################
# Metric calulcation function #
###############################
def calculate_metric(
    metric_func, var_id, gcm, ssp, needed_vars, star_path, out_path
):
    """ 
    For calculating STAR-ESDM metrics.
    """

    # Some preprocessing
    def read_star(file_path):
        _preprocess = lambda ds: ds[[ds.encoding["source"].split("/")[-1].split(".")[3]]
        ]
        ds = xr.open_mfdataset(
            file_path,
            preprocess=_preprocess,
            decode_times=False,
            chunks={"time": 365, "latitude": -1, "longitude": -1},
        )
        ds.time.attrs["calendar"] = "365_day"
        return xr.decode_cf(ds, decode_times=True)

    # Start calculation
    try:
        # Check if done
        if os.path.isfile(out_path):
            print(f"{ssp} {gcm} already done.")
            return None

        # Read
        ds_tmp = xr.merge([read_star(f"{star_path}/{ssp}/downscaled.{gcm}.r1i1p1f1.{var}*") for var in needed_vars])

        # Calculate metric
        ds_out = metric_func(ds_tmp, var_id)

        # Store
        ds_out.to_netcdf(out_path)
        print(f"{ssp} {gcm}")

    # Log if error
    except Exception as e:
        except_path = f"{project_code_path}/code/logs"
        with open(f"{except_path}/metric_calcs/STAR-ESDM_{gcm}_{ssp}_{var_id}.txt", "w") as f:
            f.write(str(e))


# %%
# %%time
#############
## CDD max ##
#############
var_id = "cdd"
metric_func = mf.calculate_dd_max
needed_vars = ["tasmin", "tasmax"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/max_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#############
## CDD sum ##
#############
var_id = "cdd"
metric_func = mf.calculate_dd_sum
needed_vars = ["tasmin", "tasmax"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/sum_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#############
## HDD sum ##
#############
var_id = "hdd"
metric_func = mf.calculate_dd_sum
needed_vars = ["tasmin", "tasmax"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/sum_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#############
## HDD max ##
#############
var_id = "hdd"
metric_func = mf.calculate_dd_max
needed_vars = ["tasmin", "tasmax"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/max_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#########################
## Average Temperature ##
#########################
var_id = "tas"
metric_func = mf.calculate_avg
needed_vars = ["tasmin", "tasmax"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/avg_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#################################
## Average Minimum Temperature ##
#################################
var_id = "tasmin"
metric_func = mf.calculate_avg
needed_vars = ["tasmin"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/avg_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#################################
## Average Maximum Temperature ##
#################################
var_id = "tasmax"
metric_func = mf.calculate_avg
needed_vars = ["tasmax"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/avg_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#########################
## Total Precipitation ##
#########################
var_id = "pr"
metric_func = mf.calculate_sum
needed_vars = ["pr"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/sum_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#########################
## Maximum Temperature ##
#########################
var_id = "tasmax"
metric_func = mf.calculate_max
needed_vars = ["tasmax"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/max_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#########################
## Minimum Temperature ##
#########################
var_id = "tasmin"
metric_func = mf.calculate_min
needed_vars = ["tasmin"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/min_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
# %%time
#########################
# Maximum Precipitation #
#########################
var_id = "pr"
metric_func = mf.calculate_max
needed_vars = ["pr"]

out_path = (
    lambda gcm,
    ssp: f"{project_data_path}/metrics/STAR-ESDM/max_{var_id}_{gcm}_r1i1p1f1_{ssp}.nc"
)

# Loop through all
for ssp in ["ssp245", "ssp585"]:
    for gcm in gcms:
        # Calculate metric
        calculate_metric(
            metric_func=metric_func,
            var_id=var_id,
            gcm=gcm,
            ssp=ssp,
            needed_vars=needed_vars,
            star_path=star_path,
            out_path=out_path(gcm, ssp),
        )

# %%
