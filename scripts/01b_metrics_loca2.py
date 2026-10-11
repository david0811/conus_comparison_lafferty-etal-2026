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
# ## Preliminaries

# %%
################
#### Paths #####
################
# Update these for reproduction
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path
loca_path = "/storage/group/pches/default/public/LOCA2" # raw loca outputs

# %%
##############
### Models ###
##############

gcms = os.listdir(f"{loca_path}/")
gcms.remove('training_data')
gcms.remove('scripts')
gcms.remove('README.md')

loca_all = {}

# Loop through gcms
for gcm in gcms:
    loca_all[gcm] = {}
    # Loop through members
    members = os.listdir(f"{loca_path}/{gcm}/0p0625deg/")
    for member in members:
        # Append SSPs
        ssps = os.listdir(f"{loca_path}/{gcm}/0p0625deg/{member}/")
        loca_all[gcm][member] = ssps

# %%
##############
### Models ###
##############
# Matches website (https://loca.ucsd.edu/loca-version-2-for-north-america-ca-jan-2023/) as of Jan 2023
print(f"# gcm: {len(gcms)}")
print(f"# gcm/expts: {np.sum([len(np.unique([item for row in [loca_all[gcm][member] for member in loca_all[gcm].keys()] for item in row])) for gcm in gcms])}")
print(f"# gcm/expts/ens: {np.sum([len(loca_all[gcm][ssp]) for gcm in gcms for ssp in loca_all[gcm]])}")

# %%
############
### Dask ###
############
from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    # account="pches",
    account="open",
    cores=1,
    memory="10GiB",
    walltime="12:00:00"
)

cluster.scale(jobs=20)  # ask for jobs

from dask.distributed import Client

client = Client(cluster)

client


# %% [markdown]
# # Calculate metrics

# %%
## File path function
def make_loca_file_path(loca_path, gcm, member, ssp, var):
    """
    Returns list of file paths for a given downscaled LOCA output.
    """
    out_path = f"{loca_path}/{gcm}/0p0625deg/{member}/{ssp}/{var}"

    if os.path.isdir(out_path):
         # Take latest version if possible
        files = glob(f"{out_path}/*_v2024*")
        # Check earlier version if empty
        if len(files) == 0:
            files = glob(f"{out_path}/*_v2022*")
            
        return files
    else:
        return []
    
## Unit conversion
def convert_units(ds):
    # Convert units
    for var in ds.keys():
        if ds[var].attrs['units'] == 'K':
            ds[var] = ds[var] - 273.15    
            ds[var].attrs["units"] = 'C'
        elif ds[var].attrs['units'] == 'kg m-2 s-1':
            ds[var] = ds[var] * 86400
            ds[var].attrs["units"] = 'mm/day'
    
    return ds


# %%
###############################
# Metric calulcation function #
###############################
def calculate_metric(metric_func, var_id, needed_vars, gcm, member, ssp, loca_path, out_path):
    """
    Inputs: selected gcm, member, ssp, variable, and metric to calculate (from LOCA)
    Outputs: calculated (annual) metric
    """ 
    # Get all file paths
    files = {}
    for var in needed_vars:
        files[var] = make_loca_file_path(loca_path, gcm, member, ssp, var)
        
    # Loop through LOCA2 time slices
    if ssp == "historical":
        time_slices = ["1950-2014"]
    else:
        time_slices = ["2015-2044", "2045-2074", "2075-2100"]
        
    for time_slice in time_slices:
        try:
            # Check if done
            save_path = out_path.replace('.nc', f'_{time_slice}.nc')
            if os.path.isfile(save_path):
                continue
            # Load
            files_to_load = [xr.open_dataset(file, chunks='auto') for var in needed_vars for file in files[var] if time_slice in file]
            ds_in = xr.merge(files_to_load, combine_attrs='drop_conflicts')
            # Convert units
            ds_in = convert_units(ds_in)
            
            # Calculate metric
            ds_out = metric_func(ds_in, var_id)
    
            # Store
            ds_out.to_netcdf(save_path)
            
        # Log if error
        except Exception as e:
            except_path = f"{project_code_path}/scripts/logs"
            with open(f"{except_path}/metric_calcs/LOCA2_{gcm}_{member}_{ssp}_{var_id}_{time_slice}.txt", "w") as f:
                f.write(str(e))


# %%
# %%time
#############
## CDD sum ##
#############
var_id = "cdd"
metric_func = mf.calculate_dd_sum
needed_vars = ['tasmin', 'tasmax']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/sum_{var_id}_{gcm}_{member}_{ssp}.nc"

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
#############
## CDD max ##
#############
var_id = "cdd"
metric_func = mf.calculate_dd_max
needed_vars = ['tasmin', 'tasmax']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/max_{var_id}_{gcm}_{member}_{ssp}.nc"

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
#############
## HDD sum ##
#############
var_id = "hdd"
metric_func = mf.calculate_dd_sum
needed_vars = ['tasmin', 'tasmax']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/sum_{var_id}_{gcm}_{member}_{ssp}.nc"

# Parallelize over dask delayed
delayed = []

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
#############
## HDD max ##
#############
var_id = "hdd"
metric_func = mf.calculate_dd_max
needed_vars = ['tasmin', 'tasmax']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/max_{var_id}_{gcm}_{member}_{ssp}.nc"

# Parallelize over dask delayed
delayed = []

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
#################################
## Average Minimum Temperature ##
#################################
var_id = "tasmin"
metric_func = mf.calculate_avg
needed_vars = ['tasmin']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/avg_{var_id}_{gcm}_{member}_{ssp}.nc"

# Parallelize over dask delayed
delayed = []

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
#################################
## Average Maximum Temperature ##
#################################
var_id = "tasmax"
metric_func = mf.calculate_avg
needed_vars = ['tasmax']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/avg_{var_id}_{gcm}_{member}_{ssp}.nc"

# Parallelize over dask delayed
delayed = []

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
#########################
## Average Temperature ##
#########################
var_id = "tas"
metric_func = mf.calculate_avg
needed_vars = ['tasmin', 'tasmax']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/avg_{var_id}_{gcm}_{member}_{ssp}.nc"

# Parallelize over dask delayed
delayed = []

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
#########################
## Maximum Temperature ##
#########################
var_id = "tasmax"
metric_func = mf.calculate_max
needed_vars = ['tasmax']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/max_{var_id}_{gcm}_{member}_{ssp}.nc"

# Parallelize over dask delayed
delayed = []

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
#########################
## Minimum Temperature ##
#########################
var_id = "tasmin"
metric_func = mf.calculate_min
needed_vars = ['tasmin']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/min_{var_id}_{gcm}_{member}_{ssp}.nc"

# Parallelize over dask delayed
delayed = []

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
#########################
## Maximum Precip ##
#########################
var_id = "pr"
metric_func = mf.calculate_max
needed_vars = ['pr']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/max_{var_id}_{gcm}_{member}_{ssp}.nc"

# Parallelize over dask delayed
delayed = []

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
# %%time
################
## Sum Precip ##
################
var_id = "pr"
metric_func = mf.calculate_sum
needed_vars = ['pr']

out_path = lambda gcm, ssp, member: f"{project_data_path}/metrics/LOCA2/sum_{var_id}_{gcm}_{member}_{ssp}.nc"

# Parallelize over dask delayed
delayed = []

# Loop through gcms
for gcm in gcms:
    # Loop through members
    for member in loca_all[gcm].keys():
        # Loop through SSPs
        for ssp in loca_all[gcm][member]:
            # Calculate metric
            calculate_metric(metric_func = metric_func,
                             var_id = var_id,
                             gcm = gcm,
                             ssp = ssp,
                             member=member,
                             needed_vars = needed_vars,
                             loca_path = loca_path,
                             out_path = out_path(gcm, ssp, member))

# %%
