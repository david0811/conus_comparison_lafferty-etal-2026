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

import metric_funcs as mf

# %% [markdown]
# ## Calculate metrics 

# %% [markdown]
# ### Preliminaries

# %%
################
#### Paths #####
################
# NOTE: this is run on a different system from other datasets
# Update these for reproduction
from utils import hopper_code_path as project_code_path
from utils import hopper_data_path as project_data_path

gard_path = "/home/shared/vs498_0001/GARD-LENS" # GARD-LENS raw
gard_gcms = ['canesm5', 'cesm2', 'ecearth3']

# %%
# Check all same
gardlens_info = {}

for gcm in gard_gcms: 
    t_mean_files = glob(f"{gard_path}/t_mean/GARDLENS_{gcm}_*.nc")
    t_range_files = glob(f"{gard_path}/t_range/GARDLENS_{gcm}_*.nc")
    pcp_files = glob(f"{gard_path}/pcp/GARDLENS_{gcm}_*.nc")
    assert len(t_mean_files) == len(t_range_files)
    assert len(t_mean_files) == len(pcp_files)

# %%
# Get all model members
models_members = glob(f"{gard_path}/t_mean/GARDLENS_*.nc")
models_members = [file.split('GARDLENS')[1].split('t_')[0][1:-1] for file in models_members]


# %%
###############################
# Metric calulcation function #
###############################
def calculate_metric(metric_func, var_id, model_member, needed_vars, gard_path, out_path):
    """
    Inputs: selected model, member, variable, and metric to calculate (from GARD-LENS)
    Outputs: calculated (annual) metric
    """
    try:
        # Check if done
        if os.path.isfile(out_path):
            # print(f"{model_member} already done.")
            return None

        # Read
        if model_member.split('_')[0] == 'ecearth3':
            time_range = '1970_2100'
        else:
            time_range = '1950_2100'
            
        # Read
        ds_tmp = xr.merge([xr.open_dataset(f"{gard_path}/{var}/GARDLENS_{model_member}_{var}_{time_range}_CONUS.nc", chunks='auto') for var in needed_vars])
    
        # Calculate metric
        ds_out = metric_func(ds_tmp, var_id)
    
        # Store
        ds_out.to_netcdf(out_path)
        print(f"{model_member}")
            
    # Log if error
    except Exception as e:
        except_path = f"{project_code_path}/scripts/logs"
        with open(f"{except_path}/{model_member}_{var_id}_GARDLENS.txt", "w") as f:
            f.write(str(e))


# %%
############
### Dask ###
############
from dask.distributed import LocalCluster
cluster = LocalCluster(n_workers = 20)
client = cluster.get_client()
client

# %% [markdown]
# ### Calculations

# %%
# %%time
##############################
# Average temperature
##############################
var_id = 'tas'
metric_func = mf.calculate_avg
needed_vars = ['t_mean']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/avg_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
##############################
# Maximum temperature: avg
##############################
var_id = 'tasmax'
metric_func = mf.calculate_avg
needed_vars = ['t_mean', 't_range']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/avg_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
##############################
# Maximum temperature: max
##############################
var_id = 'tasmax'
metric_func = mf.calculate_max
needed_vars = ['t_mean', 't_range']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/max_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
##############################
# Minimum temperature: avg
##############################
var_id = 'tasmin'
metric_func = mf.calculate_avg
needed_vars = ['t_mean', 't_range']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/avg_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
##############################
# Minimum temperature: min
##############################
var_id = 'tasmin'
metric_func = mf.calculate_min
needed_vars = ['t_mean', 't_range']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/min_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
#################
# Maximum precip
#################
var_id = 'pr'
metric_func = mf.calculate_max
needed_vars = ['pcp']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/max_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
#################
# Sum precip
#################
var_id = 'pr'
metric_func = mf.calculate_sum
needed_vars = ['pcp']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/sum_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
##############################
# Cooling degree days: max
##############################
var_id = 'cdd'
metric_func = mf.calculate_dd_max
needed_vars = ['t_mean', 't_range']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/max_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
##############################
# Cooling degree days: sum
##############################
var_id = 'cdd'
metric_func = mf.calculate_dd_sum
needed_vars = ['t_mean', 't_range']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/sum_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
##############################
# Heating degree days: max
##############################
var_id = 'hdd'
metric_func = mf.calculate_dd_max
needed_vars = ['t_mean', 't_range']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/max_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%
# %%time
##############################
# Heating degree days: sum
##############################
var_id = 'hdd'
metric_func = mf.calculate_dd_sum
needed_vars = ['t_mean', 't_range']

out_path = lambda model_member: f'{project_data_path}/metrics/GARD-LENS/sum_{var_id}_{model_member}_ssp370.nc'

for model_member in models_members:
    # Calculate metric
    calculate_metric(metric_func = metric_func,
                     var_id = var_id,
                     model_member = model_member,
                     needed_vars = needed_vars,
                     gard_path = gard_path,
                     out_path = out_path(model_member))

# %%

# %%
