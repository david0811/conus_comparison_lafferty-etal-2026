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
import multiprocessing as mp

import dask
import numpy as np
import pandas as pd
import xarray as xr
import salem

from utils import city_list, gev_metric_ids
import metric_funcs as mf

# %%
print(mp.cpu_count())

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

tgw_path = "/home/shared/vs498_0001/im3_hyperfacets_tgw/"

tgw_scenarios = os.listdir(tgw_path)


# %%
# Worker function must be defined at module level for multiprocessing
def process_file(args):
    file_path, var_id_in, var_id_out, daily_agg_func, log_path = args
    try:
        return mf.tgw_hourly_to_daily(file_path=file_path,
                                     var_id_in=var_id_in,
                                     var_id_out=var_id_out,
                                     agg_func=daily_agg_func,
                                     log_path=log_path)
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

# Metric calculation function
def calculate_metric(scenario,
                     var_id_in,
                     var_id_out,
                     daily_agg_func,
                     metric_func,
                     tgw_path=tgw_path):
    # Log path
    log_path = f"{project_code_path}/scripts/logs"
    
    try:
        # Get all hourly files sorted
        file_paths = np.sort(glob(f"{tgw_path}/{scenario}/hourly/*"))
        
        # Prepare arguments for each file
        task_args = [(file_path, var_id_in, var_id_out, daily_agg_func, log_path) 
                    for file_path in file_paths]
        
        # Set number of processes
        num_processes = mp.cpu_count() - 5
        
        # Create a pool of workers and distribute the work
        with mp.Pool(processes=num_processes) as pool:
            # Map the worker function to all tasks
            results = pool.map(process_file, task_args)
            
            # Filter out None results
            ds_daily_all = [ds for ds in results if ds is not None]
        
        # Concatenate all results if we have any
        if ds_daily_all:
            ds_daily_all = xr.concat(ds_daily_all, dim='time')
            
            # Compute annual statistics
            ds_annual = metric_func(ds_daily_all, var_id_out)
            
            return ds_annual
        else:
            print(f"No valid results for scenario {scenario}")
            return None
            
    except Exception as e:
        with open(f"{log_path}/{scenario}_TGW.txt", "w") as f:
            f.write(str(e))
        print(f"Error processing scenario {scenario}: {str(e)}")
        return None


# %% [markdown]
# ### Calculations

# %%
# %%time
# max tasmax
var_id_in = 'T2C'
var_id_out = 'tasmax'
daily_agg_func = 'max'
metric_func = mf.calculate_max

# Loop through scenarios
for scenario in tgw_scenarios:
    save_path = f"{project_data_path}/metrics/TGW/max_{var_id_out}_{scenario}.nc"
    if not os.path.exists(save_path):
        # Calculate
        ds_out = calculate_metric(scenario=scenario,
                                  var_id_in=var_id_in,
                                  var_id_out=var_id_out,
                                  daily_agg_func=daily_agg_func,
                                  metric_func=metric_func)
        # Store
        ds_out.to_netcdf(save_path)
        print(f"Done: max_{var_id_out}_{scenario}.nc")
    else:
        print(f"Already done: max_{var_id_out}_{scenario}.nc")

# %%
# %%time
# max tasmax
var_id_in = 'T2C'
var_id_out = 'tasmin'
daily_agg_func = 'min'
metric_func = mf.calculate_min

# Loop through scenarios
for scenario in tgw_scenarios:
    save_path = f"{project_data_path}/metrics/TGW/min_{var_id_out}_{scenario}.nc"
    if not os.path.exists(save_path):
        # Calculate
        ds_out = calculate_metric(scenario=scenario,
                                  var_id_in=var_id_in,
                                  var_id_out=var_id_out,
                                  daily_agg_func=daily_agg_func,
                                  metric_func=metric_func)
        # Store
        ds_out.to_netcdf(save_path)
        print(f"Done: min_{var_id_out}_{scenario}.nc")
    else:
        print(f"Already done: min_{var_id_out}_{scenario}.nc")

# %%
# %%time
# max pr
var_id_in = 'PRCP'
var_id_out = 'pr'
daily_agg_func = 'sum'
metric_func = mf.calculate_max

# Loop through scenarios
for scenario in tgw_scenarios:
    save_path = f"{project_data_path}/metrics/TGW/max_{var_id_out}_{scenario}.nc"
    if not os.path.exists(save_path):
        # Calculate
        ds_out = calculate_metric(scenario=scenario,
                                  var_id_in=var_id_in,
                                  var_id_out=var_id_out,
                                  daily_agg_func=daily_agg_func,
                                  metric_func=metric_func)
        # Store
        ds_out.to_netcdf(save_path)
        print(f"Done: max_{var_id_out}_{scenario}.nc")
    else:
        print(f"Already done: max_{var_id_out}_{scenario}.nc")

# %%
# %%time
# max cdd
var_id_in = 'T2C'
var_id_out = 'cdd'
daily_agg_func = 'cdd'
metric_func = mf.calculate_max

# Loop through scenarios
for scenario in tgw_scenarios:
    save_path = f"{project_data_path}/metrics/TGW/max_{var_id_out}_{scenario}.nc"
    if not os.path.exists(save_path):
        # Calculate
        ds_out = calculate_metric(scenario=scenario,
                                  var_id_in=var_id_in,
                                  var_id_out=var_id_out,
                                  daily_agg_func=daily_agg_func,
                                  metric_func=metric_func)
        # Store
        ds_out.to_netcdf(save_path)
        print(f"Done: max_{var_id_out}_{scenario}.nc")
    else:
        print(f"Already done: max_{var_id_out}_{scenario}.nc")

# %%
# %%time
# max hdd
var_id_in = 'T2C'
var_id_out = 'hdd'
daily_agg_func = 'hdd'
metric_func = mf.calculate_max

# Loop through scenarios
for scenario in tgw_scenarios:
    save_path = f"{project_data_path}/metrics/TGW/max_{var_id_out}_{scenario}.nc"
    if not os.path.exists(save_path):
        # Calculate
        ds_out = calculate_metric(scenario=scenario,
                                  var_id_in=var_id_in,
                                  var_id_out=var_id_out,
                                  daily_agg_func=daily_agg_func,
                                  metric_func=metric_func)
        # Store
        ds_out.to_netcdf(save_path)
        print(f"Done: max_{var_id_out}_{scenario}.nc")
    else:
        print(f"Already done: max_{var_id_out}_{scenario}.nc")

# %%
