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
import warnings
import numpy as np
import pandas as pd
import dask
import gev_bayes_utils as gevbu
import gev_city_utils as gevcu
import sa_city_utils as sacu
import trend_utils as tu

from utils import city_list, gev_metric_ids, trend_metric_ids, tgw_scenarios
from utils import roar_data_path as project_data_path
from utils import roar_code_path as project_code_path

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
    memory="3GiB",
    walltime="06:00:00",
)

cluster.scale(25)

from dask.distributed import Client
client = Client(cluster)
client

# %% [markdown]
# ## Get city timeseries

# %%
# Run for all: extremes
for city in city_list.keys():
    for metric_id in gev_metric_ids:
        sacu.get_city_timeseries_all(city, metric_id)

# %%
# Run for all: extremes with neighbors
for city in city_list.keys():
    for metric_id in gev_metric_ids:
        sacu.get_city_timeseries_all(city, metric_id, include_neighbors=True)

# %%
# Run for all: trends
for city in city_list.keys():
    for metric_id in trend_metric_ids:
        sacu.get_city_timeseries_all(city, metric_id)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Trends with bootstrap

# %%
# Fit all: cities, bootstrap
for city in city_list:
    for metric_id in trend_metric_ids:
        for n_boot in [250, 1000]:
            tu.trend_fit_city(metric_id, city, n_boot=n_boot)

# %% [markdown]
# ## Stationary GEV with bootstrap

# %% [markdown]
# ### Main

# %%
# Main ensemble
stationary = True
fit_method = "lmom"
periods_for_level = [10,25,50,100]
hist_slice = [1950,2014]
proj_slice = [2050,2100]

# Loop through all
delayed = []

for city in city_list:
    for metric_id in gev_metric_ids:
        for n_boot_proj in [100, 1000]:
            delayed.append(dask.delayed(gevcu.fit_ensemble_gev_city)
                           (city=city, 
                            metric_id=metric_id,
                            stationary=stationary,
                            fit_method=fit_method,
                            hist_slice=hist_slice,
                            proj_slice=proj_slice,
                            n_boot_proj = n_boot_proj,
                            periods_for_level=periods_for_level,
                            return_samples=True))

_ = dask.compute(*delayed)

# %%
# # TGW
# stationary = True
# fit_method = "lmom"
# periods_for_level = [10,25,50,100]
# hist_slice = [1980,2019]
# proj_slice = [2049,2099]
# return_samples = True
# n_boot_hist = 1

# stat_str = "stat" if stationary else "nonstat"
# sample_str = "_samples" if return_samples else ""

# df_out = []

# for city in city_list:
#     for metric_id in gev_metric_ids:
#         for ssp in tgw_scenarios[1:]:
#             for n_boot_proj in [100, 1000]:
#                 # Compute
#                 df = gevcu.fit_gev_city(city=city,
#                                         metric_id=metric_id,
#                                         ensemble='TGW',
#                                         gcm='none',
#                                         ssp=ssp,
#                                         member='none',
#                                         fit_method=fit_method,
#                                         hist_slice=hist_slice,
#                                         proj_slice=proj_slice,
#                                         stationary=stationary,
#                                         n_boot_proj=n_boot_proj,
#                                         n_boot_hist=n_boot_hist,
#                                         periods_for_level=periods_for_level,
#                                         return_samples=return_samples)
#                 # Append
#                 df_out.append(df)

#         # Store city/metric
#         file_name = f"TGW_{city}_{metric_id}_{hist_slice[0]}-{hist_slice[1]}_{proj_slice[0]}-{proj_slice[1]}_{fit_method}_{stat_str}_nbootproj{n_boot_proj}_nboothist{n_boot_hist}{sample_str}.csv"
#         df_out = pd.concat(df_out, ignore_index=True)
#         df_out.to_csv(f"{project_data_path}/extreme_value/cities/original_grid/freq/{file_name}", index=False)
#         df_out = []

# %% [markdown]
# ### Naive pooling

# %%
# Main ensemble
stationary = True
fit_method = "lmom"
periods_for_level = [10,25,50,100]
hist_slice = [1950,2014]
proj_slice = [2050,2100]

# Loop through all
delayed = []

for city in city_list:
    for metric_id in gev_metric_ids:
        for n_boot_proj in [100, 1000]:
            delayed.append(dask.delayed(gevcu.fit_ensemble_gev_city)
                           (city=city, 
                            metric_id=metric_id,
                            stationary=stationary,
                            fit_method=fit_method,
                            hist_slice=hist_slice,
                            proj_slice=proj_slice,
                            n_boot_proj = n_boot_proj,
                            periods_for_level=periods_for_level,
                            return_samples=True,
                            include_neighbors=True))

_ = dask.compute(*delayed)

# %%
# TGW
stationary = True
fit_method = "lmom"
periods_for_level = [10,25,50,100]
hist_slice = [1980,2019]
proj_slice = [2049,2099]
return_samples = True
include_neighbors = True
n_boot_hist = 1

stat_str = "stat" if stationary else "nonstat"
sample_str = "_samples" if return_samples else ""
neighbor_str = "_neighbors" if include_neighbors else ""

df_out = []

for city in city_list:
    for metric_id in gev_metric_ids:
        for ssp in tgw_scenarios[1:]:
            for n_boot_proj in [100, 1000]:
                # Compute
                df = gevcu.fit_gev_city(city=city,
                                        metric_id=metric_id,
                                        ensemble='TGW',
                                        gcm='none',
                                        ssp=ssp,
                                        member='none',
                                        fit_method=fit_method,
                                        hist_slice=hist_slice,
                                        proj_slice=proj_slice,
                                        stationary=stationary,
                                        n_boot_proj=n_boot_proj,
                                        n_boot_hist=n_boot_hist,
                                        periods_for_level=periods_for_level,
                                        return_samples=return_samples,
                                        include_neighbors=include_neighbors)
                # Append
                df_out.append(df)

        # Store city/metric
        file_name = f"TGW_{city}_{metric_id}_{hist_slice[0]}-{hist_slice[1]}_{proj_slice[0]}-{proj_slice[1]}_{fit_method}_{stat_str}_nbootproj{n_boot_proj}_nboothist{n_boot_hist}{sample_str}{neighbor_str}.csv"
        df_out = pd.concat(df_out, ignore_index=True)
        df_out.to_csv(f"{project_data_path}/extreme_value/cities/original_grid/freq/{file_name}", index=False)
        df_out = []

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Non-stationary GEV with bootstrap

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Location trend only

# %%
# Main ensemble: location trend only
fit_method = 'mle'
stationary = False
# n_boots = [100, 1000]
n_boots = [100]

# Loop through all
delayed = []

for city in city_list:
    for metric_id in gev_metric_ids:
        for n_boot in n_boots:
            delayed.append(dask.delayed(gevcu.fit_ensemble_gev_city)
                           (city=city, 
                            metric_id=metric_id,
                            stationary=stationary,
                            fit_method=fit_method,
                            n_boot_proj = n_boot,
                            return_samples = True))

_ = dask.compute(*delayed)

# %%
# TGW
stationary = False
fit_method = "mle"
periods_for_level = [10,25,50,100]
years = [1980, 2099]
return_samples = True
n_boot_hist = 1

stat_str = "stat" if stationary else "nonstat"
sample_str = "_samples" if return_samples else ""

df_out = []

for city in city_list:
    for metric_id in gev_metric_ids:
        for ssp in tgw_scenarios[1:]:
            for n_boot_proj in [100, 1000]:
                # Compute
                df = gevcu.fit_gev_city(city=city,
                                        metric_id=metric_id,
                                        ensemble='TGW',
                                        gcm='none',
                                        ssp=ssp,
                                        member='none',
                                        fit_method=fit_method,
                                        years=years,
                                        stationary=stationary,
                                        n_boot_proj=n_boot_proj,
                                        periods_for_level=periods_for_level,
                                        return_samples=return_samples)
                # Append
                df_out.append(df)

        # Store city/metric
        file_name = f"TGW_{city}_{metric_id}_{years[0]}-{years[1]}_{fit_method}_{stat_str}_nboot{n_boot_proj}{sample_str}.csv"
        df_out = pd.concat(df_out, ignore_index=True)
        df_out.to_csv(f"{project_data_path}/extreme_value/cities/original_grid/freq/{file_name}", index=False)
        df_out = []

# %% [markdown]
# ### Location & scale trend

# %%
# Main ensemble
fit_method = 'mle'
stationary = False
n_boots = [100, 1000]
nonstationary_scale = True

# Loop through all
delayed = []

for city in city_list.keys():
    for metric_id in gev_metric_ids[:3]:
        for n_boot in n_boots:
            delayed.append(dask.delayed(gevcu.fit_ensemble_gev_city)
                           (city=city, 
                            metric_id=metric_id,
                            stationary=stationary,
                            fit_method=fit_method,
                            n_boot_proj = n_boot,
                            return_samples = True,
                            nonstationary_scale = nonstationary_scale))

_ = dask.compute(*delayed)

# %%
# # TGW
# stationary = False
# nonstationary_scale = True
# fit_method = "mle"
# periods_for_level = [10,25,50,100]
# years = [1980, 2099]
# return_samples = True
# n_boot_hist = 1

# stat_str = "stat" if stationary else "nonstat"
# sample_str = "_samples" if return_samples else ""
# scale_str = "_scale" if nonstationary_scale else ""

# df_out = []

# for city in city_list:
#     for metric_id in gev_metric_ids:
#         for ssp in tgw_scenarios[1:]:
#             for n_boot_proj in [100, 1000]:
#                 # Compute
#                 df = gevcu.fit_gev_city(city=city,
#                                         metric_id=metric_id,
#                                         ensemble='TGW',
#                                         gcm='none',
#                                         ssp=ssp,
#                                         member='none',
#                                         fit_method=fit_method,
#                                         years=years,
#                                         stationary=stationary,
#                                         n_boot_proj=n_boot_proj,
#                                         periods_for_level=periods_for_level,
#                                         nonstationary_scale=nonstationary_scale,
#                                         return_samples=return_samples)
#                 # Append
#                 df_out.append(df)

#         # Store city/metric
#         file_name = f"TGW_{city}_{metric_id}_{years[0]}-{years[1]}_{fit_method}_{stat_str}_nboot{n_boot_proj}{sample_str}{scale_str}.csv"
#         df_out = pd.concat(df_out, ignore_index=True)
#         df_out.to_csv(f"{project_data_path}/extreme_value/cities/original_grid/freq/{file_name}", index=False)
#         df_out = []

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Bayesian GEV (old!)

# %%
# # For Bayes
# ############
# ### Dask ###
# ############
# from dask_jobqueue import SLURMCluster

# cluster = SLURMCluster(
#     # account="pches",
#     account="open",
#     cores=3,
#     processes=1,
#     job_cpu=3,
#     memory="3GiB",
#     walltime="12:00:00",
#     job_script_prologue=[
#         f"export PYTHONPATH={project_code_path}/.venv/lib/python3.12/site-packages:$PYTHONPATH",  # Put venv first
#         "export JAX_PLATFORM_NAME=cpu",
#         "export XLA_FLAGS='--xla_force_host_platform_device_count=1'",
#         # Force PyTensor to not use caching at all
#         "export PYTENSOR_FLAGS='cxx=,",
#         "mode=FAST_COMPILE,",  # Less aggressive optimization but more stable
#         "allow_gc=True,",
#         "cache_size=0'"        # Disable caching completely
#     ],
#     death_timeout=60,
#     local_directory="/tmp"
# )

# cluster.scale(5)

# from dask.distributed import Client
# client = Client(cluster)
# client

# %%
# # %%time
# # Fit info: non-stationary
# future_years = [2015,2100]
# stationary = False
# return_periods = [100]

# # Parallelize with dask delayed
# delayed = []

# # Loop thorugh all combos
# for city in city_list.keys():
#     for metric_id in gev_metric_ids:
#         tmp = dask.delayed(gevbu.fit_bayesian_gev_ensemble)(
#             city=city,
#             metric_id=metric_id,
#             years=future_years,
#             stationary=stationary,
#             shape_sigma=0.2,
#             prior_identifier='shape_sigma_02',
#             return_periods=return_periods,
#         )
#         delayed.append(tmp)

# _ = dask.compute(*delayed)

# %%
# # %%time
# # Fit info: stationary
# hist_years = [1950,2014]
# future_years = [2050,2100]
# stationary = True
# return_periods = [100]

# # Parallelize with dask delayed
# delayed = []

# # Loop thorugh all combos
# for city in city_list.keys():
#     for metric_id in gev_metric_ids:
#         for years in [hist_years, future_years]:
#             tmp = dask.delayed(gevbu.fit_bayesian_gev_ensemble)(
#                     city=city,
#                     metric_id=metric_id,
#                     years=years,
#                     stationary=stationary,
#                     return_periods=return_periods,
#                     dask=False
#             )
#             delayed.append(tmp)

# _ = dask.compute(*delayed)

# %%
# # %%time
# # Loop thorugh all combos and store
# store_path = f"{project_data_path}/extreme_value/cities/original_grid/bayes_combined/"

# return_periods = [100]

# prior_identifier = "shape_sigma_02"

# stationary = False
# stationary_string = "stat" if stationary else "nonstat"

# # for city in city_list.keys():
# for city in ['nyc', 'chicago', 'denver']:
#     for metric_id in ['max_tasmax', 'max_pr', 'min_tasmin']:
#     # for metric_id in gev_metric_ids:
#         for years in [None, [2015,2100]]:
#             # Check if done
#             change_identifier = "" if years is None else f"_change_{years[0]}-{years[1]}"
#             file_path = f"{store_path}/{city}_{metric_id}_{stationary_string}_{prior_identifier}{change_identifier}.csv"
#             if os.path.exists(file_path):
#                 continue
#             # Read
#             df = gevbu.gather_bayesian_gev_results_all(
#                 city = city,
#                 metric_id = metric_id,
#                 return_periods = return_periods,
#                 stationary = stationary,
#                 prior_identifier = prior_identifier,
#                 years = years,
#             )
#             # Store
#             df.to_csv(file_path, index=False)

# %%
