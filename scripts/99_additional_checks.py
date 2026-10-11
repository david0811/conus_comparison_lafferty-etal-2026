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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

from utils import loca_gard_mapping
from utils import roar_data_path as project_data_path
from utils import roar_code_path as project_code_path
import obs_utils as ou

# %%
# Load color maps
cm_data = np.loadtxt(f"{project_code_path}/data/lajolla.txt")[::-1]
n_colors = len(cm_data)
end_idx = int(0.8 * n_colors)
lajolla_map = LinearSegmentedColormap.from_list("lajolla", cm_data[:end_idx])

cm_data = np.loadtxt(f"{project_code_path}/data/devon.txt")[::-1]
n_colors = len(cm_data)
start_idx = int(0.3 * n_colors)
devon_map = LinearSegmentedColormap.from_list("devon", cm_data[start_idx:])

# %%
# Set fonts
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

# %% [markdown]
# # Observational comparison

# %%
# max tasmax
metric_id = 'max_tasmax'
ds_livneh, ds_gmet_mean, ds_nclim = ou.read_all_obs(metric_id)

fig, axes = ou.plot_dataset_comparison(ds_livneh, ds_gmet_mean, ds_nclim, '100yr_return_level', 'C', cmap=lajolla_map, diff_lim=8)
fig.suptitle('Observational comparison: Annual maximum temperature', fontweight='bold', fontsize=16)

plt.savefig(f'../figs/si/obs_comparison_{metric_id}.png', dpi=300, bbox_inches='tight')

# %%
# max pr
metric_id = 'max_pr'
ds_livneh, ds_gmet_mean, ds_nclim = ou.read_all_obs(metric_id)

fig, axes = ou.plot_dataset_comparison(ds_livneh, ds_gmet_mean, ds_nclim, '100yr_return_level', 'mm', cmap=devon_map, diff_cmap='BrBG', diff_lim=50)
fig.suptitle('Observational comparison: Annual maximum precipitation', fontweight='bold', fontsize=16)

plt.savefig(f'../figs/si/obs_comparison_{metric_id}.png', dpi=300, bbox_inches='tight')

# %%
# min tasmin
metric_id = 'min_tasmin'
ds_livneh, ds_gmet_mean, ds_nclim = ou.read_all_obs(metric_id)

fig, axes = ou.plot_dataset_comparison(ds_livneh, ds_gmet_mean, ds_nclim, '100yr_return_level', 'C', cmap='Blues_r', diff_lim=20)
fig.suptitle('Observational comparison: Annual minimum temperature', fontweight='bold', fontsize=16)

plt.savefig(f'../figs/si/obs_comparison_{metric_id}.png', dpi=300, bbox_inches='tight')

# %% [markdown]
# # Maxima of raw outputs

# %% [markdown]
# ## LOCA2

# %%
# LOCA2
metric_id = "max_tasmax"
loca_files_all = glob(f"{project_data_path}/metrics/LOCA2/{metric_id}_*")

# %%
# %%time
ds = xr.concat([xr.open_dataset(file).max(dim='time').assign_coords(file = i) for i, file in enumerate(loca_files_all)], dim='file')

# %%
# Plot raw
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.LambertConformal()))
ds.max(dim='file')['tasmax'].plot(cmap="Oranges", ax=ax, 
                                  vmin=30, vmax=60, levels=11,
                                  transform=ccrs.PlateCarree(),
                                  add_colorbar=True, 
                                  cbar_kwargs={"orientation": "horizontal",
                                              "shrink": 0.9,
                                              "label": "Maximum tasmax [degC]"}
                                 )

gl = ax.gridlines(draw_labels=False, x_inline=False, rotate_labels=False, alpha=0.2)
ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.25)
ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.25)
ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.25)
ax.set_extent([-120, -73, 22, 51], ccrs.Geodetic())
plt.tight_layout()
plt.show()

# %% [markdown]
# ## STAR-ESDM

# %%
# STAR-ESDM
metric_id = "max_tasmax"
star_files_all = glob(f"{project_data_path}/metrics/STAR-ESDM/{metric_id}_*")

# %%
# Drop TaiESM1
star_files_all = [file for file in star_files_all if "TaiESM1" not in file]

# %%
# %%time
ds = xr.concat([xr.open_dataset(file).max(dim='time').assign_coords(file = i) for i, file in enumerate(star_files_all)], dim='file')

# %%
# Plot raw
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.LambertConformal()))
ds.max(dim='file')['tasmax'].plot(cmap="Oranges", ax=ax, levels=11,
                                  vmin=30, vmax=60,
                                  transform=ccrs.PlateCarree(),
                                  add_colorbar=True, 
                                  cbar_kwargs={"orientation": "horizontal",
                                              "shrink": 0.9,
                                              "label": "Maximum tasmax [degC]"}
                                 )

gl = ax.gridlines(draw_labels=False, x_inline=False, rotate_labels=False, alpha=0.2)
ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.25)
ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
# ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.5)
ax.coastlines()
ax.set_extent([-120, -73, 22, 51], ccrs.Geodetic())
plt.tight_layout()
plt.show()

# %% [markdown]
# # CESM naming conventions

# %%
# We want to check these mappings are valid
# i.e. that LOCA r1i1p1f1 and GARD-LENS 1001_01
# are downscaled from the same coarse GCM outputs.
loca_gard_mapping

# %%
# Define large enough area common to both domains
lat_domain = [25,45]
lon_domain = [-125, -70]

# %%
# Average temperature
metric_id = 'avg_tas'
var_id = 'tas'

fig, axs = plt.subplots(5, 2, figsize=(10,12), sharex=True, sharey=True)
axs = axs.flatten()

for idm, member in enumerate(loca_gard_mapping.keys()):
    ax = axs[idm]
    # Read
    ds_loca = xr.open_mfdataset(f'{project_data_path}/metrics/LOCA2/{metric_id}_CESM2-LENS_{member}_*.nc')
    ds_gard = xr.open_dataset(f'{project_data_path}/metrics/GARD-LENS/{metric_id}_cesm2_{loca_gard_mapping[member]}_ssp370.nc')

    # Plot GARD
    ds_gard[var_id].sel(lat=slice(lat_domain[0], lat_domain[1]), 
                       lon=slice(lon_domain[0], lon_domain[1])).mean(dim=['lat', 'lon']).plot(ax=ax, label='GARD')

    # Plot LOCA
    ds_loca[var_id].sel(lat=slice(lat_domain[0], lat_domain[1]), 
                       lon=slice(360+lon_domain[0], 360+lon_domain[1])).mean(dim=['lat', 'lon']).plot(ax=ax, label='LOCA')

    # Tidy
    ax.legend()
    ax.grid()
    ax.set_title(f'{member} / {loca_gard_mapping[member]}')

plt.tight_layout()
plt.show()

# %%
# Total precip
metric_id = 'sum_pr'
var_id = 'pr'

fig, axs = plt.subplots(5, 2, figsize=(10,12), sharex=True, sharey=True)
axs = axs.flatten()

for idm, member in enumerate(loca_gard_mapping.keys()):
    ax = axs[idm]
    # Read
    ds_loca = xr.open_mfdataset(f'{project_data_path}/metrics/LOCA2/{metric_id}_CESM2-LENS_{member}_*.nc')
    ds_gard = xr.open_dataset(f'{project_data_path}/metrics/GARD-LENS/{metric_id}_cesm2_{loca_gard_mapping[member]}_ssp370.nc')

    # Plot GARD
    ds_gard[var_id].sel(lat=slice(lat_domain[0], lat_domain[1]), 
                       lon=slice(lon_domain[0], lon_domain[1])).mean(dim=['lat', 'lon']).plot(ax=ax, label='GARD')

    # Plot LOCA
    ds_loca[var_id].sel(lat=slice(lat_domain[0], lat_domain[1]), 
                       lon=slice(360+lon_domain[0], 360+lon_domain[1])).mean(dim=['lat', 'lon']).plot(ax=ax, label='LOCA')

    # Tidy
    ax.legend()
    ax.grid()
    ax.set_title(f'{member} / {loca_gard_mapping[member]}')

plt.tight_layout()
plt.show()

# %%
