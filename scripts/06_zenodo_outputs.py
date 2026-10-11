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
import datetime
import xarray as xr
import numpy as np

import sa_utils as sau

from utils import roar_data_path as project_data_path
import plotting_utils as pu

# %%
# Some helphers
gev_params = ['loc_intcp', 'loc_trend', 'log_scale_intcp', 'log_scale_trend', 'shape']
ensembles = ['GARD-LENS', 'STAR-ESDM', 'LOCA2']
read_funcs = [sau.read_gard, sau.read_star, sau.read_loca]
return_periods = [10, 25, 50, 100]
gev_params = ['loc_intcp', 'loc_trend', 'log_scale_intcp', 'log_scale_trend', 'shape']
gev_metric_ids = ['max_tasmax', 'min_tasmin', 'max_pr']
trend_metric_ids = ['avg_tas', 'sum_pr'] + gev_metric_ids

metric_id_units = {
    'max_tasmax': 'degC',
    'min_tasmin': 'degC',
    'avg_tasmax': 'degC',
    'avg_tasmin': 'degC',
    'avg_tas':'degC',
    'max_pr': 'mm',
    'sum_pr':'mm'
}

ensemble_identifiers = {
    'LOCA2': 'https://doi.org/10.1175/JHM-D-22-0194.1',
    'GARD-LENS': 'https://doi.org/10.1038/s41597-024-04205-z',
    'STAR-ESDM': 'https://doi.org/10.1029/2023EF004107'
}

param_names = {
    'loc_intcp':'Intercept on location parameter (time=1950)',
    'loc_trend':'Time trend on location parameter',
    'log_scale_intcp':'Intercept on log of scale parameter (time=1950)', 
    'log_scale_trend':'Time trend on log of scale parameter', 
    'shape':'Shape parameter (scipy convention)',
    'slope':'Linear ordinary least squares trend (2015-2100)'
}


# %%
def _create_encoding(ds, complevel=5, string_dtype="S16"):
    """
    Encodes string coords as S10, all else float32 or int32.
    """
    encoding = {}

    # Numeric vars
    for var in ds.data_vars:
        if ds[var].dtype.kind == 'f':
            encoding[var] = {"zlib": True, "complevel": complevel, "shuffle": True, "dtype":"float32"}

    # String vars/coords — no compression, just dtype conversion
    str_coords = [c for c in ds.coords if ds[c].dtype.kind == 'U']
    for name in str_coords:
        encoding[name] = {"dtype": string_dtype}  # match or exceed max string length

    # Other coords
    for coord in ds.coords:
        dtype = ds[coord].dtype
        if dtype == np.float64:
            encoding[coord] = {"dtype": "float32"}  # if precision allows
        elif dtype == np.int64:
            encoding[coord] = {"dtype": "int32"}    # almost always safe

    return encoding


# %% [markdown]
# ### Uncertainty results

# %%
def _add_attrs(ds, metric_id=None):
    """
    Add CF-compliant attributes 
    """
    
    # ── Global attributes ──────────────────────────────────────────────────
    ds.attrs.update({
        "title": "Uncertainty decomposition results",
        "history": f"Created {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "reference": "https://doi.org/10.22541/essoar.15003332/v1",
        "author": "David Lafferty, Cornell University",
        "contact": "dcl5300@cornell.edu",
        "Conventions": "CF-1.8",
        "LOCA2_source": ensemble_identifiers["LOCA2"],
        "GARD-LENS_source": ensemble_identifiers["GARD-LENS"],
        "STAR_ESDM_source": ensemble_identifiers["STAR-ESDM"],
        "NOTE": ("String coordinates are stored as fixed-length byte strings (NetCDF4 char arrays)."
                 " Python users: ds[coord].astype(str) to convert after loading.")
    })

    # ── Coordinate attributes ──────────────────────────────────────────────
    coord_attrs = {
        "lat": {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
        "lon": {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
        "uncertainty": {
            "long_name": "Type of uncertainty",
        },
    }
    if 'time' in ds.coords:
        coord_attrs['time'] = {
            "long_name":"Time identifier"
            }

    for coord, attrs in coord_attrs.items():
        if coord in ds.coords:
            ds[coord].attrs.update(attrs)

    # ── Data variable attributes ───────────────────────────────────────────
    for var in ds.data_vars:
        if 'return_level' in var:
            rp_str = var.split('_')[0][:-2]
            ds[var].attrs.update({
                "long_name": f"{rp_str}-year return level of {pu.title_labels[metric_id]}",
                "units": metric_id_units[metric_id],
                "return_period": rp_str,
                "return_period_units": "years",
                "method": "Maximum likelihood GEV fitting",
                "_FillValue": np.float32(1e20),
            })
        else:
            ds[var].attrs.update({
                "long_name": f"Uncertainty in 2015-2100 trend of {pu.title_labels[var].lower()}",
                "units": f"{metric_id_units[var]}/year",
                "_FillValue": np.float32(1e20),
            })

    return ds


# %%
# Slopes
def _preprocess(ds):
    uc_types = ['ssp_uc', 'gcm_uc', 'iv_uc', 'dsc_uc', 'uc_99w_main', 'uc_95w_main']
    metric_id = '_'.join(ds.encoding['source'].split('/')[-1].split('_')[:2])
    ds = ds[uc_types].to_array(name='name', dim='uncertainty').to_dataset()
    ds['uncertainty'] = ['ssp', 'gcm', 'int_var', 'downscale', '99width', '95width']
    ds = ds.rename({'name':metric_id})
    return ds

files = glob(f'{project_data_path}/results/*_2015-2100_slope_2015-2100_None_None_LOCA2grid_nearest.nc')
files = [file for file in files if 'rel' not in file]
files = [file for file in files if 'summary' not in file]

ds = _add_attrs(xr.merge([_preprocess(xr.open_dataset(file)) for file in files]))

encoding = _create_encoding(ds, complevel=1)
ds.to_netcdf(f"{project_data_path}/zenodo/uncertainty_results/annual_trends.nc", encoding=encoding)


# %%
# GEVs
def _preprocess(ds):
    uc_types = ['ssp_uc', 'gcm_uc', 'iv_uc', 'dsc_uc', 'fit_uc_mean', 'uc_99w_main', 'uc_95w_main']
    name = '_'.join(ds.encoding['source'].split('/')[-1].split('_')[3:6])
    ds = ds[uc_types].to_array(name='name', dim='uncertainty').to_dataset()
    ds['uncertainty'] = ['ssp', 'gcm', 'int_var', 'downscale', 'gev_fit', '99width', '95width']
    if 'time' not in ds.coords:
        ds = ds.assign_coords(time = '2075-1975')
    ds = ds.expand_dims(['time'])
    ds = ds.rename({'name':name})
    return ds

for metric_id in gev_metric_ids:
    files = glob(f'{project_data_path}/results/{metric_id}_1950-2100_*yr_return_level_*_mle_nonstat_scale_LOCA2grid_nearest.nc')
    files = [file for file in files if 'chfc' not in file]

    ds = _add_attrs(xr.merge([_preprocess(xr.open_dataset(file)) for file in files]), metric_id=metric_id)
    ds = ds.drop_vars(['ensemble', 'time_diff'])
    encoding = _create_encoding(ds, complevel=1)
    ds.to_netcdf(f"{project_data_path}/zenodo/uncertainty_results/gev_{metric_id}.nc", encoding=encoding)


# %% [markdown]
# ### Ensemble summaries

# %%
def _add_attrs(ds, metric_id=None):
    """
    Add CF-compliant attributes 
    """
    
    # ── Global attributes ──────────────────────────────────────────────────
    ds.attrs.update({
        "title": "Metric summaries from downscaled CMIP6 ensembles",
        "history": f"Created {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "reference": "https://doi.org/10.22541/essoar.15003332/v1",
        "author": "David Lafferty, Cornell University",
        "contact": "dcl5300@cornell.edu",
        "Conventions": "CF-1.8",
        "LOCA2_source": ensemble_identifiers["LOCA2"],
        "GARD-LENS_source": ensemble_identifiers["GARD-LENS"],
        "STAR_ESDM_source": ensemble_identifiers["STAR-ESDM"],
        "history":"Non-LOCA2 sources regridded using xarray_regrid 0.4.0 nearest neighbors algorithm",
        "NOTE": ("String coordinates are stored as fixed-length byte strings (NetCDF4 char arrays)."
                 "Python users: ds[coord].astype(str) to convert after loading.")
    })

    # ── Coordinate attributes ──────────────────────────────────────────────
    coord_attrs = {
        "lat": {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
        "lon": {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
        "quantile": {
            "long_name": "Quantile",
        },
        "ssp": {
            "long_name": "Shared Socioeconomic Pathway scenario",
        },
        "ensemble": {
            "long_name": "Downscaling ensemble identifier",
        },
    }
    if 'time' in ds.coords:
        coord_attrs['time'] = {
            "long_name":"Time identifier"
            }

    for coord, attrs in coord_attrs.items():
        if coord in ds.coords:
            ds[coord].attrs.update(attrs)

    # ── Data variable attributes ───────────────────────────────────────────
    for var in ds.data_vars:
        if 'return_level' in var:
            rp_str = var.split('_')[0][:-2]
            ds[var].attrs.update({
                "long_name": f"{rp_str}-year return level of {pu.title_labels[metric_id].lower()}",
                "units": metric_id_units[metric_id],
                "return_period": rp_str,
                "return_period_units": "years",
                "method": "Maximum likelihood GEV fitting",
                "_FillValue": np.float32(1e20),
            })
        else:
            ds[var].attrs.update({
                "long_name": f"2015-2100 trend of {pu.title_labels[var].lower()}",
                "units": f"{metric_id_units[var]}/year",
                "method": "Ordinary least squares trend estimation",
                "_FillValue": np.float32(1e20),
            })

    return ds


# %%
# Slopes
def _preprocess(ds):
    ds['ensemble'] = ['LOCA2', 'GARD-LENS', 'STAR-ESDM'] # avoid cutoffs
    metric_id = '_'.join(ds.encoding['source'].split('_')[3:5])
    ds = ds.rename({'slope':metric_id})
    return ds

files = glob(f'{project_data_path}/results/summary_*_2015-2100_None_slope_2015-2100_None_None_LOCA2grid_nearest.nc')
files = [file for file in files if 'rel' not in file]

ds = _add_attrs(xr.merge([_preprocess(xr.open_dataset(file)) for file in files]))

encoding = _create_encoding(ds, complevel=1)
ds.to_netcdf(f"{project_data_path}/zenodo/ensemble_summary/annual_trends.nc", encoding=encoding)


# %%
# GEVs
def _preprocess(ds):
    ds['ensemble'] = ['LOCA2', 'GARD-LENS', 'STAR-ESDM'] # avoid cutoffs
    if 'time' not in ds.coords:
        ds = ds.assign_coords(time = '2075-1975')
    ds = ds.expand_dims(['time'])
    return ds

for metric_id in gev_metric_ids:
    files = glob(f'{project_data_path}/results/summary_{metric_id}_1950-2100_*yr_return_level_*_mle_nonstat_scale_LOCA2grid_nearest.nc')
    files = [file for file in files if 'chfc' not in file]

    ds = _add_attrs(xr.merge([_preprocess(xr.open_dataset(file)) for file in files]), metric_id=metric_id)

    encoding = _create_encoding(ds, complevel=1)
    ds.to_netcdf(f"{project_data_path}/zenodo/ensemble_summary/gev_{metric_id}.nc", encoding=encoding)


# %% [markdown]
# ### GEV & trend results

# %%
def _add_attrs(ds, metric_id, ensemble, analysis_type = 'extreme_value'):
    """
    Add CF-compliant attributes and encoding to a non-stationary GEV
    return level dataset from downscaled CMIP6 ensembles.
    """
    
    # ── Global attributes ──────────────────────────────────────────────────
    ds.attrs.update({
        "title": "Non-stationary GEV parameters/return levels from downscaled CMIP6 ensembles",
        "history": f"Created {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "reference": "https://doi.org/10.22541/essoar.15003332/v1",
        "author": "David Lafferty, Cornell University",
        "contact": "dcl5300@cornell.edu",
        "Conventions": "CF-1.8",
        "comment": f"Original data source: {ensemble}, {ensemble_identifiers[ensemble]}",
        "history":"If non-LOCA2 source, regridded using xarray_regrid 0.4.0 nearest neighbors algorithm",
        "NOTE": ("String coordinates are stored as fixed-length byte strings (NetCDF4 char arrays)."
                 "Python users: ds[coord].astype(str) to convert after loading.")
    })
    if analysis_type == 'extreme_value':
        title = "Non-stationary GEV parameters/return levels from downscaled CMIP6 ensembles"
    elif analysis_type == 'trends':
        title = "Linear trends from downscaled CMIP6 ensembles"

    # ── Coordinate attributes ──────────────────────────────────────────────
    coord_attrs = {
        "lat": {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
        "lon": {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
        "time": {
            "long_name": "time",
            "standard_name": "time",
            "axis": "T",
            "comment": "Integer year - non-stationary GEV parameters evaluated at this year",
        },
        "gcm": {"long_name": "CMIP6 Earth System Model identifier"},
        "ssp": {"long_name": "Shared Socioeconomic Pathway scenario"},
        "ensemble": {"long_name": "Downscaling ensemble identifier"},
        "member": {"long_name": "CMIP6 ensemble member identifier"},
    }

    for coord, attrs in coord_attrs.items():
        if coord in ds.coords:
            ds[coord].attrs.update(attrs)

    # ── Data variable attributes ───────────────────────────────────────────
    for var in ds.data_vars:
        if 'return_level' in var:
            rp_str = var.split('_')[0][:-2]
            ds[var].attrs.update({
                "long_name": f"{rp_str}-year return level of {pu.title_labels[metric_id].lower()}",
                "units": metric_id_units[metric_id],
                "return_period": rp_str,
                "return_period_units": "years",
                "_FillValue": np.float32(1e20),
            })
        else:
            ds[var].attrs.update({
                "long_name": param_names[var],
                "_FillValue": np.float32(1e20),
            })
            

    return ds

# %%
# Loop through all: trends
for ensemble, read_func in zip(ensembles, read_funcs):
    for metric_id in trend_metric_ids:
        store_path = f"{project_data_path}/zenodo/trends/{ensemble}_{metric_id}_trends.nc"
        if os.path.exists(store_path):
            print(f"Already done: {ensemble} {metric_id}")
            continue

        # Read
        ds = read_func(
                metric_id = metric_id,
                grid = 'LOCA2',
                regrid_method='nearest',
                proj_slice='2015-2100',
                hist_slice=None,
                stationary=None,
                stat_name=None,
                fit_method='mle',
                bootstrap=False,
                cols_to_keep=['slope'],
                analysis_type='trends'
            )
        encoding = _create_encoding(ds)
        ds = _add_attrs(ds, metric_id, ensemble, analysis_type='trends')
        
        # Store
        ds.to_netcdf(store_path, encoding=encoding)
        print(f"Done: {ensemble} {metric_id}")
        del ds

# %%
# Loop through all: return levels
for ensemble, read_func in zip(ensembles, read_funcs):
    for metric_id in gev_metric_ids:
        for rp in return_periods:
            store_path = f"{project_data_path}/zenodo/GEV/{ensemble}_{metric_id}_nonstationary-loc-scale-{rp}year-return-level.nc"
            if os.path.exists(store_path):
                print(f"Already done: {ensemble} {metric_id} {rp}")
                continue

            # Read
            ds = read_func(
                metric_id = metric_id,
                grid = 'LOCA2',
                regrid_method='nearest',
                proj_slice='1950-2100',
                hist_slice=None,
                stationary=False,
                stat_name='nonstat_scale',
                fit_method='mle',
                bootstrap=False,
                cols_to_keep=[f'{rp}yr_return_level'],
                analysis_type='extreme_value'
            )
            encoding = _create_encoding(ds)
            ds = _add_attrs(ds, metric_id, ensemble)
        
            # Store
            ds.to_netcdf(store_path, encoding=encoding)
            print(f"Done: {ensemble} {metric_id} {rp}")
            del ds

# %%
# Loop through all: GEV params
for ensemble, read_func in zip(ensembles, read_funcs):
    for metric_id in gev_metric_ids:
        store_path = f"{project_data_path}/zenodo/GEV/{ensemble}_{metric_id}_nonstationary-loc-scale-GEV-parameters.nc"
        if os.path.exists(store_path):
            print(f"Already done: {ensemble} {metric_id}")
            continue

        # Read
        ds = read_func(
            metric_id = metric_id,
            grid = 'LOCA2',
            regrid_method='nearest',
            proj_slice='1950-2100',
            hist_slice=None,
            stationary=False,
            stat_name='nonstat_scale',
            fit_method='mle',
            bootstrap=False,
            cols_to_keep=gev_params,
            analysis_type='extreme_value'
        )
        encoding = _create_encoding(ds)
        ds = _add_attrs(ds, metric_id, ensemble)
        
        # Store
        ds.to_netcdf(store_path, encoding=encoding)
        print(f"Done: {ensemble} {metric_id}")
        del ds

# %%
