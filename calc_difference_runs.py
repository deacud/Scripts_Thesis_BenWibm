#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:08:55 2023

@author: benwib

Script responsible for the calculations of the difference between two model simulations
"""

import xarray as xr
import rioxarray as rio
from rasterio.enums import Resampling


def _calc_diff_run(ds_run, runs):
    '''
    function which calculates difference between two model datasets. Calls _resample_grid if needed.

    Parameters
    ----------
    ds_run : dict, xarray Dataset
        dictionary of model run Datasets.
    runs : list, str
        list of model runs.

    Returns
    -------
    ds_diff : xarray Dataset
        dataset with difference between model runs for containing parameters.

    '''
    # check if resampling grid needed
    if ds_run[runs[0]].DX != ds_run[runs[1]].DX:
        ds_run = _resample_grid(ds_run, runs)

    # calculate difference
    ds_diff = (ds_run[runs[0]] - ds_run[runs[1]])

    # add some useful attributes
    ds_diff = ds_diff.assign_attrs(Name=f'AROME --- difference {runs[0]} - {runs[1]}',
                                   run_ref=runs[0],
                                   run_2=runs[1],
                                   DX=ds_run[runs[0]].DX)

    return ds_diff


def _resample_grid(ds_run, runs, method='bilinear'):
    '''
    Resamples coarser grid to resolution of finer grid.

    Parameters
    ----------
    ds_run : dict, xarray Dataset
        dictionary of model run Datasets.
    runs : list, str
        list of model runs.
    method : str, optional
        interpolation method. The default is 'bilinear'.

    Returns
    -------
    ds_run : dict, xarray Dataset
        dictionary of model datasets at same grid resolution.

    '''
    # available resampling methods
    resampling = {'nearest': Resampling.nearest,
                  'cubic': Resampling.cubic,
                  'bilinear': Resampling.bilinear}

    # set CRS: 4326 (WGS84)
    for run in runs:
        ds_run[run] = ds_run[run].rio.write_crs('wgs84')

    # search for finer and coarser grid
    if ds_run[runs[0]].DX < ds_run[runs[1]].DX:
        ds_finer = ds_run[runs[0]]
        ds_coarser = ds_run[runs[1]]
    else:
        ds_finer = ds_run[runs[1]]
        ds_coarser = ds_run[runs[0]]

    # create new dataset
    ds_resampled = xr.Dataset(ds_finer.coords)

    # upsample coarser to finer grid
    keys = list(ds_coarser.keys())
    for key in keys:
        print(f'Start resampling {key} of {ds_coarser.modelrun} to {ds_finer.modelrun} ...')
        da = ds_coarser[key]
        ds_resampled[key] = da.rio.reproject_match(ds_finer,
                                                   resampling=resampling[method]).rename({'x': 'longitude',
                                                                                          'y': 'latitude'})

    # add attributes
    dict_attrs = dict(Name=f'AROME {ds_coarser.modelrun} upscaled to {ds_finer.modelrun}',
                      old_resolution=ds_coarser.rio.resolution(),
                      new_resolution=ds_resampled.rio.resolution(),
                      old_DX=ds_coarser.DX,
                      new_DX=ds_finer.DX,
                      method=method,
                      domain=ds_coarser.domain,
                      modelrun=ds_coarser.modelrun,
                      history=ds_coarser.attrs)
    ds_resampled = ds_resampled.assign_attrs(dict_attrs)

    ds_run[ds_finer.modelrun] = ds_finer
    ds_run[ds_coarser.modelrun] = ds_resampled

    return ds_run
