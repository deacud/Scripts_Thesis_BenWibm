#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:25:29 2023

@author: Benedikt Wibmer

Script which handles the calculation of gradients between two stations
"""

import numpy as np
import xarray as xr


def interp_likeRef(ds_ref, ds_loc, var_list, method='linear'):
    '''
    function to interpolate data of secondary station to height levels of reference station

    Parameters
    ----------
    ds_ref : xarray Dataset
        reference location dataset.
    ds_loc : xarray Dataset
        secondary location dataset.
    var_list : list
        list of variables to interpolate.
    method : str, optional
        method of interpolation. The default is 'linear'.

    Returns
    -------
    ds_out : xarray Dataset
        Dataset of secondary station interpolated

    '''
    # get heights of reference station
    new_heights = ds_ref.height

    # create Dataset defined with needed dimensions
    ds_out = xr.Dataset(coords=ds_ref.coords)

    # interpolate data to heights of reference station
    for var in var_list:
        try:
            # pressure log. dependence
            if var == 'pres':
                var_df = np.log(ds_loc[var])
                ds_out[f'{var}'] = np.exp(var_df.interp_like(new_heights, method=method))
            else:
                var_df = ds_loc[var]
                ds_out[f'{var}'] = var_df.interp_like(new_heights, method=method)
        except KeyError:
            print(f"{var} not found")
            continue

    return ds_out


def calc_diff_var(ds_ref, ds_loc, var):
    '''
    function to calculate difference between the vertical profiles of two location.

    Parameters
    ----------
    ds_ref : xarray Dataset
        reference location dataset.
    ds_loc : xarray Dataset
        secondary location dataset.
    var : str
        variable to calculate difference.

    Returns
    -------
    diff : xarray DataArray
        Difference profile between reference and secondary location (ds_ref[var] - ds_loc[var]).

    '''

    # check for selected variable and units
    conv_value = {'pres': 100,
                  'theta': 1,
                  'q': 0.001}
    if var in list(conv_value.keys()):
        conversion = conv_value[var]
    else:
        conversion = 1
    diff = (ds_ref[var] - ds_loc[var]) / conversion

    if var == 'dd':
        # need some adjustements for wind direction (maximum difference 180 degree)
        tmp = np.minimum(np.abs(diff), np.abs(diff + 360))  # first elementwise checking
        tmp = np.minimum(tmp, np.abs(diff - 360))           # second elementwise checking
        diff = tmp

    return diff


def main(ds_ref, ds_loc, var_list):
    '''
    Calculates gradients between two stations (ref - loc)

    Parameters
    ----------
    ds_ref : xarray Dataset
        reference location dataset.
    ds_loc : xarray Dataset
        secondary location dataset.
    var_list : list
        list of variables to calculate.

    Returns
    -------
    ds : xarray Dataset
        Dataset which contains gradients (ref - loc) for defined variables.

    '''
    # interpolate to same height levels
    ds_loc_interp = interp_likeRef(ds_ref, ds_loc, var_list)

    # calculate difference
    diff = []
    for var in var_list:
        diff.append(calc_diff_var(ds_ref, ds_loc_interp, var))

    # combine arrays
    ds = xr.merge(diff)

    # add some useful attributes
    try:
        modelrun = ds_ref.modelrun
        DX = ds_ref.DX
    except:
        if 'modelrun' in ds_loc.attrs:
            modelrun = ds_loc.modelrun
            DX = ds_loc.DX
        else:
            modelrun = ''
            DX = ''

    ds = ds.assign_attrs(Name=f'HT AROME --- gradients {ds_ref.station} - {ds_loc.station}',
                         station_ref=ds_ref.station,
                         station_loc=ds_loc.station,
                         run=modelrun,
                         DX=DX)

    return ds
