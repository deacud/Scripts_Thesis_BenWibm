#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:23:09 2023

@author: benwib

Script to read in netcdf files on hybrid pressure levels and combine data for station site
and save it as netcdf file.
"""

import xarray as xr
from os.path import join, basename
import glob2
import gc
from dask.distributed import Client, LocalCluster
import webbrowser
import time
import sys
from path_handling import get_Dataset_path, set_interpStation_name, get_Dataset_name
import re

dict_DX = {'OP500': 500,
           'OP1000': 1000,
           'OP2500': 2500,
           'IFS1000': 1000,
           'ARP1000': 1000}


def _interp_location(ds, coords, method):
    '''
    Interpolate data to defined coordinates

    Parameters
    ----------
    ds : xarray Dataset
        dataset containing Data of larger extent
    coords : tuple, float
        lon/lat coordinates to intpolate to
    method : str
        interpolation method.

    Returns
    -------
    xarray Dataset
        dataset interpolated to selected coordinates.

    '''
    # interpolate to location
    print(f'dataset: {ds}')
    ds = ds.interp(longitude=coords[0], latitude=coords[1], method=method)
    return ds.load()


def _save_toNetCDF(ds, run):
    '''
    function to save xarray Dataset to netcdf file

    Parameters
    ----------
    ds : xarray Dataset
        dataset containing data of station
    run : str
        modelrun
    '''

    # get save path and save it as netcdf file
    dir_save = get_Dataset_path(run)
    name_save = set_interpStation_name(ds)
    path_save = join(dir_save, name_save)
    print(f'Save file to {path_save}')
    ds.to_netcdf(path_save)


def _create_StationData(dir_path, var_list, run, coords, method):
    '''
    function to create interpolated model dataset at coords

    Parameters
    ----------
    dir_path : str
        path to datasets.
    var_list : list
        list with parameters.
    run : str
        modelrun.
    coords : tuple
        lon/lat coordinates.
    method : str
        interpolation method.

    Returns
    -------
    ds_station : xarray Dataset
        dataset with interpolated parameters to selected coordinates.

    '''
    # ---- 1. create Dataset defined with needed variables
    for i, var in enumerate(var_list):

        start = time.time()
        # open sliced Dataset
        filename = get_Dataset_name(run, 'hybridPressure', var=var)
        path_files = glob2.glob(join(dir_path, filename))
        print(f'dir_path, filename, path_files: {dir_path}, {filename}, {path_files}')

        # get correct file name to load
        for file in path_files:
            onlyfile = basename(file)  # get only filename
            f_extent = re.findall('\[(.*)\]', onlyfile)[0].split(',')  # extract exent of dataset
            # check if coords within extent
            if float(f_extent[0]) <= coords[0] <= float(f_extent[1]):
                if float(f_extent[2]) <= coords[1] <= float(f_extent[3]):
                    path_file = file
                else:
                    raise Exception(f"Station is outside the preprocessed domain: {coords[1]}")
            else:
                raise Exception(f"Station is outside the preprocessed domain: {coords[0]}")
                

        tmp = xr.open_dataset(path_file, chunks={'valid_time': 2})  # TODO: find best chuncksize speed/memory

        # do interpolation for variable
        print(f'Start {method} interpolation of {var} to {coords} ...')
        tmp = _interp_location(tmp[f'{var}'], coords, method)
        tmp = tmp.to_dataset()

        if i == 0:
            # create Dataset defined with needed dimensions
            ds_station = tmp
            del tmp  # free up space
            gc.collect()
        else:
            ds_station = xr.merge([ds_station, tmp])

        end = time.time()
        print(f'Elapsed time interpolation {var}: {end - start}')

    # ---- 2. add some global attributes
    DX = dict_DX.get(run)
    time_now = time.localtime()
    time_now = time.strftime("%Y-%m-%d %H:%M:%S", time_now)
    ds_station = ds_station.assign_attrs(name='AROME output parameters interpolated to coords',
                                         coords=coords,
                                         interp_method=method,
                                         modelrun=run,
                                         DX=DX,
                                         history=f'Created by Benedikt Wibmer on {time_now}')

    return ds_station


def preprocess_Station_main(dir_path=None, var_list=None, run=None, coords=None, method='linear', save=None):
    '''
    function to create interpolated model dataset at coords

    Parameters
    ----------
    dir_path : str, optional
        path to datasets.
    var_list : list, optional
        list with parameters.
    run : str, optional
        modelrun.
    coords : tuple, optional
        lon/lat coordinates.
    method : str, optional
        interpolation method. Default: 'linear' (bilinear interpolation)
    save : bool, optional
        save preprocessed file to netcdf file

    Returns
    -------
    ds : xarray Dataset
        dataset with interpolated parameters to selected coordinates.
        dimensions: level, valid_time
        data variables: parameters from var_list

    '''
    # start Dask Client
#    print('Start client')
#    try:
#        client = Client('tcp://localhost:8786', timeout='2s')
#    except OSError:
#        cluster = LocalCluster(scheduler_port=8786)
#        client = Client(cluster)
#        client.restart()
#    webbrowser.open(client.dashboard_link)

    # check for selected model run
    if not run:
        run = 'OP500'

    # check for path
    if not dir_path:
        dir_path = get_Dataset_path(run)
        print(f'dir_path: {dir_path}')

    # create Dataset
    ds = _create_StationData(dir_path, var_list, run, coords, method)

    # check for saving
    if save:
        _save_toNetCDF(ds, run)

    return ds


# %% Main

if __name__ == '__main__':

    HELP = """preprocess_Station_hybridPressure: Interpolate data to defined location and combine datasets.

    Usage:
       -h                      : print the help
       -r                      : modelrun to preprocess ('OP500, OP1000, OP2500, ARP1000, IFS1000')
                                  (default: 'OP500')
       -p                      : parameters to preprocess (e.g. 'u,v,pres,..')
       -c                      : coordinates (lon, lat) e.g. '11,47'
       -i                      : path to GRIB files (optional)
       -s                      : save interpolated file as netcdf (optional)
    """

    # parse and validate user arguments
    args = list(sys.argv)
    args_ = args
    nargs = len(args)

    run = 'OP500'
    if '-r' in args:
        run = args.pop(args.index('-r') + 1)
        args.remove('-r')

    var_list = []
    if '-p' not in args and '-h' not in args:
        raise ValueError('Need to define at least one parameter to preprocess!!')
    elif '-p' in args:
        var_list = args.pop(args.index('-p') + 1)
        var_list = var_list.replace(' ', '').split(',')
        args.remove('-p')

    if '-c' in args:
        coords = args.pop(args.index('-c') + 1)
        coords = coords.replace(' ', '').split(',')
        coords = tuple(float(c) for c in coords)
        args.remove('-c')

    dir_path = None
    if '-i' in args:
        dir_path = args.pop(args.index('-i') + 1)
        args.remove('-i')

    save = False
    if '-s' in args:
        save = True
        args.remove('-s')

    if '-h' in args:
        print(HELP)
    else:
        tmp_start = input(f'Start interpolation for {var_list} of {run} to {coords} (y/n):')
        if tmp_start == 'y':
            preprocess_Station_main(dir_path, var_list, run, coords, save=save)
