#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:33:52 2023

@author: benwib

Script to read in GRIB files and preprocess data on height levels and save it as netcdf file.
"""

import xarray as xr
import numpy as np
from os.path import join
import glob2
import time
import gc
import sys
from functools import partial
from dask.distributed import Client, LocalCluster
import webbrowser
from path_handling import get_GRIB_path, get_Dataset_path, set_Dataset_name

# dicts of unknown variables in GRIB files
# !!!ADJUST if further variables needed!!!

var_dict_height = {'dict_type': 'height',
                   'typeOfLevel': 'heightAboveGround',
                   'message': [19, 20],
                   'GRIB_paramId': [201, 202],
                   'GRIB_shortName': ['mx2t', 'mn2t'],
                   'GRIB_name': ['Maximum temperature at 2 metres since previous post-processing',
                                 'Minimum temperature at 2 metres since previous post-processing'],
                   'GRIB_units': ['K', 'K']}

dict_DX = {'OP500': 500,
           'OP1000': 1000,
           'OP2500': 2500,
           'IFS1000': 1000,
           'ARP1000': 1000}


def _preprocess_vars(ds, extent=None):
    '''
    function to preprocess variables

    Parameters
    ----------
    ds : xarray Dataset
    extent : list, optional
        extent of lon/lat coordinates to keep. Default: None

    '''

    # slice to smaller extent -> less memory and time consuming
    if extent:
        lons = extent[0:2]
        lats = extent[::-1][0:2]
        ds = ds.sel(longitude=slice(lons[0], lons[1]), latitude=slice(lats[0], lats[1]))

    return ds


def _checkTimestamps(ds, delta_m=10):
    '''
    function to correct timestamps

    Parameters
    ----------
    ds : xarray Dataset
    delta_m : int, optional
        Timestamp delta time in minutes. The default is 10 minutes.

    Returns
    -------
    ds : xarray Dataset

    '''

    # initial time
    init = ds.valid_time.values[0]
    n_steps = len(ds.valid_time)

    steps = init + np.arange(n_steps) * np.timedelta64(delta_m, 'm')
    ds = ds.assign_coords(valid_time=steps)

    return ds


def _open_mfdataset(dir_path, cfgrib_kwargs, preprocess):
    '''
    Open multiple GRIB files within directory dir_path.

    Parameters
    ----------
    dir_path : str
        path of directory where GRIB files located.
    cfgrib_kwargs_ : dict
        kwargs for cfgrib.
    preprocess : function
        function to preprocess variables. Input for xr.open_mfdataset.

    Returns
    -------
    ds : xarray.Dataset
        preprocessed Dataset of defined parameters in GRIB files depending on typeOfLevel

    '''
    # set path to grib files
    path_files = sorted(glob2.glob(join(dir_path, 'GRIBPFAROMAROM+*.grib2')))

    # open all files
    # !!!WB: split them up in 1/2 -> maybe works better
    n_half = int((len(path_files) + 1) / 2)
    files_ = [path_files[0:n_half],
              path_files[n_half:]]

    for i, p_files in enumerate(files_):
        print(f'Start processing part {i+1}/2 of files ...')
        tmp = xr.open_mfdataset(p_files, engine='cfgrib', combine='nested', concat_dim='valid_time',
                                chunks={'valid_time': 1}, parallel=True, preprocess=preprocess,
                                drop_variables=['step', 'time', 'heightAboveGround'],
                                backend_kwargs=cfgrib_kwargs)
        tmp.load()

        # concat datasets after iteration
        if i == 0:
            ds = tmp
        else:
            ds = xr.concat([ds, tmp], dim='valid_time')
        del tmp
        gc.collect()

    return ds


def _save_toNetCDF(ds, run, var, typeOfLevel):
    '''
    function to save xarray Dataset to netcdf file

    Parameters
    ----------
    ds : xarray Dataset
        dataset containing all parameters on "typeOfLevel".
    run : str
        modelrun
    var: str
        parameter name
    typeOfLevel : str
        level of preprocessed parameters.
    '''

    # get save path and save it as netcdf file
    dir_save = get_Dataset_path(run)
    name_save = set_Dataset_name(ds, typeOfLevel, var=var)
    path_save = join(dir_save, name_save)
    print(f'Save file to {path_save}')
    ds.to_netcdf(path_save)


def _preprocess_height_instant(dir_path, run, levels=[2, 10], var_dict=var_dict_height,
                               extent=None, save=False):
    '''
    function to preprocess information from GRIB files and save it as netcdf file.

    Parameters
    ----------
    dir_path : str
        path of directory where GRIB files located.
    run : str
        name of modelrun.
    levels : list
        levels to preprocess e.g. 2m and 10m. Default: [2, 10]
    var_dict : dict, optional
        dictionary with information of unknown parameters in GRIB File depending on typeOfLevel.
    extent : list, optional
        extent of lon/lat coordinates to keep.
    save : bool, optional
        save preprocessed file to netcdf file

    Returns
    -------
    ds : xarray Dataset
        preprocessed Dataset of parameters on define "typeOfLevel".

    '''
    # get type of level to preprocess
    typeOfLevel = var_dict.get('typeOfLevel')

    # ---- 1. load instant parameters depending on level
    for i, lev in enumerate(levels):
        cfgrib_kwargs = {'filter_by_keys':
                         {'typeOfLevel': typeOfLevel,
                          'stepType': 'instant',
                          'level': lev},
                         'indexpath': ''}

        # do preprocessing
        preprocess = partial(_preprocess_vars, extent=extent)

        start = time.time()
        tmp = _open_mfdataset(dir_path, cfgrib_kwargs, preprocess)

        # add levels together
        if i == 0:
            ds = tmp
        else:
            ds = xr.merge([ds, tmp])
        del tmp
        gc.collect()
        end = time.time()
        print(f'elapsed time preprocessing instant variables on level {lev}: {end - start}')

    # ---- 2. check time stamps
    # !!!WB: Problem -> no correct 10min timestamps
    ds = _checkTimestamps(ds)

    # ---- 3. add global dataset attributes
    DX = dict_DX.get(run)
    time_now = time.localtime()
    time_now = time.strftime("%Y-%m-%d %H:%M:%S", time_now)
    domain = extent if extent else 'whole'
    ds = ds.assign_attrs(name=f'AROME {typeOfLevel} parameters on {levels} m',
                         domain=domain,
                         modelrun=run,
                         DX=DX,
                         history=f'Created by Benedikt Wibmer on {time_now}')

    # ---- 4. save as netCDF file if needed
    # WB: carefull, very large files, use only when needed
    if save:
        _save_toNetCDF(ds, run, typeOfLevel)

    return ds


def preprocess_GRIB2_height_main(run=None, dir_path=None, levels=None,
                                 extent=None, save=False):
    '''
    Main entry point preprocess GRIB2 files.

    Parameters
    ----------
    run : str, optional
        name of modelrun.
    dir_path : str, optional
        path to directory of GRIB files.
    levels : list, optional
        list of levels to extract from GRIB files.
    extent : list, optional
        extent of lon/lat coordinates to keep.
    save : bool, optional
        save preprocessed file to netcdf file

    '''
    # start Dask Client
    print('Start client')
    try:
        client = Client('tcp://localhost:8786', timeout='2s')
    except OSError:
        cluster = LocalCluster(scheduler_port=8786)
        client = Client(cluster)
        client.restart()
    webbrowser.open(client.dashboard_link)

    # check for selected model run
    if not run:
        run = 'OP500'

    # check for path
    if not dir_path:
        dir_path = get_GRIB_path(run)

    print(f'Start preprocess GRIB2 files for instant variables on levels {levels} from directory: {dir_path}')
    ds = _preprocess_height_instant(dir_path, run, levels=levels, extent=extent, save=save)

    # free up space
    ds.close()
    del ds
    gc.collect()

    # close client
    client.close()


# %% Main

if __name__ == '__main__':

    HELP = """preprocess_GRIB2_height: Preprocessing and combination of AROME GRIB2 output files.

    Usage:
       -h                      : print the help
       -r                      : modelrun to preprocess ('OP500, OP1000, OP2500, ARP1000, IFS1000')
                                  (default: 'OP500')
       -l,                     : levels to preprocess (e.g. '2, 10') (default 2 and 10)
       -e                      : extent of lon/lat grid (e.g. '11, 12, 47, 48') (optional)
       -i                      : path to GRIB files (optional)
       -s                      : save preprocessed file as netcdf (optional)
    """

    # parse and validate user arguments
    args = list(sys.argv)
    args_ = args
    nargs = len(args)

    run = 'OP500'
    if '-r' in args:
        run = args.pop(args.index('-r') + 1)
        args.remove('-r')

    levels = [2, 10]
    if '-l' in args:
        levels = args.pop(args.index('-l') + 1)
        levels = levels.replace(' ', '').split(',')
        levels = [int(lev) for lev in levels]
        args.remove('-l')

    extent = None
    if '-e' in args:
        extent = args.pop(args.index('-e') + 1)
        extent = extent.replace(' ', '').split(',')
        extent = [float(e) for e in extent]
        args.remove('-e')

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
        tmp_start = input(f'Be careful (!!large data size!!) '
                          f'Start preprocessing instant heightAboveGround variables on levels {levels} '
                          f'of {run} to {extent} (y/n):')
        if tmp_start == 'y':
            preprocess_GRIB2_height_main(run=run, dir_path=dir_path, levels=levels,
                                         extent=extent, save=save)
