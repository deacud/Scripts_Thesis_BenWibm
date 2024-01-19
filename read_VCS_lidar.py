#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:42:09 2022

@author: Functions taken by Antonia Fritz,
edited by Paolo Deidda to be used with xarray and to retrieve a dataset
edited by Benedikt Wibmer to personal needs

Script to read in WLS200s vertical cross section lidar data of CROSSINN campaign.
"""

import glob2
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def process_lidar(path_data, start_date, end_date, height_km=None, resample=True, resample_time=30):
    '''
    Function to retrieve the time, coordinates and wind components
    from the Lidar files for each hour.
    Creates a dataset containing all and concatenate the datasets for the different hours.

    Parameters
    ----------
    path_data : str
        Path to the LiDAR cross section data == Path to unpacked
        WLS200s_coplanar_retrieval_YYYYMMDD.zip files from
        https://publikationen.bibliothek.kit.edu/1000127847
    start_date : str
        start date from which we retrieve the data. Format: 'yyyy-mm-dd HH'.
    end_date : str
        start date from which we retrieve the data. Format: 'yyyy-mm-dd HH'.
    height_km : float, optional
        height where to cut the dataset.
    resample : bool, optional
        activate/deactivate resampling. The default is True.
    resample_time : int, optional
        Minutes to use as resampling time. The default is 30.

    Returns
    -------
    ds : xarray dataset
        Dataset of the files from start_date to end_date with dt = resample_time.

    '''

    # define timesteps to retrieve data
    if start_date == end_date:
        timesteps = [start_date]
    else:
        timesteps = pd.date_range(start_date, end_date, freq='h', inclusive='left')
    for i, time in enumerate(timesteps):
        datetime = str(time)
        # retrieve the date in str
        date = datetime[0:4] + datetime[5:7] + datetime[8:10]
        hour = datetime[11:13]
        # look for the file
        dirs = glob2.glob(os.path.join(path_data, f'*coplanar_retrieval*{date}'))
        if len(dirs) != 1:
            raise ValueError(f'None or too many data sources: {dirs}')
        file = glob2.glob(os.path.join(dirs[0], f'CROSSINN*{hour}00-*'))
        if len(file) != 1:
            raise ValueError(f'None or too many data sources: {file}')
        # open the file
        data_nc = xr.open_mfdataset(file, decode_times=False)

        # write another dataset
        ds_temp = xr.Dataset(
            data_vars=dict(
                t_wind=(["valid_time", "index", "height"], -
                        np.transpose(data_nc.v_DD.values, (2, 1, 0))[:, ::-1]),
                w=(["valid_time", "index", "height"], np.transpose(data_nc.w_DD.values, (2, 1, 0))[:, ::-1]),
            ),
            coords=dict(
                index=data_nc.DD_mesh_horizontal_grid_point_count.values,
                distance=('index', data_nc.y_axis_DD.values/1000),
                height=(data_nc.z_axis_DD.values + 546),
                valid_time=pd.to_datetime(data_nc.timestamp, origin='unix', unit='ms'),
            ),
        )
        if i == 0:
            ds = ds_temp
        else:
            ds = xr.concat([ds, ds_temp], dim='valid_time')

    # resample data
    if resample:
        ds = ds.resample(valid_time=f'{resample_time}min', label='right',
                         closed='right').mean()  # resample over

    # add attributes
    ds.attrs = data_nc.attrs
    ds = ds.assign_attrs(dict(Name="Lidar WLS200s Kolsass, Hochhaeuser, Mairbach",
                              modelrun='WLS200s',
                              resample_time_min=resample_time if resample is True else ''))

    # check for height limit
    if height_km is not None:
        ds = ds.sel(height=slice(None, height_km*1000))

    return ds


def main(path=None, start_date='2019-09-12 12:00', end_date='2019-09-14 03:00',
         resample=False, resample_time=10, height_km=3.5):
    '''
    Main entry point program.

    Parameters
    ----------
    path : str, optional
        path to data.
    start_date: str, optional
        start date of data needed
    end_date: str, optional
        end date of data needed
    resample: bool, optional
        activate/deactivate resampling of data. Default: False
    resample_time: int, optional
        resampling time in minutes. Default: 10
    height_km: float, optional
        height in km where to cut data. Default: 3.5

    Returns
    -------
    ds : xarray dataset
        dataset with u,v,w,ff as a function of time and height.

    '''

    if not path:
        # path lidar
        path = '../../Data/CROSSINN/DATA_IOP8/Lidar_VCS'

    ds = process_lidar(path, start_date, end_date, height_km=height_km,
                       resample=resample, resample_time=resample_time)

    return ds


if __name__ == '__main__':
    ds = main()
