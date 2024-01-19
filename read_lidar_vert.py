#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:22:24 2023

@author: Paolo Deidda

Adjusted for personal needs by Benedikt Wibmer Jun 27, 2023

Script to read in SL88 lidar data of the CROSSINN campaign.
"""

from os.path import join, basename
import pandas as pd
import matplotlib.dates as mdates
import glob2
import xarray as xr


def process_lidar(path_lidar, start_date, end_date, resample=True, name='SL88', resample_time=10):
    """
    Process lidar variable. Stored in a xarray dataset, not interpolated over
    mean heights

    Parameters
    ----------
    path_lidar : str
        Path where to find lidar data.
    start_date_str : str
        Start date data.
    end_date : str
        End date data.
    resample: bool, optional
        True: the dataset is resampled with resample_time. Default: True
    resample_time: int, optional
        time of resampling (mins). Default: 10

    Returns
    -------
    ds : xarray dataset
        dataset with u,v,w,ff,dd as a function of time and height.

    """

    # get path of SL88 lidar data
    path_lidar = glob2.glob(join(path_lidar, "sl88*.nc"))

    # define start and end date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # iterate over available files
    i = 0
    for file, path in zip([basename(f) for f in path_lidar], path_lidar):
        day = pd.to_datetime(file.split(sep='_')[1])  # get date from file name
        if day.date() < start_date.date() or day.date() > end_date.date():
            continue
        with xr.open_dataset(path) as temp:

            ds_temp = xr.Dataset({'u': (['height', 'time'], temp.ucomp.values),
                                  'v': (['height', 'time'], temp.vcomp.values),
                                  'w': (['height', 'time'], temp.wcomp.values),
                                  'ff': (['height', 'time'], temp.ff.values),
                                  'dd': (['height', 'time'], temp.dd.values)},
                                 coords={'height': temp.height.values + temp.alt,
                                         'time': temp.time.values,
                                         'datenum': ('time', temp.datenum.values)},
                                 )
        if i == 0:
            i = 1
            ds = ds_temp
            continue
        ds = ds.merge(ds_temp)  # combine datasets

    # resample data to xx minute mean
    if resample is True:
        ds = ds.resample(time=f'{resample_time}min', label='right').mean()

    # adapt dataset
    ds = ds.assign_coords({'datenum': ('time', mdates.date2num(ds.time))})
    ds = ds.assign_attrs(dict(Name=f"Lidar {name}",
                              longitude=temp.lon,
                              latitude=temp.lat,
                              altitude=temp.alt,
                              station=f"Lidar {name}",
                              resample_time_min=resample_time if resample is True else ''))

    # check again for start and end date
    ds = ds.sel(time=slice(start_date, end_date))

    # invert height coordinate to match model data
    ds = ds.isel(height=slice(None, None, -1))

    return ds


def main(path=None, start_date='2019-09-12 11:50', end_date='2019-09-14 03:00',
         resample=False, resample_time=10):
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
        activate/deactivate resampling of data
    resample_time: int, optional
        resampling time. Default: 10 (minutes)

    Returns
    -------
    ds : xarray dataset
        dataset with u,v,w,ff,dd as a function of time and height.

    '''

    if not path:
        # path lidar (either SL88 or SLXR142)
        path = '../../Data/CROSSINN/DATA_IOP8/Lidars_vertical/Lidar_SL88'

    ds = process_lidar(path, start_date, end_date, resample=resample, name='SL88',
                       resample_time=resample_time)

    return ds


if __name__ == '__main__':

    ds = main()
