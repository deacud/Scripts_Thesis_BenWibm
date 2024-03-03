#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 08:18:00 2023

@author: benwib

Script to read Microwave Radiometer (MWR) observations 
"""

import xarray as xr
import numpy as np
import glob2
import pandas as pd
from os.path import join, basename
import matplotlib.dates as mdates


def process_MWR(dir_path, start_date, end_date, resample=True, name='MWR', resample_time=10, ele_corr=False):
    """
    Process MWR observations and save it as xarray Dataset.

    Parameters
    ----------
    dir_path : str
        Path where to find MWR data.
    start_date_str : str
        Start date simulation.
    end_date : str
        End date simulation.
    resample: bool
        True: the dataset is resampled with resample_time
    name: str
        Name of the MWR (just for the attrs)
    resample_time: int
        time of resampling (mins)
    ele_corr : bool
    	elevation correction

    Returns
    -------
    ds : xarray dataset
        dataset with absolute humidity (hua) and temperature (t) as a function of time and height.

    """
    path_MWR = np.sort(glob2.glob(join(dir_path, "*mwrBL00_l2_*.nc")))

    # load profile data
    for i, path in enumerate(path_MWR):
        with xr.open_dataset(path) as temp:

            # adapt height
            if 'height' in temp.dims:
                temp['height'] = temp['height'] + temp.zsl.values

            # select only data at zenith angle
            if ele_corr:
                temp = temp.where(temp.ele == 90, drop=True)

            # resample to 1 min mean
            if 'hua' in temp.data_vars:
                var = temp.hua.resample(time='1min', label='right').mean()
                var_name = 'hua'
            elif 'ta' in temp.data_vars:
                var = temp.ta.resample(time='1min', label='right').mean()
                var_name = 't'

            ds_temp = xr.Dataset({var_name: (['height', 'time'], var.values.T)},
                                 coords={'height': var.height.values,
                                         'time': var.time.values}
                                 )
        if i == 0:
            ds = ds_temp
            continue
        ds = ds.merge(ds_temp)  # combine datasets

    # resample data to xx minute mean
    if resample is True:
        ds = ds.resample(time=f'{resample_time}min', label='right').mean()

    ds = ds.assign_attrs(dict(Name=f"Microwave radiometer HATPRO",
                              longitude=temp.lon.values,
                              latitude=temp.lat.values,
                              altitude=temp.zsl.values,
                              station=f"MWR KOLS",
                              resample_time_min=resample_time if resample is True else ''))

    # check again for start and end date
    ds = ds.sel(time=slice(start_date, end_date))

    # invert height coordinate to match model data
    ds = ds.isel(height=slice(None, None, -1))

    return ds


def main(dir_path=None, start_date='2019-09-12 12:00', end_date='2019-09-14 03:00',
         resample=True, resample_time=10, ele_corr=False):
    '''
    Main entry point program.

    Parameters
    ----------
    path : str, optional
        path to data.

    Returns
    -------
    ds : xarray dataset

    '''

    if not dir_path:
        # path MWR
        dir_path = '../../Data/CROSSINN/DATA_IOP8/MWR'

    ds = process_MWR(dir_path, start_date, end_date, resample=resample, name='MWR',
                     resample_time=resample_time, ele_corr=ele_corr)

    return ds


if __name__ == '__main__':

    ds = main()
