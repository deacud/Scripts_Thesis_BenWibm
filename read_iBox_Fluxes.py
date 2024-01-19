#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:40:00 2023

@author: benwib

Script to read in iBox Flux data
"""

import numpy as np
import pandas as pd
import datetime
import os.path as path
import glob2
from read_StationMeta import get_StationMetaData


def read_fluxes(station_meta, dir_fluxes, start_time, end_time,
                columns=['rawdate', 'h1cc', 'le1cw', 'qcflag_sh1', 'qcflag_le1'], correction=True):
    '''
    Function to read in the turbulent fluxes data of ibox stations

    Parameters
    ----------
    station_meta : pandas Series
        meta data of stations
    dir_fluxes : str
        path to the directory of the iBox fluxes data
    start_time : str ('yyyy-mm-dd HH:MM:SS')
        Starting time of the period that should be read in.
    end_time : str ('yyyy-mm-dd HH:MM:SS')
        End time of the period that should be read in.
    columns : list, optional
        columns to read in from dataset. Default: ['rawdate', 'h1cc', 'le1cw', 'qcflag_sh1', 'qcflag_le1']
    correction : bool, optional
        wether manual correction of values should be performed. Default: True.
    Returns
    -------
    data : pandas Dataframe
        Daraframe containing the iBox turbulent Flux data for:
        - sshf:  sensible heat flux in W/m2
        - slhf:  latent heat flux in W/m2
        - h_m:   height of the measurements in m

    '''

    # As the iBox data sets aren't labelled consistently, a dict is needed
    # to bring them all to the same names. Only important parameters are kept.
    # Description of parameters can be found at:
    # https://acinn-data.uibk.ac.at/pages/station-list.html
    ibox_var_dict = {'KOLS': {'sw_in_avg': 'sw_in',
                              'sw_out_avg': 'sw_out',
                              'lw_in_avg': 'lw_in',
                              'lw_out_avg': 'lw_out',
                              'h1cc': 'sshf',
                              'le1cw': 'slhf',
                              'qcflag_sh1': 'qcflag_sshf',
                              'qcflag_le1': 'qcflag_slhf',
                              },
                     'HOCH': {'cnr4_sw_in_wm2': 'sw_in',
                              'cnr4_sw_out_wm2': 'sw_out',
                              'cnr4_lw_in_wm2': 'lw_in',
                              'cnr4_lw_out_wm2': 'lw_out',
                              'h2cc': 'sshf',
                              'le2cw': 'slhf',
                              'qcflag_sh2': 'qcflag_sshf',
                              'qcflag_le2': 'qcflag_slhf',
                              },
                     'WEER': {'cnr4_sw_in_wm2': 'sw_in',
                              'cnr4_sw_out_wm2': 'sw_out',
                              'cnr4_lw_in_wm2': 'lw_in',
                              'cnr4_lw_out_wm2': 'lw_out',
                              'h2cc': 'sshf',
                              'le2cw': 'slhf',
                              'qcflag_sh2': 'qcflag_sshf',
                              'qcflag_le2': 'qcflag_slhf',
                              },
                     'TERF': {'h2cc': 'sshf',
                              'le2cw': 'slhf',
                              'qcflag_sh2': 'qcflag_sshf',
                              'qcflag_le2': 'qcflag_slhf',
                              }
                     }

    # get path to needed data
    file_path = glob2.glob(path.join(dir_fluxes, f'*{station_meta["name"]}*', 'data.csv'))

    # get data
    if len(file_path) == 0:
        raise ValueError(f'No data found for {station_meta["name"]}')
    elif len(file_path) in (1, 2):
        for i, file in enumerate(file_path):
            tmp = pd.read_csv(file, delimiter=';', skiprows=1)
            # convert to datetime
            tmp['rawdate'] = pd.to_datetime(tmp['rawdate'])
            # Cut out the required period
            tmp = tmp.loc[tmp['rawdate'].between(start_time, end_time)]
            # set datetime as index
            tmp = tmp.set_index('rawdate')

            if i == 0:
                data = tmp
            else:
                data = pd.concat([data, tmp], ignore_index=False, axis=1)

    # select only relevant variables
    data = data[list(ibox_var_dict[station_meta.prf].keys())]

    # rename column names
    data.columns = list(ibox_var_dict[station_meta.prf].values())

    # Add the columns that are missing
    for c in ['sw_in', 'sw_out', 'lw_in', 'lw_out', 'sshf', 'slhf', 'qcflag_sshf', 'qcflag_slhf']:
        if c not in data.columns:
            data[c] = np.nan

    # correct data with qualityflags
    # do correciton only for Kolsass as for the rest of the stations the data seems to be ok
    if station_meta.prf == 'KOLS':
        data['sshf'] = data['sshf'].where(data['qcflag_sshf'] > -1)
        data['slhf'] = data['slhf'].where(data['qcflag_slhf'] > -1)

    # test the data for reasonable ranges
    if sum(np.abs(data['sshf']) > 300) > 0:
        if correction:
            data['sshf'][np.abs(data['sshf']) > 300] = np.nan
        else:
            raise ValueError(f"The absolute value of {sum(np.abs(data['sshf']) > 300)} " +
                             "sshf measurements (4m) is greater than 300 W/m2.")
    if sum(np.abs(data['slhf']) > 300) > 0:
        if correction:
            data['slhf'][np.abs(data['slhf']) > 300] = np.nan
        else:
            raise ValueError(f"The absolute value of {sum(np.abs(data['slhf']) > 300)} " +
                             "slhf measurements (4m) is greater than 300 W/m2.")
    if sum(np.abs(data['sw_in']) > 1000) > 0:
        if correction:
            data['sw_in'][np.abs(data['sw_in']) > 1000] = np.nan
        else:
            raise ValueError(f"The absolute value of {sum(np.abs(data['sw_in']) > 1000)} " +
                             "sw_in measurements is greater than 1000 W/m2.")
    if sum(np.abs(data['sw_out']) > 300) > 0:
        if correction:
            data['sw_out'][np.abs(data['sw_out']) > 300] = np.nan
        else:
            raise ValueError(f"The absolute value of {sum(np.abs(data['sw_out']) > 300)} " +
                             "sw_out measurements is greater than 300 W/m2.")
    if sum(np.abs(data['lw_in']) > 500) > 0:
        if correction:
            data['lw_in'][np.abs(data['lw_in']) > 500] = np.nan
        else:
            raise ValueError(f"The absolute value of {sum(np.abs(data['lw_in']) > 500)} " +
                             "lw_in measurements is greater than 500 W/m2.")
    if sum(np.abs(data['lw_out']) > 500) > 0:
        if correction:
            data['lw_out'][np.abs(data['lw_out']) > 500] = np.nan
        else:
            raise ValueError(f"The absolute value of {sum(np.abs(data['lw_out']) > 500)} " +
                             "lw_out measurements is greater than 500 W/m2.")
    if sum(data['slhf'] < - 200) > 0:
        if correction:
            data['slhf'][data['slhf'] < -200] = np.nan
        else:
            raise ValueError(f"The value of {sum(data['slhf'] < -200)} " +
                             "slhf measurements (4m) is smaller than -200 W/m2.")
    return data


def main(station, start_time='2019-09-12T12', end_time='2019-09-14T12'):

    # define paths of station data
    dir_fluxes = '../../Data/CROSSINN/DATA_IOP8/i-Box_FLUXL1AND2/'

    # define path of metadata stations and load it
    path_metadata = '../../Data/CROSSINN/DATA_IOP8/Surface_station/'
    station_meta = get_StationMetaData(path_metadata, station)

    # read data
    df = read_fluxes(station_meta, dir_fluxes, start_time, end_time)

    return df


if __name__ == '__main__':

    station = 'KOLS'
    df = main(station)
