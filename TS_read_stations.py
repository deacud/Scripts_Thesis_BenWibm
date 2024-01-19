#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to read in data from weather stations of DWD, ZAMG, ACINN, and the
south tyrolean weather service (ST) and to unify ACINN data

@author: Antonia Fritz, edited by Benedikt Wibmer

Created: 02.06.2022, Edited: 28.06.2023
"""

import numpy as np
import pandas as pd
import datetime
import os.path as path
import glob2


# %% Function to read in DWD data: temperature and wind speed
def read_dwd(station, metadata, path_DWD, start_time, end_time):
    '''
    Function to read in the station data of DWD

    Parameters
    ----------
    station : str
        Abreviation for the station as indicated in Stations.csv
    metadata : pandas DataFrame
        Data from Stations.csv read in using 'read_meta()'
    path_DWD : str
        Path to the DWD temperature and wind speed data
    start_time : str ('yyyy-mm-dd HH:MM:SS')
        Starting time of the period that should be read in.
    end_time : str ('yyyy-mm-dd HH:MM:SS')
        End time of the period that should be read in.

    Returns
    -------
    data_dwd : pandas Dataframe
        Data of DWD station. If temperature and wind speed information is
        provided, both datasets are combined to one Dataframe. The timestep
        is used as the index. The column names are used as provided by DWD:
            QN_T    quality level of temperature data set
            PP_10   air pressure at station elevation (hPa)
            TT_10   air temperature in 2m height (°C)
            TM5_10  air temperature in 5cm height (°C)
            RF_10   relative humidity in 2m height (%)
            TD_10   dew point temperature in 2m height (°C)
            QN_W    quality level of wind data set
            FF_10   mean of wind speed during the last 10 minutes (m/s)
            DD_10   mean of wind direction during the last 10 minutes (degree)
    '''
    # Test whether the provided station name is unique
    n = len(metadata[metadata.prf == station])
    if n != 1:
        raise ValueError(f'The station name {station} exists {n} times ' +
                         'in Stations.csv. Must be unique.')

    # Test whether the station is a DWD station according to Stations.csv
    if metadata[metadata.prf == station]['provider'].values[0] != 'DWD':
        raise ValueError(f'{station} is not a DWD station in Stations.csv.')

    # Extract the station ID
    dwd_ID = metadata[metadata.prf == station]['id'].values[0]

    # Convert the start and end time to DWD time format
    start_str = start_time[0:4] + start_time[5:7] + start_time[8:10] + \
        start_time[11:13] + start_time[14:16]
    end_str = end_time[0:4] + end_time[5:7] + end_time[8:10] + \
        end_time[11:13] + end_time[14:16]

    # define the filename
    file_dwd = glob2.glob(path.join(path_DWD,
                                    f'produkt_zehn_min_*_*{dwd_ID}*.txt'))
    # If no file is found:
    if len(file_dwd) == 0:
        raise ValueError(f"No data found for DWD station {station} {dwd_ID}!")
    # If exactly one file is found:
    elif len(file_dwd) in (1, 2):
        temp = []
        for i, file in enumerate(file_dwd):
            # read in the data (as string due to the eor column)
            temp_raw = np.loadtxt(file, skiprows=1, delimiter=';', dtype=str)
            # Cut out the needed data
            temp_raw = temp_raw[temp_raw[:, 1] >= start_str]
            temp_raw = temp_raw[temp_raw[:, 1] <= end_str]
            # Give the data a header
            header = np.loadtxt(file, max_rows=1, delimiter=';', dtype=str)
            data = pd.DataFrame(temp_raw, columns=header)
            # Use the datetime information as a index
            data['MESS_DATUM'] = pd.to_datetime(data.MESS_DATUM)
            data = data.set_index('MESS_DATUM')
            # rename index column
            data.index = data.index.rename('valid_time')
            # drop unneccesary columns
            data = data.drop(columns='eor')
            # check if temperature or wind file
            if 'tu' in file:
                # As 'QN' is given in both data sets, rename it to 'QN_T'
                data.rename(columns={'  QN': 'QN_T'}, inplace=True)
            elif 'ff' in file:
                data = data.drop(columns='STATIONS_ID')
                # As 'QN' is given in both data sets, rename it to 'QN_W'
                data.rename(columns={'  QN': 'QN_W'}, inplace=True)
            # Convert data type
            data = data.apply(pd.to_numeric)
            temp.append(data)

        # Combine data sets
        data_dwd = pd.concat(temp, ignore_index=False, axis=1)

        # check if all data is availabe otherwise fill it with nan
        for var in ['QN_T', 'PP_10', 'TT_10', 'TM5_10', 'RF_10', 'TD_10', 'QN_W', 'FF_10', 'DD_10']:
            if var not in data_dwd.columns:
                data_dwd[var] = np.nan

    # If more than two files are found:
    else:
        raise ValueError('More than two matching files found for ' +
                         f'station {station} ({dwd_ID}).')

    # Convert data type to numeric
    data_dwd = data_dwd.apply(pd.to_numeric)
    # Replace all DWD -999 with np.nan
    data_dwd[data_dwd == -999] = np.nan

    # Test the data for reasonable ranges
    if sum(data_dwd['TT_10'] > 50) > 0:
        raise ValueError(f"There are {sum(data_dwd['TT_10'] > 50)} " +
                         "temperature measurements (2m) greater than 50°C.")
    if sum(data_dwd['TM5_10'] > 50) > 0:
        raise ValueError(f"There are {sum(data_dwd['TM5_10'] > 50)} " +
                         "temperature measurements (5cm) greater than 50°C.")
    if sum(data_dwd['DD_10'] > 360) > 0:
        data_dwd[data_dwd['DD_10'] > 360] = np.nan
        print(f"Warning: There are {sum(data_dwd['DD_10'] > 360)} wind " +
              "directions greater than 360°. These are set to NAN.")
    if sum(data_dwd['DD_10'] < 0) > 0:
        raise ValueError(f"There are {sum(data_dwd['DD_10'] < 0)} wind " +
                         "directions smaller than 0°.")

    return data_dwd


# %% Function to read in ACINN data
def read_acinn(station, metadata, path_ACINN, start_time, end_time,
               correct_direction=True):
    '''
    Function to read in the station data of ACINN

    Parameters
    ----------
    station : str
        Abreviation for the station as indicated in Stations.csv
    metadata : pandas DataFrame
        Data from Stations.csv read in using 'read_meta()'
    path_ACINN : str
        Path to the ACINN data
    start_time : str ('yyyy-mm-dd HH:MM:SS')
        Starting time of the period that should be read in.
    end_time : str ('yyyy-mm-dd HH:MM:SS')
        End time of the period that should be read in.

    Returns
    -------
    data : pandas Dataframe
        Daraframe containing the ACINN data for:
        - t_C:    2m Temperature in Celsius
        - p_hPa:  surface pressure in hPa
        - ff_ms:  wind speed in m/s
        - dd_deg: wind direction in degree
        - h_m:   height of the wind sensor in m
        The measurements closest to the AROME output heights (agl) are selected.

    '''
    # Test whether the provided station name is unique
    n = len(metadata[metadata.prf == station])
    if n != 1:
        raise ValueError(f'The station name {station} exists {n} times ' +
                         'in Stations.csv. Must be unique.')

    # Test whether the station is an ACINN station according to Stations.csv
    if metadata[metadata.prf == station]['provider'].values[0] != 'ACINN':
        raise ValueError(f'{station} is not an ACINN station in Stations.csv.')

    # As the ACINN data sets aren't labelled consistently, a dict is needed
    # to bring them all to the same names. Only important parameters are kept.
    # Description of parameters can be found at:
    # https://acinn-data.uibk.ac.at/pages/station-list.html
    acinn_v = {
        'KOLS': {'taact1_avg': 't_C',        # Kolsass
                 'pact': 'p_hPa',
                 'wind_speed_4': 'ff_ms',
                 'avg_wdir4': 'dd_deg',
                 'h_m': 12},  # height above ground wind measurement
        'ARB': {'taact_3m_avg': 't_C',   # 'taact_1m_avg': 't1m_C',     # Arbeser, 2m temperature dismounted
                'pact': 'p_hPa',
                'ws_young_avg': 'ff_ms',
                'ws_young_wvc_2': 'dd_deg',
                'h_m': 2},
        'WEER': {'ta_avg': 't_C',            # Weerberg, only 5m temperature available
                 'p_avg': 'p_hPa',
                 'h_m': np.nan},             # no wind measuremenst available
        'TERF': {'taact_avg': 't_C',         # Terfens, only 11m temperature available
                 'pact': 'p_hPa',
                 'h_m': np.nan},             # no wind measurements available
        'HOCH': {'ta_avg': 't_C',            # Hochhäuser, only 7m temperature available
                 'p': 'p_hPa',
                 'ws_3_avg': 'ff_ms',
                 'h_m': 3},                  # no wind direction available (only at 1m)
        'ELL': {'t1_avg': 't_C',             # Ellbögen (maybe also t2_avg?)
                'p_avg': 'p_hPa',
                'ff_s_wvt': 'ff_ms',
                'dd_d1_wvt': 'dd_deg',
                'h_m': np.nan},              # unknown
        'SAT': {'t_avg': 't_C',              # Sattelberg
                'p_avg': 'p_hPa',
                'ff_s_wvt': 'ff_ms',
                'dd_d1_wvt': 'dd_deg',
                'h_m': np.nan},              # unknown
        'UNI': {'tl': 't_C',                 # University Innsbruck  Schoepfstrasse tl (579.2m)
                # 'tl2': 't_C_roof',         # Rooftop tl2 (617 m asl)
                'p': 'p_hPa',
                'ffm': 'ff_ms',
                'ddm': 'dd_deg',
                'h_m': 41},                  # wind measured on roof of University at 617.7 m asl
        'EGG': {'ta_avg': 't_C',            # temperature and humidity only at 6.18m
                'p_avg': 'p_hPa',
                'h_m': np.nan},             # no wind measurements in Eggen
    }
    # Read in the data
    # When downloading the data from the ACINN website, it is given as one (or
    # sometimes two) files named 'data.csv' inside a folder including the
    # station name.
    name = metadata[metadata['prf'] == station].name.values[0]
    file_ACINN = glob2.glob(path.join(path_ACINN, f'*{name}*', 'data.csv'))

    if len(file_ACINN) == 0:
        raise ValueError(f'No ACINN data found for {station}')
    # If stored in one two or three files
    elif len(file_ACINN) in (1, 2, 3):
        temp = []
        for i, file in enumerate(file_ACINN):
            # First data set
            temp.append(pd.read_csv(file_ACINN[i], delimiter=';', skiprows=1))
            # Convert to datetime
            temp[i]['rawdate'] = pd.to_datetime(temp[i]['rawdate'])
            # Cut out the required period
            temp[i] = temp[i].loc[temp[i]['rawdate'].between(start_time, end_time)]
            # set datetime as index
            temp[i] = temp[i].set_index('rawdate')
        # Combine data sets
        data = pd.concat(temp, ignore_index=False, axis=1)
    else:
        raise ValueError(f'More than three ACINN data found for {station}')

    # select only relevant variables
    data = data[list(acinn_v[station].keys())[:-1]]  # h_m not within data
    # rename column names
    data.columns = list(acinn_v[station].values())[:-1]
    # rename index column
    data.index = data.index.rename('valid_time')

    # Add the columns that are missing
    for c in ['t_C', 'p_hPa', 'ff_ms', 'dd_deg', 'h_m']:
        if c not in data.columns:
            data[c] = np.nan

    # Add the height of the wind sensor
    data['h_m'] = acinn_v[station]['h_m']

    # Test the data for reasonable ranges
    if sum(data['t_C'] > 50) > 0:
        raise ValueError(f"There are {sum(data['t_C'] > 50)} " +
                         "temperature measurements (2m) greater than 50°C.")
    if sum(data['dd_deg'] > 360) > 0:
        if correct_direction:
            data[data['dd_deg'] > 360] = np.nan
        else:
            raise ValueError(f"There are {sum(data['dd_deg'] > 360)} wind " +
                             "directions greater than 360°.")
    if sum(data['dd_deg'] < 0) > 0:
        raise ValueError(f"There are {sum(data['dd_deg'] < 0)} wind " +
                         "directions smaller than 0°.")
    if sum(data['p_hPa'] < 800) > 0:
        raise ValueError(f"There are {sum(data['p_hPa'] < 800)} wind " +
                         "directions smaller than 800hPa.")
    if sum(data['p_hPa'] > 1100) > 0:
        raise ValueError(f"There are {sum(data['p_hPa'] > 1100)} wind " +
                         "directions greater than 1100hPa.")

    return data


# %% Function to read in ZAMG data
def read_zamg(station, metadata, data_ZAMG, start_time, end_time):
    '''
    Function to read in the station data of ACINN

    Parameters
    ----------
    station : str
        Abreviation for the station as indicated in Stations.csv
    metadata : pandas DataFrame
        Data from Stations.csv read in using 'read_meta()'
    data_ZAMG : pandas Dataframe
        DataFrame containing ZAMG data of many stations. Read in the down-
        loaded file using 'pd.read_csv(file_ZAMG)'.
    start_time : str ('yyyy-mm-dd HH:MM:SS')
        Starting time of the period that should be read in.
    end_time : str ('yyyy-mm-dd HH:MM:SS')
        End time of the period that should be read in.

    Returns
    -------
    data : pandas Dataframe
        Dataframe containing the data of one ZAMG station:
        - *_FLAG        Quality flag of * parameter
        - DD            Wind direction
        - DDX	        Wind direction at maximum wind speed
        - FF	        Wind speed (vectorial)
        - FFAM	        Wind speed (arithmetic)
        - FFX	        Maximumd wind speed
        - GSX	        Global radiation
        - HSR	        Diffus radiation (mV)
        - HSX	        Diffus radiation(W/m²)
        - P             Pressure (hPa)
        - P0	        Reduced air pressure
        - RF	        Relative humidity
        - RR	        precipitation
        - RRM	        Niederschlagsmelder
        - SH	        Total snow height
        - SO	        sunshine duration
        - TB1		    soil temperature at -10cm
        - TB2	        soil temperature at -20cm
        - TB3		    soil temperature at -50cm
        - TL		    air temperature at 2m
        - TLMAX	        max air temperature at 2m
        - TLMIN	        min air temperature at 2m
        - TP	        dew point temperature
        - TS            air temperature at 5cm
        - TSMAX         max air temperature at 5cm
        - TSMIN         min air temperature at 5cm
        - ZEITX	        time of maximum wind speed
    '''
    # Test whether the provided station name is unique
    n = len(metadata[metadata.prf == station])
    if n != 1:
        raise ValueError(f'The station name {station} exists {n} times ' +
                         'in Stations.csv. Must be unique.')

    # Test whether the station is an ZAMG station according to Stations.csv
    if metadata[metadata.prf == station]['provider'].values[0] != 'ZAMG':
        raise ValueError(f'{station} is not a ZAMG station in Stations.csv.')

    # Extract the station ID from the metadata
    ID = int(metadata[metadata.prf == station]['id'].values[0])
    # extract the correct station
    data = data_ZAMG.loc[data_ZAMG.station == ID]
    # convert time to datetime format
    data['time'] = pd.to_datetime(data['time']).dt.tz_localize(None)
    # Cut out the required timeframe
    data = data.loc[data['time'].between(start_time, end_time)]
    # Use time as an index
    data = data.set_index('time')
    # rename index column
    data.index = data.index.rename('valid_time')

    # Test the data
    if sum(data['TLMAX'] < data['TLMIN']) > 0:
        raise ValueError(f"{sum(data['TLMAX'] < data['TLMIN']) > 0} maximum" +
                         '2m temperatures are smaller than the minimum ones.')
    if sum(data['TL'] > 50) > 0:
        raise ValueError(f"There are {sum(data['TL'] > 50)} " +
                         "temperature measurements (2m) greater than 50°C.")
    if sum(data['DD'] > 360) > 0:
        raise ValueError(f"There are {sum(data['DD'] > 360)} wind " +
                         "directions greater than 360°.")
    if sum(data['DD'] < 0) > 0:
        raise ValueError(f"There are {sum(data['DD'] < 0)} wind " +
                         "directions smaller than 0°.")

    return data


# %% Function to read in ST data
def read_st(station, metadata, path_ST, start_time, end_time):
    '''
    Function to read in the station data of the south tyrolean weather service

    Parameters
    ----------
    station : str
        Abreviation for the station as indicated in Stations.csv
    metadata : pandas DataFrame
        Data from Stations.csv read in using 'read_meta()'
    path_ST : str
        Path to the ST data
    start_time : str ('yyyy-mm-dd HH:MM:SS')
        Starting time of the period that should be read in.
    end_time : str ('yyyy-mm-dd HH:MM:SS')
        End time of the period that should be read in.

    Returns
    -------
    data : pandas Dataframe
        Dataframe containing the ST station data:
        - TimeStamp 	date/time of measurement (CEST ---> UTC = CEST - 2h)
        - NAME 	        Station Name
        - GS 	        Solar Radiation
        - HS 	        Snow height
        - LD.RED 	    Atmospheric perssion
        - LF 	        Relative humidity
        - LT 	        Air temperature
        - N 	        Precipitation
        - Q 	        Water flow
        - SD 	        Sunshine hours
        - W 	        Water temperature
        - WG 	        Wind speed
        - WG.BOE 	    Wind gust
        - WR 	        Wind direction
        - WT 	        Water temperature
    '''
    # Test whether the provided station name is unique
    n = len(metadata[metadata.prf == station])
    if n != 1:
        raise ValueError(f'The station name {station} exists {n} times ' +
                         'in Stations.csv. Must be unique.')

    # Test whether the station is an ST station according to Stations.csv
    if metadata[metadata.prf == station]['provider'].values[0] != 'ST':
        raise ValueError(f'{station} is not an ST station in Stations.csv.')

    # read in the data
    file_ST = glob2.glob(path.join(path_ST, '13stat_*_grezza_wide.csv'))
    if len(file_ST) == 0:
        raise ValueError('No ST data found.')
    elif len(file_ST) > 1:
        raise ValueError('More than one file with ST data found.')
    else:
        data = pd.read_csv(file_ST[0])
        # Select the requested station
        name = metadata[metadata['prf'] == station].name.values[0]
        data = data[data.NAME == name]
        # convert CEST to UTC
        data['TimeStamp'] = pd.to_datetime(data.TimeStamp) - datetime.timedelta(hours=2)
        # Cut the data
        data = data.loc[data['TimeStamp'].between(start_time, end_time)]
        # provide time index
        data = data.set_index('TimeStamp')
        # rename index column
        data.index = data.index.rename('valid_time')

    # Test the data
    if sum(data['LT'] > 50) > 0:
        raise ValueError(f"There are {sum(data['LT'] > 50)} " +
                         "temperature measurements (2m) greater than 50°C.")
    if sum(data['WR'] > 360) > 0:
        print(f"Warning: There are {sum(data['WR'] > 360)} wind " +
              "directions greater than 360°. These are set to NAN.")
        data[data['WR'] > 360] = np.nan
    if sum(data['WR'] < 0) > 0:
        print(f"Warning: There are {sum(data['WR'] < 0)} wind " +
              "directions smaller than 0°. These are set to NAN.")
        data[data['WR'] < 0] = np.nan
    if sum(data['N'] < 0) > 0:
        print(f"Warning: There are {sum(data['N'] < 0)} negative " +
              "precipitation measurements. These are set to NAN.")
        data[data['N'] < 0] = np.nan

    return data


# %% main function calls
def main(station='KOLS', start_time='2019-09-13', end_time='2019-09-15'):
    '''
    Main entry point script.

    Parameters
    ----------
    station : str, optional
        station name to read in data. The default is 'KOLS'.
    start_time : str, optional
        start time of data needed. The default is '2019-09-13'.
    end_time : str, optional
        end time of data needed. The default is '2019-09-15'.

    Raises
    ------
    ValueError
        if provider not available

    Returns
    -------
    df : pandas Dataframe
        Dataframe containing the observation data for defined station.

    '''

    # define paths of station data
    path_DWD = '../../Data/CROSSINN/DATA_IOP8/Surface_station/DWD_data'
    path_ZAMG = '../../Data/CROSSINN/DATA_IOP8/Surface_station/ZAMG_data'
    path_ST = '../../Data/CROSSINN/DATA_IOP8/Surface_station/ST_data'
    path_ACINN = '../../Data/CROSSINN/DATA_IOP8/Surface_station/ACINN_data'

    # define path of metadata stations
    path_metadata = '../../Data/CROSSINN/DATA_IOP8/Surface_station/'
    metadata = pd.read_csv(path.join(path_metadata, 'Stations.csv'))

    # decide from where station data to load
    provider = metadata[metadata.prf == station].provider.tolist()[0]

    if provider == 'ST':
        # read south tyrolean station data
        df = read_st(station, metadata, path_ST, start_time=start_time, end_time=end_time)
    elif provider == 'ACINN':
        # read acinn station data
        df = read_acinn(station, metadata, path_ACINN, start_time=start_time, end_time=end_time)
    elif provider == 'DWD':
        # read dwd station data
        df = read_dwd(station, metadata, path_DWD, start_time=start_time, end_time=end_time)
    elif provider == 'ZAMG':
        # read ZAMG station data
        # data_ZAMG = pd.read_csv(path.join(path_ZAMG, 'ZEHNMIN Datensatz_20190801_20190930.csv'))
        data_ZAMG = pd.read_csv(path.join(path_ZAMG, 'ZEHNMIN Datensatz_20190901_20190930.csv'))
        df = read_zamg(station, metadata, data_ZAMG, start_time=start_time, end_time=end_time)
    else:
        raise ValueError(f'Provider {provider} does not match.')

    return df


if __name__ == '__main__':
    df = main()
