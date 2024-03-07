#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edited on Mon Jun  5 15:46:30 2023

@author: Paolo Deidda and Antonia Fritz

Adjusted for personal needs by Benedikt Wibmer Jun 27, 2023

Script to read in Radiosonde data of CROSSINN campaign or UWYO radiosonde data.
"""

from os.path import join, basename
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import glob2
import xarray as xr
from scipy.interpolate import interp1d
from calculations import _calc_potTemperature, _calc_SH_from_RH, _calc_wind_components, _calc_ms_from_knots
from path_handling import get_ObsDataset_path, dir_metadata
from read_StationMeta import get_StationMetaData


def interp_scipy(df, interp_column, new_heights, method='linear'):
    '''
    Function to interpolate Radiosonde data to new height levels.

    Parameters
    ----------
    df : pandas Dataframe
        Dataframe containing data of Radiosonde measurements.
    interp_column : str
        column containing original heigh values.
    new_heights : numpy array
        New heights to which data is interpolated.
    method : str, optional
        interpolation method. The default is 'linear'.

    Returns
    -------
    pandas Dataframe
        Dataframe containing data interpolated to new height levels.

    '''
    series = {}

    for column in df:
        if column == interp_column:
            series[column] = new_heights
        else:
            interp_f = interp1d(df[interp_column], df[column], kind=method, bounds_error=False)
            series[column] = interp_f(new_heights)

    return pd.DataFrame(series)


def saturatio_wp_buck(T):
    '''
    Return saturation vapor pressure (Buck formula)
    '''
    return 0.61121*np.exp((18.678 - T/234.5)*(T/(257.14 + T)))*10


def process_radiosounds(path_RS, header, variables_rs, var_interest, new_heights):
    '''
    Process radiosoundings. The files are interpolated using new_heights.

    Parameters
    ----------
    path_RS : str
        Path to RS measuremtents.
    header : list
        list of names in order to be used as header: not to be changed.
    variables_rs: dict
        dictionary with keys: new names for the variables (no spaces),
        values: header from radiosound output
    var_interest : list
        list of variables of interest to be extracted.
    new_heights : numpy array
        New heights.

    Returns
    -------
    ds : xarray dataset
        coords: time and height, variables: variables in var_interest

    '''

    # get path of RS data
    path_rs = glob2.glob(join(path_RS, "*.txt"))
    onlyfiles = [basename(f) for f in path_rs]
    print(f'Radiosonde data files: {path_rs}')
    
    # get dates from file name
    dates_rs = pd.to_datetime(
        [i.split("UTC")[0] for i in onlyfiles], format="%Y%m%d_%H"
    )
    # dates to number (days since 1970) and sort in increasing order
    dates_num = np.sort(mdates.date2num(dates_rs))

    d = {name: pd.DataFrame() for name in var_interest}  # create df

    # iterate over files and combine it to Dataframe
    for file in onlyfiles:
        # open the rs file
        rs = pd.read_csv(join(path_RS, file), skiprows=1,
                         header=None, sep="\s+")
        rs.columns = header
        rs = rs.rename(columns={v: k for k, v in variables_rs.items()})

        # interpolate data to defined height levels
        rs = interp_scipy(rs, 'Geopot_m', new_heights)
        date = pd.to_datetime(file.split("UTC")[0], format="%Y%m%d_%H")

        for variable in var_interest:
            if variable == "t":
                d["t"][date] = rs.T_C.values + 273.15
            elif variable == "p":
                d["p"][date] = rs.P_hPa.values * 100
            elif variable == "u":
                d["u"][date], d["v"][date] = _calc_wind_components(rs.Ws_ms.values, rs.Wd_deg.values)
            elif variable == 'dd':
                d['dd'][date] = rs.Wd_deg.values
            elif variable == 'ff':
                d['ff'][date] = rs.Ws_ms.values
            elif variable == "theta":
                d['theta'][date] = _calc_potTemperature((rs.T_C.values + 273.15), (rs.P_hPa * 100))
            elif variable == 'q':
                d["q"][date] = _calc_SH_from_RH(rs.RH_perc.values/100, rs.P_hPa.values, rs.T_C.values)
            elif variable == 'td':
                d['td'][date] = rs.Dew_C.values + 273.15
            elif variable == 'v':
                continue
            else:
                print(f"{variable} not found")

    # create xarray Dataset with needed dimensions
    ds = xr.Dataset(
        coords=dict(
            height=new_heights, time=np.sort(dates_rs),
            datenum=(["time"], dates_num),
        ),
        attrs=dict(Name="Radiosoundings data",
                   longitude=11.6216,  # TODO: adapt after checking
                   latitude=47.3052,
                   station='RS KOLS'),
    )

    # fill xarray Dataset with data
    for dataframe in d:
        if d[dataframe].empty:
            continue
        d[dataframe] = d[dataframe].reindex(
            sorted(d[dataframe].columns), axis=1)
        ds[dataframe] = (["height", "time"], d[dataframe])

    return ds


def process_radiosounds_UWYO(path_RS, var_interest, timestamp, station):
    '''
    Process radiosoundings from University of Wyoming Database
    https://weather.uwyo.edu/upperair/sounding.html.

    Parameters
    ----------
    path_RS : str
        Path to RS measuremtents.
    var_interest : list
        list of variables of interest to be extracted.
    timestamp : str
        timestamp to process
    Returns
    -------
    ds : xarray dataset
        coords: time and height, variables: variables in var_interest

    '''

    # get coordinates of station
    station_meta = get_StationMetaData(dir_metadata, station)
    coords = (station_meta.lon, station_meta.lat)

    # get path to files
    path_rs = glob2.glob(join(path_RS, "*.txt"))
    onlyfiles = np.array([basename(f) for f in path_rs])

    # get dates of data
    dates_rs = pd.to_datetime(
        [i.split("UTC")[0] for i in onlyfiles], format="%Y%m%d_%H"
    )

    # continue only if timestamp available
    if timestamp not in dates_rs:
        print(f'{timestamp} not available in Radiosounding observation for {station}')
        ds = None
    else:
        # select only needed timestamp
        mask_date = dates_rs == timestamp
        path_file = np.array(path_rs)[mask_date][0]

        # create empty pandas dataframes
        d = {name: pd.DataFrame() for name in var_interest}  # create dfs

        # open the rs file
        rs = pd.read_csv(path_file, header=1, skiprows=[2, 3], sep='\s+')
        rs = rs.dropna()

        # get needed data
        for variable in var_interest:
            if variable == "t":
                d["t"] = rs.TEMP.values + 273.15
            elif variable == "p":
                d["p"] = rs.PRES.values * 100
            elif variable == "u":
                if 'SKNT' in rs:
                    d["u"], d["v"] = _calc_wind_components(rs.SKNT.values, rs.DRCT.values, ff_units='knots')
                elif 'SPED' in rs:
                    d["u"], d["v"] = _calc_wind_components(rs.SPED.values, rs.DRCT.values)
            elif variable == 'dd':
                d['dd'] = rs.DRCT.values
            elif variable == 'ff':
                if 'SKNT' in rs:
                    d['ff'] = _calc_ms_from_knots(rs.SKNT.values)
                elif 'SPED' in rs:
                    d['ff'] = rs.SPED.values
            elif variable == "theta":
                d['theta'] = rs.THTA.values
            elif variable == 'q':
                d["q"] = _calc_SH_from_RH(rs.RELH.values/100, rs.PRES.values, rs.TEMP.values)
            elif variable == 'td':
                d['td'] = rs.DWPT.values + 273.15
            elif variable == 'v':
                continue
            else:
                print(f"{variable} not found")

        # create xarray Dataset
        ds = xr.Dataset(
            coords=dict(height=rs.HGHT.values, time=pd.to_datetime(timestamp)),
            attrs=dict(Name="Radiosounding data",
                       longitude=coords[0],
                       latitude=coords[1],
                       station=f'RS {station[3:]}'),
        )

        # assign data
        for dataframe in d:
            ds[dataframe] = (["height"], d[dataframe])

        ds = ds.expand_dims(dim='time')

    return ds


def read_RS_CROSSINN(path_RS, var_interest, heights):
    '''
    Open Radiosounds and create a xarray dataframe with the variables of interest.
    The files are interpolated using new_heights.

    The function calls "process_radiosounds", which actually creates the dataset.
    This function has the hard coded head for the CROSSINN campaign rs.

    The possible variables are ['t', 'p', 'u', 'v', 'ff', 'dd', 'theta', 'q'].
    Parameters
    ----------
    path_RS : str
        Path of the RS files.
    var_interest : list
        Variables of interest between ['t', 'p', 'u', 'v', 'ff', 'dd', 'theta', 'q'].
    heights : np.array / list
        heights for interpolating the RS output.

    Returns
    -------
    ds : Xarray dataset
        Xarray dataset containing the varibles chosen in 'var_interest'.

    '''
    # header of the radiosoundings
    header = ['Time [sec]', 'P [hPa]', 'T [°C]', 'Hu [%]', 'Ws [m/s]',
              'Wd [°]', 'Alt [m]', 'Geopot [m]', 'Dew [°C]']
    # conversion to a format without spaces
    variables_rs = {
        "delta_time": 'Time [sec]',
        "P_hPa": 'P [hPa]',
        "T_C": 'T [°C]',
        "RH_perc": 'Hu [%]',
        "Ws_ms": 'Ws [m/s]',
        "Wd_deg": 'Wd [°]',
        "Alt_m": 'Alt [m]',
        "Geopot_m": 'Geopot [m]',
        "Dew_C": 'Dew [°C]',
    }

    # actually create the xarray dataset
    ds = process_radiosounds(path_RS, header, variables_rs, var_interest, heights)

    return ds


def main(path_RS=None, station='KOLS', timestamp=None):
    '''
    Main entry point program.

    Parameters
    ----------
    path_RS : str, optional
        path to data.
    station: str, optional
        name of station
    timestamp: str, optional
       timestamp to use. Only needed for UWYO Radiosounding data.

    Returns
    -------
    ds_rs : xarray dataset
        coords: time and height, variables: variables in var_interest

    '''
    if not path_RS:
        path_RS = get_ObsDataset_path('RS')

    # The possible variables are ['t', 'p', 'u', 'v', 'ff', 'dd', 'theta', 'q']
    var_interest = ['t', 'p', 'u', 'v', 'ff', 'dd', 'theta', 'q', 'td']

    # heights: heights where to interpolate the RS output
    heights = np.arange(550, 5001, 10)

    print(f'station: {station}')
    if station == 'KOLS':
        ds_rs = read_RS_CROSSINN(path_RS, var_interest, heights)
    else:
        ds_rs = process_radiosounds_UWYO(path_RS, var_interest, timestamp, station)

    return ds_rs


if __name__ == '__main__':
    ds_rs = main()
