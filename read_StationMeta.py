#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:15:47 2023

@author: benwib

Script containing functions to read in station meta data defined in Stations.csv
"""

import pandas as pd
from os.path import join


def get_StationMetaData(path_metadata, station):
    '''
    function to load metadata of station from Stations.csv file

    Parameters
    ----------
    path_metadata : str
        path to directory of Stations.csv file.
    station : str
        abbreviation of station e.g. 'KOLS'.

    Returns
    -------
    station_meta : pandas Series
        Dataframe containing the metadata of station:
        - name 	        station long name
        - prf 	        station short name
        - lat           Latitude of station
        - lon           Longitude of station
        - id            station id
        - comment       comments
        - alt           station elevation in m
        - provider      provider name e.g. ACINN, ZAMG, DWD, ST
    '''

    # load metadata
    metadata = pd.read_csv(join(path_metadata, 'Stations.csv'))

    # get metadata of station
    station_meta = metadata[metadata.prf == station]

    # rename columns
    station_meta = station_meta.rename(columns={'Lat': 'lat', 'Lon': 'lon', 'elevation (m)': 'alt'})

    return station_meta.reset_index(drop=True).squeeze()


def get_StationMetaProvider(path_metadata, provider):
    '''
    function to load metadata of all stations of provider from Stations.csv file

    Parameters
    ----------
    path_metadata : str
        path to directory of Stations.csv file.
    provider : list
        list of providers. e.g. ['ACINN', 'ZAMG', 'DWD', 'ST'].

    Returns
    -------
    station_meta : pandas Series
        Dataframe containing the metadata of station:
        - name 	        station long name
        - prf 	        station short name
        - lat           Latitude of station
        - lon           Longitude of station
        - id            station id
        - comment       comments
        - alt           station elevation in m
        - provider      provider name e.g. ACINN, ZAMG, DWD, ST
    '''
    # load metadata
    metadata = pd.read_csv(join(path_metadata, 'Stations.csv'))

    # get metadata of all station from provider
    station_meta = metadata[metadata.provider.isin(provider)]

    # rename columns
    station_meta = station_meta.rename(columns={'Lat': 'lat', 'Lon': 'lon', 'elevation (m)': 'alt'})

    return station_meta.reset_index(drop=True).squeeze()
