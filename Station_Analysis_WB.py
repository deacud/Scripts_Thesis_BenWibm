#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:31:48 2023

@author: Benedikt Wibmer

Analysis of timeseries model data vs. station observations
"""

import TS_read_stations
import xarray as xr
from os.path import join, exists
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import gc
from calculations import _calc_ff, _calc_dd, _calc_geopotHeight, _calc_mslPressure
from read_StationMeta import get_StationMetaData, get_StationMetaProvider
from path_handling import get_Dataset_path, get_Dataset_name, get_interpStation_name, dir_metadata
import glob2
from preprocess_Station_hybridPressure import preprocess_Station_main
from datetime import datetime

# set global attributes
dir_plots = '../../Plots/Stations/'

dict_diff = {'t2m': {'ylabel': '$\Delta$ 2 m temperature (°C)',
                     'ylimit': (-8, 8), },
             't': {'ylabel': '$\Delta$ temperature (°C)',
                   'ylimit': (-8, 8)},
             'ff': {'ylabel': '$\Delta$ 10 m wind speed (m s$^{-1}$)',
                    'ylimit': (-5, 5)},
             'dd': {'ylabel': '$\Delta$ 10 m wind direction (°)',
                    'ylimit': (-10, 190)},
             'pmsl': {'ylabel': '$\Delta$ presurre deviation (hPa)',
                      'ylimit': (-3, 3)}}


def get_ModelData(station_meta, var_list, run, method='linear'):
    '''
    gets model data on height levels, interpolated to needed station location

    Parameters
    ----------
    station_meta : pandas series
        metadata of station.
    var_list : list
        list with needed parameters
    run : str
        modelrun.
    method : str, optional
        interpolation method. The default is 'linear' which is a bilinear interpolation.

    Raises
    ------
    KeyError
        When one of parameters in var_list not contained in dataset.

    Returns
    -------
    ds : xarray Dataset
        Dataset with needed parameters interpolated to station location.

    '''

    # ---- 1. open heightAboveGround dataset
    dir_path = get_Dataset_path(run)
    filename = get_Dataset_name(run, 'heightAboveGround')
    path_file = glob2.glob(join(dir_path, filename))
    ds = xr.open_mfdataset(path_file, chunks={'valid_time': 1})

    # ---- 2. check for pressure data if needed
    if 'sp' in var_list:
        filename = get_Dataset_name(run, 'surface', var='sp')
        path_file = glob2.glob(join(dir_path, filename))
        tmp = xr.open_mfdataset(path_file, chunks={'valid_time': 1})
        ds['sp'] = tmp.sp / 100  # convert to hPa

    # ---- 3. select only needed variables
    try:
        ds = ds[var_list]
    except KeyError:
        raise KeyError(f'One of needed parameters {var_list} is missing in Dataset. Check Dataset!')

    # ---- 4. interpolate data to needed location
    ds = ds.interp(longitude=station_meta.lon, latitude=station_meta.lat, method=method)
    ds.load()

    # ---- 5. check parameters
    # check temperature data
    if 't2m' in var_list:
        ds['t2m'] = ds['t2m'] - 273.15  # convert to degree celsius

    # check if wind data available
    if all([x in var_list for x in ['u10', 'v10']]):
        # calculate wind speed
        ds['ff'] = _calc_ff(ds['u10'], ds['v10'])

        # calculate wind direction
        ds['dd'] = (['valid_time'], _calc_dd(ds['u10'], ds['v10']))

    # calulate delta Z model vs. station
    deltaZ = _get_deltaZ_modelTerrain(run, station_meta, method=method)

    # calculate mean sea level pressure
    if 'sp' in var_list:
        ds['pmsl'] = _calc_mslPressure(ds['sp'], ds['t2m'] + 273.15, station_meta.alt + deltaZ)

    # ---- 6. add attributes
    ds.attrs['interpolation'] = method
    ds.attrs['deltaZ'] = deltaZ

    return ds


def get_ModelData_lowestModelLevel(station_meta, var_list, run):
    '''
    gets model data on lowest model level, interpolated to needed station location

    Parameters
    ----------
    station_meta : pandas series
        metadata of station.
    var_list : list
        list with needed parameters
    run : str
        modelrun.

    Raises
    ------
    KeyError
        When one of parameters in var_list not contained in dataset.

    Returns
    -------
    ds : xarray Dataset
        Dataset with needed parameters interpolated to station location.

    '''

    # ---- 1. open interpolated hybrid pressure dataset
    dir_path = get_Dataset_path(run)
    coords = (station_meta.lon, station_meta.lat)
    filename = get_interpStation_name(run, coords)
    path_file = join(dir_path, filename)

    # ---- 2. check for file
    if not exists(path_file):
        ds = preprocess_Station_main(var_list=['u', 'v', 't', 'z', 'pres', 'q', 'tke'],
                                     run=run, coords=coords, save=True)
    else:
        ds = xr.open_dataset(path_file)

    # select lowest model level
    ds = ds.sel(level=90)

    # ---- 3. select needed variables
    try:
        ds = ds[var_list]
    except KeyError:
        raise KeyError(f'One of needed parameters {var_list} is missing in Dataset. Check Dataset!')

    # ---- 4. check parameters
    # check temperature data
    if 't' in var_list:
        ds['t'] = ds['t'] - 273.15  # convert to degree celsius

    # check if wind data available
    if all([x in var_list for x in ['u', 'v']]):
        # calculate wind speed
        ds['ff'] = _calc_ff(ds['u'], ds['v'])

        # calculate wind direction
        ds['dd'] = (['valid_time'], _calc_dd(ds['u'], ds['v']))

    if 'z' in var_list:
        ds['Z'] = _calc_geopotHeight(ds['z'])

        # calulate delta Z model vs. station
        deltaZ = ds['Z'].isel(valid_time=0).values - station_meta['alt']
    else:
        deltaZ = None

    # ---- 5. add attributes
    ds.attrs['interpolation'] = ds.attrs['interp_method']
    ds.attrs['deltaZ'] = deltaZ

    # ---- 6. add longitude and latitude as data var
    ds['longitude'] = coords[0]
    ds['latitude'] = coords[1]

    return ds


def get_StationOBS_Dataset(station_meta,
                           start_time='2019-09-12 11:51', end_time='2019-09-14 03:00',
                           resample=True, res_time='10min'):
    '''
    function to load station observations and average it over resample time.

    Parameters
    ----------
    station_meta : pandas series
        metadata of station
    start_time : str
        first timestep of observations.
    end_time : str
        last timestep of observations.
    resample : bool, optional
        activate/deactivate resampling. The default is True.
    res_time : str, optional
        resampling time if resample==True. The default is '10min'.

    Returns
    -------
    df_resampled : pandas DataFrame
        Dataframe containing the observations of the station.

    '''

    # ---- 1. load station data
    df = TS_read_stations.main(station=station_meta.prf, start_time=start_time, end_time=end_time)

    # ---- 2. get provider information
    provider = station_meta.provider

    # ---- 3. check for needed columns and rename them to be concise
    if provider == 'ACINN':
        columns_corr = {'t_C': 't2m',
                        'p_hPa': 'pres',
                        'ff_ms': 'ff',
                        'dd_deg': 'dd'}
    elif provider == 'ZAMG':
        columns_corr = {'TL': 't2m',
                        'P': 'pres',
                        'FF': 'ff',
                        'DD': 'dd',
                        }
    elif provider == 'DWD':
        columns_corr = {'TT_10': 't2m',
                        'PP_10': 'pres',
                        'FF_10': 'ff',
                        'DD_10': 'dd',
                        }
    elif provider == 'ST':
        columns_corr = {'LT': 't2m',
                        'LD.RED': 'pres',
                        'WG': 'ff',
                        'WR': 'dd',
                        }

    # ---- 4. rename columns and select them from dataset
    df = df.rename(columns=columns_corr)[columns_corr.values()]

    # calculate mean sea level pressure
    df['pmsl'] = _calc_mslPressure(df['pres'], df['t2m'] + 273.15, station_meta.alt)

    # ---- 5. check for resampling
    if resample:
        # resample it to xx minute values
        df_resampled = df.resample(res_time, closed='right', label='right').mean()
    else:
        df_resampled = df.copy()

    # rename index column
    df_resampled.index.names = ['valid_time']

    return df_resampled


def _get_deltaZ_modelTerrain(run, station_meta, method='linear'):
    '''
    function to calculate difference between model and real topography for station location
    '''
    # set correct paths to file
    dir_path = get_Dataset_path(run)
    name_ds = get_Dataset_name(run, 'surface', var='z')
    path_file = glob2.glob(join(dir_path, name_ds))

    # open file
    with xr.open_mfdataset(path_file, chunks={'valid_time': 1}) as ds:
        ds = ds.isel(valid_time=0)
        ds['Z'] = _calc_geopotHeight(ds['z'])

    # get model data for needed coordinates
    ds = ds[['Z']].interp(longitude=station_meta.lon, latitude=station_meta.lat, method=method).load()

    # calculate difference between model and real topography
    deltaZ = ds['Z'] - station_meta['alt']

    return float(deltaZ.values)


def _plot_var(fig, ax, ds_model, df_obs, var, ds_model_lowestML=None):
    '''
    function to create timeseries plot of variable (model vs. observations)
    '''

    # define some plotting options
    dict_var = {'t2m': {'ylabel': '2 m temperature (°C)',
                        'ylimit': (0, 30)},
                'ff': {'ylabel': '10 m wind speed (m s$^{-1}$)',
                       'ylimit': (0, 10)},
                'dd': {'ylabel': '10 m wind direction (°)',
                       'ylimit': (0, 360)},
                't': {'ylabel': 'temperature (°C)',
                      'ylimit': (0, 30)}}

    # define color lines
    color_run = {'OP500': 'tab:blue',
                 'OP1000': 'tab:orange',
                 'OP2500': 'tab:green'}

    # check for observational parameter naming
    if var == 't':
        var_obs = 't2m'
    else:
        var_obs = var

    # check if ds_model is list
    if isinstance(ds_model, dict):
        # iterate over different model runs
        keys = list(ds_model.keys())
        for k in keys:
            model = ds_model[k]
            run = model.attrs['modelrun']
            color = color_run[run]

            # check also for lowest ML data
            model_lowestML = None
            if ds_model_lowestML:
                model_lowestML = ds_model_lowestML[k]

            # plot model data
            if var != 'dd':
                model[var].plot(label=run, lw=1.5, color=color, ax=ax)
                if model_lowestML and ('t' in var):
                    model_lowestML['t'].plot(label='', lw=1.5, ls='--', color=color, ax=ax)
            else:
                ax.scatter(model['valid_time'], model['dd'], label=run, marker='.', color=color)
    else:
        run = ds_model.attrs['modelrun']
        model = ds_model
        color = color_run[run]
        if var != 'dd':
            model[var].plot(label=run, lw=1.5, color=color, ax=ax)
        else:
            ax.scatter(model['valid_time'], model['dd'], label=run, marker='.', color=color)

    # plot station data
    if var != 'dd':
        df_obs[var_obs].plot(label='Obs', color='k', ax=ax)
    else:
        ax.scatter(df_obs.index, df_obs['dd'], color='k', marker='.', label='Obs')

    # set y-axis label
    ax.set_ylabel(dict_var[var]['ylabel'])
    ax.set_ylim(dict_var[var]['ylimit'])

    # set y-axis attributes
    if var == 'dd':
        ax.set_yticks(np.arange(0, 361, 45))
        ax.set_yticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])

    return fig, ax


def _plot_var_deviation(ds_model_dev, df_obs_dev, var, stations):
    '''
    function to create deviation timeseries plot between two stations (station1 - station2)
    '''

    # make plot
    fig, ax = plt.subplots(figsize=(15, 5))

    # define some names
    if var == 'pmsl':
        ylabel = 'msl pressure deviation (hPa)'
        ylimit = (-5, 5)

    # check if ds_model is list
    if isinstance(ds_model_dev, dict):
        # iterate over different model runs
        keys = list(ds_model_dev.keys())
        for k in keys:
            model = ds_model_dev[k]
            run = ds_model_dev[k].attrs['modelrun']

            # plot model data
            model[var].plot(label=run, lw=1.5, ax=ax)
    else:
        model = ds_model_dev
        run = ds_model_dev.attrs['modelrun']

        # plot model data
        model[var].plot(label=run, lw=1.5, ax=ax)

    # plot station data
    df_obs_dev[var].plot(label='Obs', color='k', ax=ax)

    # set y-axis label
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylimit)

    # add legend
    ax.legend()

    # set labels, limits etc.
    title = f'{stations[0]} - {stations[1]}'
    start_x = df_obs_dev.index[0]
    end_x = df_obs_dev.index[-1]
    ax_kwargs = {'ylim': ylimit,
                 'xlim': (start_x, end_x),
                 'ylabel': ylabel,
                 'xlabel': 'UTC',
                 'title': title}
    ax.set(**ax_kwargs)

    # set/change format of xaxis ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator(),
                                                             formats=['%Y', '%b', '%H:%M\n%d-%b',
                                                                      '%H:%M', '%H:%M', '%S.%f'],
                                                             show_offset=False))
    ax.figure.autofmt_xdate(rotation=0, ha='center')

    # add grid
    ax.grid(which='major')

    # add horizontal line at zero
    ax.hlines(0, start_x, end_x,
              colors='k', ls='--', alpha=0.5)

    # add area of IOP8
    ax.axvspan(datetime(2019, 9, 13), datetime(2019, 9, 14), color='tab:red', alpha=0.1, zorder=0)
    ax.text(datetime(2019, 9, 13, 0, 10), ax.get_ylim()[1] - 0.5, 'IOP8', color='tab:red',
            fontweight='bold', fontsize=12, fontstyle='italic', va='top')

    # make a nice layout
    fig.tight_layout()

    return fig, ax


def _plot_differenceToObs(fig, ax, ds_model, df_obs, var):
    '''
    function for timeseries plot of difference model to observations
    '''
    ylabel = dict_diff[var]['ylabel']
    ylimit = dict_diff[var]['ylimit']

    # check for observational parameter naming
    if var == 't':
        var_obs = 't2m'
    else:
        var_obs = var

    # check if ds_model is list
    if isinstance(ds_model, dict):
        # iterate over different model runs
        keys = list(ds_model.keys())
        for k in keys:
            model = ds_model[k]
            run = model.attrs['modelrun']

            # calculate difference model - obs
            ds_diff = model[var] - df_obs[var_obs]

            # select only IOP8 for statistics
            ds_diff_IOP8 = ds_diff.sel(valid_time=slice('2019-09-13T00:00:00', '2019-09-14T00:00:00'))

            # plot difference
            if var != 'dd':

                # add some statistics
                me = np.mean(ds_diff_IOP8)  # mean error
                rmse = np.sqrt(np.mean(ds_diff_IOP8**2))  # root mean square error

                label = run + f' - ME: {np.round(me.values, 2)}, RMSE: {np.round(rmse.values, 2)}'
                ds_diff.plot(label=label, lw=1.5, ax=ax)

            else:
                # need some adjustements for wind direction (maximum difference 180 degree)
                ds_diff.data = np.min([np.abs(ds_diff), np.abs(
                    ds_diff + 360), np.abs(ds_diff - 360)], axis=0)
                ds_diff_IOP8 = ds_diff.sel(valid_time=slice('2019-09-13T00:00:00', '2019-09-14T00:00:00'))

                # add some statistics
                me = np.mean(ds_diff_IOP8)  # mean error
                rmse = np.sqrt(np.mean(ds_diff_IOP8**2))  # root mean square error
                label = run + f' - ME: {np.round(me.values, 2)}, RMSE: {np.round(rmse.values, 2)}'

                # plot model data
                ax.scatter(ds_diff['valid_time'], ds_diff, label=label, marker='.')

    else:
        run = ds_model.attrs['modelrun']
        model = ds_model

        # calculate difference model - obs
        ds_diff = model[var] - df_obs[var_obs]

        # plot difference
        ds_diff.plot(label=run, lw=1.5, ax=ax)

    # plot horizontal line at 0
    ax.hlines(0, ds_diff.valid_time[0], ds_diff.valid_time[-1], colors='k', ls='--', alpha=0.5)

    # add sector lines
    if var == 'dd':
        ax.hlines([0, 45, 90, 135, 180], ds_diff.valid_time[0], ds_diff.valid_time[-1],
                  colors='k', ls='--', alpha=0.5)
        ax.set_yticks([0, 45, 90, 135, 180])

    # set y-axis label
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylimit)

    return fig, ax


def _save_plot(dir_plots, name_plot):
    '''
    function to save plot
    '''
    path_save = join(dir_plots, name_plot)
    print(f'Save Figure to: {path_save}')
    plt.savefig(path_save, bbox_inches='tight')
    plt.close()


def plot_StationData(ds_model, df_obs, station_meta, ds_model_lowestML=None, diff=False,
                     var='ff', run='', save=True, dir_plots=None, ax_kwargs=None):
    '''
    function which handles different plotting options and creates plot

    Parameters
    ----------
    ds_model : xarray Dataset
        dataset model for location of station.
    df_obs : pandas Dataframe
        dataframe observations station.
    station_meta : pandas Series
        station meta data.
    ds_model_lowestML : xarray Dataset, optional
        dataset containing information on lowest model level. The default is None.
    diff: bool, optional
        make difference to observations plot. The default is False.
    var : str or list of str, optional
        variable to plot. The default is 'ff'.
    run : str or list of str, optional
        model run names for saving. The default is ''.
    save : bool, optional
        activate/deactivate saving of plot. The default is True.
    dir_plots : str, optional
        path to directory of plots. The default is None.
    ax_kwargs : dict, optional
        dictionary for axis plotting options. The default is None.

    Returns
    -------
    fig, ax

    '''

    # ---- 1. create figure
    fig, ax = plt.subplots(figsize=(15, 5))

    # ---- 2. call plotting routine
    if not diff:
        if var == ['ff', 'dd']:
            # combined plot
            fig, ax = _plot_var(fig, ax, ds_model, df_obs, var[0])
            ax2 = ax.twinx()
            fig, ax2 = _plot_var(fig, ax2, ds_model, df_obs, var[1])
        else:
            fig, ax = _plot_var(fig, ax, ds_model, df_obs, var, ds_model_lowestML=ds_model_lowestML)
    else:
        fig, ax = _plot_differenceToObs(fig, ax, ds_model, df_obs, var)

    # --- 3. add legend, labels, limits etc.
    ax.legend()

    # get model data for labeling
    str_dZ = ''
    if isinstance(ds_model, dict):
        model = ds_model[run[0]]
        for key in list(ds_model.keys()):
            str_dZ += f'$\Delta h_{{{key}}}$ = {ds_model[key].attrs["deltaZ"]:.1f} m; '
    else:
        model = ds_model
        str_dZ += f'$\Delta h_{{{model.attrs["modelrun"]}}}$ = {model.attrs["deltaZ"]:.1f} m; '

    # check for ax kwargs
    if ax_kwargs is None:
        title = f'{station_meta["name"]} - @ {model.latitude.values:.3f} °N, ' \
                f'{model.longitude.values:.3f} °E; @ {station_meta.alt} m asl; \n' \
                f'{str_dZ}'

        start_x = model.valid_time[0].values
        end_x = model.valid_time[-1].values
        ax_kwargs = {'xlim': (start_x, end_x),
                     'xlabel': 'UTC',
                     'title': title}

    # set axis properties
    if ax_kwargs:
        ax.set(**ax_kwargs)

    # set/change format of xaxis ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator(),
                                                             formats=['%Y', '%b', '%H:%M\n%d-%b',
                                                                      '%H:%M', '%H:%M', '%S.%f'],
                                                             show_offset=False))
    ax.figure.autofmt_xdate(rotation=0, ha='center')

    # add grid
    try:
        ax2.grid(which='major')  # if available more important
    except:
        ax.grid(which='major')

    # add area of IOP8
    ax.axvspan(datetime(2019, 9, 13), datetime(2019, 9, 14), color='tab:red', alpha=0.1, zorder=0)
    ax.text(datetime(2019, 9, 13, 0, 10), ax.get_ylim()[1] * 0.98, 'IOP8', color='tab:red',
            fontweight='bold', fontsize=12, fontstyle='italic', va='top')

    # make a nice layout
    fig.tight_layout()

    # ---- 4. save figure
    if save and dir_plots:
        if not diff:
            name_plot = f'{station_meta.prf}_OBS_{run}_{var}.svg'  # name of plot
        else:
            name_plot = f'{station_meta.prf}_OBS_{run}_{var}_diff.svg'  # name of plot
        _save_plot(dir_plots, name_plot)

    return fig, ax


def main_StationAnalysis(runs, method='linear', with_lowest_ML=True):
    '''
    main entry point station analysis/plots

    Parameters
    ----------
    runs : list
        list containing modelruns to plot.
    method : str, optional
        interpolation method. The default is 'linear'.
    with_lowest_ML : bool, optional
        add timeseries on lowest model level. The default is True.

    Returns
    -------
    None.

    '''
    # ---- 1.  model parameters ---
    var_list = ['t2m', 'u10', 'v10']

    # ---- 2.  station paramters ---
    stations = list(get_StationMetaProvider(dir_metadata, provider=['ACINN'])['prf'])
    stations = stations + ['LAN', 'IMS', 'HAI', 'LOWI', 'JEN', 'KIE', 'KUF']  # add further station innvalley

    # ---- 3. iterate over stations
    for station in stations:
        station_meta = get_StationMetaData(dir_metadata, station)
        df_obs = get_StationOBS_Dataset(station_meta)  # load observations station

        # ---- 4. load model data ---
        ds_model = {}
        ds_model_lowestML = {}
        for i, run in enumerate(runs):
            print(station, i, run)
            ds_model[run] = get_ModelData(station_meta, var_list, run,
                                          method=method)  # add model to data container
            ds_model_lowestML[run] = get_ModelData_lowestModelLevel(station_meta, ['z', 't', 'u', 'v'], run)

        # ---- 5. make plots ---
        # standard plots
        for var in ['t2m', ['ff', 'dd']]:
            fig, ax = plot_StationData(ds_model, df_obs, station_meta,
                                       var=var, run=runs, save=True, dir_plots=dir_plots)
            if var == 't2m' and with_lowest_ML:
                fig, ax = plot_StationData(ds_model, df_obs, station_meta, ds_model_lowestML=ds_model_lowestML,
                                           var=var, run=runs, save=False, dir_plots=dir_plots)
                name_plot = f'{station_meta.prf}_OBS_{runs}_{var}_withlowestML.svg'  # name of plot
                _save_plot(dir_plots, name_plot)

        # difference to obs plots
        for var in ['t2m', 'ff', 'dd']:
            fig, ax = plot_StationData(ds_model, df_obs, station_meta, diff=True,
                                       var=var, run=runs, save=True, dir_plots=dir_plots)

        # --- free up memory space ---
        del ds_model, df_obs, fig, ax
        gc.collect()


def main_StationAnalysis_lowestModelLevel(runs, method='linear'):
    '''
    main entry point station analysis on lowest model level

    Parameters
    ----------
    runs : list
        list containing modelruns to plot.
    method : str, optional
        interpolation method. The default is 'linear'.

    Returns
    -------
    None.

    '''
    # ---- 1. model parameters ---
    var_list = ['z', 't', 'u', 'v']

    # ---- 2. station paramters ---
    # @KOLS: t in 2m
    # @UNI: t in 2m
    # @HOCH: t in 7m
    # @EGG: t in 6m
    # @TERF: t in 11 m
    # @WEER: t in 5 m
    stations = ['KOLS', 'UNI', 'HOCH', 'EGG', 'TERF', 'WEER']

    # ---- 3. iterate over stations
    for station in stations:
        station_meta = get_StationMetaData(dir_metadata, station)
        df_obs = get_StationOBS_Dataset(station_meta)  # load observations station

        # ---- 4. load model data ---
        ds_model = {}
        for i, run in enumerate(runs):
            print(station, i, run)
            ds_model[run] = get_ModelData_lowestModelLevel(station_meta, var_list, run)

        # ---- 5. make plots ---
        for var in ['t', ['ff', 'dd']]:
            fig, ax = plot_StationData(ds_model, df_obs, station_meta,
                                       var=var, run=runs, save=False, dir_plots=dir_plots)
            name_plot = f'{station_meta.prf}_OBS_{runs}_{var}_lowestML.svg'  # name of plot
            _save_plot(dir_plots, name_plot)

        # difference to observations plots
        for var in ['t', 'ff', 'dd']:
            fig, ax = plot_StationData(ds_model, df_obs, station_meta, diff=True,
                                       var=var, run=runs, save=False, dir_plots=dir_plots)
            name_plot = f'{station_meta.prf}_OBS_{runs}_{var}_diff_lowestML.svg'  # name of plot
            _save_plot(dir_plots, name_plot)

        # --- free up memory space ---
        del ds_model, df_obs, fig, ax
        gc.collect()


def main_StationAnalysis_deviations(runs, var, stations=['KOLS', 'MUC'], method='linear', save=True):
    '''
    main entry point station analysis deviation between two stations

    Parameters
    ----------
    runs : list
        list containing modelruns to plot.
    var : str
        parameter to plot.
    stations : str, optional
        stations to compare. The default is ['KOLS', 'MUC'].
    method : str, optional
        interpolation method. The default is 'linear'.
    save : bool, optional
        save plots. The default is True.

    Returns
    -------
    None.

    '''
    # ---- 1. model parameters ---
    var_list = ['t2m', 'u10', 'v10', 'sp']

    # ---- 2. iterate over stations ---
    df_obs = {}
    ds_model = {}
    for station in stations:
        station_meta = get_StationMetaData(dir_metadata, station)
        df_obs[station] = get_StationOBS_Dataset(station_meta)  # load observations station

        # ---- 3. load model data ---
        tmp = {}
        for i, run in enumerate(runs):
            print(station, i, run)
            tmp[run] = get_ModelData(station_meta, var_list, run,
                                     method=method)  # add model to data container
        # add to data container
        ds_model[station] = tmp

    # ---- 4. calculate deviation between stations ---
    df_obs_dev = df_obs[stations[0]] - df_obs[stations[1]]

    ds_model_dev = {}
    for run in runs:
        ds_model_dev[run] = ds_model[stations[0]][run] - ds_model[stations[1]][run]
        ds_model_dev[run].attrs['modelrun'] = run

    # ---- 5. do deviation plot ---
    fig, ax = _plot_var_deviation(ds_model_dev, df_obs_dev, 'pmsl', stations)

    # ---- 6. save plot ---
    if save:
        name_plot = f'dev_{stations[0]}_{stations[1]}_{runs}_{var}.svg'
        _save_plot(dir_plots, name_plot)


# %% ---- main ----

# OP500 vs OP1000 vs OP2500
runs = ['OP500', 'OP1000', 'OP2500']
main_StationAnalysis(runs)
# main_StationAnalysis_lowestModelLevel(runs)  # not needed anymore
main_StationAnalysis_deviations(runs, 'pmsl', stations=['KOLS', 'KUF'])
