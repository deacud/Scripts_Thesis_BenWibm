#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:06:24 2023

@author: Benedikt Wibmer

Script for creating Hight-Time plots
"""

from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from matplotlib import colormaps as cm
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.offsetbox import AnchoredText
from scipy.interpolate import interp1d
import read_RS
import read_lidar_vert
import read_MWR
import calc_gradients_WB
from preprocess_Station_hybridPressure import preprocess_Station_main
from calculations import (_calc_ff, _calc_geopotHeight, _calc_potTemperature, _calc_dd, _calc_w_from_omega,
                          _calc_knots_from_ms)
from path_handling import (get_Dataset_path, get_interpStation_name,
                           get_ObsDataset_path, dir_metadata, get_Dataset_name)
import glob2
from custom_colormaps import cmap_user
from customized_barbs import barbs
from read_StationMeta import get_StationMetaData

# set global parameters
dir_plots = '../../Plots/HT/'  # path of plots


def _check_interpStationDataset(dir_path, run, station_meta, var_list):
    '''
    function to check for dataset containing the interpolated data for the station location
    '''

    # define coordinates and paths
    coords = (station_meta.lon, station_meta.lat)
    filename = get_interpStation_name(run, coords)
    path_file = join(dir_path, filename)

    # check for file
    if not exists(path_file):
        ds = preprocess_Station_main(var_list=var_list, run=run, coords=coords, save=True)
    else:
        ds = xr.load_dataset(path_file)

    # check for needed variables
    error = []
    for var in var_list:
        if var in ['w']:  # w not defined on hybrid pressure levels (add it afterwards)
            continue
        try:
            ds[var]
        except KeyError:
            error.append(var)
            continue
    if error:
        raise KeyError(
            f'Missing parameters {error} in interpolated Station Dataset. Check Datasets for them!')

    return ds


def _check_ObsDataset(dir_path, run):
    '''
    function to check for observational data

    Parameters
    ----------
    dir_path : str
        path to directory of observations.
    run : str
        abbreviation for observation instrument. (e.g. RS, SL88, MWR)

    Raises
    ------
    NotImplementedError
        if accessed observation not implemented.

    Returns
    -------
    ds : xarray Dataset
        Dataset containing the observations.

    '''

    dict_ds_save = {'RS': 'ds_RS_IOP8.nc',
                    'SL88': 'ds_lidar_IOP8.nc',
                    'MWR': 'ds_MWR_IOP8.nc'}

    path_file = join(dir_path, dict_ds_save.get(run))

    if not exists(path_file):
        if run == 'RS':
            # create radiosonde dataset
            ds = read_RS.main(dir_path)
        elif run == 'SL88':
            # create lidar datset
            ds = read_lidar_vert.main(dir_path)
        elif run == 'MWR':
            # create MWR dataset
            ds = read_MWR.main(dir_path)
        else:
            raise NotImplementedError(f'run=="{run}"')
        # save dataset
        _ = ds.to_netcdf(path_file)
    else:
        ds = xr.load_dataset(path_file)

    return ds


def _get_pLevel_Data(dir_path, run, station_meta, var_list=['z', 'w']):
    '''
    function to extract needed data, not defined on hybrid levels, from pressure levels (e.g. w)
    and interpolate it to station location

    '''
    # define coordinates
    coords = (station_meta.lon, station_meta.lat)
    ds = []
    for var in var_list:
        # define paths
        name_ds = get_Dataset_name(run, 'isobaricInhPa', var=var)
        path_file = glob2.glob(join(dir_path, name_ds))

        # load data
        ds.append(xr.open_mfdataset(path_file))

    # merge to combined dataset
    ds = xr.merge(ds)

    # add geopotential height
    ds['Z'] = _calc_geopotHeight(ds['z'])

    # interpolate data to station
    ds_station = ds.interp(longitude=coords[0], latitude=coords[1], method='linear')

    return ds_station.load()


def get_Data(dir_path, run, station_meta=None, var_list=None):
    '''
    get data for station (model or observation)

    Parameters
    ----------
    dir_path : str
        path to directory of netcdf datasets.
    run : str
        abbreviation modelrun or observation.
    station_meta : pandas Series, optional
        dataset containing station meta data.
    var_list : list, optional
        list with parameters needed.

    Returns
    -------
    ds : xarray Dataset
        station Dataset interpolated to regular heights
    ds_init : xarray Dataset
        initial station Dataset

    '''

    # Model
    if run not in ['RS', 'SL88', 'MWR']:
        # get station dataset
        ds_init = _check_interpStationDataset(dir_path, run, station_meta, var_list)

        # add further needed variables
        ds_init['theta'] = _calc_potTemperature(ds_init['t'], ds_init['pres'])  # potential temperature
        ds_init['Z'] = _calc_geopotHeight(ds_init['z'])  # geopotentail height

        # interpolate to regular heights
        ds = interpToRegularGrid(ds_init)

        # add windspeed information
        ds['ff'] = _calc_ff(ds['u'], ds['v'])
        wdir = _calc_dd(ds['u'], ds['v'])
        ds['dd'] = (['height', 'time'], wdir)

        # get vertical wind speed data if needed (only on p-level available)
        if 'w' in var_list:
            # get data on p-levels
            tmp_plevels = _get_pLevel_Data(dir_path, run, station_meta)
            # interpolate from p-levels to height levels
            tmp_plevels = interp_pLevels_ToRegularGrid(tmp_plevels, new_heights=ds.height.values)
            # convert from omega to w
            ds['w'] = _calc_w_from_omega(tmp_plevels.w, ds['pres'], ds['t'], q=ds['q'])

        # add station meta
        ds.attrs['station'] = station_meta.prf

    # Observations
    else:
        ds = _check_ObsDataset(dir_path, run)
        ds_init = None

    return ds, ds_init


def interpToRegularGrid(ds_profile, keep_AROME_levels=True, dh=10, method='linear'):
    '''
    Process profile datset and interpolate over mean heights (out of timesteps)

    Parameters
    ----------
    ds_profile : xarray dataset
        dataset of profile with dimensions valid_time, level and it's data.
    keep_AROME_levels : bool, optional
        keep model levels of AROME or define user specified spacing between min, max with dh.
        The default is True.
    dh : int/float, optional
        spacing between model levels. Only available when keep_AROME_levels == False.
        The default is None.
    method : str, optional
        interpolation method for scipy.interp1d (e.g. linear, nearest).
        The default is 'linear'.

    Raises
    ------
    ValueError
        dh not defined if keep_AROME_levels == False.

    Returns
    -------
    ds : xarray dataset
        dataset with dimensions time, height and variables

    '''

    print(f'Create {ds_profile.modelrun} dataset interpolated to regular heights ...')
    var_list = list(ds_profile.keys())

    # get height and dates out profile
    height = ds_profile['Z'].values
    dates = ds_profile.valid_time.values

    # get min and max height of grid (avoid extrapolation)
    min_height = np.max(height.T[-1])  # min height I can have to interpolate
    max_height = np.min(height.T[0])  # max height to interpolate

    # check if we should keep the AROME model levels or define user specified spacing
    if keep_AROME_levels is True:
        new_heights = height.mean(axis=0)  # get mean height on each level
        new_heights[-1] = min_height  # set lower boundary condition
        new_heights[0] = max_height  # set upper boundary condtion
    else:
        if dh is not None:
            new_heights = np.arange(max_height, min_height, -dh)
        else:
            raise ValueError('dh need to be specified if keep_AROME_levels == False')

    # create Dataset defined with needed dimensions
    ds = xr.Dataset(coords=dict(height=new_heights, time=dates,
                                datenum=(['time'], mdates.date2num(dates)),),
                    attrs=dict(Name='AROME output parameters interpolated to regular heights',
                               longitude=ds_profile.longitude.values,
                               latitude=ds_profile.latitude.values,
                               modelrun=ds_profile.modelrun,
                               DX=ds_profile.DX,
                               history=ds_profile.name),)

    # get needed variables
    for var in var_list:
        if var in ['longitude', 'latitude', 'Z']:
            continue
        try:
            var_df = ds_profile[var]
        except:
            print(f"{var} not found")
            continue

        # interpolate data to defined height levels
        df_temp = pd.DataFrame()
        for i, time in enumerate(dates):
            print(f'start interpolation {var} timestep: {i}')
            interp_funct = interp1d(height[i], var_df[i], kind=method)
            df_temp[time] = interp_funct(new_heights)

        ds[f"{var}"] = (["height", "time"], df_temp)

    return ds


def interp_pLevels_ToRegularGrid(ds_profile, new_heights=None, dh=10, method='linear'):
    '''
    Process profile dataset on pressure levels and interpolate to new heights

    Parameters
    ----------
    ds_profile : xarray dataset
        dataset of profile with dimensions valid_time, level and it's data.
    new_heights : array, optional
        heights to which dataset is interpolated.
    dh : int/float, optional
        spacing between model levels. Only available when new_heights == None.
        The default is 10.
    method : str, optional
        interpolation method for scipy.interp1d (e.g. linear, nearest).
        The default is 'linear'.

    Raises
    ------
    ValueError
        dh not defined if new_heights == None.

    Returns
    -------
    ds : xarray dataset
        dataset with dimensions time, height and variables

    '''

    print(f'Create {ds_profile.modelrun} dataset interpolated to regular heights ...')
    var_list = list(ds_profile.keys())

    # get height and dates out profile
    height = ds_profile['Z'].values
    dates = ds_profile.valid_time.values

    # check for new heights
    if new_heights is None:
        # get min and max height of grid (avoid extrapolation)
        min_height = np.max(height.T[-1])  # min height I can have to interpolate
        max_height = np.min(height.T[0])  # max height to interpolate

        if dh is not None:
            new_heights = np.arange(max_height, min_height, -dh)
        else:
            raise ValueError('dh need to be specified if keep_AROME_levels == False')

    # create Dataset defined with needed dimensions
    ds = xr.Dataset(coords=dict(height=new_heights, time=dates,
                                datenum=(['time'], mdates.date2num(dates)),),
                    attrs=dict(Name='AROME output parameters interpolated to regular heights from p-levels',
                               longitude=ds_profile.longitude.values,
                               latitude=ds_profile.latitude.values,
                               modelrun=ds_profile.modelrun,
                               DX=ds_profile.DX,
                               history=ds_profile.name),)

    # get needed variables
    for var in var_list:
        if var in ['longitude', 'latitude', 'Z']:
            continue
        try:
            var_df = ds_profile[var]
        except:
            print(f"{var} not found")
            continue

        # interpolate data to defined height levels
        df_temp = pd.DataFrame()
        for i, time in enumerate(dates):
            print(f'start interpolation {var} timestep: {i}')
            interp_funct = interp1d(height[i], var_df[i], kind=method)
            df_temp[time] = interp_funct(new_heights)

        ds[f"{var}"] = (["height", "time"], df_temp)

    return ds


def customize_xtick_labels(ax):
    '''
    function to adapt xtick labels for personal needs
    '''
    # define date format first which we can adapt afterwards
    date_form = DateFormatter('%H:%M\n%d-%b')
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # define major ticks every third hour
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # define minor ticks every hour

    # adapt x-labels to personal needs
    new_labels = []
    for i, label in enumerate(ax.get_xticklabels()):
        lab = label.get_text()
        if not '00:00' in lab:  # keep only Hour and Minutes if not 00 UTC
            tmp_new = lab[0:5]
        else:                   # keep only date if 00 UTC
            tmp_new = lab
        new_labels.append(tmp_new)

    # update labels
    ax.set_xticklabels(new_labels, rotation=0, ha='center')

    return ax


def add_barbs_legend(fig):
    '''
    Add barb legend to the plot
    '''
    legend_axes = fig.add_axes([0.65, 0.935, 0.18, 0.03])
    legend_axes.patch.set_visible(False)
    legend_axes.axis('off')
    legend_axes.axes.get_xaxis().set_visible(False)
    legend_axes.axes.get_yaxis().set_visible(False)
    speeds_u = [0, -2, -5, -10, -25, -50]
    speeds_v = [0, 0, 0, 0, 0, 0]
    loc_x = [0.12, 0.22, 0.32, 0.42, 0.52, 0.62]
    loc_y = [0, 0, 0, 0, 0, 0]
    barbs(legend_axes, loc_x, loc_y, speeds_u, speeds_v,
          pivot='tip', fill_empty=True, length=4.5, lw=.5, rounding=True,
          sizes=dict(emptybarb=0.025, spacing=0.2), zorder=3)
    legend_axes.plot(loc_x, loc_y, '.', color='black', markersize=2)
    legend_axes.annotate('Calm', xy=(loc_x[0], 1), fontsize=8, ha='center')
    legend_axes.annotate('<5', xy=(loc_x[1], 1), fontsize=8)
    legend_axes.annotate(' 5', xy=(loc_x[2], 1), fontsize=8)
    legend_axes.annotate('10', xy=(loc_x[3], 1), fontsize=8)
    legend_axes.annotate('25', xy=(loc_x[4], 1), fontsize=8)
    legend_axes.annotate('50 kn East', xy=(loc_x[5], 1), fontsize=8)

    legend_axes.set_ylim([-1.3, 2.5])
    legend_axes.set_xlim([0.08, 0.8])

    return fig


def time_height_plot_wind(ds, figsize=(15, 4), run=None, station_meta=None, ax_kwargs=None, title=None,
                          quiver=False, quiver_l=10, scale=400, barb_l=4):
    '''
    function to create time height plot of winds.

    Parameters
    ----------
    ds : xarray dataset
        dataset whit dimensions height, time and the needed variables.
    figsize : tuple, optional
        size of figure. The default is (10, 5).
    run : str, optional
        model run name for labeling. The default is None.
    station_meta : pandas Series, optional
        station meta data
    ax_kwargs : dict, optional
        defines xlim and ylim of plot. The default is None.
    title : str, optional
        set title
    quiver : bool, optional
        make quiver or barb plot. The default is False (barb plot).
    quiver_l : int, optional
        quiverkey length if used. The default is 10.
    scale : int, optional
        scale for quivers if used. The default is 400.
    barb_l : int, optional
        length of barbs if used. The default is 4.

    Returns
    -------
    fig, ax, qv

    '''
    # define plot options quiverkey and colorbar
    quiverkey_kwargs = {'X': 0.95, 'Y': 1.03,
                        'U': quiver_l,
                        'label': f'{quiver_l}' + ' m s$^{-1}$',
                        'labelpos': 'W',
                        }

    colorbar_kwargs = {'label': 'horizontal wind speed (m s$^{-1}$)',
                       'pad': 0.01,  # fraction between colorbar and plot (default: 0.05)
                       }

    # check for ax kwargs
    if ax_kwargs is None:
        start_x = np.datetime64('2019-09-12 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        ylim = (np.floor(station_meta['alt']/100)*100, 2000 if run == 'SL88' else 3500)
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': ylim}

    # additional plotting options
    dy_ticks = 250 if ax_kwargs['ylim'][1] <= 2000 else 500
    station = station_meta['prf'] if station_meta['prf'] not in ['SL88', 'RS'] else 'KOLS'

    # set target grid points for quiver/barb plot
    target_time = pd.date_range(ax_kwargs['xlim'][0], ax_kwargs['xlim'][1], freq='60min')
    target_heights = np.arange(ax_kwargs['ylim'][0], ax_kwargs['ylim'][1] + 100, 100)

    # levels colormap
    levels = np.arange(0, 17, 1)
    cmap = cm.get_cmap('magma_r')

    # --- make plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- plot wind speed colorcoded ---
    cf_ff = ds.ff.plot.contourf(x='time', y='height', cmap=cmap, levels=levels, extend='max',
                                cbar_kwargs=colorbar_kwargs, ax=ax)

    # --- plot quivers or barbs of wind ---
    # plot only nearest quivers at target grid points
    pu = ds.u.sel(time=target_time, height=target_heights, method='nearest')
    pv = ds.v.sel(time=target_time, height=target_heights, method='nearest')

    # quiver or barbs
    if quiver:
        qv = ax.quiver(pu.time, pv.height, pu, pv, pivot='middle', linewidth=0.5,
                       headlength=2, headaxislength=2, headwidth=2, width=0.0025, scale=scale)
        qk = plt.quiverkey(qv, **quiverkey_kwargs)
    else:
        # convert m/s to knots and plot barb plot
        pu_kn = _calc_knots_from_ms(pu)
        pv_kn = _calc_knots_from_ms(pv)
        # adapted barbs method in customized_barbs.py
        qv = barbs(ax, pu_kn.time, pv_kn.height, pu_kn, pv_kn, pivot='middle', rounding=True,
                   fill_empty=True, sizes=dict(emptybarb=0.025), length=barb_l, lw=.5)

    # --- plot isentropes ---
    if run not in 'SL88':
        levels_theta = np.arange(288, 320, 1)
        c_theta = ds.theta.plot.contour(levels=levels_theta, colors='k', linewidths=0.5)
        ax.clabel(c_theta, levels=levels_theta[::2], inline=True, fmt='%1.0f')
        # Adjust linewidth
        for i, line in enumerate(c_theta.collections):
            if i % 2 == 0:
                line.set_linewidth(1)  # Increase linewidth for every second line

    # set x and y limit
    if ax_kwargs:
        ax.set(**ax_kwargs)

    # set/change format of xaxis ticks
    ax = customize_xtick_labels(ax)
    ax.set_yticks(np.arange(ax_kwargs['ylim'][0], ax_kwargs['ylim'][1] + dy_ticks, dy_ticks))

    # add horizontal line @ max height of Lidar
    if run != 'SL88':
        ax.hlines(2000, *ax.get_xlim(), linestyle=':', color='k', alpha=0.5)

    # add labels
    ax.set(ylabel='height (m asl)', xlabel='UTC')
    if not title:
        title = (f'{station} @ {ds.attrs["latitude"]:.3f} °N, ' +
                 f'{ds.attrs["longitude"]:.3f} °E')
    ax.set_title(title)

    # add textbox
    at = AnchoredText(run, loc='upper left',
                      prop=dict(fontsize=10, color='white', weight='bold'),
                      frameon=True, pad=0.2)
    at.patch.set(facecolor='grey', edgecolor='k', alpha=0.9)
    ax.add_artist(at)

    # add barb legend
    if not quiver:
        add_barbs_legend(fig)

    # make nice looking layout
    ax.grid()
    ax.grid(linestyle='--', alpha=0.2, zorder=-10)
    fig.tight_layout()

    return fig, ax, qv


def time_height_plot_var(ds, var, figsize=(10, 5), quiver_l=10, add_quiver=True, add_isentropes=True,
                         run=None, station_meta=None, ax_kwargs=None, cmap=None, extend='both', scale=400):
    '''
    function to create time height plot of needed variable.

    Parameters
    ----------
    ds : xarray dataset
        dataset whit dimensions height, time and the needed variables.
    figsize : tuple, optional
        size of figure. The default is (10, 5).
    quiver_l : int, optional
        quiverkey length. The default is 5.
    run : str, optional
        model run name for labeling. The default is None.
    station_meta : pandas Series, optional
        station meta data.
    ax_kwargs : dict, optional
        defines xlim and ylim of plot. The default is None.
    cmap : str, optional
        colormap
    extend : str, optional
        extend of colormap
    scale : int, optional
        scale for quivers if used. The default is 400.

    Returns
    -------
    fig, ax, qv

    '''
    # define plot options quiverkey and colorbar
    quiverkey_kwargs = {'X': 0.95, 'Y': 1.03,
                        'U': quiver_l,
                        'label': f'{quiver_l}' + ' m s$^{-1}$',
                        'labelpos': 'W',
                        }

    # dictionary with label options depending on needed variable
    var_dict = {'t': {'label': 'temperature (°C)',
                      'levels': (0, 31, 1), },
                'ff': {'label': 'horizontal wind speed (m s$^{-1}$)',
                       'levels': (0, 17, 1), },
                'pres': {'label': 'pressure (hPa)',
                         'levels': (700, 1000, 25)},
                'w': {'label': 'vertical wind speed (m s$^{-1}$)',
                      'levels': (-1.2, 1.3, 0.1)},
                'q': {'label': 'specific humidity (g kg$^{-1}$)',
                      'levels': (0, 10.1, 1)}}

    colorbar_kwargs = {'label': var_dict.get(var).get('label'),
                       'pad': 0.01,  # fraction between colorbar and plot (default: 0.05)
                       }

    # check for ax kwargs
    if ax_kwargs is None:
        start_x = np.datetime64('2019-09-12 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 5000)}

    # set target grid points for quiver plot
    target_time = pd.date_range(ax_kwargs['xlim'][0], ax_kwargs['xlim'][1], freq='30min')
    target_heights = np.arange(ax_kwargs['ylim'][0], ax_kwargs['ylim'][1] + 100, 100)

    # levels colormap
    levels_ = var_dict.get(var).get('levels')
    levels = np.arange(levels_[0], levels_[1], levels_[2])
    if cmap:
        cmap = cm.get_cmap(cmap)
    else:
        cmap = cm.get_cmap('magma_r')

    # check for correct units of var
    ds = ds.copy()
    if var == 't':
        ds['t'] = ds['t'] - 273.15  # convert to degree Celsius
    elif var == 'pres':
        ds['pres'] = ds['pres'] / 100  # convert to hPa
    elif var == 'q':
        ds['q'] = ds['q'] * 1000  # convert to g/kg

    # --- make plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- plot parameter colorcoded ---
    cf_var = ds[var].plot.contourf(x='time', y='height', cmap=cmap, levels=levels, extend=extend,
                                   cbar_kwargs=colorbar_kwargs)

    # --- plot quivers of wind ---
    if add_quiver:
        # plot only nearest quivers at target grid points
        pu = ds.u.sel(time=target_time, height=target_heights, method='nearest')
        pv = ds.v.sel(time=target_time, height=target_heights, method='nearest')

        # pu, pv = ds.u[::n_quiver[0], ::n_quiver[1]], ds.v[::n_quiver[0], ::n_quiver[1]]  # old WB

        qv = ax.quiver(pu.time, pv.height, pu, pv, pivot='middle', linewidth=0.5,
                       headlength=2, headaxislength=2, headwidth=2, width=0.0025, scale=scale)
        qk = plt.quiverkey(qv, **quiverkey_kwargs)

    # --- plot isentropes ---
    if add_isentropes:
        levels_theta = np.arange(288, 320, 1)
        c_theta = ds.theta.plot.contour(levels=levels_theta, colors='k', linewidths=0.5)
        ax.clabel(c_theta, levels=levels_theta[::2], inline=True)
        # Adjust linewidth
        for i, line in enumerate(c_theta.collections):
            if i % 2 == 0:
                line.set_linewidth(1)  # Increase linewidth for every second line

    # set x and y limit
    if ax_kwargs:
        ax.set(**ax_kwargs)

    # set/change format of xaxis ticks
    date_form = DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    # add labels
    ax.set(ylabel='height (m asl)', xlabel='UTC')
    ax.set_title(f'{station_meta.prf}: Time-Height Plot @ {ds.attrs["latitude"]:.3f} °N, ' +
                 f'{ds.attrs["longitude"]:.3f} °E')

    # add textbox
    at = AnchoredText(run, loc='upper left',
                      prop=dict(fontsize=10, color='white', weight='bold'),
                      frameon=True, pad=0.2)
    at.patch.set(facecolor='grey', edgecolor='k', alpha=0.9)
    ax.add_artist(at)

    # make nice looking layout
    ax.grid()
    ax.grid(linestyle='--', alpha=0.2, zorder=-10)
    fig.tight_layout()

    return fig, ax


def time_height_plot_diff(ds, var, figsize=(10, 5),
                          run=None, ax_kwargs=None, cmap=None):
    '''
    function to create time height plot of difference of needed variable.

    Parameters
    ----------
    ds : xarray dataset
        dataset whit dimensions height, time and the needed variables.
    figsize : tuple, optional
        size of figure. The default is (10, 5).
    quiver_l : int, optional
        quiverkey length. The default is 5.
    run : str, optional
        model run name for labeling. The default is None.
    ax_kwargs : dict, optional
        defines xlim and ylim of plot. The default is None.
    cmap : str, optional
        colormap
    Returns
    -------
    fig, ax, qv

    '''

    # dictionary with label options depending on needed variable
    var_dict = {'t': {'label': r'$\Delta$t (K)',
                      'levels': (-5, 5.1, 0.5), },
                'theta': {'label': r'$\Delta\theta$ (K)',
                          'levels': (-5, 5.1, 0.5), },
                'pres': {'label': r'$\Delta$p (hPa)',
                         'levels': (-1.6, 1.7, 0.1), },
                'ff': {'label': r'$\Delta$ff (m s$^{-1}$)',
                       'levels': (-6, 6.1, 1), },
                'q': {'label': r'$\Delta$q (g kg$^{-1}$)',
                      'levels': (-5, 5.1, 0.5)}
                }

    colorbar_kwargs = {'label': var_dict.get(var).get('label'),
                       'pad': 0.01,  # fraction between colorbar and plot (default: 0.05)
                       }

    # check for ax kwargs
    if ax_kwargs is None:
        start_x = np.datetime64('2019-09-12 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(ds.height.min()), 5000)}

    # levels colormap
    levels_ = var_dict.get(var).get('levels')
    levels = np.arange(levels_[0], levels_[1], levels_[2])
    if cmap:
        cmap = cm.get_cmap(cmap)
    else:
        cmap = cm.get_cmap('magma_r')

    # --- make plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- plot parameter colorcoded ---
    cf_var = ds[var].plot.contourf(x='time', y='height', cmap=cmap, levels=levels, extend='both',
                                   cbar_kwargs=colorbar_kwargs)

    # set x and y limit
    if ax_kwargs:
        ax.set(**ax_kwargs)

    # set/change format of xaxis ticks
    date_form = DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    # add labels
    ax.set(ylabel='height (m asl)', xlabel='UTC')
    ax.set_title(f'Time-Height Plot: {ds.attrs["station_ref"]} - {ds.attrs["station_loc"]}')

    # add textbox
    at = AnchoredText(run, loc='upper left',
                      prop=dict(fontsize=10, color='white', weight='bold'),
                      frameon=True, pad=0.2)
    at.patch.set(facecolor='grey', edgecolor='k', alpha=0.9)
    ax.add_artist(at)

    # make nice looking layout
    ax.grid()
    ax.grid(linestyle='--', alpha=0.2, zorder=-10)
    fig.tight_layout()

    return fig, ax


def time_height_plot_diff_wind(ds, var, figsize=(15, 4),
                               run=None, ax_kwargs=None, cmap=None):
    '''
    function to create time height plot of difference wind.

    Parameters
    ----------
    ds : xarray dataset
        dataset whit dimensions height, time and the needed variables.
    figsize : tuple, optional
        size of figure. The default is (10, 5).
    quiver_l : int, optional
        quiverkey length. The default is 5.
    run : str, optional
        model run name for labeling. The default is None.
    ax_kwargs : dict, optional
        defines xlim and ylim of plot. The default is None.
    cmap : str, optional
        colormap
    Returns
    -------
    fig, ax, qv

    '''

    # dictionary with label options depending on needed variable
    var_dict = {'ff': {'label': r'$\Delta$ff (m s$^{-1}$)',
                       'levels': (-6, 6.1, 1), },
                }

    colorbar_kwargs = {'label': var_dict.get(var).get('label'),
                       'pad': 0.01,  # fraction between colorbar and plot (default: 0.05)
                       }

    # check for ax kwargs
    if ax_kwargs is None:
        start_x = np.datetime64('2019-09-12 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        if 'RS' in run:
            ylim = (500, 2000)
        else:
            ylim = (500, 2000)
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': ylim}

    dy_ticks = 250 if ax_kwargs['ylim'][1] <= 2000 else 500

    # levels colormap
    levels_ = var_dict.get(var).get('levels')
    levels = np.arange(levels_[0], levels_[1], levels_[2])
    if cmap:
        cmap = cm.get_cmap(cmap)
    else:
        cmap = cm.get_cmap('magma_r')

    # --- make plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- plot parameter colorcoded ---
    cf_var = ds[var].plot.pcolormesh(x='time', y='height', cmap=cmap, levels=levels, extend='both',
                                     cbar_kwargs=colorbar_kwargs)

    # set x and y limit
    if ax_kwargs:
        ax.set(**ax_kwargs)

    # set/change format of xaxis ticks
    ax = customize_xtick_labels(ax)
    ax.set_yticks(np.arange(ax_kwargs['ylim'][0], ax_kwargs['ylim'][1] + dy_ticks, dy_ticks))

    # add labels
    ax.set(ylabel='height (m asl)', xlabel='UTC')
    ax.set_title('')

    # add wind error plot
    time = ds.time.values
    height = ds.height.values
    dd = ds.dd.values
    ff = ds.ff.values

    markers = ['.', 's', '^']
    colors = ['tab:green', 'tab:blue', 'tab:red']
    levels = [[45, 90], [90, 135], [135, 180]]

    for i, level in enumerate(levels):
        # wind error criterium
        mask = np.where((dd > level[0]) & (dd <= level[1]) & (np.abs(ff) >= 1))
        label = '(' + f'{level[0]}' + '°, ' + f'{level[1]}' + '°]'
        ax.scatter(time[mask[1]], height[mask[0]], marker=markers[i], color=colors[i], s=2,
                   label=label)

    # add textbox
    at = AnchoredText(run, loc='upper left',
                      prop=dict(fontsize=10, color='white', weight='bold'),
                      frameon=True, pad=0.2)
    at.patch.set(facecolor='grey', edgecolor='k', alpha=0.9)
    ax.add_artist(at)

    # add legend
    ax.legend(ncols=len(levels), loc='lower left', fontsize='small', bbox_to_anchor=(0, 1))

    # make nice looking layout
    ax.grid()
    ax.grid(linestyle='--', alpha=0.2, zorder=-10)
    fig.tight_layout()

    return fig, ax


def save_plot(dir_plots, name_plot):
    '''
    function to save plot

    Parameters
    ----------
    dir_plots : str
        path to directory of plots.
    name_plot : str
        name of plot to save.

    Returns
    -------
    None.

    '''
    path_save = join(dir_plots, name_plot)
    print(f'Save plot to: {path_save}')
    plt.savefig(path_save, bbox_inches='tight')
    plt.close()


# %% --- main OP500 ---
if __name__ == '__main_':

    # define paramters global
    var_list = ['u', 'v', 't', 'z', 'pres', 'q', 'tke', 'w']
    run = 'OP500'
    ds_OP500 = {}

    # define paths
    dir_path = get_Dataset_path(run)

    # -----------------------------------------------
    # ---- KOLSASS, HAIMING, ROS etc.
    # -----------------------------------------------
    # define parameters
    stations = ['KOLS']  # , 'HAI', 'ROS', 'UNI', 'RS_LOWI', 'RS_MUC']

    for station in stations:

        # load metadata station
        station_meta = get_StationMetaData(dir_metadata, station)
        coords = (station_meta.lon, station_meta.lat)

        # get profile Dataset model
        ds_model, ds_init = get_Data(dir_path, run, station_meta, var_list)

        # make the plot
        fig, ax, qv = time_height_plot_wind(ds_model, run=run, station_meta=station_meta)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot night
        start_x = np.datetime64('2019-09-12 21:00')
        end_x = np.datetime64('2019-09-13 12:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax, qv = time_height_plot_wind(ds_model, figsize=(10, 5), run=run, station_meta=station_meta,
                                            ax_kwargs=ax_kwargs, barb_l=5)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}_zoom_night.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot day
        start_x = np.datetime64('2019-09-13 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax, qv = time_height_plot_wind(ds_model, figsize=(10, 5), run=run, station_meta=station_meta,
                                            ax_kwargs=ax_kwargs, barb_l=5)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}_zoom_day.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # save data in container
        ds_OP500[station] = ds_model

    # ------------------------------------------------
    # ---- KOLSASS - HAIMING, ROSENHEIM - KOLSASS, ROSENHEIM - HAIMING, MUNICH - KOLSASS
    # ------------------------------------------------
    # define parameters
    var_list = ['pres', 'theta']
    station_list = [['KOLS', 'HAI'], ['ROS', 'KOLS'], ['ROS', 'HAI'], ['RS_MUC', 'KOLS']]

    for station in station_list:
        # calculate difference between stations
        ds_diff = calc_gradients_WB.main(ds_OP500[station[0]], ds_OP500[station[1]], var_list)

        # make the plots
        for var in var_list:
            fig, ax = time_height_plot_diff(ds_diff, var, run=run, cmap='RdBu_r')
            name_plot = f'HT_{var}_{station[0]}_{station[1]}_{run}.svg'  # name of plot
            save_plot(dir_plots, name_plot)
            plt.close()

        # save data in container
        ds_OP500[f'{station[0]} - {station[1]}'] = ds_diff

    # ----------------------------------------------
    # ---- plots KOLSASS w, q
    # ----------------------------------------------
    station = 'KOLS'
    var_list = ['w', 'q']
    cmaps = ['PuOr_r', cmap_user.get('DarkMint').reversed()]
    extend = ['both', 'max']
    for i, var in enumerate(var_list):
        fig, ax = time_height_plot_var(ds_OP500[station], var, run=run,
                                       station_meta=station_meta, cmap=cmaps[i], extend=extend[i])
        name_plot = f'HT_{var}_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)

# %% --- main OP1000 ---
if __name__ == '__main_':

    # define paramters global
    var_list = ['u', 'v', 't', 'z', 'pres', 'q', 'tke', 'w']
    run = 'OP1000'
    ds_OP1000 = {}

    # define paths
    dir_path = get_Dataset_path(run)

    # -----------------------------------------------
    # ---- KOLSASS, HAIMING, ROS
    # -----------------------------------------------
    # define parameters
    stations = ['KOLS']  # , 'HAI', 'ROS', 'UNI', 'RS_LOWI', 'RS_MUC']

    for station in stations:

        # load metadata station
        station_meta = get_StationMetaData(dir_metadata, station)
        coords = (station_meta.lon, station_meta.lat)

        # get profile Dataset model
        ds_model, ds_init = get_Data(dir_path, run, station_meta, var_list)

        # make the plot
        fig, ax, qv = time_height_plot_wind(ds_model, run=run, station_meta=station_meta)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot night
        start_x = np.datetime64('2019-09-12 21:00')
        end_x = np.datetime64('2019-09-13 12:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax, qv = time_height_plot_wind(ds_model, figsize=(10, 5), run=run, station_meta=station_meta,
                                            ax_kwargs=ax_kwargs, barb_l=5)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}_zoom_night.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot day
        start_x = np.datetime64('2019-09-13 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax, qv = time_height_plot_wind(ds_model, figsize=(10, 5), run=run, station_meta=station_meta,
                                            ax_kwargs=ax_kwargs, barb_l=5)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}_zoom_day.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # save data in container
        ds_OP1000[station] = ds_model

    # ------------------------------------------------
    # ---- KOLSASS - HAIMING, ROSENHEIM - KOLSASS
    # ------------------------------------------------
    # define parameters
    var_list = ['pres', 'theta']
    station_list = [['KOLS', 'HAI'], ['ROS', 'KOLS'], ['ROS', 'HAI'], ['RS_MUC', 'KOLS']]

    for station in station_list:
        # calculate difference between stations
        ds_diff = calc_gradients_WB.main(ds_OP1000[station[0]], ds_OP1000[station[1]], var_list)

        # make the plots
        for var in var_list:
            fig, ax = time_height_plot_diff(ds_diff, var, run=run, cmap='RdBu_r')
            name_plot = f'HT_{var}_{station[0]}_{station[1]}_{run}.svg'  # name of plot
            save_plot(dir_plots, name_plot)
            plt.close()

        # save data in container
        ds_OP1000[f'{station[0]} - {station[1]}'] = ds_diff

    # ----------------------------------------------
    # ---- plots KOLSASS w, q
    # ----------------------------------------------
    station = 'KOLS'
    var_list = ['w', 'q']
    cmaps = ['PuOr_r', cmap_user.get('DarkMint').reversed()]
    extend = ['both', 'max']
    for i, var in enumerate(var_list):
        fig, ax = time_height_plot_var(ds_OP1000[station], var, run=run,
                                       station_meta=station_meta, cmap=cmaps[i], extend=extend[i])
        name_plot = f'HT_{var}_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)

# %% --- main ARP1000 ---
if __name__ == '__main_':

    # define paramters global
    var_list = ['u', 'v', 't', 'z', 'pres', 'q', 'tke', 'w']
    run = 'ARP1000'
    ds_ARP1000 = {}

    # define paths
    dir_path = get_Dataset_path(run)

    # -----------------------------------------------
    # ---- KOLSASS, HAIMING, ROS
    # -----------------------------------------------
    # define parameters
    stations = ['KOLS', 'HAI', 'ROS', 'UNI', 'RS_LOWI', 'RS_MUC']

    for station in stations:

        # load metadata station
        station_meta = get_StationMetaData(dir_metadata, station)
        coords = (station_meta.lon, station_meta.lat)

        # get profile Dataset model
        ds_model, ds_init = get_Data(dir_path, run, station_meta, var_list)

        # make the plot
        fig, ax, qv = time_height_plot_wind(ds_model, run=run, station_meta=station_meta)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot
        start_x = np.datetime64('2019-09-13 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax, qv = time_height_plot_wind(ds_model, run=run, station_meta=station_meta, ax_kwargs=ax_kwargs)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}_zoom.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # save data in container
        ds_ARP1000[station] = ds_model

    # ------------------------------------------------
    # ---- KOLSASS - HAIMING, ROSENHEIM - KOLSASS
    # ------------------------------------------------
    # define parameters
    var_list = ['pres', 'theta']
    station_list = [['KOLS', 'HAI'], ['ROS', 'KOLS'], ['ROS', 'HAI'], ['RS_MUC', 'KOLS']]

    for station in station_list:
        # calculate difference between stations
        ds_diff = calc_gradients_WB.main(ds_ARP1000[station[0]], ds_ARP1000[station[1]], var_list)

        # make the plots
        for var in var_list:
            fig, ax = time_height_plot_diff(ds_diff, var, run=run, cmap='RdBu_r')
            name_plot = f'HT_{var}_{station[0]}_{station[1]}_{run}.svg'  # name of plot
            save_plot(dir_plots, name_plot)
            plt.close()

        # save data in container
        ds_ARP1000[f'{station[0]} - {station[1]}'] = ds_diff

    # ----------------------------------------------
    # ---- plots KOLSASS w, q
    # ----------------------------------------------
    station = 'KOLS'
    var_list = ['w', 'q']
    cmaps = ['PuOr_r', cmap_user.get('DarkMint').reversed()]
    extend = ['both', 'max']
    for i, var in enumerate(var_list):
        fig, ax = time_height_plot_var(ds_ARP1000[station], var, run=run,
                                       station_meta=station_meta, cmap=cmaps[i], extend=extend[i])
        name_plot = f'HT_{var}_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)


# %% --- main IFS1000 ---
if __name__ == '__main_':

    # define paramters global
    var_list = ['u', 'v', 't', 'z', 'pres', 'q', 'tke', 'w']
    run = 'IFS1000'
    ds_IFS1000 = {}

    # define paths
    dir_path = get_Dataset_path(run)

    # -----------------------------------------------
    # ---- KOLSASS, HAIMING, ROS
    # -----------------------------------------------
    # define parameters
    stations = ['KOLS', 'HAI', 'ROS', 'UNI', 'RS_LOWI', 'RS_MUC']

    for station in stations:

        # load metadata station
        station_meta = get_StationMetaData(dir_metadata, station)
        coords = (station_meta.lon, station_meta.lat)

        # get profile Dataset model
        ds_model, ds_init = get_Data(dir_path, run, station_meta, var_list)

        # make the plot
        fig, ax, qv = time_height_plot_wind(ds_model, run=run, station_meta=station_meta)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot
        start_x = np.datetime64('2019-09-13 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax, qv = time_height_plot_wind(ds_model, run=run, station_meta=station_meta, ax_kwargs=ax_kwargs)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}_zoom.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # save data in container
        ds_IFS1000[station] = ds_model

    # ------------------------------------------------
    # ---- KOLSASS - HAIMING, ROSENHEIM - KOLSASS
    # ------------------------------------------------
    # define parameters
    var_list = ['pres', 'theta']
    station_list = [['KOLS', 'HAI'], ['ROS', 'KOLS'], ['ROS', 'HAI'], ['RS_MUC', 'KOLS']]

    for station in station_list:
        # calculate difference between stations
        ds_diff = calc_gradients_WB.main(ds_IFS1000[station[0]], ds_IFS1000[station[1]], var_list)

        # make the plots
        for var in var_list:
            fig, ax = time_height_plot_diff(ds_diff, var, run=run, cmap='RdBu_r')
            name_plot = f'HT_{var}_{station[0]}_{station[1]}_{run}.svg'  # name of plot
            save_plot(dir_plots, name_plot)
            plt.close()

        # save data in container
        ds_IFS1000[f'{station[0]} - {station[1]}'] = ds_diff

    # ----------------------------------------------
    # ---- plots KOLSASS w, q
    # ----------------------------------------------
    station = 'KOLS'
    var_list = ['w', 'q']
    cmaps = ['PuOr_r', cmap_user.get('DarkMint').reversed()]
    extend = ['both', 'max']
    for i, var in enumerate(var_list):
        fig, ax = time_height_plot_var(ds_IFS1000[station], var, run=run,
                                       station_meta=station_meta, cmap=cmaps[i], extend=extend[i])
        name_plot = f'HT_{var}_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)

# %% --- main OP2500 ---
if __name__ == '__main_':

    # define paramters global
    var_list = ['u', 'v', 't', 'z', 'pres', 'q', 'tke', 'w']
    run = 'OP2500'
    ds_OP2500 = {}

    # define paths
    dir_path = get_Dataset_path(run)

    # -----------------------------------------------
    # ---- KOLSASS, HAIMING, ROS etc.
    # -----------------------------------------------
    # define parameters
    stations = ['KOLS']  # , 'HAI', 'ROS', 'UNI', 'RS_LOWI', 'RS_MUC']

    for station in stations:

        # load metadata station
        station_meta = get_StationMetaData(dir_metadata, station)
        coords = (station_meta.lon, station_meta.lat)

        # get profile Dataset model
        ds_model, ds_init = get_Data(dir_path, run, station_meta, var_list)

        # make the plot
        fig, ax, qv = time_height_plot_wind(ds_model, run=run, station_meta=station_meta)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot night
        start_x = np.datetime64('2019-09-12 21:00')
        end_x = np.datetime64('2019-09-13 12:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax, qv = time_height_plot_wind(ds_model, figsize=(10, 5), run=run, station_meta=station_meta,
                                            ax_kwargs=ax_kwargs, barb_l=5)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}_zoom_night.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot day
        start_x = np.datetime64('2019-09-13 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax, qv = time_height_plot_wind(ds_model, figsize=(10, 5), run=run, station_meta=station_meta,
                                            ax_kwargs=ax_kwargs, barb_l=5)

        # save figure
        name_plot = f'HT_Wind_{station}_{run}_zoom_day.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # save data in container
        ds_OP2500[station] = ds_model

    # ------------------------------------------------
    # ---- KOLSASS - HAIMING, ROSENHEIM - KOLSASS, ROSENHEIM - HAIMING
    # ------------------------------------------------
    # define parameters
    var_list = ['pres', 'theta']
    station_list = [['KOLS', 'HAI'], ['ROS', 'KOLS'], ['ROS', 'HAI'], ['RS_MUC', 'KOLS']]

    for station in station_list:
        # calculate difference between stations
        ds_diff = calc_gradients_WB.main(ds_OP2500[station[0]], ds_OP2500[station[1]], var_list)

        # make the plots
        for var in var_list:
            fig, ax = time_height_plot_diff(ds_diff, var, run=run, cmap='RdBu_r')
            name_plot = f'HT_{var}_{station[0]}_{station[1]}_{run}.svg'  # name of plot
            save_plot(dir_plots, name_plot)
            plt.close()

        # save data in container
        ds_OP2500[f'{station[0]} - {station[1]}'] = ds_diff

    # ----------------------------------------------
    # ---- plots KOLSASS w, q
    # ----------------------------------------------
    station = 'KOLS'
    var_list = ['w', 'q']
    cmaps = ['PuOr_r', cmap_user.get('DarkMint').reversed()]
    extend = ['both', 'max']
    for i, var in enumerate(var_list):
        fig, ax = time_height_plot_var(ds_OP2500[station], var, run=run,
                                       station_meta=station_meta, cmap=cmaps[i], extend=extend[i])
        name_plot = f'HT_{var}_{station}_{run}.svg'  # name of plot
        save_plot(dir_plots, name_plot)

# %% --- main Radiosonde ---
if __name__ == 'main':

    # define parameters
    run = 'RS'

    # check for  Dataset
    dir_rs = get_ObsDataset_path(run)
    ds_rs, _ = get_Data(dir_rs, run)

    # upsample RS data to match model data
    # ds_rs = ds_rs.resample(time='10min', label='left').interpolate('linear')

    # make wind plot
    station = 'RS'
    station_meta = get_StationMetaData(dir_metadata, station)
    fig, ax, qv = time_height_plot_wind(ds_rs, run=run, station_meta=station_meta)

    # save figure
    name_plot = 'HT_Wind_RS.svg'  # name of plot
    save_plot(dir_plots, name_plot)

    # make zoom plot night
    start_x = np.datetime64('2019-09-12 21:00')
    end_x = np.datetime64('2019-09-13 12:00')
    ax_kwargs = {'xlim': (start_x, end_x),
                 'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

    fig, ax, qv = time_height_plot_wind(ds_rs, figsize=(10, 5), run=run, station_meta=station_meta,
                                        ax_kwargs=ax_kwargs, barb_l=5)

    # save figure
    name_plot = f'HT_Wind_RS_zoom_night.svg'  # name of plot
    save_plot(dir_plots, name_plot)

    # make zoom plot day
    start_x = np.datetime64('2019-09-13 12:00')
    end_x = np.datetime64('2019-09-14 03:00')
    ax_kwargs = {'xlim': (start_x, end_x),
                 'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

    fig, ax, qv = time_height_plot_wind(ds_rs, figsize=(10, 5), run=run, station_meta=station_meta,
                                        ax_kwargs=ax_kwargs, barb_l=5)

    # save figure
    name_plot = f'HT_Wind_RS_zoom_day.svg'  # name of plot
    save_plot(dir_plots, name_plot)

    # make specific humidity plot
    cmap = cmap_user.get('DarkMint').reversed()
    fig, ax = time_height_plot_var(ds_rs, 'q', cmap=cmap, run=run, station_meta=station_meta, extend='max')

    # save figure
    name_plot = 'HT_q_RS.svg'  # name maskof plot
    save_plot(dir_plots, name_plot)

# %% --- main Lidar ---
if __name__ == 'main':

    # define parameters
    run = 'SL88'

    # check for Dataset
    dir_lidar = get_ObsDataset_path(run)
    ds_lidar, _ = get_Data(dir_lidar, run)

    # resample lidar to 10 min values
    ds_lidar = ds_lidar.resample(time='10min', label='right', closed='right').mean().transpose()

    # make plot
    station = 'SL88'
    station_meta = get_StationMetaData(dir_metadata, station)
    fig, ax, qv = time_height_plot_wind(ds_lidar, run=run, station_meta=station_meta)

    # save figure
    name_plot = 'HT_Wind_Lidar_SL88.svg'  # name of plot
    save_plot(dir_plots, name_plot)

    # make zoom plot night
    start_x = np.datetime64('2019-09-12 21:00')
    end_x = np.datetime64('2019-09-13 12:00')
    ax_kwargs = {'xlim': (start_x, end_x),
                 'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

    fig, ax, qv = time_height_plot_wind(ds_lidar, figsize=(10, 5), run=run, station_meta=station_meta,
                                        ax_kwargs=ax_kwargs, barb_l=5)

    # save figure
    name_plot = f'HT_Wind_Lidar_SL88_zoom_night.svg'  # name of plot
    save_plot(dir_plots, name_plot)

    # make zoom plot day
    start_x = np.datetime64('2019-09-13 12:00')
    end_x = np.datetime64('2019-09-14 03:00')
    ax_kwargs = {'xlim': (start_x, end_x),
                 'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

    fig, ax, qv = time_height_plot_wind(ds_lidar, figsize=(10, 5), run=run, station_meta=station_meta,
                                        ax_kwargs=ax_kwargs, barb_l=5)

    # save figure
    name_plot = f'HT_Wind_Lidar_SL88_zoom_day.svg'  # name of plot
    save_plot(dir_plots, name_plot)

    # make vertical wind speed plot
    fig, ax = time_height_plot_var(ds_lidar, 'w', run=run, station_meta=station_meta,
                                   cmap='PuOr_r', add_quiver=False, add_isentropes=False)

    # save figure
    name_plot = 'HT_w_Lidar_SL88.svg'  # name of plot
    save_plot(dir_plots, name_plot)

# %% --- main MWR ---
if __name__ == 'main':

    # define parameters
    run = 'MWR'

    # check for Dataset
    dir_MWR = get_ObsDataset_path(run)
    ds_MWR, _ = get_Data(dir_MWR, run)

    # make plot
    station = 'MWR'
    station_meta = get_StationMetaData(dir_metadata, station)
    var_list = ['t']
    for var in var_list:
        fig, ax = time_height_plot_var(ds_MWR, var, run=run, station_meta=station_meta,
                                       add_quiver=False, add_isentropes=False)

        # save figure
        name_plot = f'HT_{var}_MWR.svg'  # name of plot
        save_plot(dir_plots, name_plot)

# %% --- difference Model run - Lidar ---
if __name__ == 'main':

    # define paramters global
    var_list = ['u', 'v', 't', 'z', 'pres', 'q', 'tke']
    runs = ['OP500', 'OP1000', 'OP2500']
    station = 'KOLS'
    ds_model = {}
    ds_init = {}

    # get lidar data
    dir_lidar = get_ObsDataset_path('SL88')
    ds_lidar, _ = get_Data(dir_lidar, 'SL88')

    # resample lidar to 10 min values
    ds_lidar = ds_lidar.resample(time='10min', label='right', closed='right').mean()

    # get model data
    for run in runs:
        # define paths
        dir_path = get_Dataset_path(run)

        # load metadata station
        station_meta = get_StationMetaData(dir_metadata, station)
        coords = (station_meta.lon, station_meta.lat)

        # get profile Dataset model
        ds_model[run], ds_init[run] = get_Data(dir_path, run, station_meta, var_list)

        # calculate difference between lidar and model data
        ds_diff = calc_gradients_WB.main(ds_model[run], ds_lidar, ['ff', 'dd'])

        # make the plot
        fig, ax = time_height_plot_diff_wind(ds_diff, 'ff', run=f'{run} - SL88', cmap='PuOr_r')

        # save figure
        name_plot = f'HT_Wind_diff_{run}_LidarSL88.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot night
        start_x = np.datetime64('2019-09-12 21:00')
        end_x = np.datetime64('2019-09-13 12:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax = time_height_plot_diff_wind(ds_diff, 'ff', figsize=(10, 5), run=f'{run} - SL88',
                                             ax_kwargs=ax_kwargs, cmap='PuOr_r')

        # save figure
        name_plot = f'HT_Wind_diff_{run}_LidarSL88_zoom_night.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make zoom plot day
        start_x = np.datetime64('2019-09-13 12:00')
        end_x = np.datetime64('2019-09-14 03:00')
        ax_kwargs = {'xlim': (start_x, end_x),
                     'ylim': (np.floor(station_meta.alt/100)*100, 2000)}

        fig, ax = time_height_plot_diff_wind(ds_diff, 'ff', figsize=(10, 5), run=f'{run} - SL88',
                                             ax_kwargs=ax_kwargs, cmap='PuOr_r')

        # save figure
        name_plot = f'HT_Wind_diff_{run}_LidarSL88_zoom_day.svg'  # name of plot
        save_plot(dir_plots, name_plot)

# %% --- difference Model run - RS ---
if __name__ == 'main':

    # define paramters global
    var_list = ['u', 'v', 't', 'z', 'pres', 'q', 'tke']
    runs = ['OP500', 'OP1000', 'OP2500']
    station = 'KOLS'
    ds_model = {}
    ds_init = {}

    # get RS data
    dir_RS = get_ObsDataset_path('RS')
    ds_RS, _ = get_Data(dir_RS, 'RS')

    # resample RS to 10 min values, interpolate linear
    ds_RS = ds_RS.resample(time='10min', label='left').interpolate('linear')

    # get model data
    for run in runs:
        # define paths
        dir_path = get_Dataset_path(run)

        # load metadata station
        station_meta = get_StationMetaData(dir_metadata, station)
        coords = (station_meta.lon, station_meta.lat)

        # get profile Dataset model
        ds_model[run], ds_init[run] = get_Data(dir_path, run, station_meta, var_list)

        # calculate difference between model data and RS observations
        ds_diff = calc_gradients_WB.main(ds_model[run], ds_RS, ['t', 'q', 'ff', 'dd'])

        # make the plot for wind speed
        fig, ax = time_height_plot_diff_wind(ds_diff, 'ff', run=f'{run} - RS', cmap='PuOr_r')

        # save figure
        name_plot = f'HT_Wind_diff_{run}_RS.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make the plot for temperature
        fig, ax = time_height_plot_diff(ds_diff, 't', run=f'{run} - RS', cmap='RdBu_r')

        # save figure
        name_plot = f'HT_t_diff_{run}_RS.svg'  # name of plot
        save_plot(dir_plots, name_plot)

        # make the plot for specific humidity
        fig, ax = time_height_plot_diff(ds_diff, 'q', run=f'{run} - RS', cmap='BrBG')

        # save figure
        name_plot = f'HT_q_diff_{run}_RS.svg'  # name of plot
        save_plot(dir_plots, name_plot)

# %% --- difference model run - MWR ---
if __name__ == 'main':

    # define paramters global
    var_list = ['u', 'v', 't', 'z', 'pres', 'q', 'tke']
    runs = ['OP500', 'OP1000', 'OP2500']
    station = 'KOLS'
    ds_model = {}
    ds_init = {}

    # get lidar data
    dir_MWR = get_ObsDataset_path('MWR')
    ds_MWR, _ = get_Data(dir_MWR, 'MWR')

    # get model data
    for run in runs:
        # define paths
        dir_path = get_Dataset_path(run)

        # load metadata station
        station_meta = get_StationMetaData(dir_metadata, station)
        coords = (station_meta.lon, station_meta.lat)

        # get profile Dataset model
        ds_model[run], ds_init[run] = get_Data(dir_path, run, station_meta, var_list)

        # calculate difference between lidar and model data
        ds_diff = calc_gradients_WB.main(ds_model[run], ds_MWR, ['t'])

        # make the plot
        fig, ax = time_height_plot_diff(ds_diff, 't', run=f'{run} - MWR', cmap='RdBu_r')

        # save figure
        name_plot = f'HT_t_diff_{run}_MWR.svg'  # name of plot
        save_plot(dir_plots, name_plot)

# %% --- difference Model runs ---
if __name__ == 'main':

    # define paramters global
    var_list = ['u', 'v', 't', 'z', 'pres', 'q', 'tke']
    runs = ['OP500', 'OP1000', 'OP2500']
    station = 'KOLS'
    ds_model = {}
    ds_init = {}

    # get model data
    for run in runs:
        # define paths
        dir_path = get_Dataset_path(run)

        # load metadata station
        station_meta = get_StationMetaData(dir_metadata, station)
        coords = (station_meta.lon, station_meta.lat)

        # get profile Dataset model
        ds_model[run], ds_init[run] = get_Data(dir_path, run, station_meta, var_list)

    diff_runs = [[runs[0], runs[1]], [runs[0], runs[2]]]
    for d_r in diff_runs:
        # calculate difference between model runs
        ds_diff = calc_gradients_WB.main(ds_model[d_r[0]], ds_model[d_r[1]], ['ff', 'dd'])

        # make the plot
        fig, ax = time_height_plot_diff_wind(ds_diff, 'ff', run=f'{d_r[0]} - {d_r[1]}', cmap='PuOr_r')

        # save figure
        name_plot = f'HT_Wind_diff_{d_r[0]}_{d_r[1]}_{station}.svg'  # name of plot
        save_plot(dir_plots, name_plot)
