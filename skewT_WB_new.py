#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:42:47 2023

@author: benwib

Script for creating skewT plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metpy.plots import SkewT
import metpy.calc as mpcalc
from metpy.units import units
from calculations import _calc_potTemperature, _calc_geopotHeight, _calc_ff, _calc_dd
from read_StationMeta import get_StationMetaData
import glob2
from os.path import join
import xarray as xr
import read_RS
import read_lidar_vert
import read_MWR
from path_handling import (get_Dataset_path, get_ObsDataset_path, get_Coupling_IFS_path, dir_metadata,
                           get_Dataset_name, get_interpStation_name, get_GRIB_path)
from matplotlib import gridspec

# define global variables
dir_plots = '../../Plots/Profiles/'


def _get_modelData(dir_path, var_list, run, timestep, station_meta, method='linear'):
    '''
    function to create profile dataset of model data on hybrid pressure levels.

    Parameters
    ----------
    dir_path : str
        path to directory of data.
    var_list : list
        list of variables needed.
    run : str
        modelrun.
    timestep : str
        timestep needed.
    station_meta : pandas Series
        meta data of station
    method : str, optional
        interpolation method. The default is 'linear'.

    Returns
    -------
    ds_init : xarray Dataset
        Dataset containing profile data on hybrid pressure levels

    '''

    # get coordinates of station
    coords = (station_meta.lon, station_meta.lat)

    # get path to file
    filename = get_interpStation_name(run, coords)  # load already interpolated Dataset from HT
    path_file = glob2.glob(join(dir_path, filename))

    # open dataset or extract data from GRIB File
    if path_file != []:
        ds_init = xr.open_mfdataset(path_file, chunks={'valid_time': 1})
    else:
        ds_init = _read_profile(dir_path, run, timestep, station_meta, method=method)

    # slice it to needed timestep
    ds_init = ds_init.sel(valid_time=timestep)

    # add further needed variables
    ds_init['theta'] = _calc_potTemperature(ds_init['t'], ds_init['pres'])  # potential temperature
    ds_init['Z'] = _calc_geopotHeight(ds_init['z'])  # geopotentail height
    ds_init['ff'] = _calc_ff(ds_init['u'], ds_init['v'])  # wind speed
    ds_init['dd'] = _calc_dd(ds_init['u'], ds_init['v'])  # wind direction

    # add station meta
    ds_init.attrs['station'] = station_meta.prf

    return ds_init


def _read_profile(dir_path, run, timestep, station_meta,
                  init_time='2019-09-12T12:00:00', method='linear'):
    '''
    function to read in profile data from GRIB files

    Parameters
    ----------
    dir_path : str
        path to directory of data.
    run : str
        modelrun.
    timestep : str
        timestep needed.
    station_meta : pandas Series
        meta data station.
    init_time : str, optional
        initial timestep GRIB data. The default is '2019-09-12T12:00:00'.
    method : str, optional
        interpolation method. The default is 'linear'.

    Returns
    -------
    ds : xarray Dataset
        Dataset containing profile data.

    '''

    print('create data from GRIB File ...')
    # open only needed timestep
    # get timestep
    step = int((pd.to_datetime(timestep) - pd.to_datetime(init_time)) / pd.Timedelta('1 hour'))

    # set path to grib files
    path_GRIB = get_GRIB_path(run)
    path_file = sorted(glob2.glob(join(path_GRIB, f'GRIBPFAROMAROM+00{step:02d}_00.grib2')))

    # open dataset
    cfgrib_kwargs = {'filter_by_keys':
                     {'typeOfLevel': 'hybridPressure',
                      },  # 'shortName': var},
                     'indexpath': ''}
    ds = xr.open_mfdataset(path_file, engine='cfgrib', combine='nested', concat_dim='valid_time',
                           chunks={'valid_time': 1, 'hybridPressure': 28}, parallel=True,
                           drop_variables=['step', 'time'], backend_kwargs=cfgrib_kwargs)

    # interpolate data to needed coordinates
    coords = (station_meta.lon, station_meta.lat)
    ds = ds.interp(longitude=coords[0], latitude=coords[1], method=method)

    # rename vertical coordinate
    ds = ds.rename({'hybridPressure': 'level'})

    # add needed attributes
    ds = ds.assign_attrs(modelrun=run)

    return ds


def _get_ObsDataset(dir_path, run, timestep):
    '''
    function to get observational radisonde data

    Parameters
    ----------
    dir_path : str
        path to directory of data.
    run : str
        name of observations (e.g. RS, SL88 (lidar), MWR (micro wave radiometer)).
    timestep : str
        timestep needed.

    Returns
    -------
    ds : xarray Dataset
        Dataset containing profile data of observations.

    '''

    # check which observational data is needed
    if run == 'RS':
        # create radiosonde dataset
        ds = read_RS.main(dir_path)
        # interpolate data between available timesteps hourly
        ds = ds.resample(time='1h').interpolate('linear')
    elif run == 'SL88':
        # create lidar datset
        ds = read_lidar_vert.main(dir_path)
        # resample to get nicer timesteps
        ds = ds.resample(time='10min', label='right', closed='right').interpolate('linear')
    elif run == 'MWR':
        ds = read_MWR.main(dir_path)
    elif 'RS_' in run:
        name_station = run
        ds = read_RS.main(dir_path, station=name_station, timestamp=timestep)

    # check if dataset contains timestep otherwise return None
    if ds:
        if pd.to_datetime(timestep) in ds.time:
            # select timestep
            ds = ds.sel(time=timestep)

            # add attribute information
            ds.attrs['modelrun'] = run
        else:
            ds = None

    return ds


def _get_IFSData(dir_path, station, timestep, init_time='2019-09-12T12:00:00'):
    '''
    function to extract IFS data

    Parameters
    ----------
    dir_path : str
        path to directory of data.
    station : str
        name of station.
    timestep : str
        timestep needed.
    init_time : str, optional
        initial timestep data. The default is '2019-09-12T12:00:00'.

    Returns
    -------
    ds : xarray Dataset
        Dataset containing profile data of IFS data.

    '''

    # get timestep
    step = int((pd.to_datetime(timestep) - pd.to_datetime(init_time)) / pd.Timedelta('1 hour'))
    filename = f'{station}_profile_t_{step:03d}.txt'
    path_file = join(dir_path, filename)

    # get data
    df = pd.read_csv(path_file, skiprows=1, delim_whitespace=True, names=['pres', 't'])
    df['valid_time'] = timestep

    # convert data to xarray
    ds = df.to_xarray()

    # calculate potential temperature
    ds['theta'] = _calc_potTemperature(ds['t'], ds['pres'])

    # add attribute information
    ds.attrs['modelrun'] = 'IFS'

    return ds


def _plot_skewT(ds_profiles,
                ds_profile_OBS=None,
                ds_profile_IFS=None,
                title=None,
                zoom={'ymin': None, 'ymax': 500, 'xmin': -15, 'xmax': 30},
                figsize=(10., 6.),
                colors=None,
                legend_kwargs=None,
                fig=None, ax=None):
    '''
    function to create skewT plot

    '''

    line_colors = ['tab:blue', 'tab:orange', 'tab:green']
    line_style = ['-', '--']
    x_loc_barb = [0.92, 0.94, 0.96]
    x_loc_barb_obs = 0.98

    # Create a figure and SkewT object or add it to existing plot
    if not fig:
        fig = plt.figure(figsize=figsize)
        skew = SkewT(fig)
        subplot = False
    else:
        skew = SkewT(fig=fig, subplot=ax.get_gridspec()[0])
        ax.remove()
        ax = skew.ax
        subplot = True

    for i, ds_profile in enumerate(ds_profiles.values()):
        keys = list(ds_profile.keys())

        # check if temperatur is in correct units of degree C
        if (ds_profile['t'] > 200).any():
            t = (ds_profile['t'].values - 273.15) * units.degreeC
        else:
            t = ds_profile['t'].values * units.degreeC

        # add units - needed by Metpy
        if 'pres' in keys:
            p = ds_profile['pres'].values / 100 * units.hPa

        # Plot the temperature profile
        lc = line_colors[i]
        skew.plot(p, t, lc, linestyle=line_style[0],
                  label=f'{ds_profile.attrs["modelrun"]}', linewidth=1.5)

        # if humidity profile is available use it
        if 'q' in keys:
            # convert specific humidity to dewpoint
            q = ds_profile['q'].values * units('g/g')
            td = mpcalc.dewpoint_from_specific_humidity(p, t, q)
            skew.plot(p, td, lc, linestyle=line_style[1],
                      label='', linewidth=1.5)

        # if wind data is available use it
        # if 'u' in keys:
            # u = ds_profile['u'].values * units.meter_per_second
            # v = ds_profile['v'].values * units.meter_per_second
            # skew.plot_barbs(p, u, v, plot_units=units.knots, barbcolor=lc, xloc=x_loc_barb[i], length=5)

    # add observational data
    if ds_profile_OBS:
        t = (ds_profile_OBS['t'].values - 273.15) * units.degreeC
        p = ds_profile_OBS['p'].values / 100 * units.hPa
        td = (ds_profile_OBS['td'].values - 273.15) * units.degreeC

        # plot t
        skew.plot(p, t, 'k', linestyle=line_style[0],
                  label=f'{ds_profile_OBS.attrs["modelrun"]}', linewidth=1.5)

        # plot td
        skew.plot(p, td, 'k', linestyle=line_style[1],
                  label='', linewidth=1.5)

        # plot wind
        # u = ds_profile_OBS['u'].values * units.meter_per_second
        # v = ds_profile_OBS['v'].values * units.meter_per_second
        # plot only every second barb
        # skew.plot_barbs(p[::6], u[::6], v[::6], plot_units=units.knots, xloc=x_loc_barb_obs, length=5)

    # add IFS Coupling profile
    if ds_profile_IFS:
        t = (ds_profile_IFS['t'].values - 273.15) * units.degreeC
        p = ds_profile_IFS['pres'].values * units.hPa

        # plot t
        skew.plot(p, t, 'tab:red', linestyle=line_style[0],
                  label=f'IFS', linewidth=1.5)

    # Add relevant special lines
    kw_args_lines = dict(alpha=0.4, linewidth=1, linestyles='-')
    skew.plot_dry_adiabats(colors='limegreen', **kw_args_lines)
    skew.plot_moist_adiabats(colors='darkmagenta', **kw_args_lines)
    mr = np.array([0.001, 0.002, 0.004, 0.007,
                   0.01, 0.016, 0.024, 0.032]).reshape(-1, 1)
    skew.plot_mixing_lines(mixing_ratio=mr, colors='black', **kw_args_lines)

    # add labels for mixing lines
    for val in mr:
        p_surf = 1000 * units.hPa
        dewpt = mpcalc.dewpoint(mpcalc.vapor_pressure(p_surf, val[0] * units('g/g')))
        skew.ax.text(dewpt, p_surf, str(val[0]*1e3), color='black', alpha=0.4, rotation=90,
                     fontsize='small', horizontalalignment='center', verticalalignment='bottom')

    # Add some labels and title
    title = (f'{ds_profile.attrs["station"]}: ' +
             str(ds_profile.valid_time.dt.strftime("%Y-%m-%d %H:%M UTC").values))
    skew.ax.set(title=title, ylabel='pressure (hPa)', xlabel='temperature (°C)')
    skew.ax.legend(loc='upper right')
    skew.ax.grid(True)

    # zoom active?
    if zoom is not None:
        if 'ymin' in zoom:
            skew.ax.set_ylim(bottom=zoom['ymin'])
        if 'ymax' in zoom:
            skew.ax.set_ylim(top=zoom['ymax'])
        if 'xmax' in zoom:
            skew.ax.set_xlim(right=zoom['xmax'])
        if 'xmin' in zoom:
            skew.ax.set_xlim(left=zoom['xmin'])

    # Add corresponding height values for defined p-levels
    if ds_profile_OBS:
        p_levels = [p.max().m, 900, 850, 800, 750, 700, 650, 600]
        h_levels = []
        for p_lev in p_levels:
            # interpolate height values to regular p-levels
            h_lev = np.interp(p_lev, ds_profile_OBS.p.values[::-1] / 100, ds_profile_OBS.height.values[::-1])
            h_levels.append(f'{h_lev:.1f} m')
        secax = skew.ax.secondary_yaxis(0.08, functions=(lambda p: p, lambda p: p))  # make second axis
        secax.yaxis.set_major_locator(plt.FixedLocator(p_levels))  # set ticks
        secax.yaxis.set_minor_locator(plt.NullLocator())
        secax.yaxis.set_major_formatter(plt.ScalarFormatter())
        secax.yaxis.set_ticklabels(h_levels, fontsize=6)  # add labels for corresponding height
        secax.set_frame_on(False)  # remove frame
        secax.tick_params(left=False)  # remove tickslines

    # give skew to ax
    ax = skew.ax

    # make some adaptions if subplot
    if subplot:
        ax.set(title='')
    else:
        fig.tight_layout()

    return (fig, ax)


def _plot_windprofile(ds_profiles, ds_profiles_OBS=None, var='ff', pressure=False,
                      title=None,
                      zoom_h={'ymin': 0, 'ymax': 5000, 'xmin': 0, 'xmax': 16},
                      zoom_p={'ymin': 1000, 'ymax': 500, 'xmin': 0, 'xmax': 16},
                      figsize=(10., 10.),
                      colors=None,
                      fig=None, ax=None):
    '''
    function to create wind speed or wind direction profile plot

    '''

    line_colors = ['tab:blue', 'tab:orange', 'tab:green']
    xlabel = {'ff': 'wind speed (m s$^{-1}$)',
              'dd': 'wind direction (°)'}

    # create figure otherwise use existing axes
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
        subplot = False
    else:
        subplot = True

    # loop over profiles if more than one
    for i, ds_profile in enumerate(ds_profiles.values()):
        # calculate wind speed and wind direction
        u = ds_profile['u'] * units.meter_per_second
        v = ds_profile['v'] * units.meter_per_second
        ff = ds_profile['ff'] * units.meter_per_second
        dd = mpcalc.wind_direction(u, v, convention='from')

        # select param to plot
        param = ff if var == 'ff' else dd

        # plot wind profile with pressure or height as vertical axis
        if pressure:
            ax.plot(param, ds_profile['pres']/100, color=line_colors[i], label=ds_profile.modelrun, )
            ax.invert_yaxis()
            ax.set_yscale('log')
        else:
            ax.plot(param, ds_profile['Z'], color=line_colors[i], label=ds_profile.modelrun)

    linestyle = [':', '--']
    color = ['k', 'grey']
    if ds_profiles_OBS.values() is not None:
        for i, ds_profile_OBS in enumerate(ds_profiles_OBS.values()):
            if ds_profile_OBS:
                # calculate wind speed and wind direction
                u = ds_profile_OBS['u'] * units.meter_per_second
                v = ds_profile_OBS['v'] * units.meter_per_second
                ff = ds_profile_OBS['ff'] * units.meter_per_second
                dd = mpcalc.wind_direction(u, v, convention='from')

                # select param to plot
                param = ff if var == 'ff' else dd

                # plot wind profile with pressure or height as vertical axis
                if pressure:
                    if ds_profile_OBS.modelrun == 'SL88':
                        continue
                    ax.plot(param, ds_profile_OBS['p']/100, label=ds_profile_OBS.modelrun,
                            color=color[i], linestyle=linestyle[i])
                else:
                    ax.plot(param, ds_profile_OBS.height, label=ds_profile_OBS.modelrun,
                            color=color[i], linestyle=linestyle[i])

    # Add some labels and title
    title = (f'{ds_profile.attrs["station"]}: ' +
             str(ds_profile.valid_time.dt.strftime("%Y-%m-%d %H:%M UTC").values))
    ylabel = 'pressure (hPa)' if pressure else 'Height (m asl)'
    ax.set(title=title, ylabel=ylabel, xlabel=xlabel[var])
    ax.legend()
    ax.grid(True)

    # zoom active?
    zoom = zoom_p if pressure else zoom_h
    if zoom is not None:
        if 'ymin' in zoom:
            ax.set_ylim(bottom=zoom['ymin'])
        if 'ymax' in zoom:
            ax.set_ylim(top=zoom['ymax'])
        if 'xmin' in zoom:
            ax.set_xlim(left=zoom['xmin'])
        if 'xmax' in zoom:
            ax.set_xlim(right=zoom['xmax'])

        yticks_dx = -100 if pressure else 500
        yticks = np.arange(zoom['ymin'], zoom['ymax'] + yticks_dx, yticks_dx)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)

    # adapt xticks for wind direction plot
    if var == 'dd':
        xticks = np.arange(0, 361, 90)
        xtick_labels = ['N', 'E', 'S', 'W', 'N']
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

    # make some adaptions if plot is used as subplot
    if subplot:
        ax.set(ylabel='', title='')
    else:
        fig.tight_layout()

    return (fig, ax)


def _plot_profile_var(ds_profiles, ds_profile_OBS,
                      var,
                      figsize=(10., 10.),
                      colors=None):
    '''
    function to create profile plot of addressed var

    '''

    line_colors = ['tab:blue', 'tab:orange', 'tab:green']
    dict_labels = {'theta': {'xlabel': 'potential temperature (K)',
                             'xlim': (285, 320),
                             'ylim': (0, 5000)},
                   't': {'xlabel': 'temperature (K)',
                         'xlim': (270, 300),
                         'ylim': (0, 5000)}}

    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    # loop over profiles if more than one
    for i, ds_profile in enumerate(ds_profiles.values()):
        # plot parameter
        ax.plot(ds_profile[var], ds_profile['Z'], color=line_colors[i], label=ds_profile.modelrun)

    # plot observations
    if ds_profile_OBS:
        if var in ds_profile_OBS.keys():
            ax.plot(ds_profile_OBS[var], ds_profile_OBS.height, label=ds_profile_OBS.modelrun,
                    color='k', linestyle='--')

    # Add some labels and title
    title = (f'{ds_profile.attrs["station"]}: ' +
             str(ds_profile.valid_time.dt.strftime("%Y-%m-%d %H:%M UTC").values))
    ax.set(title=title, ylabel='Height (m asl)', xlabel=dict_labels[var]['xlabel'])
    ax.legend()
    ax.grid(True)

    # zoom active?
    xlim = dict_labels[var]['xlim']
    ylim = dict_labels[var]['ylim']
    ax.set(ylim=ylim, xlim=xlim)
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 500))

    return (fig, ax)


def _plot_combined_skewT(ds_profiles, ds_profiles_OBS, RS, station):
    '''
    function to combine skewT and wind profiles

    '''

    # -----------------
    # --- 1. define layout plot ---
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=(10, 2.5, 2.5),
                          left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax2.tick_params(axis='y', labelleft=False)
    ax3.tick_params(axis='y', labelleft=False)

    # -----------------
    # --- 2. do plots
    fig, ax1 = _plot_skewT(ds_profiles, ds_profiles_OBS[RS], fig=fig, ax=ax1)
    fig, ax2 = _plot_windprofile(ds_profiles, ds_profiles_OBS, var='ff', fig=fig, pressure=True, ax=ax2)
    fig, ax3 = _plot_windprofile(ds_profiles, ds_profiles_OBS, var='dd', fig=fig, pressure=True, ax=ax3,
                                 zoom_p={'xmin': 0, 'xmax': 360, 'ymin': 1000, 'ymax': 400})

    # -----------------
    # --- 3. adapt positions
    ax2.sharey(ax1)
    ax3.sharey(ax1)
    pos_ax1 = ax1.get_position().bounds
    pos_ax2 = ax2.get_position().bounds
    pos_ax3 = ax3.get_position().bounds
    pos_ax2 = [pos_ax2[0], pos_ax1[1], pos_ax2[2], pos_ax1[3]]
    pos_ax3 = [pos_ax3[0], pos_ax1[1], pos_ax3[2], pos_ax1[3]]
    ax2.set_position(pos_ax2)
    ax3.set_position(pos_ax3)

    # -----------------
    # --- 4. add title etc.
    runs = list(ds_profiles.keys())
    title = (str(ds_profiles[runs[0]].valid_time.dt.strftime("%Y-%m-%d %H:%M UTC").values) +
             f' | @{ds_profiles[runs[0]].attrs["station"]}')
    ax1.set_title(title, loc='left')

    return fig, (ax1, ax2, ax3)


def _save_plot(dir_plots, name_plot):
    '''
    saves plot to directory with defined name
    '''
    # define save path
    path_save = join(dir_plots, name_plot)
    print(f'Save Figure to: {path_save}')
    plt.savefig(path_save)
    plt.close()


def main_skewT(station, var_list, timestep, runs, with_obs=True, save=True):
    '''
    Main entry point to create skewT plots.

    Parameters
    ----------
    station : str
        station name.
    var_list : list
        list of needed parameters.
    timestep : str
        timestep to plot.
    runs : list
        list of model runs to plot.
    with_obs : bool, optional
        activate/deactivate plot of observations. The default is True.
    save : bool, optional
        activate/deactivate saving of plot. The default is True.

    '''

    # get station meta data
    station_meta = get_StationMetaData(dir_metadata, station)

    # ---- 1. get model profiles
    ds_profiles = {}
    for run in runs:
        dir_path = get_Dataset_path(run)
        ds_profiles[run] = _get_modelData(dir_path, var_list, run, timestep, station_meta)

    # ---- 2. get observation profiles
    ds_profiles_OBS = {}
    if station == 'KOLS':
        observations = ['RS', 'SL88']
        RS = 'RS'
    else:
        observations = [station]
        RS = station
    if with_obs:
        for obs in observations:
            dir_path = get_ObsDataset_path(obs)
            ds_profiles_OBS[obs] = _get_ObsDataset(dir_path, obs, timestep)
    else:
        ds_profiles_OBS[RS] = None

    # ---- 3. call skewT plotting routine
    fig, ax = _plot_skewT(ds_profiles, ds_profiles_OBS[RS])
    if save:
        name_plot = f'skewT_{station}_{timestep}_{runs}.svg'
        _save_plot(dir_plots, name_plot)

    # ---- 4. call wind profile plotting routine
    fig, ax = _plot_windprofile(ds_profiles, ds_profiles_OBS)
    if save:
        name_plot = f'Windprofile_{station}_{timestep}_{runs}.svg'
        _save_plot(dir_plots, name_plot)

    # ---- 5. make combined plot
    fig, ax = _plot_combined_skewT(ds_profiles, ds_profiles_OBS, RS, station)
    if save:
        name_plot = f'skewT_wind_{station}_{timestep}_{runs}.svg'
        _save_plot(dir_plots, name_plot)

    # ---- 6. plot some additional variable
    var = 't'
    fig, ax = _plot_profile_var(ds_profiles, ds_profiles_OBS[RS], var)
    if save:
        name_plot = f'{var}_profile_{station}_{timestep}_{runs}.svg'
        _save_plot(dir_plots, name_plot)


def main_skewT_domainForcing(station, var_list, timestep, runs, with_obs=True, save=True):
    '''
    Main entry point to create skewT plot with additional IFS data.

    Parameters
    ----------
    station : str
        station name.
    var_list : list
        list of needed parameters.
    timestep : str
        timestep to plot.
    runs : list
        list of model runs to plot.
    with_obs : bool, optional
        activate/deactivate plot of observations. The default is True.
    save : bool, optional
        activate/deactivate saving of plot. The default is True.

    '''

    # get station meta data
    station_meta = get_StationMetaData(dir_metadata, station)

    # ---- 1. get model profiles
    ds_profiles = {}
    for run in runs:
        dir_path = get_Dataset_path(run)
        ds_profiles[run] = _get_modelData(dir_path, var_list, run, timestep, station_meta)

    # ---- 2. get observations
    ds_profiles_OBS = {}
    if station == 'KOLS':
        observations = ['RS', 'SL88']
        RS = 'RS'
    elif station == 'REG':
        observations = ['RS_KUEM']
        RS = 'RS_KUEM'
    else:
        observations = [station]
        RS = station
    if with_obs:
        for obs in observations:
            dir_path = get_ObsDataset_path(obs)
            ds_profiles_OBS[obs] = _get_ObsDataset(dir_path, obs, timestep)
    else:
        ds_profiles_OBS[RS] = None

    # --- 3. get IFS Data
    dir_path = get_Coupling_IFS_path()
    ds_profile_IFS = _get_IFSData(dir_path, station, timestep)

    # ---- 4. call skewT plotting routine
    fig, ax = _plot_skewT(ds_profiles, ds_profiles_OBS[RS], ds_profile_IFS=ds_profile_IFS)
    if save:
        name_plot = f'skewT_{station}_{timestep}_{runs}_IFS.svg'
        _save_plot(dir_plots, name_plot)


# %% RS_KOLS plots - OP500, OP1000, OP2500
# define paths
runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = ['2019-09-12T23:00:00', '2019-09-13T00:00:00', '2019-09-13T03:00:00',
             '2019-09-13T06:00:00', '2019-09-13T09:00:00',
             '2019-09-13T11:00:00', '2019-09-13T12:00:00', '2019-09-13T13:00:00',
             '2019-09-13T15:00:00', '2019-09-13T17:00:00',
             '2019-09-13T20:00:00', '2019-09-13T23:00:00']

# timesteps = np.arange(np.datetime64('2019-09-12T12:00:00'),
#                       np.datetime64('2019-09-14T04:00:00'),
#                       np.timedelta64(3, 'h'))

# get station metadata
station = 'KOLS'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT(station, var_list, timestep, runs, save=False)

# %% RS_KOLS plots - OP1000, ARP1000, IFS1000
runs = ['OP1000', 'ARP1000', 'IFS1000']

# define timestep
timesteps = ['2019-09-12T23:00:00', '2019-09-13T03:00:00',
             '2019-09-13T06:00:00', '2019-09-13T09:00:00',
             '2019-09-13T11:00:00', '2019-09-13T13:00:00',
             '2019-09-13T15:00:00', '2019-09-13T17:00:00',
             '2019-09-13T20:00:00', '2019-09-13T23:00:00']


# get station metadata
station = 'KOLS'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT(station, var_list, timestep, runs)

# %% RS_LOWI plots
# define paths
runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = ['2019-09-13T03:00:00', '2019-09-14T03:00:00']

# get station metadata
station = 'RS_LOWI'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT(station, var_list, timestep, runs)


# %% RS_MUC plots
# define paths
runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = ['2019-09-12T12:00:00', '2019-09-13T00:00:00', '2019-09-13T12:00:00', '2019-09-13T18:30:00',
             '2019-09-14T00:00:00']

# get station metadata
station = 'RS_MUC'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT(station, var_list, timestep, runs, with_obs=False)


# %% RS_STUT - Plots with IFS Coupling profile

runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = np.arange(np.datetime64('2019-09-12T12:00:00'),
                      np.datetime64('2019-09-14T04:00:00'),
                      np.timedelta64(1, 'h'))

# define station
station = 'RS_STUT'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT_domainForcing(station, var_list, timestep, runs, save=True)

# %% RS_LIPI - Plots with IFS Coupling profile

runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = np.arange(np.datetime64('2019-09-12T12:00:00'),
                      np.datetime64('2019-09-14T04:00:00'),
                      np.timedelta64(1, 'h'))

# define station
station = 'RS_LIPI'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT_domainForcing(station, var_list, timestep, runs, save=True)

# %% RS_LOWL - Plots with IFS Coupling profile

runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = np.arange(np.datetime64('2019-09-12T12:00:00'),
                      np.datetime64('2019-09-14T04:00:00'),
                      np.timedelta64(1, 'h'))

# define station
station = 'RS_LOWL'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT_domainForcing(station, var_list, timestep, runs, save=True)

# %% RS_MUC - Plots with IFS Coupling profile

runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = np.arange(np.datetime64('2019-09-12T12:00:00'),
                      np.datetime64('2019-09-14T04:00:00'),
                      np.timedelta64(1, 'h'))

# define station
station = 'RS_MUC'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT_domainForcing(station, var_list, timestep, runs, save=True)

# %% RS_PAYE - Plots with IFS Coupling profile

runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = np.arange(np.datetime64('2019-09-12T12:00:00'),
                      np.datetime64('2019-09-14T04:00:00'),
                      np.timedelta64(1, 'h'))

# define station
station = 'RS_PAYE'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT_domainForcing(station, var_list, timestep, runs, save=True)


# %% RS_REGENSBURG - Plots with IFS Coupling profile

runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = np.arange(np.datetime64('2019-09-12T12:00:00'),
                      np.datetime64('2019-09-14T04:00:00'),
                      np.timedelta64(1, 'h'))

# define station
station = 'REG'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT_domainForcing(station, var_list, timestep, runs, save=True)


# %% RS KOLS - Plots with IFS Coupling profile

runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = np.arange(np.datetime64('2019-09-12T12:00:00'),
                      np.datetime64('2019-09-14T04:00:00'),
                      np.timedelta64(1, 'h'))

# define station
station = 'KOLS'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT_domainForcing(station, var_list, timestep, runs, save=True)

# %% RS ALT - Plots with IFS Coupling profile

runs = ['OP500', 'OP1000', 'OP2500']

# define timestep
timesteps = np.arange(np.datetime64('2019-09-13T21:00:00'),
                      np.datetime64('2019-09-14T04:00:00'),
                      np.timedelta64(1, 'h'))

# define station
station = 'RS_ALT'

# define variables to plot
var_list = ['t', 'q', 'u', 'v', 'z', 'pres']

# make plots @RS Launches
for timestep in timesteps:
    main_skewT_domainForcing(station, var_list, timestep, runs, save=True)
