#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:30:23 2023

@author: benwib

Script for doing statistics between station observations and model data
"""

import TS_read_stations
import xarray as xr
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from calculations import _calc_geopotHeight, _calc_ff, _calc_dd
from read_StationMeta import get_StationMetaProvider
from path_handling import get_Dataset_path, get_Dataset_name, dir_metadata
import pandas as pd
import glob2
from cartoplot_xarray import cartoplot
import cartopy.crs as ccrs
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import scipy
from adjustText import adjust_text
import seaborn as sns
from config_file import dict_extent

# set global attributes
dir_plots = '../../Plots/Station_Stats/'

dict_labeling = {'t2m': {'title': '2 m temperature',
                         'ylabel': '2 m temperature (°C)',
                         'ylim': (0, 6)},
                 'ff': {'title': '10 m wind speed',
                        'ylabel': '10 m wind speed (m s$^{-1}$)',
                        'ylim': (0, 6),
                        'xlim': (-3.5, 3.5),
                        'loc_leg': 'upper right'},
                 'dd': {'title': '10 m wind direction',
                        'ylabel': '10 m wind direction (deg)',
                        'ylim': (0, 180),
                        'xlim': (20, 130),
                        'loc_leg': 'lower right'}}


colors_station_cat = ("#88A3D2", "#8382BE", "#805CA5")
station_cat = ['flat', 'valley', 'innvalley', 'summit']


def _get_ModelTerrain(run, var='z'):
    '''
    function to get model terrain data
    '''
    # set correct paths to file
    dir_path = get_Dataset_path(run)
    name_ds = get_Dataset_name(run, 'surface', var=var)
    path_file = glob2.glob(join(dir_path, name_ds))

    # open file
    with xr.open_mfdataset(path_file, chunks={'valid_time': 1}) as ds:
        ds = ds.isel(valid_time=0)
        ds['Z'] = _calc_geopotHeight(ds['z'])

    return ds


def _get_ModelData(run, var_list=['t2m', 'r2', 'u10', 'v10']):
    '''
    function to get model data for accessed variables
    '''
    # set correct paths to file
    dir_path = get_Dataset_path(run)
    name_ds = get_Dataset_name(run, 'heightAboveGround')
    path_file = glob2.glob(join(dir_path, name_ds))

    # open file
    ds = xr.open_mfdataset(path_file, chunks={'valid_time': 1})

    # select needed variables
    try:
        ds = ds[var_list]
    except KeyError:
        raise KeyError(f'One of needed parameters {var_list} is missing in Dataset. Check Dataset!')

    return ds


def _get_StationObs(stations_meta, start_time='2019-09-12 11:51', end_time='2019-09-14 03:00',
                    resample=True, res_time='10min'):
    '''
    function to get station observations and resample it
    '''
    # load observation data
    stations_obs = {}
    for prf, provider in zip(stations_meta.prf, stations_meta.provider):
        tmp = TS_read_stations.main(prf, start_time=start_time, end_time=end_time)

        # check for needed columns and rename them to be concise
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
        # rename columns a select them from dataset
        tmp = tmp.rename(columns=columns_corr)[columns_corr.values()]

        # resample to xx-mean
        tmp = tmp.resample(res_time, closed='right', label='right').mean()

        # rename index column
        tmp.index.names = ['valid_time']

        # add it to datacontainer
        stations_obs[prf] = tmp

    return stations_obs


def _get_deltaZ(ds, stations_meta, method='linear'):
    '''
    function to get difference model terrain height and station height
    '''
    # get model data for needed coordinates
    model = []
    for stat, lon, lat in zip(stations_meta.prf, stations_meta.lon, stations_meta.lat):
        tmp = ds.interp(longitude=lon, latitude=lat, method=method)
        model.append(
            {
                'name': stat,
                'lon': float(tmp.longitude.values),
                'lat': float(tmp.latitude.values),
                'alt': float(tmp.Z.values),
            }
        )

    # convert to pandas dataframe
    df_model = pd.DataFrame(model)

    # calculate difference between model and real topography
    df_model['delta_Z'] = df_model['alt'] - stations_meta['alt']

    # add true height
    df_model['alt_true'] = stations_meta['alt']

    # add category of station
    df_model['category'] = pd.Categorical(stations_meta['category'], categories=station_cat)

    return df_model


def _interp_model_stations(ds, stations_meta, var, method='linear'):
    '''
    function to interpolate model data to station location
    '''
    # intialize pandas Dataframe
    df_model_stations = pd.DataFrame()
    df_model_stations['valid_time'] = ds.valid_time.values

    # select only needed data
    if var in ['t2m', 'r2']:
        ds_var = ds[[var]]
    elif var in ['ff', 'dd']:
        ds_var = ds[['u10', 'v10']]

    # get model data for needed variable for each station location
    for stat, lon, lat in zip(stations_meta.prf, stations_meta.lon, stations_meta.lat):
        print(f'Interpolate to station: {stat} ...')
        tmp_interp = ds_var.interp(longitude=lon, latitude=lat, method=method).load()

        # check for needed data
        if 't2m' in var:
            tmp_interp['t2m'] = tmp_interp['t2m'] - 273.15  # convert to degree celsius

        elif 'ff' in var:
            # calculate wind speed
            tmp_interp['ff'] = _calc_ff(tmp_interp['u10'], tmp_interp['v10'])
        elif 'dd' in var:
            # calculate wind direction
            tmp_interp['dd'] = (['valid_time'], _calc_dd(tmp_interp['u10'], tmp_interp['v10']))

        # select needed data and save it
        df_model_stations[stat] = tmp_interp[var]

    # set index
    df_model_stations = df_model_stations.set_index('valid_time')

    return df_model_stations


def _statistics_model_obs(df_model_stations, stations_obs, stations_meta, var):
    '''
    function to do statistics like mean error, RMSE etc. between model and station observations
    '''
    # define daytime, nighttime
    whole_day = ('12:00', '11:59')
    daytime = ('05:00', '17:30')  # sunrise @06:48 LT, sunset @19:30 LT (Kolsass)
    nighttime = ('17:30', '05:00')
    time_split = {'whole_day': whole_day,
                  'daytime': daytime,
                  'nighttime': nighttime}
    stats = {}

    # select only data within IOP8
    df_model_IOP8 = df_model_stations.loc['2019-09-13T00:00:00':'2019-09-14T00:00:00']

    # initilaize dicts
    me, mae, rmse, R, R2, ff_mean = ({}, {}, {}, {}, {}, {})
    df_obs_IOP8 = pd.DataFrame()
    for t_key, t_value in time_split.items():
        for prf in stations_meta.prf:
            # select only observations within IOP8 for current station
            tmp_obs_IOP8 = stations_obs[prf].loc['2019-09-13T00:00:00':'2019-09-14T00:00:00']
            df_obs_IOP8[prf] = tmp_obs_IOP8[var]  # save it to dataframe

            # get data depending on time period
            obs = df_obs_IOP8[prf].between_time(*t_value)
            pred = df_model_IOP8[prf].between_time(*t_value)

            # drop NaN values
            pred = pred.where(np.isfinite(obs)).dropna()
            obs = obs.dropna()

            # calculate stats whole period
            if len(obs) == 0:
                me[prf] = np.nan
                mae[prf] = np.nan
                rmse[prf] = np.nan
                R[prf] = np.nan
                ff_mean[prf] = np.nan
            else:
                diff = (pred - obs)
                if var == 'dd':
                    # need some adjustements for wind direction (maximum difference 180 degree)
                    diff = np.min([np.abs(diff), np.abs(diff + 360), np.abs(diff - 360)], axis=0)
                    ff_mean[prf] = np.mean(tmp_obs_IOP8['ff'].between_time(*t_value))
                # see Oettl et al. 2021
                me[prf] = np.mean(diff)                         # mean error (bias)
                mae[prf] = np.mean(np.abs(diff))                # mean absolute error
                rmse[prf] = np.sqrt(np.mean(diff**2))           # root mean square error
                R[prf] = scipy.stats.pearsonr(pred, obs)[0]     # pearson correlation coefficient
                R2[prf] = R[prf]**2                             # coefficient of determination

        # merge dicts to pandas DataFrame
        if var == 'dd':
            tmp = pd.DataFrame([me, mae, rmse, R, R2, ff_mean], index=[
                               'me', 'mae', 'rmse', 'R', 'R2', 'ff_mean']).T
        else:
            tmp = pd.DataFrame([me, mae, rmse, R, R2], index=['me', 'mae', 'rmse', 'R', 'R2']).T
        stats[t_key] = tmp.reset_index().rename(columns={'index': 'station'})

        # add station altitude
        stats[t_key]['alt_true'] = stations_meta['alt']

        # add station category
        stats[t_key]['category'] = pd.Categorical(stations_meta['category'], categories=station_cat)

    return stats, df_model_IOP8, df_obs_IOP8


def _get_timeseries_stats(df_model_IOP8, df_obs_IOP8, stations_meta, var):
    '''
    function fir computing temporal statistics (e.g. timeseries of mean error)
    '''
    # calculate differences between model and observations
    df_diff_IOP8 = df_model_IOP8 - df_obs_IOP8
    if var == 'dd':
        # need some adjustements for wind direction (maximum difference 180 degree)
        for col in df_diff_IOP8.columns:
            tmp_diff = df_diff_IOP8[col]
            df_diff_IOP8[col] = np.min([np.abs(tmp_diff), np.abs(
                tmp_diff + 360), np.abs(tmp_diff - 360)], axis=0)

    # get data for different categories
    tmp_cat = {}
    df_diff_cat = {}
    for cat in ['innvalley', 'summit', 'valley', 'flat']:
        if cat == 'valley':
            tmp_cat[cat] = stations_meta['prf'].loc[(stations_meta.category == cat) | (
                stations_meta.category == 'innvalley')]
        else:
            tmp_cat[cat] = stations_meta['prf'][stations_meta.category == cat]
        df_diff_cat[cat] = df_diff_IOP8[tmp_cat[cat]]

    # calculate timeseries of ME, RMSE etc.
    df_ts_stats_IOP8 = pd.DataFrame()
    df_ts_stats_IOP8['ME'] = np.mean(df_diff_IOP8, axis=1)
    df_ts_stats_IOP8['RMSE'] = np.sqrt(np.mean(df_diff_IOP8**2, axis=1))
    df_ts_stats_IOP8['MAE'] = np.mean(np.abs(df_diff_IOP8), axis=1)

    # calculate timeseries of ME, RMSE etc. depending on category
    for cat in ['innvalley', 'summit', 'valley', 'flat']:
        df_ts_stats_IOP8[f'ME_{cat}'] = np.mean(df_diff_cat[cat], axis=1)
        df_ts_stats_IOP8[f'RMSE_{cat}'] = np.sqrt(np.mean(df_diff_cat[cat]**2, axis=1))
        df_ts_stats_IOP8[f'MAE_{cat}'] = np.mean(np.abs(df_diff_cat[cat]), axis=1)

    # select daytime and nighttime
    daytime = ('05:00', '17:30')  # sunrise @06:48 LT, sunset @19:30 LT (Kolsass)
    nighttime = ('17:30', '05:00')

    df_ts_stats_IOP8_day = df_ts_stats_IOP8.between_time(*daytime)
    df_ts_stats_IOP8_night = df_ts_stats_IOP8.between_time(*nighttime)

    return df_ts_stats_IOP8, df_ts_stats_IOP8_day, df_ts_stats_IOP8_night


def _calc_EEA_EPA_crit(stats_all):
    '''
    calculation of EEA and EPA criteria
    '''
    # calculate how many stations fulfill criteria
    stats_calc = stats_all.dropna()
    n_stations = len(stats_calc)
    EEA_crit = (np.abs(stats_calc['me']) < 0.5) & (stats_calc['rmse'] < 2.0)
    EPA_crit = (np.abs(stats_calc['me']) < 1.5) & (stats_calc['rmse'] < 2.5)
    EEA_score = np.sum(EEA_crit) / n_stations * 100
    EPA_score = np.sum(EPA_crit) / n_stations * 100

    # return stations that do not fulfill EPA criterion
    stats_EPA_false = stats_calc[~EPA_crit]

    return EEA_score, EPA_score, stats_EPA_false


def _calc_hitRate_dd_crit(stats_all):
    '''
    calculation of wind direction hit rate after Oettl 2021
    '''
    stats_calc = stats_all.dropna()
    n_stations = len(stats_calc)
    hit_rate = (np.sum(stats_calc['me'] <= (46/np.maximum(stats_calc['ff_mean'], 0.5)) + 25)
                / n_stations * 100)

    return hit_rate


def _sort_dataset(df, cats=['summit', 'valley', 'flat']):
    '''
    function for sorting dataset depending on station category and station altitude
    '''
    # copy dataframe to not overwrite
    df_sorted = df.copy()

    # sort whole dataframe by only 3 categories (summit, valley, flat) and altitude
    df_sorted.loc[df_sorted.category == 'innvalley', 'category'] = 'valley'
    df_sorted = df_sorted.sort_values(['category', 'alt_true'],
                                      ascending=[False, False], ignore_index=True)
    df_sorted['category'] = pd.Categorical(df_sorted['category'].values,
                                           categories=cats)

    # extract innvalley data first
    df_innvalley = df[df.category == 'innvalley']

    return df_sorted, df_innvalley


def _plot_modelrun_info(ax, run):
    '''
    add modelrun information
    '''
    from matplotlib.offsetbox import AnchoredText
    at = AnchoredText(run, loc='upper left', prop=dict(size=12), frameon=True)
    at.patch.set(facecolor='white', edgecolor='grey', alpha=0.8)
    ax.add_artist(at)

    return ax


def _plot_terrain_deltaZ(ds, df_stations_model, extent='Stations'):
    '''
    function to plot HCS with stations and their height difference model vs. true altitude
    '''
    # set contour levels terrain
    clevels = np.arange(0, 3700, 200)

    # make terrain plot
    fig, ax, _ = cartoplot(ds['Z'], extent=dict_extent[extent], figsize=(8, 6),
                           plot_method='pcolormesh', cmap='Greys', clevels=clevels,
                           minmax=[clevels[0], clevels[-1]],
                           proj='Miller', title='', colorbar=None, extend='max')

    # define colormap boundaries
    bounds = np.arange(-250, 251, 25)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')

    # add scatter of model - real topography (delta_Z)
    # iterate over different category
    markers = ['X', 'o', 'o', '^']
    size = [40, 30, 30, 40]

    for i, stat_cat in enumerate(station_cat):
        tmp_cat = df_stations_model[df_stations_model['category'] == stat_cat]
        edgecolors = 'k' if stat_cat == 'innvalley' else 'dimgray'
        sc = ax.scatter(tmp_cat['lon'], tmp_cat['lat'], s=size[i], marker=markers[i],
                        c=tmp_cat['delta_Z'], cmap='coolwarm_r', linewidths=0.75,
                        edgecolors=edgecolors, norm=norm, transform=ccrs.PlateCarree(), zorder=10)

    # add colorbar at bottom
    colorbar_ax_kw = dict(size='3%', pad=0.4)
    cax = make_axes_locatable(ax).append_axes('bottom',
                                              axes_class=plt.Axes,
                                              **colorbar_ax_kw)
    cb = plt.colorbar(sc,
                      orientation='horizontal',
                      label='$\Delta ~ Z_{model - station}$ (m)',
                      cax=cax)

    # plot for legend
    sc_leg = {}
    for i, stat_cat in enumerate(station_cat):
        edgecolors = 'k' if stat_cat == 'innvalley' else 'dimgray'
        sc_leg[stat_cat] = ax.scatter(tmp_cat['lon'].values[0], tmp_cat['lat'].values[0], s=size[i], c='white',
                                      edgecolors=edgecolors, linewidths=0.75,
                                      marker=markers[i], label=stat_cat, zorder=-1, transform=ccrs.PlateCarree())

    ax.legend((sc_leg['flat'], sc_leg['valley'], sc_leg['innvalley'], sc_leg['summit']),
              ('flat', 'valley', 'inn valley', 'summit'),
              scatterpoints=1, loc='lower right', title='station category', alignment='left')
    # make a nice layout
    fig.tight_layout()

    return fig, ax


def _plot_stats_deltaZ(df_stations_model, run, cats=['summit', 'valley', 'flat']):
    '''
    function for plotting station statistics altitude difference (model vs. true altitude station)
    '''

    # ---- 0. data preparation ---
    # sort dataframe by altitude and use only 3 categories (flat, valley, summit)
    df_sorted, df_innvalley = _sort_dataset(df_stations_model)

    # ---- 1. define layout plot ---
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=(25, 2),
                          left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    ax2.tick_params(axis='y', labelleft=False)

    # ---- 2. make bar plot: model - real topography ---
    bar = df_sorted.plot.bar(ax=ax, x='name', y='delta_Z', color='dimgray',
                             legend=False, zorder=10, label='_nolegend_')

    # ---- 3. highlight inn valley stations ---
    index_inn = df_sorted[df_sorted['name'].isin(df_innvalley['name'])].index.values
    for i in index_inn:
        bar.get_children()[i].set_edgecolor('k')
        # ax.get_xticklabels()[i].set_fontweight('semibold')

    # ---- 4. add summary boxplot ---
    bbox_kwargs = dict(medianprops=dict(linestyle='-', linewidth=1.5, color='tab:red'),
                       meanprops=dict(marker='.', markerfacecolor='k', markeredgecolor='k', markersize=4),
                       showmeans=True,
                       fliersize=0,
                       width=.5,
                       linewidth=.75,
                       boxprops=dict(edgecolor='k', alpha=.3)
                       )
    sns.boxplot(x='category', y='delta_Z', data=df_sorted, palette=colors_station_cat,
                **bbox_kwargs, ax=ax2)
    ax2.set_xticklabels(labels=cats, rotation=90, fontdict={'fontsize': 8})
    ax2.set(xlabel='', ylabel='')

    # ---- 5. color background based on station category ---
    for i, (stat_cat, c) in enumerate(zip(cats, colors_station_cat)):
        indexes = df_sorted[df_sorted.category == stat_cat].index.values
        if len(indexes) == 0:
            continue
        xmin, xmax = (indexes.min(), indexes.max())
        label = f'{stat_cat}'
        ax.axvspan(xmin - 0.5, xmax + 0.5, alpha=0.25, color=c, label=label)

    # ---- 6. set labels, limits etc. ---
    limits = np.abs(np.round(df_sorted.delta_Z.min())) + 10
    ylimit = (-limits, limits)
    xlimit = (-0.5, len(df_sorted)-0.5)
    ax.set(ylim=ylimit,
           xlim=xlimit,
           xlabel='',
           ylabel='$\Delta ~ Z_{model - station}$ (m)')
    ax.legend()

    # ---- 7. add horizontal line at zero ---
    ax.hlines(0, xlimit[0], xlimit[1], colors='k', ls='-.', alpha=0.5, linewidth=0.2)

    # ---- 8. add vertical line at Kolsass ---
    ax.vlines(df_sorted[df_sorted.name == 'KOLS'].index, ymin=ylimit[0], ymax=ylimit[1],
              colors='tab:red', ls='--', alpha=0.5)

    # ---- 9. add stats ---
    stats = df_sorted.delta_Z.describe()
    from matplotlib.offsetbox import AnchoredText
    stat = ['max', '75%', '50%', '25%', 'min', 'mean']
    text_label = np.round(stats[stat], 2).to_string()
    at = AnchoredText(text_label, loc='lower right', prop=dict(size=10), frameon=True)
    at.patch.set(facecolor='white', edgecolor='grey', alpha=0.8)
    ax.add_artist(at)

    # ---- 10. add information about model resolution
    _plot_modelrun_info(ax, run)
    fig.tight_layout()

    return fig, ax


def _plot_stats_errors(stats, run, var, cats=['summit', 'valley', 'flat']):
    '''
    function for plotting station statistics of variable (model vs. observations)
    '''

    # -----------------
    # ---- 0. prepare data ---
    # get label and limits depending on var
    lab_lim = dict_labeling[var]

    # sort dataframe
    stats_sorted = {}
    stats_innvalley = {}
    for key in stats.keys():
        tmp_sorted, tmp_innvalley = _sort_dataset(stats[key])
        stats_sorted[key] = tmp_sorted
        stats_innvalley[key] = tmp_innvalley

    # define colors for scatter plot
    color_markers = ("#C2C2C2", "#8A8A8A", "#474747")
    markers = ['o', 'x']

    # -----------------
    # ---- 1. define layout plot ---
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=(25, 3),
                          left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    ax2.tick_params(axis='y', labelleft=False)

    # -----------------
    # ---- 2. scatter plot of bias and rmse ---
    errors = ['me', 'rmse']
    for j, err in enumerate(errors):
        for i, key in enumerate(stats_sorted.keys()):
            label = f'{err.upper()} - {key}'
            stats_sorted[key].plot.scatter(ax=ax, x='station', y=err, marker=markers[j],
                                           color=color_markers[i], label=label,
                                           zorder=10, rot=90)
    legend1 = ax.legend(ncols=6, bbox_to_anchor=(0.5, 1),
                        loc='lower center', fontsize='small')
    ax.add_artist(legend1)  # Add the legend manually to the Axes.

    # -----------------
    # ---- 3. highlight inn valley stations ---
    index_inn = stats_sorted[key][stats_sorted[key]['station'].isin(
        stats_innvalley[key]['station'])].index.values
    for i in index_inn:
        ax.get_xticklabels()[i].set_fontweight('semibold')

    # ---------------
    # ---- 4. add summary boxplot ---
    bbox_kwargs = dict(medianprops=dict(linestyle='-', linewidth=1.5, color='tab:red'),
                       meanprops=dict(marker='.', markerfacecolor='k', markeredgecolor='k', markersize=4),
                       showmeans=True,
                       fliersize=0,
                       # width=.5,
                       linewidth=1,
                       boxprops=dict(alpha=0.5))

    # plot me and rmse of inn valley stations
    df_me = pd.DataFrame()
    for key in stats_innvalley.keys():
        df_me[key] = stats_innvalley[key]['me']
    df_me['error'] = 'ME'

    df_rmse = pd.DataFrame()
    for key in stats_innvalley.keys():
        df_rmse[key] = stats_innvalley[key]['rmse']
    df_rmse['error'] = 'RMSE'

    # concat dataframes
    df_error = pd.concat([df_me, df_rmse])
    df_error = df_error.melt(id_vars=['error'], value_vars=['whole_day', 'daytime', 'nighttime'],
                             var_name='time')
    # make boxplot
    my_pal = ("#4A83BA", "#9BB3D4")
    sns.boxplot(data=df_error, x='time', y='value', hue='error', palette=my_pal,
                **bbox_kwargs, gap=.25, ax=ax2)
    plt.legend(loc='upper center', title=None, fontsize=8)

    # sns.boxplot(data=df_me, **bbox_kwargs, ax=ax2)
    ax2.set_xticklabels(labels=['whole day', 'daytime', 'nighttime'], rotation=90, fontdict={'fontsize': 8})
    ax2.set(ylabel='', xlabel='')
    ax2.set_title('inn valley', fontsize=8, fontweight='bold')

    # --------------
    # ---- 5. color background based on station category ---
    bg = []
    for i, (stat_cat, c) in enumerate(zip(cats, colors_station_cat)):
        tmp = stats_sorted['whole_day']
        indexes = tmp[tmp.category == stat_cat].index.values
        if len(indexes) == 0:
            continue
        xmin, xmax = (indexes.min(), indexes.max())
        label = f'{stat_cat}'
        bg.append(ax.axvspan(xmin - 0.5, xmax + 0.5, alpha=0.25, color=c))
    legend2 = ax.legend(bg, cats, loc='lower right')

    # -----------------
    # ---- 6. set labels, limits etc. ---
    limits = lab_lim['ylim'][1]
    ylimit = (-10, limits) if var == 'dd' else (-limits, limits)
    xlimit = (-0.5, len(stats_sorted[key])-0.5)
    ax.set(ylim=ylimit,
           xlim=xlimit,
           xlabel='',
           ylabel=lab_lim['ylabel'])

    # -----------------
    # ---- 7. add horizontal line at zero ---
    ax.hlines(0, xlimit[0], xlimit[1], colors='k', ls='--', alpha=0.5)

    # -----------------
    # ---- 8. add vertical line at Kolsass ---
    ax.vlines(stats_sorted[key][stats_sorted[key].station == 'KOLS'].index, ymin=ylimit[0], ymax=ylimit[1],
              colors='tab:red', ls='--', alpha=0.5)

    # -----------------
    # ---- 9. add information about model resolution
    _plot_modelrun_info(ax, run)
    fig.tight_layout()

    return fig, ax


def _plot_wind_stats_errors(stats, run, var, cats=['summit', 'innvalley', 'valley', 'flat']):
    '''
    function for plotting wind station statistics (EEA, EPA criterion and dd hit rate)
    '''
    # -----------------
    # ---- 0. prepare data
    # extract needed labels and limits and define some needed parameters
    lab_lim = dict_labeling[var]

    # select only statistics of whole day
    stats_all = stats['whole_day']

    # calculate EEA and EPA wind speed criterion for stations
    if var == 'ff':
        EEA_score, EPA_score, stats_EPA_false = _calc_EEA_EPA_crit(stats_all)

    # calculate hitRate dd (Oettl et al. 2021)
    if var == 'dd':
        hit_rate_dd = _calc_hitRate_dd_crit(stats_all)

    # -----------------
    # ---- 1. make the plot
    fig, ax = plt.subplots()

    # scatter plot stations me vs. rmse
    annotations = []
    markers = ['^', 'o', 'o', 'X']
    size = [30, 20, 20, 30]
    colors = ("#88A3D2", "#8382BE", "#8382BE", "#805CA5")
    for i, (stat_cat, c) in enumerate(zip(cats, colors)):

        # select stations within category
        tmp = stats_all[stats_all['category'] == stat_cat]
        if len(tmp) == 0:
            continue

        # define label
        label = stat_cat

        # do scatter plot
        edgecolors = 'k' if stat_cat == 'innvalley' else 'dimgray'
        tmp.plot.scatter(x='me', y='rmse', ax=ax, marker=markers[i], label=label, s=size[i],
                         c=c, edgecolor=edgecolors, lw=0.75, zorder=10)

        # label stations outside EPA criterion if wind speed plot
        if var == 'ff':
            tmp_outside = tmp[tmp.index.isin(stats_EPA_false.index)]
            for index in tmp_outside.index:
                annotations.append(ax.annotate(tmp_outside['station'][index],
                                               (tmp_outside['me'][index], tmp_outside['rmse'][index]),
                                               fontsize=8, fontweight='demi',
                                               color=c, zorder=10))

    # adjust scatter plot annotations in a nice way
    adjust_text(annotations, ax=ax)

    # -----------------
    # ---- 2. add title, labels etc.
    n_stations = len(stats_all.dropna())
    title = lab_lim['title'] + ' ($n_{stations}$ = ' + f'{n_stations})'
    ylabel = lab_lim['ylabel']
    ax.set(title=title,
           ylabel=f'RMSE ({ylabel})',
           xlabel=f'ME ({ylabel})')

    # -----------------
    # ---- 3. plot EEA and US-EPA quality criteria
    if var == 'ff':
        EEA = [-0.5, 0, 1, 2]  # x0, y0, width, height
        color_EEA = [0.753, 0.365, 0.365, 0.1]  # RGB values and alpha
        rect_EEA = patches.Rectangle((EEA[0], EEA[1]), EEA[2], EEA[3],
                                     edgecolor=color_EEA[0:3], facecolor=color_EEA, lw=2)
        p_EEA = ax.add_patch(rect_EEA)

        EPA = [-1.5, 0, 3, 2.5]  # x0, y0, width, height
        color_EPA = [0.318, 0.478, 0.788, 0.1]  # RGB values and alpha
        rect_EPA = patches.Rectangle((EPA[0], EPA[1]), EPA[2], EPA[3],
                                     edgecolor=color_EPA[0:3], facecolor=color_EPA, lw=2)
        p_EPA = ax.add_patch(rect_EPA)

        # add information to plot
        p_EEA.set_label(f'EEA criterion = {EEA_score:.1f}%')
        p_EPA.set_label(f'EPA criterion = {EPA_score:.1f}%')

    # -----------------
    # ---- 4. plot hit_rate dd
    if var == 'dd':
        formula = r'$\frac{1}{n} \sum_{i=0}^{n} (ME_i \leq \frac{46}{max(\overline{ff_{stat}}_i, 0.5)} + 25)$'
        text = formula + f'= {hit_rate_dd:.1f} %'
        ax.annotate(text, (22, 145), fontsize=10)

    # -----------------
    # ---- 5. set limits, legend
    ax.set(ylim=lab_lim['ylim'],
           xlim=lab_lim['xlim'])
    ax.legend(title='$category_{station}$', alignment='left', loc=lab_lim['loc_leg'])

    # -----------------
    # ---- 6. add information about model resolution
    _plot_modelrun_info(ax, run)
    fig.tight_layout()

    return fig, ax


def _plot_timeseries_stats(df_ts_stats_IOP8, df_ts_stats_IOP8_day, df_ts_stats_IOP8_night, var, run):
    '''
    function for plotting temporal statistics of ME, etc.
    '''
    # -----------------
    # ---- 1. do some calculations ---
    mean_ME = df_ts_stats_IOP8.mean()

    # -----------------
    # ---- 2. make a plot of statistic timeseries ---
    fig, ax = plt.subplots()
    for c in df_ts_stats_IOP8.columns:
        if 'ME_' in c:
            df_ts_stats_IOP8[c].plot(style='--', ax=ax, label=f'{c[3:]} - mean: {mean_ME[c]:.2f}')
        elif c == 'ME':
            df_ts_stats_IOP8[c].plot(style='-', color='k', ax=ax, label=f'all - mean: {mean_ME[c]:.2f}')

    # -----------------
    # ---- 3. add legend, labels, limits etc. ---
    ax.legend(loc='upper right', title='mean error', alignment='left')
    ylim = (-5, 180) if var == 'dd' else (-4, 4)
    ax.set(ylim=ylim,
           xlabel='UTC')
    ax.grid('both')

    # -----------------
    # ---- 4. add horizontal line at zero ---
    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], colors='k', ls='--', alpha=0.5)

    # -----------------
    # ---- 5. add information about model resolution
    _plot_modelrun_info(ax, run)
    fig.tight_layout()

    return fig, ax


def _plot_correlation(stats, var, cats=['summit', 'innvalley', 'valley', 'flat']):
    '''
    function for doing correlation plot
    '''
    # -----------------
    # ---- 0. prepare data
    # extract needed labels and limits and define some needed parameters
    lab_lim = dict_labeling[var]

    # select only statistics of whole day
    stats_all = stats['whole_day']

    # -----------------
    # ---- 1. make the plot
    fig, ax = plt.subplots()

    # scatter plot stations me vs. rmse
    markers = ['^', 'o', 'o', 'X']
    size = [30, 20, 20, 30]
    colors = ("#88A3D2", "#8382BE", "#8382BE", "#805CA5")
    for i, (stat_cat, c) in enumerate(zip(cats, colors)):

        # select stations within category
        tmp = stats_all[stats_all['category'] == stat_cat]
        if len(tmp) == 0:
            continue

        # define label
        label = stat_cat

        # do scatter plot
        edgecolors = 'k' if stat_cat == 'innvalley' else 'dimgray'
        tmp.plot.scatter(x='me', y='R2', ax=ax, marker=markers[i], label=label, s=size[i],
                         c=c, edgecolor=edgecolors, lw=0.75, zorder=10)

    # -----------------
    # ---- 2. add title, labels etc.
    n_stations = len(stats_all.dropna())
    title = lab_lim['title'] + ' ($n_{stations}$ = ' + f'{n_stations})'
    ylabel = lab_lim['ylabel']
    xlim = (0, 180) if var == 'dd' else (-4, 4)
    ax.set(title=title,
           ylabel=f'R²',
           xlabel=f'ME ({ylabel})',
           ylim=(-0.1, 1),
           xlim=xlim)

    # -----------------
    # ---- 3. plot stats
    text = (r'$\overline{ME}$' + f'= {stats_all["me"].mean():.1f}'
            + r'; $\overline{R²}$' + f'= {stats_all["R2"].mean():.1f}')
    ax.annotate(text, (-4, 0.9), fontsize=10)

    return fig, ax


def _save_stats(stats, dir_plots, run, var):
    '''
    function for saving statistics to csv file
    '''
    # get statistics depending on time period
    for key in stats.keys():
        tmp_sorted, tmp_innvalley = _sort_dataset(stats[key])

        # save statistics to csv file
        tmp_sorted.to_csv(dir_plots + f'{run}_{var}_stats_all_{key}.csv')
        tmp_innvalley.to_csv(dir_plots + f'{run}_{var}_stats_innvalley_{key}.csv')


def _save_plot(dir_plots, name_plot):
    '''
    saves plot to directory with defined name
    '''
    # define save path
    path_save = join(dir_plots, name_plot)
    print(f'Save Figure to: {path_save}')
    plt.savefig(path_save, bbox_inches='tight')
    plt.close()


def main(runs=['OP500'], provider=['ACINN', 'ZAMG', 'DWD', 'ST'], var='t2m', save=True, method='linear'):
    '''
    main entry point station statistic analysis
    '''
    # ---- 1. load station metadata
    stations_meta = get_StationMetaProvider(dir_metadata, provider)

    # ---- 2. load stations observation data
    stations_obs = _get_StationObs(stations_meta)

    # ---- 3. iterate over modelruns
    for run in runs:
        # load model terrain data
        ds_terrain = _get_ModelTerrain(run)

        # get difference model real topography for stations
        df_deltaZ = _get_deltaZ(ds_terrain, stations_meta, method=method)

        # make terrain plot
        fig, ax = _plot_terrain_deltaZ(ds_terrain, df_deltaZ)
        if save:
            name_plot = f'{run}_terrain_stations_altdiff_{method}.svg'
            _save_plot(dir_plots, name_plot)

        # make stats delta_Z plot
        fig, ax = _plot_stats_deltaZ(df_deltaZ, run)
        if save:
            name_plot = f'{run}_stations_stats_altitude_{method}.svg'
            _save_plot(dir_plots, name_plot)

        # load model data
        ds_model = _get_ModelData(run)

        # extract model data for observation sites
        df_model_stations = _interp_model_stations(ds_model, stations_meta, var, method=method)

        # do statistics
        stats, df_model_IOP8, df_obs_IOP8 = _statistics_model_obs(
            df_model_stations, stations_obs, stations_meta, var)

        # make stats error plot
        fig, ax = _plot_stats_errors(stats, run, var)
        if save:
            name_plot = f'{run}_stations_stats_errors_{var}_{method}.svg'
            _save_plot(dir_plots, name_plot)

            # save statistics
            _save_stats(stats, dir_plots, run, var)

        # make wind stats error plot
        if var in ['ff', 'dd']:
            fig, ax = _plot_wind_stats_errors(stats, run, var)
            if save:
                name_plot = f'{run}_stations_RMSEvsME_{var}_{method}.svg'
                _save_plot(dir_plots, name_plot)

                # save statistics
                _save_stats(stats, dir_plots, run, var)

        # calculate timeseries statistics and do plot
        df_ts_stats_IOP8, df_ts_stats_IOP8_day, df_ts_stats_IOP8_night = _get_timeseries_stats(
            df_model_IOP8, df_obs_IOP8, stations_meta, var)
        fig, ax = _plot_timeseries_stats(df_ts_stats_IOP8, df_ts_stats_IOP8_day,
                                         df_ts_stats_IOP8_night, var, run)
        if save:
            name_plot = f'{run}_timeseries_ME_{var}.svg'
            _save_plot(dir_plots, name_plot)

        fig, ax = _plot_correlation(stats, var)
        if save:
            name_plot = f'{run}_correlation_R2_ME_{var}.svg'
            _save_plot(dir_plots, name_plot)


# %% call main
if __name__ == '__main_':

    runs = ['OP500', 'OP1000', 'OP2500']
    var_list = ['t2m', 'ff', 'dd']
    for var in var_list:
        main(runs=runs, var=var)
