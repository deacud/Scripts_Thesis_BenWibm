#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:17:15 2023

@author: Benedikt Wibmer

Script for horizontal cross section analysis and plots
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob2
from os.path import join
from calculations import (_calc_ff, _calc_geopotHeight, _calc_potTemperature,
                          _calc_w_from_omega, _calc_EnergyBalance,
                          _calc_RadiationBalance, _calc_knots_from_ms)
from path_handling import get_Dataset_path, get_Dataset_name, dir_metadata
from cartoplot_xarray import cartoplot
import cartopy.crs as ccrs  # Projections list
import salem
from read_StationMeta import get_StationMetaData
from calc_difference_runs import _calc_diff_run
from custom_colormaps import cmap_user
from scipy.ndimage import gaussian_filter
from config_file import dict_extent
from customized_barbs import barbs

# set global parameters
dir_plots = '../../Plots/HCS/'  # path of plots

dict_cbar_kwargs = {'ff': {'label': 'horizontal wind speed (m s$^{-1}$)',
                           'extend': 'max'},
                    't': {'label': 'temperature (°C)',
                          'extend': 'both'},
                    'theta': {'label': 'pot. temperature (K)',
                              'extend': 'both'},
                    'w_ms': {'label': 'vertical wind speed (m s$^{-1}$)',
                             'extend': 'both'},
                    't2m': {'label': '2m temperature (°C)',
                            'extend': 'both'},
                    'r2': {'label': '2m relative humidity (%)',
                           'extend': 'min'},
                    'sh2': {'label': '2m specific humidity (g kg$^{-1}$)',
                            'extend': 'max'},
                    'ff10': {'label': '10m wind speed (m s$^{-1}$)',
                             'extend': 'max'},
                    'q': {'label': 'specific humidity (g kg$^{-1}$)',
                          'extend': 'max'},
                    'Z': {'label': 'geopotential height (m)',
                          'extend': 'both'},
                    'tcc': {'label': 'total cloud cover',
                            'extend': 'max'},
                    }

dict_cbar_diff_kwargs = {'ff': {'label': '$\Delta$ horizontal wind speed (m s$^{-1}$)',
                                'extend': 'both'},
                         't': {'label': '$\Delta$ temperature (°C)',
                               'extend': 'both'},
                         'theta': {'label': '$\Delta$ pot. temperature (K)',
                                   'extend': 'both'},
                         'w_ms': {'label': '$\Delta$ vertical wind speed (m s$^{-1}$)',
                                  'extend': 'both'},
                         't2m': {'label': '$\Delta$ 2m temperature (°C)',
                                 'extend': 'both'},
                         'r2': {'label': '$\Delta$ 2m relative humidity (%)',
                                'extend': 'both'},
                         'sh2': {'label': '$\Delta$ 2m specific humidity (g/kg)',
                                 'extend': 'both'},
                         'ff10': {'label': '$\Delta$ 10m wind speed (m s$^{-1}$)',
                                  'extend': 'both'},
                         }

dict_cbar_diff_to_KOLS_kwargs = {'ff': {'label': '$\Delta$ horizontal wind speed to Kolsass (m s$^{-1}$)',
                                        'extend': 'both'},
                                 't': {'label': '$\Delta$ temperature to Kolsass (°C)',
                                       'extend': 'both'},
                                 'theta': {'label': '$\Delta$ pot. temperature to Kolsass (K)',
                                           'extend': 'both'},
                                 'q': {'label': '$\Delta$ specific humidity to Kolsass (g/kg)',
                                       'extend': 'both'},
                                 'w_ms': {'label': '$\Delta$ vertical wind speed to Kolsass (m s$^{-1}$)',
                                          'extend': 'both'},
                                 't2m': {'label': '$\Delta$ 2m temperature to Kolsass(°C)',
                                         'extend': 'both'},
                                 'r2': {'label': '$\Delta$ 2m relative humidity to Kolsass (%)',
                                        'extend': 'both'},
                                 'sh2': {'label': '$\Delta$ 2m specific humidity to Kolsass (g/kg)',
                                         'extend': 'both'},
                                 'ff10': {'label': '$\Delta$ 10m wind speed to Kolsass (m s$^{-1}$)',
                                          'extend': 'both'},
                                 'prmsl': {'label': '$\Delta$ msl pressure to Kolsass (hPa)',
                                           'extend': 'both'}
                                 }

dict_plotting = {'ff': {'levels': np.arange(0, 19, 1),
                        'cmap': 'magma_r'},
                 't': {'levels': np.arange(6, 31, 1),
                       'cmap': 'RdYlBu_r'},
                 'theta': {'levels': np.arange(288, 313, 1),
                           'cmap': 'RdYlBu_r'},
                 'w_ms': {'levels': np.arange(-1.5, 1.6, 0.1),
                          'cmap': 'RdBu_r'},
                 't2m': {'levels': np.arange(0, 31, 2),
                         'cmap': 'RdBu_r'},
                 'r2': {'levels': np.arange(50, 101, 10),
                        'cmap': 'Greens'},
                 'sh2': {'levels': np.arange(0, 10.1, 1),
                         'cmap': cmap_user.get('DarkMint').reversed()},
                 'ff10': {'levels': np.arange(0, 11, 1),
                          'cmap': 'magma_r'},
                 'q': {'levels': np.arange(0, 10.1, 1),
                       'cmap': cmap_user.get('DarkMint').reversed()},
                 'Z': {'levels': np.arange(1600, 1670, 5),
                       'cmap': 'RdBu_r'},
                 'tcc': {'levels': [0, 30, 50, 70, 90],
                         'cmap': cmap_user.get('Blues').reversed()}}

dict_plotting_diff = {'ff': {'levels': np.arange(-5, 5.1, 0.5),
                             'cmap': 'RdBu_r'},
                      't': {'levels': np.arange(-3, 3.1, 0.2),
                            'cmap': 'RdBu_r'},
                      'theta': {'levels': np.arange(-3, 3.1, 0.2),
                                'cmap': 'RdBu_r'},
                      'w_ms': {'levels': np.arange(-1.5, 1.6, 0.1),
                               'cmap': 'RdBu_r'},
                      't2m': {'levels': np.arange(-3, 3.1, 0.2),
                              'cmap': 'RdBu_r'},
                      'r2': {'levels': np.arange(-2, 2, 0.1),
                             'cmap': 'RdBu_r'},
                      'sh2': {'levels': np.arange(-2, 2, 0.1),
                              'cmap': 'RdBu_r'},
                      'ff10': {'levels': np.arange(-2, 2, 0.1),
                               'cmap': 'RdBu_r'}}

dict_plotting_diff_to_KOLS = {'ff': {'levels': np.arange(-5, 5.1, 0.5),
                                     'cmap': 'RdBu_r'},
                              't': {'levels': np.arange(-3, 3.1, 0.2),
                                    'cmap': 'RdBu_r'},
                              'theta': {'levels': np.arange(-3, 3.1, 0.5),
                                        'cmap': 'RdBu_r'},
                              'q': {'levels': np.arange(-3, 3, 0.1),
                                    'cmap': 'BrBG'},
                              'w_ms': {'levels': np.arange(-1.5, 1.6, 0.1),
                                       'cmap': 'RdBu_r'},
                              't2m': {'levels': np.arange(-3, 3.1, 0.2),
                                      'cmap': 'RdBu_r'},
                              'r2': {'levels': np.arange(-2, 2, 0.1),
                                     'cmap': 'RdBu_r'},
                              'sh2': {'levels': np.arange(-2, 2, 0.1),
                                      'cmap': 'RdBu_r'},
                              'ff10': {'levels': np.arange(-2, 2, 0.1),
                                       'cmap': 'RdBu_r'},
                              'prmsl': {'levels': np.arange(-4, 4.2, 0.2),
                                        'cmap': 'RdBu'}}

# clevels_isohypses = {925: np.arange(800, 1200, 5),
#                      850: np.arange(1000, 2000, 5),
#                      800: np.arange(1800, 2200, 5)}

clevels_isohypses = {300: np.arange(772, 1272, 4),
                     500: np.arange(472, 1272, 4),
                     700: np.arange(272, 1272, 4),
                     800: np.arange(72, 1272, 4),
                     850: np.arange(72, 1272, 4),
                     925: np.arange(72, 1272, 4)}

nth_gridpoint = {'OP500': {'OP500': 20,
                           'OP1000': 10,
                           'OP2500': 5,
                           'ARP1000': 10,
                           'IFS1000': 10},
                 'OP2500': {'OP500': 50,
                            'OP1000': 25,
                            'OP2500': 15,
                            'ARP1000': 25,
                            'IFS1000': 25},
                 'InnValley': {'OP500': 5,
                               'OP1000': 2,
                               'OP2500': 1,
                               'ARP1000': 2,
                               'IFS1000': 2},
                 'Kolsass': {'OP500': 2,
                             'OP1000': 1,
                             'OP2500': 1,
                             'ARP1000': 1,
                             'IFS1000': 1}}


def _getData(run, var_list, timestep, typeOfLevel, level=None):
    '''
    function to get model data on defined level type
    '''

    # ---- 1: define path to file directory
    dir_path = get_Dataset_path(run)

    # ---- 2: load data
    for i, var in enumerate(var_list):
        # get name of file
        name_ds = get_Dataset_name(run, typeOfLevel, var=var, level=level)

        # get file path
        path_file = glob2.glob(join(dir_path, name_ds))

        # open file and select timestep
        tmp = xr.open_mfdataset(path_file, chunks={'valid_time': 1}).sel(valid_time=timestep)

        # combine data
        if i == 0:
            ds = tmp
        else:
            ds = xr.merge([ds, tmp])

    # ---- 3: calculate additional parameters if possible
    keys = list(ds.keys())

    # geopotential height in m
    if 'z' in keys:
        ds['Z'] = _calc_geopotHeight(ds['z'])

    # potential temperature in K
    if level and 't' in keys:
        ds['theta'] = _calc_potTemperature(ds['t'], level*100)

    # vertical wind speed in m/s
    if level and all([x in keys for x in ['w', 't', 'q']]):
        ds['w_ms'] = _calc_w_from_omega(ds['w'], level*100, ds['t'], ds['q'])

    # horizontal wind speed in m/s
    if all([x in keys for x in ['u', 'v']]):
        ds['ff'] = _calc_ff(ds['u'], ds['v'])
    if all([x in keys for x in ['u10', 'v10']]):
        ds['ff10'] = _calc_ff(ds['u10'], ds['v10'])

    # do some unit conversion
    if 't' in keys:
        ds['t'] = ds['t'] - 273.15  # conversion to deg C
    if 't2m' in keys:
        ds['t2m'] = ds['t2m'] - 273.15  # conversion to deg C
    if 'sh2' in keys:
        ds['sh2'] = ds['sh2'] * 1000  # conversion to g/kg
    if 'q' in keys:
        ds['q'] = ds['q'] * 1000  # conversion to g/kg
    if 'prmsl' in keys:
        ds['prmsl'] = ds['prmsl'] / 100  # conversion to hPa

    return ds


def _getOrography(run, timestep):
    '''
    function to get terrain data
    '''
    # define path to file directory
    dir_path = get_Dataset_path(run)

    # load surface dataset
    name_z_surf = get_Dataset_name(run, 'surface', var='z')
    path_file = glob2.glob(join(dir_path, name_z_surf))

    ds_surf = xr.open_mfdataset(path_file, chunks={'valid_time': 1}).sel(valid_time=timestep)
    ds_surf['Z'] = _calc_geopotHeight(ds_surf['z'])

    return ds_surf


def _mask_withOrorgraphy(ds, ds_surf, ds_diff=None):
    '''
    function to mask data with orography
    '''

    # select only values which are above Orography
    mask = (ds['Z'] >= ds_surf['Z']).values

    # check if difference dataset should be masked or not
    if ds_diff is not None:
        ds_masked = ds_diff.where(mask)
    else:
        ds_masked = ds.where(mask)

    return ds_masked


def _plot_wind_arrows(ax, ds, extent, run, quiver_l=10, quiver=True):
    '''
    function to handle wind arrow plotting
    '''

    # ---- 1. define some plotting stuff
    quiverkey_kwargs = {'X': 0.95, 'Y': 1.03,
                        'U': quiver_l,
                        'label': f'{quiver_l}' + ' m s$^{-1}$',
                        'labelpos': 'W',
                        'coordinates': 'axes',
                        'labelsep': 0.05,
                        }

    barbs_kwargs = {'fill_empty': True,
                    'sizes': {'emptybarb': 0.25},
                    'length': 4,
                    'linewidth': .5}

    # check for variable names
    keys = list(ds.keys())
    u, v = ('u10', 'v10') if 'u10' in keys else ('u', 'v')

    # ---- 2. add quiver plot
    # reduce grid size to extent -> to plot nearly same grid points after resampling
    extent_list = dict_extent[extent]
    ds_extent = ds.sel(longitude=slice(extent_list[0], extent_list[1]),
                       latitude=slice(extent_list[3], extent_list[2]))

    # define how many quivers are plotted
    nx = nth_gridpoint[extent][run]  # depending on extent
    pu = ds_extent[u].isel(longitude=slice(None, None, nx), latitude=slice(None, None, nx))
    pv = ds_extent[v].isel(longitude=slice(None, None, nx), latitude=slice(None, None, nx))

    # make quiver plot
    if quiver:
        qv = ax.quiver(pu.longitude.values, pu.latitude.values, pu.values, pv.values, pivot='middle',
                       headlength=2, headaxislength=2, headwidth=2.5, width=0.00225, scale=200,
                       transform=ccrs.PlateCarree())

        # set quiverkey
        qk = ax.quiverkey(qv, **quiverkey_kwargs)
    else:
        # convert m/s to knots and plot barb plot
        pu_kn = _calc_knots_from_ms(pu)
        pv_kn = _calc_knots_from_ms(pv)
        qv = barbs(ax, pu_kn.longitude.values, pv_kn.latitude.values, pu_kn.values, pv_kn.values,
                   pivot='middle', rounding=True, transform=ccrs.PlateCarree(),
                   **barbs_kwargs)  # new barbs method

    return ax


def _plot_isohypses(ax, ds, level):
    '''
    funciton handling isohypses plotting
    '''

    # ---- 1. define some plotting stuff
    dict_sigma = {500: 20,
                  1000: 10,
                  2500: 4}
    # get contour levels
    clevels = clevels_isohypses[int(level)]
    Z = ds['Z'] / 10  # convert to dm

    # ---- 2. gaussian filter to smooth isohypses
    sigma = dict_sigma[ds.attrs['DX']]  # set sigma depending on model resolution
    Z_filt = gaussian_filter(Z, sigma=sigma)

    # ---- 3. plot isohypses
    c = ax.contour(Z.longitude.values, Z.latitude.values, Z_filt, levels=clevels, colors='whitesmoke', linewidths=1,
                   transform=ccrs.PlateCarree())
    # c = ds['Z'].plot.contour(ax=ax, levels=clevels, colors='k', linewidths=0.5, transform=ccrs.PlateCarree())
    ax.clabel(c, levels=clevels[::1], inline=True, fontsize=8)

    # Adjust linewidth
    for i, line in enumerate(c.collections):
        if i % 2 == 0:
            line.set_linewidth(1.5)  # Increase linewidth for every second line

    return ax


def _plot_station(ax, station_meta, label=None, marker_color='tab:red'):
    '''
    function to handle plotting of stations
    '''

    x = station_meta['lon']
    y = station_meta['lat']
    color = marker_color
    ax.scatter(x, y, marker='o', c=color, s=10, transform=ccrs.PlateCarree())
    if label:
        ax.annotate(station_meta['prf'], (x, y), ha='center', c=color, transform=ccrs.PlateCarree())

    return ax


def _add_terrain_isohypses(ax, ds_surf):
    '''
    function to add terrain isohypses
    '''
    # plot isohypses
    clevels = np.array([1000, 2000])
    c = ds_surf['Z'].plot.contour(ax=ax, levels=clevels, colors='tab:gray', linewidths=0.5,
                                  transform=ccrs.PlateCarree())
    # Adjust linewidth
    for i, line in enumerate(c.collections):
        if i == 1:
            line.set_linewidth(1)  # Increase linewidth for 2000 m height line

    return ax


def _add_gridpoints(ax, ds):
    '''
    function to add grid points of lon/lat mesh
    '''
    if 'x' in ds.coords:
        lon = ds.x.values
        lat = ds.y.values
    else:
        lon = ds.longitude.values
        lat = ds.latitude.values

    LON, LAT = np.meshgrid(lon, lat)

    ax.scatter(LON, LAT, s=5, color='tab:blue', transform=ccrs.PlateCarree())

    x = 11.6222
    y = 47.3053
    name = 'KOLS'
    ax.scatter(x, y, marker='o', c='tab:red', s=5, transform=ccrs.PlateCarree())
    ax.text(x, y-0.005, name, ha='center', c='tab:red', transform=ccrs.PlateCarree())

    return ax


def _add_temperature_isolines(ax, ds, var):
    '''
    function to add isolines of temperature
    '''
    c = ds[var].plot.contour(colors='tab:grey', levels=np.arange(-40, 41, 5), linewidths=0.5,
                             transform=ccrs.PlateCarree(), ax=ax)
    ax.clabel(c, levels=np.arange(-40, 41, 5), inline=True, inline_spacing=3, fontsize=8)

    return ax


def _plot_HCS_var(ds_init, ds, var, run=None, ds_surf=None, extent=None, quiver_l=10, cmap='magma_r',
                  add_quiver=None, add_isohypses=None, add_terrain_lines=None,
                  station_meta=None):
    '''
    function to plot horizontal cross section of defined variable
    '''

    # ---- 1. define some plotting properties
    cbar_kwargs = dict_cbar_kwargs[var]
    clevels, cmap = dict_plotting[var].values()  # colorbar levels, colormap
    extent_list = dict_extent[extent]

    # ---- 2. make wind speed plot
    fig, ax, result = cartoplot(ds[var], proj='Miller', figsize=(8, 5), extent=extent_list,
                                plot_method='pcolormesh', cmap=cmap, extend=cbar_kwargs['extend'],
                                clevels=clevels,
                                title='', run=run,
                                colorbar_kwargs=cbar_kwargs)

    # ---- 3. add plotting options
    if var in ['ff', 'ff10'] or add_quiver:
        ax = _plot_wind_arrows(ax, ds, extent, run)

    if add_isohypses:
        ax = _plot_isohypses(ax, ds_init, ds_init.level)

    if add_terrain_lines:
        ax = _add_terrain_isohypses(ax, ds_surf)

    if station_meta is not None:
        ax = _plot_station(ax, station_meta)

    if var == 't':
        ax = _add_temperature_isolines(ax, ds, var)

    # ---- 4. set labeling, title etc.
    if 'level' in ds.coords:
        title = '\n'.join([str(ds.valid_time.dt.strftime('%Y-%m-%d %H:%M UTC').values),
                          f'@{ds.level.values} hPa'])
    else:
        title = str(ds.valid_time.dt.strftime('%Y-%m-%d %H:%M UTC').values)
    ax.set_title(title)

    # ---- 5. make a nice layour
    fig.tight_layout()

    return fig, ax


def _plot_HCS_diff_var(ds, var, ds_surf=None, extent=None, quiver_l=10, cmap='magma_r',
                       add_quiver=None, add_isohypses=None, add_terrain_lines=None,
                       station_meta=None):
    '''
    function to plot horizontal cross section difference between two model simulations of defined variable
    '''

    # ---- 1. define some plotting properties
    cbar_kwargs = dict_cbar_diff_kwargs[var]
    clevels, cmap = dict_plotting_diff[var].values()  # colorbar levels, colormap
    extent_list = dict_extent[extent]

    # ---- 2. make plot
    fig, ax, result = cartoplot(ds[var], proj='Miller', figsize=(8, 5), extent=extent_list,
                                plot_method='pcolormesh', cmap=cmap, extend=cbar_kwargs['extend'],
                                clevels=clevels,
                                title='',
                                colorbar_kwargs=cbar_kwargs)

    # ---- 3. add plotting options
    if add_terrain_lines:
        ax = _add_terrain_isohypses(ax, ds_surf)

    if station_meta is not None:
        if isinstance(station_meta, list):
            for stat_meta in station_meta:
                ax = _plot_station(ax, stat_meta)
        else:
            ax = _plot_station(ax, station_meta)

    # ---- 4. set labeling, title etc.
    if 'level' in ds.coords:
        title = '\n'.join([f'{ds.run_ref} - {ds.run_2}: ' +
                           str(ds.valid_time.dt.strftime('%Y-%m-%d %H:%M UTC').values),
                          f'@{ds.level.values} hPa'])
    else:
        title = f'{ds.run_ref} - {ds.run_2}: ' + str(ds.valid_time.dt.strftime('%Y-%m-%d %H:%M UTC').values)
    ax.set_title(title)

    # ---- 5. make a nice layour
    fig.tight_layout()

    return fig, ax


def _plot_HCS_diff_to_KOLS(ds_diff, var, ds_init=None, ds_surf=None, run=None, extent=None, quiver_l=10, cmap='magma_r',
                           add_quiver=None, add_isohypses=None, add_terrain_lines=None,
                           station_meta=None):
    '''
    function to plot horizontal cross section of difference to Kolsass for defined variable
    '''
    # ---- 1. define some plotting properties
    cbar_kwargs = dict_cbar_diff_to_KOLS_kwargs[var]
    clevels, cmap = dict_plotting_diff_to_KOLS[var].values()  # colorbar levels, colormap
    extent_list = dict_extent[extent]

    # ---- 2. make wind speed plot
    fig, ax, result = cartoplot(ds_diff[var], proj='Miller', figsize=(8, 5), extent=extent_list,
                                plot_method='pcolormesh', cmap=cmap, extend=cbar_kwargs['extend'],
                                clevels=clevels,
                                title='', run=run,
                                colorbar_kwargs=cbar_kwargs)

    # ---- 3. add plotting options
    if var in ['ff', 'ff10'] or add_quiver:
        ax = _plot_wind_arrows(ax, ds_init, extent, run=run, quiver_l=quiver_l)

    if add_isohypses:
        ax = _plot_isohypses(ax, ds_init, ds_init.level)

    if add_terrain_lines:
        ax = _add_terrain_isohypses(ax, ds_surf)

    if station_meta is not None:
        if isinstance(station_meta, list):
            marker_color = ['tab:red', 'tab:green', 'tab:green']
            for i, station in enumerate(station_meta):
                ax = _plot_station(ax, station, marker_color=marker_color[i])
        else:
            ax = _plot_station(ax, station_meta)

    # ---- 4. set labeling, title etc.
    if ('level' in ds_diff.coords) and (ds_diff.level != 0):
        title = '\n'.join([str(ds_diff.valid_time.dt.strftime('%Y-%m-%d %H:%M UTC').values),
                          f'@{ds_diff.level.values} hPa'])
    else:
        title = str(ds_diff.valid_time.dt.strftime('%Y-%m-%d %H:%M UTC').values)
    ax.set_title(title)

    # ---- 5. make a nice layour
    fig.tight_layout()

    return fig, ax


def _save_plot(dir_plots, name_plot):
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


def main_HCS(run=None, var_list=None, var=None, timestep=None,
             typeOfLevel=None, level=None, extent=None, mask=True,
             add_quiver=None, add_isohypses=None, add_terrain_lines=None,
             save=None):
    '''
    main entry point for horizontal cross section plot

    Parameters
    ----------
    run : str, optional
        modelrun.
    var_list : list, optional
        list of needed parameters.
    var : str, optional
        variable to plot.
    timestep : str, optional
        timestep to plot.
    typeOfLevel : str, optional
        type of level on which variable is plotted.
    level : , optional
        level on which variable is plotted.
    extent : str, optional
        extent of domain of plot.
    mask : bool, optional
        mask data with orography. The default is True.
    add_quiver : bool, optional
        add quivers.
    add_isohypses : bool, optional
        add isohypses.
    add_terrain_lines : bool, optional
        add terrain lines.
    save : bool, optional
        save plot.

    Returns
    -------
    ds_masked : xarray Dataset
        Dataset used for plotting HCS
    '''

    print(f'Make HCS {run} for {var} ...')
    # ---- 1. get model data
    ds = _getData(run, var_list, timestep, typeOfLevel, level=level)

    # ---- 2. get Orography data
    ds_surf = _getOrography(run, timestep)

    if typeOfLevel != 'heightAboveGround' and mask:
        # ---- 3. mask model data
        ds_masked = _mask_withOrorgraphy(ds, ds_surf)
    else:
        ds_masked = ds

    # ---- 4. get station meta data
    station_meta = get_StationMetaData(dir_metadata, 'KOLS')

    # ---- 5. make HCS plot
    fig, ax = _plot_HCS_var(ds, ds_masked, var, run=run, ds_surf=ds_surf, extent=extent, add_quiver=add_quiver,
                            add_isohypses=add_isohypses, add_terrain_lines=add_terrain_lines,
                            station_meta=station_meta)

    # ---- 6. save plot
    if save:
        if typeOfLevel == 'isobaricInhPa':
            name_plot = f'HCS_{run}_{var}_{int(ds_masked.level.values)}hPa_{extent}_{timestep}.svg'
        elif typeOfLevel == 'heightAboveGround':
            name_plot = f'HCS_{run}_{var}_{extent}_{timestep}.svg'
        elif typeOfLevel == 'surface':
            name_plot = f'HCS_{run}_{var}_{extent}_{timestep}.svg'
        _save_plot(dir_plots, name_plot)

    return ds_masked


def main_difference_HCS(runs=None, var_list=None, var=None, timestep=None,
                        typeOfLevel=None, level=None, extent=None,
                        add_quiver=None, add_isohypses=None,  add_terrain_lines=None,
                        save=None):
    '''
    main entry point for horizontal cross section difference plot between two modelruns

    Parameters
    ----------
    runs : list, optional
        list containing the modelruns to compare.
    var_list : list, optional
        list of needed parameters.
    var : str, optional
        variable to plot.
    timestep : str, optional
        timestep to plot.
    typeOfLevel : str, optional
        type of level on which variable is plotted.
    level : , optional
        level on which variable is plotted.
    extent : str, optional
        extent of domain of plot.
    add_quiver : bool, optional
        add quivers.
    add_isohypses : bool, optional
        add isohypses.
    add_terrain_lines : bool, optional
        add terrain lines.
    save : bool, optional
        save plot.

    Returns
    -------
    ds_masked : xarray Dataset
        Dataset used for plotting HCS difference
    '''

    print(f'Make HCS difference plot {runs[0]} - {runs[1]} for {var} ...')
    # ---- 1. get model data for different runs
    ds_run = {}
    for run in runs:
        ds_run[run] = _getData(run, var_list, timestep, typeOfLevel, level=level)

    # ---- 2. calculate difference between model runs
    ds_diff = _calc_diff_run(ds_run, runs)

    # ---- 3. get Orography data
    ds_surf = _getOrography(runs[0], timestep)

    if typeOfLevel != 'heightAboveGround':
        # ---- 4. mask model data
        ds_masked = _mask_withOrorgraphy(ds_run[runs[0]], ds_surf, ds_diff=ds_diff)
    else:
        ds_masked = ds_diff

    # ---- 5. get station meta data
    station_meta = []
    for station in ['KOLS', 'RS_MUC', 'RS_STUT', 'RS_ALT', 'REG']:
        station_meta.append(get_StationMetaData(dir_metadata, station))

    # ---- 6. make HCS plot
    fig, ax = _plot_HCS_diff_var(ds_masked, var, ds_surf=ds_surf, extent=extent, station_meta=station_meta,
                                 add_terrain_lines=add_terrain_lines)

    # ---- 7. save plot
    if save:
        if typeOfLevel == 'isobaricInhPa':
            name_plot = f'HCS_diff_{runs[0]}_{runs[1]}_{var}_{int(ds_diff.level.values)}hPa_{extent}_{timestep}.svg'
        elif typeOfLevel == 'heightAboveGround':
            name_plot = f'HCS_diff_{runs[0]}_{runs[1]}_{var}_{extent}_{timestep}.svg'
        _save_plot(dir_plots, name_plot)

    return ds_masked


def main_difference_to_KOLS(run=None, var_list=None, var=None, timestep=None,
                            typeOfLevel=None, level=None, extent=None,
                            add_quiver=None, add_isohypses=None,  add_terrain_lines=None,
                            save=None):
    '''
    main entry point for horizontal cross section plot with difference to the location of Kolsass

    Parameters
    ----------
    run : str, optional
        modelrun.
    var_list : list, optional
        list of needed parameters.
    var : str, optional
        variable to plot.
    timestep : str, optional
        timestep to plot.
    typeOfLevel : str, optional
        type of level on which variable is plotted.
    level : , optional
        level on which variable is plotted.
    extent : str, optional
        extent of domain of plot.
    add_quiver : bool, optional
        add quivers.
    add_isohypses : bool, optional
        add isohypses.
    add_terrain_lines : bool, optional
        add terrain lines.
    save : bool, optional
        save plot.

    Returns
    -------
    ds_diff_masked : xarray Dataset
        Dataset used for plotting HCS difference to Kolsass
    '''

    # ---- 1. get model data for run
    ds = _getData(run, var_list, timestep, typeOfLevel, level=level)

    # --- 2. calculate difference to location Kolsass
    KOLS = get_StationMetaData(dir_metadata, 'KOLS')
    ds_KOLS = ds.interp(longitude=KOLS.lon, latitude=KOLS.lat, method='linear')
    ds_diff = ds - ds_KOLS

    # --- 3. get Orography data
    ds_surf = _getOrography(run, timestep)

    if typeOfLevel not in ['heightAboveGround', 'meanSea']:
        # ---- 4. mask model data
        ds_diff_masked = _mask_withOrorgraphy(ds, ds_surf, ds_diff)
        ds_masked = _mask_withOrorgraphy(ds, ds_surf)
    else:
        ds
        ds_diff_masked = ds_diff
        ds_masked = ds

    # ---- 5. make HCS plot
    KUF = get_StationMetaData(dir_metadata, 'KUF')  # mark also Kufstein
    ROS = get_StationMetaData(dir_metadata, 'ROS')  # mark also Rosenheim
    fig, ax = _plot_HCS_diff_to_KOLS(ds_diff_masked, var, run=run, ds_init=ds_masked, ds_surf=ds_surf,
                                     extent=extent, station_meta=[KOLS, KUF, ROS], add_quiver=add_quiver,
                                     add_isohypses=add_isohypses, add_terrain_lines=add_terrain_lines)

    # ---- 6. save plot
    if save:
        if typeOfLevel == 'isobaricInhPa':
            name_plot = f'HCS_diffToKOLS_{run}_{var}_{int(ds_diff_masked.level.values)}hPa_{extent}_{timestep}.svg'
        elif typeOfLevel in ['heightAboveGround', 'meanSea']:
            name_plot = f'HCS_diffToKOLS_{run}_{var}_{extent}_{timestep}.svg'
        _save_plot(dir_plots, name_plot)

    return ds_diff_masked


# %% --- main HCS plots ---
if __name__ == '__main_':

    # define some needed parameters
    runs = ['OP500', 'OP1000', 'OP2500']  # , 'ARP1000', 'IFS1000']
    timesteps = np.arange(np.datetime64('2019-09-12T18:00:00'),
                          np.datetime64('2019-09-14T04:00:00'),
                          np.timedelta64(3, 'h'))
    extent = 'OP500'

    # ------------------------------
    # ---- plots @925hPa
    # ------------------------------
    var_list = ['u', 'v', 'w', 'z', 't', 'q']
    level = 925
    typeOfLevel = 'isobaricInhPa'
    params = ['ff']  # ['ff', 't', 'theta', 'w_ms', 'q']
    for var in params:
        for run in runs:
            for timestep in timesteps:
                main_HCS(run=run, var_list=var_list, var=var, timestep=timestep,
                         typeOfLevel=typeOfLevel, level=level, extent=extent,
                         add_isohypses=False, save=True)

    # ------------------------------
    # ---- plots @850hPa
    # ------------------------------
    extent = 'OP500'
    var_list = ['u', 'v', 'w', 'z', 't', 'q']
    level = 850
    typeOfLevel = 'isobaricInhPa'
    params = ['t']  # ['ff', 't', 'theta']
    for var in params:
        for run in runs:
            for timestep in timesteps:
                out = main_HCS(run=run, var_list=var_list, var=var, timestep=timestep,
                               typeOfLevel=typeOfLevel, level=level, extent=extent,
                               mask=False, add_isohypses=True, save=True)

    extent = 'OP500'
    var_list = ['u', 'v', 'w', 'z', 't', 'q']
    level = 850
    typeOfLevel = 'isobaricInhPa'
    params = ['ff']  # ['ff', 't', 'theta']
    for var in params:
        for timestep in timesteps:
            out = main_HCS(run='OP2500', var_list=var_list, var=var, timestep=timestep,
                           typeOfLevel=typeOfLevel, level=level, extent=extent,
                           mask=True, add_isohypses=True, add_quiver=True, save=False)

    # -------------------------------
    # ---- plots height above ground
    # -------------------------------
    var_list = ['u10', 'v10', 't2m', 'r2', 'sh2']
    typeOfLevel = 'heightAboveGround'
    params = ['ff10', 't2m', 'r2', 'sh2']
    for var in params:
        for run in runs:
            for timestep in timesteps:
                main_HCS(run=run, var_list=var_list, var=var, timestep=timestep,
                         typeOfLevel=typeOfLevel, extent=extent,
                         add_terrain_lines=True, save=True)

    # -------------------------------
    # ---- plots surface
    # -------------------------------
    var_list = ['tcc', 'z']
    typeOfLevel = 'surface'
    params = ['tcc']
    for var in params:
        for run in runs:
            for timestep in timesteps:
                ds = main_HCS(run=run, var_list=var_list, var=var, timestep=timestep,
                              typeOfLevel=typeOfLevel, extent=extent, mask=False,
                              add_terrain_lines=True, save=True)

# %% --- main HCS difference plots ---
if __name__ == '__main_':

    timesteps = np.arange(np.datetime64('2019-09-12T18:00:00'),
                          np.datetime64('2019-09-14T04:00:00'),
                          np.timedelta64(3, 'h'))
    extent = 'OP500'
    var_list = ['u', 'v', 'w', 'z', 't', 'q']

    # ------------------------------------------------------------------------
    # ---- OP500 - OP1000, OP500 - OP2500, OP1000 - OP2500
    # ------------------------------------------------------------------------
    runs_list = [['OP500', 'OP1000'], ['OP500', 'OP2500'], ['OP1000', 'OP2500']]
    # ['OP1000', 'ARP1000'], ['OP1000', 'IFS1000']]

    # @925hPa
    # ------------------------------
    level = 925
    typeOfLevel = 'isobaricInhPa'
    params = ['theta']
    for runs in runs_list:
        for var in params:
            for timestep in timesteps:
                ds_masked = main_difference_HCS(runs=runs, var_list=var_list, var=var, timestep=timestep,
                                                typeOfLevel=typeOfLevel, level=level,
                                                extent=extent, save=True)

    # @850hPa
    # ------------------------------
    level = 850
    typeOfLevel = 'isobaricInhPa'
    params = ['t']
    for runs in runs_list:
        for var in params:
            for timestep in timesteps:
                ds_masked = main_difference_HCS(runs=runs, var_list=var_list, var=var, timestep=timestep,
                                                typeOfLevel=typeOfLevel, level=level,
                                                extent=extent, save=True)

    # @heightAboveGround
    # ------------------------------
    typeOfLevel = 'heightAboveGround'
    params = ['t2m', 'ff10']
    for runs in runs_list:
        for var in params:
            for timestep in timesteps:
                ds_masked = main_difference_HCS(runs=runs, var_list=var_list, var=var, timestep=timestep,
                                                typeOfLevel=typeOfLevel, extent=extent,
                                                add_terrain_lines=True, save=True)


# %% --- main HCS difference to Kolsass plots ---
if __name__ == '__main_':

    # define some needed parameters
    runs = ['OP500', 'OP1000', 'OP2500']
    timesteps = np.arange(np.datetime64('2019-09-12T21:00:00'),
                          np.datetime64('2019-09-14T04:00:00'),
                          np.timedelta64(3, 'h'))
    extent = 'InnValley'

    # ------------------------------
    # ---- plots @925hPa
    # ------------------------------
    var_list = ['u', 'v', 'w', 'z', 't', 'q']
    level = 925
    typeOfLevel = 'isobaricInhPa'
    params = ['theta']
    for var in params:
        for run in runs:
            for timestep in timesteps:
                main_difference_to_KOLS(run=run, var_list=var_list, var=var, timestep=timestep,
                                        typeOfLevel=typeOfLevel, level=level, extent=extent, add_quiver=True,
                                        save=True)

    # ------------------------------
    # ---- plots @900hPa
    # ------------------------------
    var_list = ['u', 'v', 'w', 'z', 't', 'q']
    level = 900
    typeOfLevel = 'isobaricInhPa'
    params = ['theta']
    for var in params:
        for run in runs:
            for timestep in timesteps:
                main_difference_to_KOLS(run=run, var_list=var_list, var=var, timestep=timestep,
                                        typeOfLevel=typeOfLevel, level=level, extent=extent, add_quiver=True,
                                        save=True)

    # ------------------------------
    # ---- plots @850hPa
    # ------------------------------
    var_list = ['u', 'v', 'w', 'z', 't', 'q']
    level = 850
    typeOfLevel = 'isobaricInhPa'
    params = ['theta']
    for var in params:
        for run in runs:
            for timestep in timesteps:
                main_difference_to_KOLS(run=run, var_list=var_list, var=var, timestep=timestep,
                                        typeOfLevel=typeOfLevel, level=level, extent=extent, add_quiver=True,
                                        save=True)

    # ------------------------------
    # ---- plots @mean sea level
    # ------------------------------
    var_list = ['prmsl']
    typeOfLevel = 'meanSea'
    params = ['prmsl']
    for var in params:
        for run in runs:
            for timestep in timesteps:
                main_difference_to_KOLS(run=run, var_list=var_list, var=var, timestep=timestep,
                                        typeOfLevel=typeOfLevel, extent=extent, add_terrain_lines=True,
                                        save=False)
