#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:29:44 2023

@author: benwib

Script for terrain plots
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from cartoplot_xarray import cartoplot
from calculations import _calc_geopotHeight
import cartopy.crs as ccrs
from read_StationMeta import get_StationMetaData, get_StationMetaProvider
from os.path import join
import xarray as xr
from path_handling import get_Dataset_path, dir_metadata, get_Dataset_name
from adjustText import adjust_text
from config_file import dict_extent
import glob2

# define global parameters
dir_plots = '../../Plots/Domain/'  # path where to save plots


def _get_ModelTerrain(run, var='z'):
    '''
    function to load model terrain data
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


def _plot_terrain(ds, extent, xticks, yticks,
                  station_meta=None, station_marker=None, station_color=None, station_label=None,
                  figsize=(6, 6),
                  save=True, dir_plots=None, name_plot=None):
    '''
    function to create terrain plot.

    Parameters
    ----------
    ds : xarray dataset
        dataset which contains information of surface geopotential height.
    extent : list
        extent with lon/lat coordinates e.g. [lon1, lon2, lat1, lat2].
    xticks : array
        array of xticks to use.
    yticks : array
        array of yticks to use.
    station_meta : list, optional
        metadata information of stations to plot. The default is None.
    figsize : tuple, optional
        The default is (6, 6).
    save : bool, optional
        activate/deactivate save of plot. The default is True.
    dir_plots : str, optional
        path to directory of plots to be saved. The default is None.
    name_plot : str, optional
        name of plot to be saved. The default is None.

    Returns
    -------
    fig, ax

    '''

    # set contour levels
    clevels = np.arange(0, 3700, 200)

    # extract information of model run
    run = ds.attrs['modelrun']

    # call plotting routine
    fig, ax, result = cartoplot(ds.Z, extent, figsize=figsize, proj='Miller',
                                plot_method='contourf', cmap='Greys', clevels=clevels,
                                minmax=[clevels[0], clevels[-1]],
                                title='', run=run, xticks=xticks, yticks=yticks,
                                colorbar_kwargs={'label': 'surface elevation (m asl)'}, extend='max',
                                station=station_meta, station_marker=station_marker,
                                station_color=station_color, station_label=station_label,
                                save=save, dir_plots=dir_plots, name_plot=name_plot)

    return fig, ax


def _plot_terrain_stations(ds, stations_meta, extent='Stations',
                           station_cat=['flat', 'valley', 'innvalley', 'summit']):
    '''
    function to create terrain plot with stations

    Parameters
    ----------
    ds : xarray Dataset
        Dataset with terrain information.
    stations_meta : pandas Dataframe
        Dataset of stations meta data.
    extent : str, optional
        extent of domain to plot. The default is 'Stations'.
    station_cat : list, optional
        list with station categorys to plot. The default is ['flat', 'valley', 'innvalley', 'summit'].

    Returns
    -------
    fig, ax

    '''

    # set contour levels terrain
    clevels = np.arange(0, 3700, 200)

    # comprise extent -> faster
    extent = dict_extent[extent]
    ds = ds.sel(longitude=slice(extent[0], extent[1]), latitude=slice(extent[3], extent[2]))

    # mask values below zero
    Z_masked = ds['Z'].where(ds['Z'] > 0, 0)

    # define colormap
    fig, ax, results = cartoplot(Z_masked, extent=extent, figsize=(8, 6), proj='Miller',
                                 plot_method='pcolormesh', cmap='Greys', clevels=clevels,
                                 minmax=[clevels[0], clevels[-1]], colorbar=None,
                                 title='', extend='max')

    # update values below zero to appear white -> does not work with xarray otherwise
    pf = results['plot_elements']
    pf.cmap.set_under('white')
    fig.canvas.draw()

    # add stations depending on category
    # iterate over different category
    markers = ['X', 'o', 'o', '^']
    size = [30, 20, 20, 30]
    colors = ("#88A3D2", "#8382BE", 'tab:red', "#805CA5")
    facecolors = ('tab:orange', 'tab:red', 'k', 'tab:blue')
    facecolors = ("#419F44", "#4D68A0", 'k', "#B76F39")
    edgecolors = 'white'

    sc = {}
    for i, stat_cat in enumerate(station_cat):
        tmp_cat = stations_meta[stations_meta['category'] == stat_cat]
        sc[stat_cat] = ax.scatter(tmp_cat['lon'], tmp_cat['lat'], s=size[i], marker=markers[i],
                                  c=facecolors[i], linewidths=0.5,
                                  edgecolors=edgecolors, transform=ccrs.PlateCarree(), zorder=10)

    # add legend
    ax.legend((sc['flat'], sc['valley'], sc['innvalley'], sc['summit']),
              ('flat', 'valley', 'inn valley', 'summit'),
              scatterpoints=1, loc='lower right', title='station category', alignment='left')
    # make a nice layout
    fig.tight_layout()

    return fig, ax


def _add_inset(ax, extent):
    '''
    function to add an inset to actual plot.

    Parameters
    ----------
    ax
    extent : list
        extent of inset with lon/lat coordinates e.g. [lon1, lon2, lat1, lat2].

    Returns
    -------
    ax

    '''

    w_inset = extent[1] - extent[0]
    h_inset = extent[3] - extent[2]
    rect = patches.Rectangle((extent[0], extent[2]), w_inset, h_inset,
                             edgecolor="tab:red", facecolor='none', lw=2,
                             transform=ccrs.PlateCarree())
    ax.add_patch(rect)

    return ax


def _add_gridpoints(ax, ds, station_meta):
    '''
    function to add grid points of lon/lat mesh

    Parameters
    ----------
    ax
    ds : xarray dataset
        dataset containing information about lon and lat.

    Returns
    -------
    ax

    '''
    lon = ds.longitude.values
    lat = ds.latitude.values

    LON, LAT = np.meshgrid(lon, lat)

    ax.scatter(LON, LAT, s=10, color='tab:blue', transform=ccrs.PlateCarree())

    text = []
    for station in station_meta:
        x = station['lon']
        y = station['lat']
        name = station['name']
        ax.scatter(x, y, marker='o', c='tab:red', s=20, transform=ccrs.PlateCarree())
        text.append(ax.text(x, y, name, ha='center', c='tab:red', transform=ccrs.PlateCarree()))

    adjust_text(text, ax=ax)

    return ax


def _add_terrain_isohypses(ax, ds):
    '''
    function to add terrain isohypses
    '''

    # plot isohypses
    clevels = np.array([1000, 2000])
    c = ds['Z'].plot.contour(ax=ax, levels=clevels, colors='tab:grey', linewidths=0.5, add_labels=False,
                             transform=ccrs.PlateCarree())
    # Adjust linewidth
    for i, line in enumerate(c.collections):
        if i == 1:
            line.set_edgecolor('k')
            line.set_alpha(0.5)
            # line.set_linewidth(1)  # Increase linewidth for 2000 m height line

    return ax


def _save_plot(dir_plots, name_plot):
    '''
    saves plot to directory with defined name
    '''
    # define save path
    path_save = join(dir_plots, name_plot)
    print(f'Save Figure to: {path_save}')
    plt.savefig(path_save, bbox_inches='tight')
    plt.close()


def main_terrain(run='OP1000'):
    '''
    main entry point terrain plot without stations

    Parameters
    ----------
    run : str, optional
        modelrun. The default is 'OP1000'.

    Returns
    -------
    ds : xarray Dataset
        Dataset with terrain information.

    '''

    # set needed parameters
    stations = ['KOLS', 'UNI']
    station_marker = ['o', 'o']
    station_color = ['k', 'tab:red']
    station_label = [True, True]
    var = 'Z'  # gepotential height

    # get topography from surface geopotential
    print('Load data...')
    ds = _get_ModelTerrain(run)

    # load metadata stations
    station_meta = []
    for station in stations:
        station_meta.append(get_StationMetaData(dir_metadata, station))

    # -----------------------------------------------
    # ---- Inn Valley
    # -----------------------------------------------
    print('Create Inn Valley plot...')
    name_plot = f'{run}_InnValley_{var}.pdf'  # name of plot

    # set the extent of the map
    extent_Innvalley = dict_extent['InnValley']
    xticks = np.arange(extent_Innvalley[0], extent_Innvalley[1], 0.2)
    yticks = np.arange(extent_Innvalley[2], extent_Innvalley[3], 0.2)

    # make terrain plot Inn valley
    fig, ax = _plot_terrain(ds, extent_Innvalley, xticks, yticks,
                            station_meta=station_meta, station_marker=station_marker,
                            station_color=station_color, station_label=station_label,
                            figsize=(6, 6),
                            save=True, dir_plots=dir_plots, name_plot=name_plot)

    # -----------------------------------------------
    # ---- whole extent
    # -----------------------------------------------
    print('Create whole extent plot...')
    name_plot = f'{run}_domain_{var}.pdf'  # name of plot

    # set the extent of the map
    extent = [ds.longitude.min()-0.5, ds.longitude.max()+0.5, ds.latitude.min()-0.5, ds.latitude.max()+0.5]
    xticks = np.arange(6, 23, 2)
    yticks = np.arange(44, 51, 2)

    # make terrain plot
    fig, ax = _plot_terrain(ds, extent, xticks, yticks,
                            figsize=(8, 5),
                            save=False, dir_plots=dir_plots, name_plot=name_plot)

    # add inset
    _add_inset(ax, extent_Innvalley)

    plt.savefig(join(dir_plots, name_plot))
    plt.close()

    # -----------------------------------------------
    # ---- extent Kolsass with gridpoints
    # -----------------------------------------------
    print('Create Kolsass plot with gridpoints...')
    name_plot = f'{run}_Kolsass_{var}_withgridpoints.pdf'

    # set the extent of the map
    extent_KOLS = dict_extent['Kolsass']
    xticks = np.arange(extent_KOLS[0], extent_KOLS[1], 0.05)
    yticks = np.arange(extent_KOLS[2], extent_KOLS[3], 0.05)

    # make terrain plot Kolsass
    fig, ax = _plot_terrain(ds, extent_KOLS, xticks, yticks,
                            figsize=(6, 6),
                            save=False, dir_plots=dir_plots, name_plot=name_plot)

    # add grid points and stations
    stations = get_StationMetaProvider(dir_metadata, provider=['ACINN']).prf
    station_meta = []
    for station in stations:
        station_meta.append(get_StationMetaData(dir_metadata, station))
    ax = _add_gridpoints(ax, ds, station_meta)

    # add isohypses terrain
    # ax = add_terrain_isohypses(ax, ds)

    plt.savefig(join(dir_plots, name_plot))
    plt.close()

    return ds


def main_terrain_stations(run='OP500', provider=['ACINN', 'ZAMG', 'DWD', 'ST'], extent='OP500', save=True):
    '''
    main entry point to plot terrain overview with stations

    Parameters
    ----------
    run : str, optional
        modelrun. The default is 'OP500'.
    provider : list, optional
        list of providers for stations. The default is ['ACINN', 'ZAMG', 'DWD', 'ST'].
    extent : str, optional
        extent of domain. The default is 'OP500'.
    save : bool, optional
        activate/deactivate save option of plot. The default is True.

    Returns
    -------
    None.

    '''
    # load station metadata
    stations_meta = get_StationMetaProvider(dir_metadata, provider)

    # get model terrain data
    ds_terrain = _get_ModelTerrain(run)

    # make plot
    fig, ax = _plot_terrain_stations(ds_terrain, stations_meta, extent=extent)

    # add isohypses
    # ax = add_terrain_isohypses(ax, ds_terrain)

    # add inn valley inset
    extent_Innvalley = dict_extent['InnValley']
    ax = _add_inset(ax, extent_Innvalley)

    # color axes frame
    ax.patch.set(lw=2, ec='k')

    if save:
        name_plot = f'{run}_terrain_withStations.pdf'
        _save_plot(dir_plots, name_plot)


# %% --- OP500 ---
ds_OP500 = main_terrain(run='OP500')
main_terrain_stations(run='OP500')

# %% --- OP1000 ---
ds_OP1000 = main_terrain(run='OP1000')

# %% --- ARP1000 ---
ds_ARP1000 = main_terrain(run='ARP1000')

# %% --- IFS1000 ---
ds_IFS1000 = main_terrain(run='IFS1000')

# %% --- OP2500 ---
ds_OP2500 = main_terrain(run='OP2500')
