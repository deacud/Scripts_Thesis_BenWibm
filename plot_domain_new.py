#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:25:42 2023

@author: benwib

Script for doing the plotting of the simulation domains
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import matplotlib.patches as mpatches
import cartopy.feature as cfeature
from os.path import join
import glob2
import xarray as xr
import salem
from path_handling import get_Dataset_path
from calculations import _calc_geopotHeight
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path as mpath


# define globals
dir_plots = '../../Plots/Domain/'

dict_domain = {'OP500': dict(lon_0=11,                      # center lon
                             lat_0=47,                      # center lat
                             llcrnrlon=7.078734536442348,   # lower left corner lon
                             llcrnrlat=44.79954912121823,   # lower left corner lat
                             urcrnrlon=15.246492245282312,  # upper right corner lon
                             urcrnrlat=49.055418102469304,  # upper right corner lat
                             ),
               'OP1000': dict(lon_0=13.8,
                              lat_0=47.4,
                              llcrnrlon=4.7815810959490666,
                              llcrnrlat=42.216679208283225,
                              urcrnrlon=24.606051869909287,
                              urcrnrlat=51.73280044114799,
                              ),
               'OP2500': dict(lon_0=13.8,
                              lat_0=47.4,
                              llcrnrlon=4.877266079477371,
                              llcrnrlat=42.306035813956804,
                              urcrnrlon=24.459185089904924,
                              urcrnrlat=51.66395695269972,
                              ),
               'IFS': dict(lon_0=17,
                           lat_0=46.24470064,
                           llcrnrlon=1.663589726833235,
                           llcrnrlat=33.45113280556127,
                           urcrnrlon=40.20435542428083,
                           urcrnrlat=55.91163933792602)}

dict_DX = {'OP500': '0.5 km',
           'OP1000': '1.0 km',
           'OP2500': '2.5 km',
           'IFS': 'IFS'}


def prepare_plot(extent=None, figsize=None, clong=0.0, clat=40.0, proj='PlateCarree'):
    '''
    Function which returns prepared axes for plot.

    Parameters
    ----------
    extent : list, optional
        list of strings of extent e.g. [lon_min, lon_max, lat_min, lat_max]. The default is None.
    figsize : tuple, optional
        size of figure. The default is None.
    clong : float, optional
        central longitude for projection. The default is 0.0.
    clat : float, optional
        central latitude for projection. The default is 40.0.
    proj : str, optional
        projection to use e.g 'PlateCarree, Lambert, EuroPP'. The default is 'PlateCarree'.

    Returns
    -------
    fig : matplotlib figure
    ax : cartopy geoaxes

    '''

    # select projection
    if proj == 'PlateCarree':
        projection = ccrs.PlateCarree(central_longitude=clong)
    elif proj == 'Lambert':
        projection = ccrs.LambertConformal(central_longitude=clong, central_latitude=clat)
    elif proj == 'EuroPP':
        projection = ccrs.EuroPP()

    # make nice figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection})

    # add some features
    ax.add_feature(cfeature.LAND.with_scale('10m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), alpha=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), alpha=0.8)
    ax.stock_img()

    # add grid lines
    xl = ax.gridlines(linestyle=':', draw_labels=True)
    xl.top_labels = False
    xl.right_labels = False

    # set extent of map
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    return fig, ax


def get_domain(ds):
    '''
    Function which creates polygon lines for domain.

    Parameters
    ----------
    ds : xarray Dataset
        dataset which contains lon/lat information.

    Returns
    -------
    polyline : numpy ndarray
        array that contains lon lat information of polygon about domain with shape (xx,2).

    '''
    # get lon/lat data
    lon = ds.longitude.data
    lat = ds.latitude.data

    # get grid information
    grid = ds.salem.grid
    nlat = grid.ny
    nlon = grid.nx
    lon0 = grid.x0
    lat0 = grid.y0

    # Define edges of regional data:
    lon_lower = lon  # from left edge to right
    lon_right = np.full(lat.shape, lon[nlon-1])
    lon_upper = lon[::-1]  # from right edge to left
    lon_left = np.full(lat.shape, lon0)

    lat_lower = np.full(lon.shape, lat0)
    lat_right = lat  # from lower to upper edge
    lat_upper = np.full(lon.shape, lat[nlat-1])
    lat_left = lat[::-1]  # from upper to lower edge

    # Generate the data for the edges of the regional grid
    line_lons = np.concatenate((lon_lower, lon_right, lon_upper, lon_left))
    line_lats = np.concatenate((lat_lower, lat_right, lat_upper, lat_left))

    # create polyline
    polyline = np.column_stack([line_lons, line_lats])

    return polyline


def _basemap_prepare_plot(extent=[-10, 30, 46, 56], figsize=None, clong=13.8, clat=47.4, proj='lcc'):
    '''
    function to prepare basemap plot
    '''
    # create figue
    fig, ax = plt.subplots(figsize=figsize)

    # add base Basemap with defined extent
    map_base = Basemap(projection=proj,
                       resolution='i',
                       lon_0=clong,
                       lat_0=clat,
                       llcrnrlon=extent[0],
                       llcrnrlat=extent[1],
                       urcrnrlon=extent[2],
                       urcrnrlat=extent[3],
                       epsg=31287,
                       ax=ax)

    # add some features
    map_base.drawcoastlines()
    map_base.drawcountries()
    map_base.drawparallels(range(-90, 90, 5), labels=[1, 0, 0, 0])
    map_base.drawmeridians(range(-180, 180, 5), labels=[0, 0, 0, 1])
    map_base.shadedrelief()

    return fig, ax, map_base


def _basemap_add_domain(fig, ax, map_base, run, color='k'):
    '''
    add matplotlib basemap domain
    '''

    # get domain parameters
    domain = dict_domain[run]
    DX = dict_DX[run]

    # create domain basemap
    map_domain = Basemap(projection='lcc',
                         resolution='i',
                         **domain,
                         )

    # get domain edge points in overview basemap
    lbx1, lby1 = map_base(*map_domain(map_domain.xmin, map_domain.ymin, inverse=True))
    ltx1, lty1 = map_base(*map_domain(map_domain.xmin, map_domain.ymax, inverse=True))
    rtx1, rty1 = map_base(*map_domain(map_domain.xmax, map_domain.ymax, inverse=True))
    rbx1, rby1 = map_base(*map_domain(map_domain.xmax, map_domain.ymin, inverse=True))

    # get vetices and define codes
    verts = [
        (lbx1, lby1),  # left, bottom
        (ltx1, lty1),  # left, top
        (rtx1, rty1),  # right, top
        (rbx1, rby1),  # right, bottom
        (lbx1, lby1),  # ignored
    ]

    codes = [mpath.MOVETO,
             mpath.LINETO,
             mpath.LINETO,
             mpath.LINETO,
             mpath.CLOSEPOLY,
             ]

    # add patch
    path = mpath(verts, codes)
    patch = mpatches.PathPatch(path, edgecolor=color, fc='none', lw=2, label=DX)
    ax.add_patch(patch)

    return fig, ax, map_domain


def domainPlot_main(runs=['OP500', 'OP1000', 'OP2500'], save=False):
    '''
    function which creates an overview with the different domains.

    Parameters
    ----------
    runs : list, optional
        list of modelruns
    save : bool, optional
        save plot
    '''

    # get polygons of domain extent
    poly_dict = {}
    for run in runs:
        file_name = f'ds_{run}_z_surface_whole.nc'
        dir_path = get_Dataset_path(run)
        filepath = join(dir_path, file_name)

        # load dataset
        ds = xr.open_dataset(filepath, chunks={}).isel(valid_time=0)
        ds['Z'] = _calc_geopotHeight(ds['z'])

        poly_dict[run] = get_domain(ds)

    # prepare figure
    extent = [-10, 25, 38, 56]
    fig, ax = prepare_plot(extent=extent, proj='EuroPP', clong=10)

    # add polyline for regional domain
    lw, fc = 2, 'y'    # linewidth, facecolor
    colors = ['k', 'C0', 'C3']  # edgecolor

    # create domain
    for lbl, c in zip(runs, colors):
        poly_domain = poly_dict[lbl]
        mp = mpatches.Polygon(poly_domain,
                              closed=True,
                              fill=False,
                              linewidth=lw,
                              edgecolor=c,
                              facecolor=fc,
                              label=lbl,
                              transform=ccrs.Geodetic())
        ax.add_patch(mp)

    # set labels and title
    ax.set_title('Domains model simulations')
    ax.legend()

    # set nice layout
    fig.tight_layout()

    # save plot
    if save:
        name_plot = f'domain_overview_{runs}.svg'
        path_save = join(dir_plots, name_plot)
        print(f'Save Figure to: {path_save}')
        plt.savefig(path_save, bbox_inches='tight')
        plt.close()


def basemap_domain_main(runs=['OP500', 'OP1000', 'OP2500', 'IFS'], save=True):
    '''
    function which creates different domains using matplotlib basemap function

    Parameters
    ----------
    runs : list, optional
        domains to plot. The default is ['OP500', 'OP1000', 'OP2500'].
    save : bool, optional
        save plot. The default is True.

    Returns
    -------
    None.

    '''
    # prepare plot
    fig, ax, map_base = _basemap_prepare_plot()

    # add domains
    map_domain = {}
    colors = ['k', 'tab:blue', 'tab:red', 'tab:orange']  # edgecolor
    for i, run in enumerate(runs):
        fig, ax, map_domain[run] = _basemap_add_domain(fig, ax, map_base, run, color=colors[i])

    # add legend and make nice layout
    ax.legend(loc='upper left')
    fig.tight_layout()

    # save plot
    if save:
        name_plot = f'domain_overview_forcing_{runs}.svg'
        path_save = join(dir_plots, name_plot)
        print(f'Save Figure to: {path_save}')
        plt.savefig(path_save, bbox_inches='tight')
        plt.close()


# %% main
if __name__ == '__main__':

    # call routine
    domainPlot_main(save=True)
    basemap_domain_main()
