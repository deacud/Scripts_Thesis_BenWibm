#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:44:09 2023

@author: benwib

Script handling the plotting of 2D (lon, lat) data of xarray Dataset
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
import numpy as np
from os.path import join
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from scalebar import scale_bar


# define some global parameters needed for plotting
default_NEfeatures = [dict(category='cultural',
                           name='admin_0_countries_lakes',
                           facecolor='none',
                           edgecolor='k',
                           scale='10m'), ]

default_scatter_kw = {'s': 20,
                      'marker': ',',
                      'linewidths': 0}
default_contour_kw = {'linewidths': 1}
default_contourf_kw = {'add_colorbar': False,
                       'rasterized': True}
default_pcolormesh_kw = {'add_colorbar': False,
                         'rasterized': True}
default_imshow_kw = {'interpolation': None,
                     # 'aspect': 'auto',
                     'origin': 'upper',
                     'add_colorbar': False}
default_clabel_kw = {'fmt': '%0i'}
default_gridlines_kw = {'draw_labels': True,
                        'x_inline': False,
                        'y_inline': False,
                        'linewidth': 1,
                        'color': 'k',
                        'linestyle': '--',
                        'alpha': 0.3,
                        'rotate_labels': False}


def _cartoplot_init(extent=None, figsize=None, facecolor='gainsboro',
                    projection=ccrs.PlateCarree(),
                    NE_features=default_NEfeatures,
                    xticks=None, yticks=None):
    """
    This function returns prepared axes for the regional plot.
    """

    # create figure
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection, facecolor=facecolor)

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # add defined Natural Earth features
    for feat in default_NEfeatures:
        ax.add_feature(cfeature.NaturalEarthFeature(**feat))

    gl = ax.gridlines(**default_gridlines_kw)
    gl.top_labels = False
    gl.right_labels = False

    if xticks is not None:
        gl.xlocator = mticker.FixedLocator(xticks)
    if yticks is not None:
        gl.ylocator = mticker.FixedLocator(yticks)

    # # add x and y ticks if PlateCarree projection
    # if xticks is not None and yticks is not None and isinstance(projection, type(ccrs.PlateCarree())):
    #     ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    #     ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    return fig, ax, projection


def _cartoplot_treat_minmax(da, minmax):
    '''
    function to handle min and max values
    '''

    # 1: get min/max for plot either from user or from data
    pmin = da.data.min()
    pmax = da.data.max()

    if minmax is not None:
        try:
            pmin = float(minmax[0])
        except ValueError:
            pass
        try:
            pmax = float(minmax[1])
        except ValueError:
            pass

    return pmin, pmax


def _cartoplot_shape(x, y, data, plot_method):
    """
    shape data according to plot_method
    """

    xf = x
    yf = y
    zf = data

    if plot_method == 'scatter':
        # flatten data for scatter
        xf = x.flatten()
        yf = y.flatten()
        zf = data.flatten()

    return xf, yf, zf


def _cartoplot_actualPlot(da, ax,
                          contourlabel,
                          plot_method,
                          plot_kwargs):
    '''
    function which handles the actual plotting depending on the plot method
    '''

    if plot_method == 'contourf':
        plot_kwargs.update(default_contourf_kw)
        pf = da.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), **plot_kwargs)

    elif plot_method == 'pcolormesh':
        plot_kwargs.update(default_pcolormesh_kw)
        pf = da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), **plot_kwargs)

    elif plot_method == 'imshow':
        plot_kwargs.update(default_imshow_kw)
        pf = da.plot.imshow(ax=ax, transform=ccrs.PlateCarree(), **plot_kwargs)

    elif plot_method == 'contour':
        plot_kwargs.update(default_contour_kw)
        pf = da.plot.contour(ax=ax, transform=ccrs.PlateCarree(), **plot_kwargs)
        if contourlabel:
            clabel_kw = default_clabel_kw
            ax.clabel(pf, **clabel_kw)
    else:
        raise NotImplementedError(f'plot_method=="{plot_method}"')

    return pf


def _cartoplot_colorbar(ax, pf,
                        colorbar,
                        colorbar_ax_kw,
                        colorbar_kwargs):
    """
    function for handling the colorbar
    """

    # define colorbar
    if colorbar_ax_kw is None:
        colorbar_ax_kw = dict(size='3%', pad=0.2)
    cax = make_axes_locatable(ax).append_axes(colorbar,
                                              axes_class=plt.Axes,
                                              **colorbar_ax_kw)
    # get orientation
    orientation = 'vertical' if colorbar in ('right', 'left') else 'horizontal'

    # get ticks
    # TODO

    # make colorbar
    cb = plt.colorbar(pf,
                      orientation=orientation,
                      cax=cax,
                      **colorbar_kwargs)
    return cb


def _cartoplot_text(da,
                    ax,
                    title,
                    run):
    """
    function to handle title, textboxes etc..
    """
    if title is None:
        title = "\n".join([da.long_name, str(da.valid_time.dt.strftime('%Y-%m-%d %H:%M UTC').values)])
    ax.set_title(title)

    # add information about model resolution
    if run is not None:

        # add textbox
        from matplotlib.offsetbox import AnchoredText
        at = AnchoredText(run, loc='upper left', prop=dict(size=12), frameon=True)
        at.patch.set(facecolor='white', edgecolor='grey', alpha=0.8)
        ax.add_artist(at)

    return ax


def _cartoplot_station(ax, station, kwargs, label):
    '''
    function to add station locations
    '''
    x = station['lon']
    y = station['lat']
    ax.scatter(x, y, marker=kwargs['marker'], c=kwargs['color'],
               s=20, zorder=10, transform=ccrs.PlateCarree())
    if label:
        ax.annotate(station['prf'], (x, y), ha='center', c=kwargs['color'], transform=ccrs.PlateCarree())

    return ax


def _cartoplot_scalebar(ax, length=75.5, location=(0.1, 0.1), linewidth=3):
    '''
    function to add a scalebar
    '''

    ax = scale_bar(ax, (0.1, 0.1), length=length, linewidth=linewidth)

    return ax


def _cartoplot_save(dir_plots, name_plot):
    '''
    function to save plot
    '''
    path_save = join(dir_plots, name_plot)
    plt.savefig(path_save)


def cartoplot(da,
              extent=None,
              figsize=None,
              proj='PlateCarree',
              plot_method='contourf',
              minmax=None,
              contourlabel=True,
              title=None,
              run=None,
              xticks=None,
              yticks=None,
              # colormapping
              cmap='viridis',
              clevels=None,
              extend='both',
              normalize=None,
              # colorbar
              colorbar='right',
              colorbar_ax_kw=None,
              colorbar_kwargs={},
              # station
              station=None,
              station_marker=['o', 's', '^', 'v'],
              station_color=['k', 'tab:red', 'tab:blue', 'tab:green'],
              station_label=None,
              # scalebar
              scalebar=None,
              # saving
              save=False,
              dir_plots=None,
              name_plot=None):
    '''
    main entry point which calls subfunctions depending on plotting options
    '''

    # ---- 1: prepare figure
    # select projection
    clong = 11.4
    clat = 47.25
    if proj == 'PlateCarree':
        projection = ccrs.PlateCarree(central_longitude=clong)
    elif proj == 'Lambert':
        projection = ccrs.LambertConformal(central_longitude=clong, central_latitude=clat)
    elif proj == 'EuroPP':
        projection = ccrs.EuroPP()
    elif proj == 'Geodetic':
        projection = ccrs.Geodetic()
    elif proj == 'Miller':
        projection = ccrs.Miller(central_longitude=clong)
    elif proj == 'Mercator':
        projection = ccrs.Mercator(central_longitude=clong)
    elif proj == 'TransverseMercator':
        projection = ccrs.TransverseMercator(central_longitude=clong, central_latitude=clat)

    fig, ax, projection = _cartoplot_init(extent, figsize=figsize, projection=projection,
                                          xticks=xticks, yticks=yticks)

    result = dict(fig=fig, ax=ax)

    # ---- 2: get the min/max
    pmin, pmax = _cartoplot_treat_minmax(da, minmax)

    # ---- 3: colormapping
    # norm = TwoSlopeNorm(vmin=pmin, vcenter=0, vmax=pmax)

    plot_kwargs = dict(cmap=cmap,
                       vmin=pmin,
                       vmax=pmax,
                       levels=clevels,
                       extend=extend)

    # ---- 4: make the actual plot
    pf = _cartoplot_actualPlot(da, ax, contourlabel, plot_method, plot_kwargs)
    # pf.cmap.set_under('white')
    result['plot_elements'] = pf

    # ---- 5: colorbar
    if colorbar and plot_method != 'contour':
        cb = _cartoplot_colorbar(ax,
                                 pf,
                                 colorbar,
                                 colorbar_ax_kw,
                                 colorbar_kwargs)
        result['colorbar'] = cb

    # ---- 6: texts
    ax = _cartoplot_text(da, ax, title, run)

    # ---- 7: station location
    if station is not None:
        marker = station_marker
        color = station_color
        label = station_label
        for i, stat in enumerate(station):
            station_kwargs = dict(marker=marker[i], color=color[i])
            ax = _cartoplot_station(ax, stat, station_kwargs, label=label)

    # ---- 8: add a scalebar
    if scalebar:
        ax = _cartoplot_scalebar(ax)

    # ---- 9: set nice layout
    fig.tight_layout()

    # ---- 10: save plot
    if save is True and dir_plots is not None and name_plot is not None:
        _cartoplot_save(dir_plots, name_plot)
        plt.close()
    elif save is True and dir_plots is None:
        print('Need to specify directory where figure should be saved.')
    elif save is True and name_plot is None:
        print('Need to specify name of plot.')

    return fig, ax, result
