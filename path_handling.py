#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:39:08 2023

@author: Benedikt Wibmer

File which controls path handling
"""

from os.path import join, dirname

# parent directory of model runs
#dir_parent = '/media/benwib/Benni_SSD/MasterThesis/Data_AROME/'  # !!!Adapt!!!
dir_parent = '/perm/aut0883/claef1k/DATA/20190912/12/MEM_00/'
# parent directory of observation data
#dir_CROSSINN = '/home/benwib/Studium/Master_Thesis/Data/CROSSINN/'  # !!!Adapt!!!
dir_CROSSINN = '/perm/aut0883/data/CROSSINN/'


# directories of different model runs
#folder_run = dict(OP500='0.5km/',
#                  OP1000='1.0km/OP/',
#                  ARP1000='1.0km/ARP/',
#                  IFS1000='1.0km/IFS/',
#                  OP2500='2.5km/')
folder_run = dict(OP1000='')

# directories of observation datasets
dir_OBS = dict(RS=join(dir_CROSSINN, 'DATA_IOP8/RS/IOP8/'),
               SL88=join(dir_CROSSINN, 'DATA_IOP8/Lidars_vertical/Lidar_SL88/'),
               WLS200s=join(dir_CROSSINN, 'DATA_IOP8/Lidar_VCS/'),
               MWR=join(dir_CROSSINN, 'DATA_IOP8/MWR/'),
               RS_LOWI=join(dir_CROSSINN, 'DATA_IOP8/RS_other/LOWI/'),
               RS_MUC=join(dir_CROSSINN, 'DATA_IOP8/RS_other/MUC/'),
               RS_STUT=join(dir_CROSSINN, 'DATA_IOP8/RS_other/STUT/'),
               RS_IDAR=join(dir_CROSSINN, 'DATA_IOP8/RS_other/IDAR/'),
               RS_PRAG=join(dir_CROSSINN, 'DATA_IOP8/RS_other/PRAG/'),
               RS_PAYE=join(dir_CROSSINN, 'DATA_IOP8/RS_other/PAYE/'),
               RS_LIPI=join(dir_CROSSINN, 'DATA_IOP8/RS_other/LIPI/'),
               RS_LOWL=join(dir_CROSSINN, 'DATA_IOP8/RS_other/LOWL/'),
               RS_KUEM=join(dir_CROSSINN, 'DATA_IOP8/RS_other/KUEM/'),
               RS_ALT=join(dir_CROSSINN, 'DATA_IOP8/RS_other/ALT/')
               )

# directory station metadata
dir_metadata = join(dir_CROSSINN, 'DATA_IOP8/Surface_station/')  # path station metadata

# directories of IFS Coupling profiles
folder_IFS = 'IFS_Coupling_Profiles/'


def get_GRIB_path(run):
    '''
    return path to grib files

    Parameters
    ----------
    run : str
        modelrun.
    '''
    return join(dir_parent, folder_run[run], 'GRIB2')


def get_Dataset_path(run):
    '''
    return path to netcdf Datasets

    Parameters
    ----------
    run : str
        modelrun.
    '''
    return dirname(get_GRIB_path(run))


def get_ObsDataset_path(obs):
    '''
    return path to observation Datasets

    Parameters
    ----------
    obs : str
        observation e.g. RS, SL88, MWR, RS_LOWI, RS_MUC
    '''
    return dir_OBS[obs]


def get_Coupling_IFS_path():
    '''
    return path to profiles extracted from IFS coupling files

     Parameters
     ----------
     run : str
         modelrun.
     '''
    return join(dir_parent, folder_IFS)


def set_Dataset_name(ds, typeOfLevel, var=None, level=None):
    '''
    set Dataset name depending on options

    Parameters
    ----------
    ds : xarray Dataset
        Dataset to get saving name
    typeOfLevel : str
        type of model level
    var : str, optional
        parameter name. The default is None.
    level : int, optional
        pressure level. The default is None.
    '''
    if typeOfLevel == 'surface':
        ds_name = f'ds_{ds.modelrun}_{var}_{typeOfLevel}_{ds.domain}.nc'

    elif typeOfLevel == 'hybridPressure':
        ds_name = f'ds_{ds.modelrun}_{var}_{typeOfLevel}_{ds.domain}.nc'

    elif typeOfLevel == 'heightAboveGround':
        ds_name = f'ds_{ds.modelrun}_{typeOfLevel}_instant_{ds.domain}.nc'

    elif typeOfLevel == 'isobaricInhPa':
        lev = f'{level}hPa_' if level else ''
        ds_name = f'ds_{ds.modelrun}_{var}_{typeOfLevel}_{lev}{ds.domain}.nc'

    elif typeOfLevel == 'meanSea':
        ds_name = f'ds_{ds.modelrun}_{var}_{typeOfLevel}_{ds.domain}.nc'

    return ds_name


def get_Dataset_name(run, typeOfLevel, var=None, level=None, extent=None):
    '''
    get Dataset name depending on options

    Parameters
    ----------
    run : str
        modelrun
    typeOfLevel : str
        type of model level
    var : str, optional
        parameter name. The default is None.
    level : int, optional
        pressure level. The default is None.
    extent : str, optional
        extent for hybrid pressure. e.g. whole, or MUC
    '''
    if typeOfLevel == 'surface':
        ds_name = f'ds_{run}_{var}_{typeOfLevel}*.nc'

    elif typeOfLevel == 'hybridPressure':
        if extent == 'whole':
            ds_name = f'ds_{run}_{var}_{typeOfLevel}_[10*.nc'
        elif extent == 'MUC':
            ds_name = f'ds_{run}_{var}_{typeOfLevel}_[11*.nc'
        else:
            ds_name = f'ds_{run}_{var}_{typeOfLevel}*.nc'

    elif typeOfLevel == 'heightAboveGround':
        ds_name = f'ds_{run}_{typeOfLevel}_instant*.nc'

    elif typeOfLevel == 'isobaricInhPa':
        lev = f'{level}hPa_' if level else '['
        ds_name = f'ds_{run}_{var}_{typeOfLevel}_{lev}*.nc'

    elif typeOfLevel == 'meanSea':
        ds_name = f'ds_{run}_{var}_{typeOfLevel}*.nc'

    return ds_name


def set_interpStation_name(ds, typeOfLevel='hybridPressure'):
    '''
    set interpolation Dataset name

    Parameters
    ----------
    ds : xarray Dataset
        Dataset to get saving name
    typeOfLevel : str, optional
        type of model level. Default: 'hybridPressure'
    '''
    if typeOfLevel == 'hybridPressure':
        coords = tuple(ds.attrs['coords'])
        ds_name = f'ds_{ds.modelrun}_interp_{typeOfLevel}_{coords}.nc'

    return ds_name


def get_interpStation_name(run, coords, typeOfLevel='hybridPressure'):
    '''
    get interpolation Dataset name

    Parameters
    ----------
    run : str
        modelrun
    coords : tuple
        coordinates
    typeOfLevel : str, optional
        type of model level. Default: 'hybridPressure'
    '''
    if typeOfLevel == 'hybridPressure':
        ds_name = f'ds_{run}_interp_{typeOfLevel}_{coords}.nc'

    return ds_name
