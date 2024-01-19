#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:48:55 2023

@author: Benedikt Wibmer

Script which contains set of different parameter calculations
"""

from bronx.meteo.constants import P0, Rd, Cpd, g0, Rv
import numpy as np
import metpy.calc as mpcalc


def _calc_potTemperature(t, p):
    '''
    calculate potential temperature.
    '''
    return t * (P0 / p)**(Rd/Cpd)


def _calc_geopotHeight(z):
    '''
    calculate geopotential height.
    '''

    return z / g0


def _calc_ff(u, v):
    '''
    calculate wind speed from u and v component.
    '''

    return np.sqrt(u**2 + v**2)


def _calc_dd(u, v):
    '''
    calculate wind direction from u and v wind components

    Parameters
    ----------
    u : int/float or array of int/float
        zonal wind component (wind from W is positive, wind from E is negative)
    v : int/float or array of int/float
        meridional wind component (wind from S is positive, wind from N is negative).

    Returns
    -------
    wdir : float or array of floats
        wind direction in degree (meterological convention).
        wdir = (270 - atan2(v, u) * 180/pi)

    Notes
    -----
    N-wind = 360°
    E-wind = 90°
    S-wind = 180°
    W-wind = 270°
    calm winds (u = v = 0), function returns direction of 0°

    '''

    wdir = (270 - np.arctan2(v, u) * (180 / np.pi))
    origshape = wdir.shape

    # check for values larger than 360° (like % 360 but want 360° to be N)
    if len(origshape) == 3:
        wdir = wdir.where(wdir <= 360, wdir % 360)
        # TODO: add what happens when 0
    else:
        wdir = np.atleast_1d(wdir)
        mask = np.array(wdir > 360)
        if np.any(mask):
            wdir[mask] -= 360

        # check for calm winds
        calm_mask = (np.asanyarray(np.abs(u)) == 0.) & (np.asanyarray(np.abs(v)) == 0.)
        if np.any(calm_mask):
            wdir[calm_mask] = 0

        wdir = wdir.reshape(origshape)

    return wdir


def _calc_sw_net(sw_in, sw_out):
    '''
    calculate the sw net radiaiton
    '''
    return sw_in - sw_out


def _calc_lw_net(lw_in, lw_out):
    '''
    calculate the lw net radiaiton
    '''
    return lw_in - lw_out


def _calc_RadiationBalance(sw_net, lw_net):
    '''
    calculate radiation balance from net shortwave and longwave radiaiton.
    '''
    return sw_net + lw_net


def _calc_EnergyBalance(radBalance, shf, lhf):
    '''
    calculate energy balance from radiation balance and turbulence fluxes
    '''
    return radBalance - (shf + lhf)


def _calc_RH_from_SH(q, p, t):
    '''
    calculates relative humidity from specific humidity using the Arden Buck equation
    '''
    # calculate mixing ratio from specific humidity
    w = q / (1 - q)

    # calculate saturation vapor pressure
    e_s = _calc_saturation_vapor_pressure(t)

    # calculate saturation mixing ratio
    ws = 0.622 * e_s/(p - e_s)

    # calculate relative humidity
    return w / ws


def _calc_RH_from_AH(AH, t):
    '''
    calculates relative humidity from specific humidity using the Arden Buck equation
    '''
    # calculate saturation vapor pressure
    e_s = _calc_saturation_vapor_pressure(t)

    # calculate saturation absolute humidity
    AH_s = (e_s) / (Rv * (t + 273.15))

    # calculate relative humidity
    return AH / AH_s


def _calc_SH_from_RH(rh, p, t):
    '''
    calculates specific humidity from relative humidity
    with rh, p in hPa, t in degree Celsius
    '''

    # calculate saturation vapor pressure
    e_s = _calc_saturation_vapor_pressure(t)

    # calculate vapor pressure
    e = e_s * rh

    # calculate mixing ratio
    w = 0.622 * e / (p - e)

    # calculate specific humidity
    return w / (1 + w)


def _calc_saturation_vapor_pressure(t):
    '''
    Arden Buck equation for saturation vapor pressure for moist air

    Buck 1996:
    e_s = 6.1121 * exp((18.678 - t / 234.5) * (t / (257.14 + t)))

    with t in degree Celsius, e_s in hPa

    '''
    return 6.1121*np.exp((18.678 - t/234.5)*(t/(257.14 + t)))


def _calc_virtual_temperature(w, t):
    '''
    virtual temperature describes temperature that dry air would have if its pressure and density were 
    equal to those of given sample of moist air.

    '''
    return t * ((w + 0.622) / (0.622 * (1 + w)))


def _calc_w_from_omega(omega, p, t, q=0):
    '''
    converts vertical velocity with respect to pressure to height assuming hydrostatic conditions.

    '''

    # calculate mixing ratio from specific humidity
    w = q / (1 - q)

    # calculate density from virtual temperature
    tv = _calc_virtual_temperature(w, t)
    rho = p / (Rd * tv)

    return (omega / (-g0 * rho))


def _calc_ms_from_knots(ff_knots):
    '''
    converts wind speed from knots to m/s
    '''
    return ff_knots * 0.514


def _calc_knots_from_ms(ff_ms):
    '''
    converts wind speed from m/s to knots
    '''
    return ff_ms / 0.514


def _calc_wind_components(ff, dd, ff_units=None):
    '''
    calculates u, v wind components from wind speed (m/s) and direction.
    '''

    if ff_units == 'knots':
        ff = _calc_ms_from_knots(ff)

    u = -ff * np.sin(dd * np.pi / 180)
    v = -ff * np.cos(dd * np.pi / 180)

    return u, v


def _calc_tangentialnormalWind(u, v):
    '''
    calculates the tangential and normal wind component for cross section.
    Makes use of metpy cross_section_component function.
    '''

    t_wind, n_wind = mpcalc.cross_section_components(u, v)

    return t_wind, n_wind


def _calc_mslPressure(p, t, height):
    '''
    calculates the pressure reduced to mean sea level from height [m], pressure [hPa], temperature [K]
    t in 

    '''

    # calculate the scale height
    H = Rd * t / g0

    # calculate pmsl
    p_msl = p * np.exp(height/H)

    return p_msl


def _calc_reduce_p_toHeight(p, t, height, height_new):

    # calculate the scale height
    H = Rd * t / g0

    # calculate height difference
    dh = height - height_new

    # calculate reduced pressure
    p_red = p * np.exp(dh/H)

    return p_red
