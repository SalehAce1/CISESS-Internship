# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:19:05 2016

@author: imoradi
"""
import numpy as np
    
def polar_to_cartesian(phi, theta, rho):
    # phi = lon, theta = lat, rho = alt
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    x = np.cos(phi) * np.cos(theta) * rho
    y = np.cos(theta) * np.sin(phi) * rho
    z = np.sin(theta) * rho # z is 'up'
    return x,y,z
    
    
def cartesian_to_polar(x, y, z):
    # http://keisan.casio.com/exec/system/1359533867
    r = np.sqrt(x*x + y*y + z*z)
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.arctan2(np.sqrt(x*x + y*y), z)
    lat = 90 - np.rad2deg(lat)
    return lat, lon, r

def convert_spherical_to_cartesian(radius, azimuth, elevation):
    """
    Converts spherical coordinates given in radians into Cartesian coordinates.
    """
    x = radius * np.cos(azimuth) * np.cos(elevation)
    y = radius * np.sin(azimuth) * np.cos(elevation)
    z = radius * np.ones_like(azimuth) * np.sin(elevation)
    return x, y, z
    
def cbrt(x):
    if x >= 0: 
        return pow(x, 1.0/3.0)
    else:
        return -pow(abs(x), 1.0/3.0)
        
#https://code.google.com/p/pysatel/source/browse/trunk/coord.py?r=22
def ecef2geodetic(x, y, z):
    """Convert ECEF coordinates to geodetic.
    J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates \
    to geodetic coordinates," IEEE Transactions on Aerospace and \
    Electronic Systems, vol. 30, pp. 957-961, 1994."""

    # Constants defined by the World Geodetic System 1984 (WGS84)
    a = 6378.137
    b = 6356.7523142
    esq = 6.69437999014
    e1sq = 6.73949674228
    f = 1 / 298.257223563

    r = np.sqrt(x * x + y * y)
    Esq = a * a - b * b
    F = 54 * b * b * z * z
    G = r * r + (1 - esq) * z * z - esq * Esq
    C = (esq * esq * F * r * r) / (pow(G, 3))
    S = cbrt(1 + C + np.sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = np.sqrt(1 + 2 * esq * esq * P)
    r_0 =  -(P * esq * r) / (1 + Q) + np.sqrt(0.5 * a * a*(1 + 1.0 / Q) - \
        P * (1 - esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = np.sqrt(pow((r - esq * r_0), 2) + z * z)
    V = np.sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z)
    Z_0 = b * b * z / (a * V)
    h = U * (1 - b * b / (a * V))
    lat = np.arctan((z + e1sq * Z_0) / r)
    lon = np.arctan2(y, x)
    return (np.rad2deg(lat), np.rad2deg(lon))
    
def LLHtoECEF(lat, lon, alt):
    # see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html

    rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
    f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
    cosLat = np.cos(lat)
    sinLat = np.sin(lat)
    FF     = (1.0-f)**2
    C      = 1/np.sqrt(cosLat**2 + FF * sinLat**2)
    S      = C * FF

    x = (rad * C + alt)*cosLat * np.cos(lon)
    y = (rad * C + alt)*cosLat * np.sin(lon)
    z = (rad * S + alt)*sinLat

    return (x, y, z)