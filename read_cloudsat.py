#!/home/imoradi/packages/anaconda3/bin/python3.4
# Copyright 2013, Isaac Moradi

import os
from glob import glob
import sys
import copy
sys.path.append("/gpfsm/dnb31/imoradi/workspace/utils/pyWeather")
sys.path.append("/gpfsm/dnb31/imoradi/workspace/utils/collocate/collocation")

import numpy as np
import scipy as sp  
import scipy.io as sio  
import datetime, time
import pdb
import pandas as pd
import xarray as xr

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# do not remove this to avoid error in reading HDF
#from pyhdf.HDF import *
#from pyhdf.VS import *

def convert_to_Tb(rad, freq):
    alpha1 = 1.1910427e-05;
    alpha2 = 1.4387752;

    #Convert rad_input to mW/m^2/sr/cm^-1%    
    if np.max(rad)<1e-3:
      rad = rad*1e7;
    #end

    tb = alpha2 * freq / np.log(1+(alpha1* (freq ** 3) / rad));

    return tb


def apodize(realLW): 
    """
    Hamming apodization function for CrIS spectra. See CrIS SDR ATBD 3.7    
    """
    # apodization parameters
    Hanming_a = 0.23
    w0=Hanming_a
    w1=1.-2*Hanming_a
    w2=Hanming_a
   
    realLW = np.asarray(realLW, dtype=np.float64)
   
    shapeLW = realLW.shape
    realLW = realLW.reshape(-1, shapeLW[-1])
   
    apLW = np.zeros_like(realLW)
    apLW[:, 0] = w1*realLW[:, 0] + w0*realLW[:, 1]
    apLW[:, 1:shapeLW[-1]-1] = w0*realLW[:, 0:shapeLW[-1]-2] + w1*realLW[:, 1:shapeLW[-1]-1] + w2*realLW[:, 2:shapeLW[-1]]
    apLW[:, shapeLW[-1]-1] = w1*realLW[:, shapeLW[-1]-1] + w0*realLW[:, shapeLW[-1]-2]
   
    apLW = apLW[:, 2:shapeLW[-1]-2]
    apLW = apLW.reshape(shapeLW[:-1]+(-1, ))
           
    return apLW

# https://hdfeos.org/zoo/OTHER/2010128055614_21420_CS_2B-GEOPROF_GRANULE_P_R04_E03.hdf.py
def read_hdf_file(fname):
    from pyhdf.SD import SD, SDC
    from pyhdf import HDF, VS, V
    from scipy import interpolate

    hdf = SD(fname, SDC.READ)

    # Read datasets.
    DATAFIELD_NAME = 'Radar_Reflectivity';
    dset = hdf.select(DATAFIELD_NAME)
    data = dset[:,:]

    ht = hdf.select('Height')
    height = ht[:,:]

    # Read attributes.
    '''
    attrs = dset.attributes(full=1)
    lna=attrs["long_name"]
    long_name = lna[0]
    sfa=attrs["factor"]
    scale_factor = sfa[0]        
    vra=attrs["valid_range"]
    valid_min = vra[0][0]        
    valid_max = vra[0][1]        
    ua=attrs["units"]
    units = ua[0]
   
    attrs_h = ht.attributes(full=1)
    uah=attrs_h["units"]
    units_h = uah[0]
    '''

    h = HDF.HDF(fname)
    vs = h.vstart()

    # Read the scales
    xid = vs.find('Radar_Reflectivity.factor')
    scale_factor = np.array(vs.attach(xid)[:]).flatten()
    xid = vs.find('Radar_Reflectivity.offset')
    offset = np.array(vs.attach(xid)[:]).flatten()
    xid = vs.find('Radar_Reflectivity.valid_range')
    valid_range = np.array(vs.attach(xid)[:]).flatten()
    valid_min = valid_range[0]
    valid_max = valid_range[1]
    xid = vs.find('start_time')
    start_time = np.array(vs.attach(xid)[:]).flatten()
    start_time = datetime.datetime.strptime(start_time[0], '%Y%m%d%H%M%S')
    
    xid = vs.find('Latitude')
    latid = vs.attach(xid)
    latid.setfields('Latitude')
    nrecs, _, _, _, _ = latid.inquire()
    latitude = latid.read(nRec=nrecs)
    latid.detach()

    lonid = vs.attach(vs.find('Longitude'))
    lonid.setfields('Longitude')
    nrecs, _, _, _, _ = lonid.inquire()
    longitude = lonid.read(nRec=nrecs)
    lonid.detach()

    timeid = vs.attach(vs.find('Profile_time'))
    timeid.setfields('Profile_time')
    nrecs, _, _, _, _ = timeid.inquire()
    time = timeid.read(nRec=nrecs)
    profile_time = []
    for ss in time:
        profile_time.append(start_time+datetime.timedelta(seconds=ss[0]))

    '''
    units_t =  timeid.attr('units').get()
    longname_t = timeid.attr('long_name').get()
    '''
    timeid.detach()

    # Process valid range.
    invalid = np.logical_or(data < valid_min, data > valid_max)
    dataf = data.astype(float)
    dataf[invalid] = np.nan

    # Apply scale factor according to [1].
    dataf = dataf / scale_factor

    # interpolate to common height
    '''
    dataf1 = np.ma.array(dataf, mask=np.isnan(dataf))
    for n in np.arange(dataf1.shape[0]):
        dataf1[n,:] = np.interp(height[0,:], height[n,:], dataf1[n,:])
    '''

    # build output dictionary
    csdata = {}
    csdata['lat'] = np.squeeze(np.array(latitude))
    csdata['lon'] = np.squeeze(np.array(longitude))
    csdata['height'] = np.squeeze(np.array(height))
    csdata['time'] = np.squeeze(profile_time)
    csdata['obs'] = np.squeeze(dataf)
    #from matplotlib import pyplot as plt
    #plt.imshow(csdata['obs'],aspect=0.0100); plt.savefig('test.png');plt.close()
    #pdb.set_trace()

    return csdata

def read_cloudsat(cs_dir, dt_start, dt_end, period='hourly'):

    dt_range = pd.date_range(dt_start, dt_end, freq="1H")
    cs_fnames = []
    date = []
    for idt in dt_range:
        #2012297175936_34524_CS_2B-GEOPROF_GRANULE_P_R04_E06_ATL-18L
        fstr = '%s*_R05_*.hdf'%idt.strftime('%Y%j%H')
        f1 = glob(os.path.join(cs_dir, fstr))
        if f1: 
           cs_fnames = cs_fnames + f1 # don't change this to append
           date.append('%s'%idt.strftime('%Y%m%d%H%M'))

    if not cs_fnames: return []

    for f, d in zip(cs_fnames, date):
        print(f, d)
        cs_data = read_hdf_file(f)

        if not 'cs_data1' in vars():  # not defined yet
            cs_data1 = cs_data
        else:
            for k in (cs_data.keys() & cs_data1.keys()): # common keys
                cs_data1[k] = np.append(cs_data1[k], cs_data[k], axis=0)

    if not 'cs_data1' in vars():
        return []
    # create xarray dataset
    nobs = cs_data1['obs'].shape[0]
    nlev = cs_data1['obs'].shape[1]
    nchan = 1
    channel = np.arange(1,nchan+1)  # cloudsat has one channel
    chan_elem = np.arange(1,nlev+1) # chan_elem are different high levels that go from 1 to n_height
    nelem_chan = np.zeros(nchan) + nlev  # number of elements like number of levels in CloudSat in each channel - 1 for radiances
    obs_id = np.arange(nobs)
    cs_data = xr.Dataset()
    cs_data['obs_id'] =  xr.DataArray(obs_id, dims={'obs_id': obs_id})
    cs_data['channel'] = xr.DataArray(channel, dims={'channel': channel})
    cs_data['chan_elem'] = xr.DataArray(chan_elem, dims={'chan_elem': chan_elem})
    cs_data['nelem_chan'] = xr.DataArray(nelem_chan, dims={'channel': channel})
    for k in cs_data1:
        if ('obs' in k) or ('height' in k): 
           dims = {'obs_id': obs_id, 'chan_elem': chan_elem}
        else:          
           dims = {'obs_id': obs_id}

        cs_data[k] = xr.DataArray(cs_data1[k], dims=dims)

    # reverse the coordinates and add obs_id
    cs_data = cs_data.transpose('obs_id', 'chan_elem', 'channel')

    # rename chan to channel and  also add attributes
    attr = {}
    for i in cs_data.channel.values:
        attr['channel_id%02d'%i] = 'cloudsat%d'%i 
    cs_data.channel.attrs = attr
    
    return cs_data

