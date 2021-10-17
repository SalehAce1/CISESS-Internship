import os
from os import fsencode, fsdecode, listdir
import datetime as dt
from ftplib import FTP_TLS
import numpy as np
from mayavi import mlab
import convert
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import netCDF4


DEBUG = True
base_dir = "./pub/gpmdata/"
base_file = "2A.GPM.DPRX.V8-20200326."
link = "arthurhou.pps.eosdis.nasa.gov"

def log(o: object):
    if not DEBUG: return
    print("[DEBUG]: ", o)

def get_files(datetime: str, dest: str, username: str, password: str):
    """Gets hdf files from the GPM FTP server based on date

    Args:
        date (str): Date of the files for the day you want in yyyy/mm/dd format
        dest (str): Destination folder we want to download to
        username (str): GPM username
        password (str): GPM password
    """

    # Convert string date to datetime
    form = "%Y/%m/%d"
    date = dt.datetime.strptime(datetime, form)
    # Making base file names we want to get
    target_name = "{}{}{:02d}{:02d}-".format(base_file, date.year, date.month, date.day)
    # Get files for that day
    ftp = FTP_TLS(link)
    ftp.login(user=username, passwd=password)
    ftp.cwd(base_dir + datetime + "/Xradar/")
    file_names = ftp.nlst()
    for file_name in file_names:
        if target_name not in file_name: continue
        local_filename = os.path.join(dest, file_name)
        if os.path.exists(local_filename):
            log("Skipping " + file_name)
            continue
        file = open(local_filename, 'wb')
        ftp.retrbinary('RETR '+ file_name, file.write)
        log("Downloaded " + file_name)
        file.close()
    ftp.quit()

def compress_files(in_dir, out_dir):
    """Compresses HDF5 files into nc files, storing only the data we need

    Args:
        in_dir (str): Path to where the HDF5 files are.
        out_dir (str): path to where the nc files should be written to.
    """
    data_path = in_dir
    done_path = out_dir
    directory = fsencode(data_path)
    for file in listdir(directory):
        target = data_path + fsdecode(file)
        print("Doing", target)
        data = netCDF4.Dataset(target, diskless=True, persist=False)

        swath = data["FS"]
        pre = swath["PRE"]
        lon = swath["Longitude"]
        lat = swath["Latitude"]
        obs = pre["zFactorMeasured"]
        alt = pre["height"]
        scan = swath["ScanTime"]
        year = scan["Year"]
        month = scan["Month"]
        day = scan["DayOfMonth"]
        hour = scan["Hour"]
        min = scan["Minute"]
        sec = scan["Second"]

        new_dataset = netCDF4.Dataset(done_path + fsdecode(file), 'w', format='NETCDF4')
        new_dataset.createDimension("nscan", obs[:].shape[0])
        new_dataset.createDimension("nray", obs[:].shape[1])
        new_dataset.createDimension("nbin", obs[:].shape[2])
        #new_dataset.createDimension("nfreq", obs[:].shape[3])

        new_lon = new_dataset.createVariable("Longitude", np.float32, dimensions=("nscan", "nray"))
        new_lon[:] = lon[:]
        new_lat = new_dataset.createVariable("Latitude", np.float32, dimensions=("nscan", "nray"))
        new_lat[:] = lat[:]

        new_h = new_dataset.createVariable("height", np.float32, dimensions=("nscan", "nray", "nbin"))
        new_h[:] = alt[:]
        new_obs = new_dataset.createVariable("zFactorMeasured", np.float32, dimensions=("nscan", "nray", "nbin"))
        new_obs[:] = obs[:, :, :, 0]

        newY = new_dataset.createVariable("Year", np.short, dimensions=("nscan"))
        newY[:] = year[:]
        newM = new_dataset.createVariable("Month", np.byte, dimensions=("nscan"))
        newM[:] = month[:]
        newD = new_dataset.createVariable("DayOfMonth", np.byte, dimensions=("nscan"))
        newD[:] = day[:]
        newH = new_dataset.createVariable("Hour", np.byte, dimensions=("nscan"))
        newH[:] = hour[:]
        newMin = new_dataset.createVariable("Minute", np.byte, dimensions=("nscan"))
        newMin[:] = min[:]
        newSec = new_dataset.createVariable("Second", np.byte, dimensions=("nscan"))
        newSec[:] = sec[:]
        
        data.close()
        new_dataset.close()
        
        filename = done_path + fsdecode(file)
        comp = dict(zlib=True, complevel=5)
        xr_new = xr.open_dataset(filename)
        encoding = {var: comp for var in xr_new.variables}
        filename = filename.replace('HDF5', 'nc')
        xr_new.to_netcdf(filename, encoding=encoding)
        xr_new.close()

def plot_earth_wrapped(earth_tex_path="./Textures/EarthMap_2500x1250.jpg", star_tex_path="./Textures/starmap.png"):
    """Plots a 3D model of the earth with a starmap in the background in Mayavi.

    Args:
        earth_tex_path (str): Path to the earth texture, file type should be jpg.
        star_tex_path (str): Path to the star texture, file type should be png. 
    
    Returns:
        The mlab figure the earth is plotted on.
    """

    from tvtk.api import tvtk
    from mayavi.sources.api import BuiltinSurface

    earth_r = 1
    stars_r = 50
    eps     = 1e-4
    mlab.figure(1, size=(1920, 1080))
    fig = mlab.gcf()
    mlab.clf()
    
    fig.scene.disable_render = True

    # plot earth
    earth_flat = BuiltinSurface(source='plane')
    earth_flat.data_source.set(
                        origin=(earth_r, np.pi-eps, -np.pi),
                        point1=(earth_r, np.pi-eps,  np.pi), 
                        point2=(earth_r, eps,       -np.pi),
                        x_resolution=74, 
                        y_resolution=38,
                    ) 

      
    earth_round = mlab.pipeline.user_defined(earth_flat, 
            filter=tvtk.TransformPolyDataFilter(transform=tvtk.SphericalTransform())
            )

    
    earth = mlab.pipeline.surface(earth_round)
    earth_img       = tvtk.JPEGReader(file_name=earth_tex_path)
    earth_texture = tvtk.Texture(input_connection=earth_img.output_port,
                              interpolate=1)
    earth.actor.actor.texture = earth_texture

    
    # plot stars
    stars_flat = BuiltinSurface(source='plane')
    stars_flat.data_source.set(
                                    origin=(stars_r, np.pi-eps, -np.pi),
                                    point1=(stars_r, np.pi-eps,  np.pi), 
                                    point2=(stars_r, eps,       -np.pi),
                                    x_resolution=37, 
                                    y_resolution=19,
                                )

    
    stars_round = mlab.pipeline.user_defined(stars_flat,
            filter=tvtk.TransformPolyDataFilter(transform=tvtk.SphericalTransform())
            )

    
    stars = mlab.pipeline.surface(stars_round)
    stars_img     = tvtk.PNGReader(file_name=star_tex_path)
    stars_texture = tvtk.Texture(input_connection=stars_img.output_port,
                              interpolate=1)
    stars.actor.actor.texture = stars_texture

    # Plot some circles onto earth for orientation
    theta = np.linspace(0, 2*np.pi, 100)
    above_earth_fac = 1.001
    for angle_degree in (-60, -30, 0, 30, 60):
        angle = angle_degree * np.pi / 180
        x, y, z = convert.convert_spherical_to_cartesian(above_earth_fac, theta, angle)
        mlab.plot3d(x, y, z, color=(1, 1, 1), opacity=0.1, tube_radius=None)

    
    for angle_degree in (90, 0):
        angle = angle_degree * np.pi / 180
        x, y, z = convert.convert_spherical_to_cartesian(above_earth_fac, angle, theta)
        mlab.plot3d(x, y, z, color=(1, 1, 1), opacity=0.1, tube_radius=None)

    return fig

def plot_gpm_data(data, fig, step=1, save_location="./3d_images/", file_type=".HDF5"):
    """Given GPM dataset in HDF5 format and a figure to plot it on, it plots the reflectivity data in step steps, and saves each step to save_location.

    Args:
        data (netCDF4 Dataset): 2A.GPM.DPRX.V8 data from GPM.
        fig (mlab figure): The figure the data should be plotted on.
        step (int): Number of steps it should take to plot the whole figure, by default equals 1 which means the entire data will be plotted in one step.
        save_location (str): Where the screenshot in each step should be saved to.
        file_type (str): Raw ".HDF5" file or compressed ".nc" file. 
    """
    obs = lat = lon = st = None

    if file_type == ".HDF5":
        swath = data["FS"]
        pre = swath["PRE"]
        # Reflectivity data
        obs = pre["zFactorMeasured"][:, :, :, 0].T
        # Latitude data
        lat = swath["Latitude"][:]
        # Longitude data
        lon = swath["Longitude"][:]
        # Times data
        st = data["FS"]["ScanTime"]
    elif file_type == ".nc":
        # Reflectivity data
        obs = data["zFactorMeasured"][:].T
        # Latitude data
        lat = data["Latitude"][:]
        # Longitude data
        lon = data["Longitude"][:]
        # Times data
        st = data

    # Limiting the reflectivity by xmin and xmax and scaling it by 0.01
    xmin = 0
    xmax = np.inf
    obs[obs <= xmin] = xmin 
    obs[obs >= xmax] = xmax
    obs = obs * 0.01
    # Compressing reflectivity heights into 1 layer
    obs_mean = np.nanmean(obs, axis=0)

    # Calculating the number of steps
    total_len = obs.shape[2]
    step_size = int(total_len/step)

    # Slight repositioning of latitude data to make the camera center on the head of where the satellite is at currently
    c_lat = lat.max() - lat + 20

    # Loop for the number of steps we need to do
    for i in range(0, total_len, step_size):
        a,b = i, i + step_size
        # Calculate for each footprint
        for footprint_ind in range(49):
            # Convert lon-lat data in current step and footprint to cartesian
            x, y, z = convert.polar_to_cartesian(lon[a:b, footprint_ind], lat[a:b, footprint_ind], 1.001)
            # Get reflectivity data for current step and footprint
            s = obs_mean[footprint_ind, a:b]

            # TODO Implement VIL instead of just taking mean
            # s = obs[:, footprint_ind, a:b]
            # h = alt[:, footprint_ind, a:b]
            # Calculating vertically integrated liquid (VIL)
            # z_mean =  0.5*(s[:-1] + s[1:])
            # dh = np.abs(h[1:] - h[:-1])
            # vil = np.nansum((3.44 * np.power(10.0, -6.0) * np.power(z_mean, 4.0/7.0)) * dh, axis=0)

            mlab.plot3d(x, y, z, s, tube_radius=None, figure=fig, opacity=0.6, vmin=0.11, vmax=0.12)

        # Calculate where the center of the current step's longitude data is so we can set the camera there 
        lon_mid_ind = (a+b)//2
        if lon_mid_ind >= lon.shape[0]: lon_mid_ind = lon.shape[0] - 1
        lon_mid = lon[lon_mid_ind, footprint_ind] % 360

        # Calculate where the center of the current step's latitude data is so we can set the camera there
        lat_mid_ind = (a+b)//2
        if lat_mid_ind >= lat.shape[0]: lat_mid_ind = lat.shape[0] - 1
        lat_mid = c_lat[lat_mid_ind, footprint_ind]
        lat_mid2 = lat_mid % 360

        # Set the position of camera
        mlab.view(lon_mid, lat_mid2, 2.2)
        # Getting the name of each image screenshot based on date the data is from and its index and saving the image
        month = st["Month"][lon_mid_ind]
        day = st["DayOfMonth"][lon_mid_ind]
        hour = st["Hour"][lon_mid_ind]
        minute = st["Minute"][lon_mid_ind]
        second = st["Second"][lon_mid_ind]
        start_name = f'{month:02}-{day:02}-{hour:02}-{minute:02}-{second:02}'
        mlab.savefig(save_location + start_name + ".png")

def plot_lonlatref_2d(data, lat_min, lat_max, title=None, save_path="./2d_images/", file_type=".HDF5"):
    """Plots a contour plot of Longitude x Latitude x Reflectivity, given GPM data.

    Args:
        data (netCDF4 Dataset): 2A.GPM.DPRX.V8 data from GPM.
        lat_min (float): Minimum value of latitude range.
        lat_max (float): Maximum value of latitude range.
        title (str): Title for plot.
        save_path (str): Where the screenshot should be saved to.
        file_type (str): Raw ".HDF5" file or compressed ".nc" file. 
    """
    my_cmap = [(57/255, 78/255, 157/255), (0, 159/255, 60/255), (248/255, 244/255, 0),(1, 0, 0), (1, 1, 1)]
    my_cmap = colors.LinearSegmentedColormap.from_list("Reflectivity", my_cmap, N=26)
    obs = lat = alt = ds = None
    if file_type == ".HDF5":
        swath = data["FS"]
        pre = swath["PRE"]
        # Reflectivity data
        obs = pre["zFactorMeasured"][:, :, :, 0]
        # Latitude data
        lat = swath["Latitude"][:]
        alt = pre["height"][:]
        # Times data
        ds = data["FS"]["ScanTime"]
    elif file_type == ".nc":
        lat = data["Latitude"][:]
        obs = data["zFactorMeasured"][:]
        alt = data["height"][:]
        ds = data

    # Limit data to be between the min and max lat
    selected_inds = (lat >= lat_min) & (lat <= lat_max)
    obs = obs[selected_inds, 24, :].T
    alt = alt[selected_inds,24,:].T / 1000
    lat = lat[selected_inds]
    lat = np.array([[x] * 176 for x in lat]).T
    obs[obs < -48] = np.nan
    obs[obs > 48] = np.nan
    # Give date as title, if one is not given
    if title is None:
        year = ds["Year"][0]
        month = ds["Month"][0]
        day = ds["DayOfMonth"][0]
        title = f'{year}-{month:02}-{day:02}'
    plt.contourf(lat, alt, obs, cmap=my_cmap, levels=26)
    plt.title(title)
    plt.xlabel("Latitude, deg")
    plt.ylabel("Altitude, km")
    plt.colorbar(label="Reflectivity (dbz)")
    plt.ylim((0, 21))
    plt.xlim((lat.min(), lat.max()))
    full_path = save_path + title
    plt.savefig(full_path, facecolor='white', transparent=False)

def plot_front_2d(data, x0, x1, title=None, save_path="./2d_images/", file_type=".HDF5"):
    """Plots a contour plot of Footprint x Altitude x Reflectivity, given GPM data.

    Args:
        data (netCDF4 Dataset): 2A.GPM.DPRX.V8 data from GPM.
        x0 (int): Range of indices the function should plot graphs for.
        x1 (int): Range of indices the function should plot graphs for.
        title (str): Title for plot.
        save_path (str): Where the screenshot should be saved to.
        file_type (str): Raw ".HDF5" file or compressed ".nc" file. 
    """
    my_cmap = [(57/255, 78/255, 157/255), (0, 159/255, 60/255), (248/255, 244/255, 0),(1, 0, 0), (1, 1, 1)]
    my_cmap = colors.LinearSegmentedColormap.from_list("Reflectivity", my_cmap, N=26)
    obs = alt = ds = None

    if file_type == ".HDF5":
        swath = data["FS"]
        pre = swath["PRE"]
        obs = pre["zFactorMeasured"][:, :, :, 0]
        alt = pre["height"][:]
        ds = data["FS"]["ScanTime"]
    elif file_type == ".nc":
        obs = data["zFactorMeasured"][:]
        alt = data["height"][:]
        ds = data

    # Limit data to be between the min and max range
    # Shape is (range, footprint, height)
    obs = obs[x0:x1, :, :].T
    alt = alt[x0:x1, :, :].T / 1000
    bottom_z = 18
    obs[obs < bottom_z] = np.nan
    #obs[obs > 48] = np.nan

    # Give date as title, if one is not given
    if title is None:
        year = ds["Year"][0]
        month = ds["Month"][0]
        day = ds["DayOfMonth"][0]
        title = f'{year}-{month:02}-{day:02}'

    footprint = np.array([np.full((176,), x) for x in range(0, 245, 5)]).T
    for layer_i in range(obs.shape[2]):
        # (footprint, height)
        obs_i = obs[:, :, layer_i]
        alt_i = alt[:, :, layer_i]
        plt.contourf(footprint, alt_i, obs_i, cmap=my_cmap, levels=26)
        plt.title(title)
        plt.xlabel("Footprint, km")
        plt.ylabel("Altitude, km")
        plt.colorbar(label="Reflectivity (dbz)")
        plt.ylim((0, 21))
        plt.xlim((0, 245))
        plt.clim((bottom_z))
        full_path = save_path + f"/{layer_i:05}"
        plt.savefig(full_path, facecolor='white', transparent=False)
        plt.clf()

def plot_front_3d(data, x0, x1, fig, title=None, save_location="./3d_images/", file_type=".HDF5"):
    """Given GPM dataset in HDF5 format and a figure to plot it on, it plots the reflectivity data in step steps, and saves each step to save_location.

    Args:
        data (netCDF4 Dataset): 2A.GPM.DPRX.V8 data from GPM.
        fig (mlab figure): The figure the data should be plotted on.
        step (int): Number of steps it should take to plot the whole figure, by default equals 1 which means the entire data will be plotted in one step.
        save_location (str): Where the screenshot in each step should be saved to.
        file_type (str): Raw ".HDF5" file or compressed ".nc" file. 
    """
    obs = alt = ds = lon = lat = None

    if file_type == ".HDF5":
        swath = data["FS"]
        pre = swath["PRE"]
        obs = pre["zFactorMeasured"][:, :, :, 0]
        alt = pre["height"][:]
        lat = swath["Latitude"][:]
        lon = swath["Longitude"][:]
        ds = data["FS"]["ScanTime"]
    elif file_type == ".nc":
        obs = data["zFactorMeasured"][:]
        alt = data["height"][:]
        lat = data["Latitude"][:]
        lon = data["Longitude"][:]
        ds = data

    # Limit data to be between the min and max range
    # Shape is (range, footprint, height)
    obs = obs[x0:x1, :, :].T
    lon = lon[x0:x1, :].T
    lat = lat[x0:x1, :].T
    alt = alt[x0:x1, :, :].T / 1000
    bottom_z = 18
    obs[obs < bottom_z] = np.nan

    if title is None:
        year = ds["Year"][0]
        month = ds["Month"][0]
        day = ds["DayOfMonth"][0]
        title = f'{year}-{month:02}-{day:02}'

    #(height, footprint)
    footprint = np.array([np.full((176,), x) for x in range(0, 245, 5)]).T
    for layer_i in range(obs.shape[2]):
        
        #lat_i = (lat_i / lat_i.max()) + 1.001
        # (height, footprint)
        for h_i in range(len(alt)):
            print(obs.shape, alt.shape, lon.shape, lat.shape)
            obs_i = obs[h_i, :, layer_i]
            alt_i = alt[h_i, :, layer_i]
            alt_i = (alt_i / alt_i.max()) + 0.001
            lon_i = lon[:, layer_i]
            lat_i = lat[:, layer_i]
            x, y, z = convert.polar_to_cartesian(lon_i, lat_i, alt_i)
            mlab.plot3d(x, y, z, obs_i, tube_radius=None, figure=fig, opacity=0.6, vmin=0.11, vmax=0.12)
            mlab.view()
            #mlab.savefig(save_location + "wow" + str(layer_i) + ".png")
    mlab.show()

import re 
class RadarDisplay:
    
    FILETYPE = ".HDF5"
    PREFIX = "2A.GPM.DPRX.V8-20200326."
    SUFFIX = ".V06X.HDF5"

    def __init__(self, data_path: str, username: str, password: str, debug_mode=False):
        self.data_path = data_path
        self.username = username
        self.password = password
        self.debug = debug_mode
        directory = listdir(fsencode(data_path))
        pattern = r"^(\d{8})-S(\d{6})-E(\d{6}).*$"
        dt_format = "%Y%m%d %H%M%S"
        st_p = len(data_path) + len(self.PREFIX)
        ed_p = len(self.SUFFIX)

        self.files = [data_path + fsdecode(f) for f in directory if self.FILETYPE in fsdecode(f)]

        self.f_to_dt = {}
        for f_path_extr in self.files:
            f_path = f_path_extr[st_p:-ed_p]
            grps = re.search(pattern, f_path)
            f_start = dt.datetime.strptime(grps.group(1) + " " + grps.group(2), dt_format)
            f_end = dt.datetime.strptime(grps.group(1) + " " + grps.group(3), dt_format)
            self.f_to_dt[f_path_extr] = (f_start, f_end)

    def download_files(self, start: dt, end: dt, files: list[str]):
        """Gets hdf files from the GPM FTP server based on date

        Args:
            date (str): Date of the files for the day you want in yyyy/mm/dd format
            dest (str): Destination folder we want to download to
            username (str): GPM username
            password (str): GPM password
        """
        base_dir = "./pub/gpmdata/"
        base_dir = "{}{}/{:02d}/{:02d}/Xradar".format(base_dir, start.year, start.month, start.day)
        link = "arthurhou.pps.eosdis.nasa.gov"
        st_name = "{}{}{:02d}{:02d}-".format(self.PREFIX, start.year, start.month, start.day)
        ed_name = "{}{}{:02d}{:02d}-".format(self.PREFIX, end.year, end.month, end.day)

        # Get files for that day
        ftp = FTP_TLS(link)
        ftp.login(user=self.username, passwd=self.password)
        ftp.cwd(base_dir)
        file_names = ftp.nlst()
        for file_name in file_names:
            if target_name not in file_name: continue
            local_filename = os.path.join(dest, file_name)
            if os.path.exists(local_filename):
                log("Skipping " + file_name)
                continue
            file = open(local_filename, 'wb')
            ftp.retrbinary('RETR '+ file_name, file.write)
            log("Downloaded " + file_name)
            file.close()
        ftp.quit()

    def get_files_by_dt(self, start: dt, end: dt):
        files = []

        for f_path, f_dts in self.f_to_dt.items():
            f_st, f_ed = f_dts
            if f_st >= start and f_ed <= end:
                files.append(f_path)
        
        delta = end - start
        for i in range(delta.days + 1):
            day = start + dt.timedelta(days=i)
            # Now download files for start to end but only the ones we dont have
            # append those paths to files as well
        return files

    def log(self, o: object):
        if not self.debug: return
        print("[DEBUG]:", o)