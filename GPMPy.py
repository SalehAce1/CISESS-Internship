import os
import datetime as dt
from ftplib import FTP_TLS
import numpy as np
from mayavi import mlab
import convert

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

def plot_gpm_data(data, fig, step=1, save_location="./images/"):
    """Given GPM dataset in HDF5 format and a figure to plot it on, it plots the reflectivity data in step steps, and saves each step to save_location.

    Args:
        data (netCDF4 Dataset): 2A.GPM.DPRX.V8 data from GPM.
        fig (mlab figure): The figure the data should be plotted on.
        step (int): Number of steps it should take to plot the whole figure, by default equals 1 which means the entire data will be plotted in one step.
        save_location (str): Where the screenshot in each step should be saved to.
    """

    swath = data["FS"]
    pre = swath["PRE"]
    # Reflectivity data
    obs = pre["zFactorMeasured"][:, :, :, 0].T
    # Latitude data
    lat = swath["Latitude"][:]
    # Longitude data
    lon = swath["Longitude"][:]

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
        month = data["FS"]["ScanTime"]["Month"][lon_mid_ind]
        day = data["FS"]["ScanTime"]["DayOfMonth"][lon_mid_ind]
        hour = data["FS"]["ScanTime"]["Hour"][lon_mid_ind]
        minute = data["FS"]["ScanTime"]["Minute"][lon_mid_ind]
        second = data["FS"]["ScanTime"]["Second"][lon_mid_ind]
        start_name = f'{month:02}-{day:02}-{hour:02}-{minute:02}-{second:02}'
        mlab.savefig(save_location + start_name + ".png")
