import os
from os import fsencode, fsdecode, listdir
from os.path import splitext
import datetime as dt
from ftplib import FTP_TLS
import numpy as np
from mayavi import mlab
import convert
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import netCDF4
from typing import List, Tuple
from enum import Enum
import pathlib
from PIL import Image
import cv2
import re 


class RadarDisplay:
    
    FILETYPE = ".HDF5"
    PREFIX = "2A.GPM.DPRX.V8-20200326."
    SUFFIX = ".V06X.HDF5"
    IMG_PATH = "./combined_images/"

    class FileType(Enum):
        HDF5 = ".HDF5"
        NC = ".nc"

    def __init__(self, data_path: str, username: str, password: str, debug_mode=False):
        self.data_path = data_path
        self.username = username
        self.password = password
        self.debug = debug_mode
        directory = listdir(fsencode(data_path))

        self.files = [fsdecode(f) for f in directory if self.FILETYPE in fsdecode(f)]
        self.f_to_dt = {}

        for f_path_extr in self.files:
            f_start, f_end = self.get_dt_from_name(f_path_extr)
            self.f_to_dt[f_path_extr] = (f_start, f_end)

    def get_dt_from_name(self, filename: str) -> Tuple[dt.datetime, dt.datetime]:
        pattern = r"^(\d{8})-S(\d{6})-E(\d{6}).*$"
        dt_format = "%Y%m%d %H%M%S"
        st_p = len(self.PREFIX)
        ed_p = len(self.SUFFIX)
        f = filename[st_p:-ed_p]
        grps = re.search(pattern, f)
        f_start = dt.datetime.strptime(grps.group(1) + " " + grps.group(2), dt_format)
        f_end = dt.datetime.strptime(grps.group(1) + " " + grps.group(3), dt_format)
        
        return (f_start, f_end) 

    def download_files(self, start: dt, end: dt, files: List[str]):
        """Gets hdf files from the GPM FTP server based on date

        Args:
            date (str): Date of the files for the day you want in yyyy/mm/dd format
            dest (str): Destination folder we want to download to
            username (str): GPM username
            password (str): GPM password
        """
        base_dir = "./pub/gpmdata/"
        link = "arthurhou.pps.eosdis.nasa.gov"
        type_pattern = "^{}.*{}$".format(self.PREFIX, self.SUFFIX) 
        ftp = FTP_TLS(link)
        ftp.login(user=self.username, passwd=self.password)
        ftp.cwd(base_dir)

        delta = end - start
        for i in range(delta.days + 1):
            curr_day = start + dt.timedelta(days=i)
            curr_dir = "{}/{:02d}/{:02d}/Xradar".format(curr_day.year, curr_day.month, curr_day.day)
            ftp.cwd(curr_dir)
            file_names = ftp.nlst()

            for file_name in file_names:
                # ignore if we have it
                if file_name in files: continue
                # ignore if not the correct type of file
                if re.match(type_pattern, file_name) is None: continue
                # ignore if not within timeframe
                (s_dt, e_dt) = self.get_dt_from_name(file_name)
                s_dt = s_dt.replace(minute=0, second=0)
                e_dt = e_dt.replace(minute=0, second=0)
                if s_dt < start or e_dt > end: continue
                # get file and save to path
                local_filename = os.path.join(self.data_path, file_name)
                # if os.path.exists(local_filename):
                #     log("Skipping " + file_name)
                #     continue
                files.append(file_name)
                self.files.append(file_name)
                self.f_to_dt[file_name] = (s_dt, e_dt)
                file = open(local_filename, 'wb')
                ftp.retrbinary('RETR '+ file_name, file.write)
                log("Downloaded " + file_name)
                file.close()
            
            ftp.cwd("../../../../")

        ftp.quit()

    def get_files_by_dt(self, start: dt, end: dt) -> List[str]:
        files = []

        for f_path, f_dts in self.f_to_dt.items():
            f_st, f_ed = f_dts
            # ignore minutes and seconds for now
            f_st = f_st.replace(minute=0, second=0)
            f_ed = f_ed.replace(minute=0, second=0)
            if f_st >= start and f_ed <= end:
                files.append(f_path)
        
        self.download_files(start, end, files)

        return files


    def combine_imgs_by_path(self, images1:str, images2:str, out_files, ind: int):
        for i in range(len(images1)):
            images = [Image.open(x) for x in [images1[i], images2[i]]]
            widths, heights = zip(*(i.size for i in images))

            total_width = sum(widths)
            max_height = max(heights)

            new_im = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]

            path = f"final{ind + i:05}.png"
            out_files.append(path)
            new_im.save(self.IMG_PATH + path)

    def combine_video(self, files: List[str], fps:int):
        # Turn images into videos
        img_array = []
        for filename in files:
            img = cv2.imread(self.IMG_PATH + filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        out = cv2.VideoWriter('./vid_out/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def plot_combined(self, start: dt, end: dt, frames: int, fps: int):
        files = self.get_files_by_dt(start, end)
        files.sort()
        imgs1 = []
        imgs2 = []
        finalimgs = []
        i = 0
        for file in files:
            data = netCDF4.Dataset(self.data_path + file, diskless=True, persist=False)
            title = "{} To {}".format(str(start), str(end))
            file_type = self.get_file_type(file)
            imgs1 = self.plot_front_2d(data, start, end, frames, title, i, file_type=file_type, save_path=self.IMG_PATH)
            if imgs1 is None: continue
            plt.clf()
            fig = self.plot_earth_wrapped()
            imgs2 = self.plot_3d(data, start, end, frames, fig, title, i, file_type=file_type, save_path=self.IMG_PATH)
            plt.clf()
            data.close()
            mlab.close()
            self.combine_imgs_by_path(imgs1, imgs2, finalimgs, i)
            i += frames + 1

        self.combine_video(finalimgs, fps)
        


    def plot_front_2d(self, data, start:dt.datetime, end: dt.datetime, frames: int, title:str, ind:int,
                      save_path:str="./2d_images/", file_type:FileType=FileType.HDF5):
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

        if file_type == self.FileType.HDF5:
            swath = data["FS"]
            pre = swath["PRE"]
            obs = pre["zFactorMeasured"][:, :, :, 0]
            alt = pre["height"][:]
            ds = data["FS"]["ScanTime"]
        elif file_type == self.FileType.NC:
            obs = data["zFactorMeasured"][:]
            alt = data["height"][:]
            ds = data

        # We need to get the range based on time
        year = ds["Year"][:]
        month = ds["Month"][:]
        day = ds["DayOfMonth"][:]
        hour = ds["Hour"][:]
        minute = ds["Minute"][:]
        second = ds["Second"][:]
        all_dates = np.array([year, month, day, hour, minute, second]).T
        all_dt = np.array([dt.datetime(*x) for x in all_dates])
        all_ind = np.where((all_dt >= start) & (all_dt <= end))[0]
        # Stop if times did not fit range
        if len(year[all_ind]) == 0: return None
        # Limit data to be between the min and max range
        # Shape is (range, footprint, height)
        obs = obs[all_ind, :, :]
        alt = alt[all_ind, :, :]
        step = len(obs) // frames
        obs = obs[::step, :, :].T
        alt = alt[::step, :, :].T / 1000
        bottom_z = 18
        obs[obs < bottom_z] = np.nan

        footprint = np.array([np.full((176,), x) for x in range(0, 245, 5)]).T
        files = []
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
            full_path = save_path + f"/2d{ind+layer_i:05}.png"
            files.append(full_path)
            plt.savefig(full_path, facecolor='white', transparent=False)
            plt.clf()

        return files


    def plot_3d(self, data, start:dt.datetime, end: dt.datetime, frames: int, fig, title:str, ind:int, 
                      save_path:str="./2d_images/", file_type:FileType=FileType.HDF5):
        """Given GPM dataset in HDF5 format and a figure to plot it on, it plots the reflectivity data in step steps, and saves each step to save_location.

        Args:
            data (netCDF4 Dataset): 2A.GPM.DPRX.V8 data from GPM.
            fig (mlab figure): The figure the data should be plotted on.
            step (int): Number of steps it should take to plot the whole figure, by default equals 1 which means the entire data will be plotted in one step.
            save_location (str): Where the screenshot in each step should be saved to.
            file_type (str): Raw ".HDF5" file or compressed ".nc" file. 
        """
        obs = lat = lon = st = None

        if file_type == self.FileType.HDF5:
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
        elif file_type == self.FileType.NC:
            # Reflectivity data
            obs = data["zFactorMeasured"][:].T
            # Latitude data
            lat = data["Latitude"][:]
            # Longitude data
            lon = data["Longitude"][:]
            # Times data
            st = data

        year = st["Year"][:]
        month = st["Month"][:]
        day = st["DayOfMonth"][:]
        hour = st["Hour"][:]
        minute = st["Minute"][:]
        second = st["Second"][:]
        all_dates = np.array([year, month, day, hour, minute, second]).T
        all_dt = np.array([dt.datetime(*x) for x in all_dates])
        all_ind = np.where((all_dt >= start) & (all_dt <= end))[0]
        # Stop if times did not fit range
        if len(year[all_ind]) == 0: return None

        obs = obs[:,:,all_ind]
        lat = lat[all_ind]
        lon = lon[all_ind]
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
        step_size = int(total_len/frames)

        # Loop for the number of steps we need to do
        layer_i = 0
        files = []
        for i in range(0, total_len, step_size):
            a,b = i, i + step_size
            # Calculate for each footprint
            for footprint_ind in range(49):
                # Convert lon-lat data in current step and footprint to cartesian
                x, y, z = convert.polar_to_cartesian(lon[a:b, footprint_ind], lat[a:b, footprint_ind], 1.001)
                # Get reflectivity data for current step and footprint
                s = obs_mean[footprint_ind, a:b]
                mlab.plot3d(x, y, z, s, tube_radius=None, opacity=0.6, vmin=0.11, vmax=0.12)

            # Calculate where the center of the current step's longitude data is so we can set the camera there 
            lon_mid_ind = (a+b)//2
            if lon_mid_ind >= lon.shape[0]: lon_mid_ind = lon.shape[0] - 1
            lon_mid = lon[lon_mid_ind, footprint_ind] % 360

            # Set the position of camera
            mlab.view(lon_mid, 75, 3.8)
            full_path = save_path + f"/3d{ind+layer_i:05}.png"
            files.append(full_path)
            mlab.savefig(full_path)
            layer_i += 1
        
        return files

    def plot_earth_wrapped(self, earth_tex_path="./Textures/EarthMap_2500x1250.jpg", star_tex_path="./Textures/starmap.png"):
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
        mlab.figure(1, size=(640, 480))
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

    def get_file_type(self, file_path: str) -> FileType:
        file_type = pathlib.Path(file_path).suffix
        return self.FileType.HDF5 if file_type == self.FileType.HDF5.value else self.FileType.NC 

    def log(self, o: object):
        if not self.debug: return
        print("[DEBUG]:", o)