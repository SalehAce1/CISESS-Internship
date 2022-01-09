import os
from os import fsencode, fsdecode, listdir, remove
from os.path import splitext
import datetime as dt
from ftplib import FTP_TLS, FTP
from platform import python_compiler
import ssl
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
import pandas as pd
import re
import read_cloudsat as rc
import drpy.core.gpmdpr as drp
from multiprocessing.dummy import Pool as ThreadPool


class RadarDisplay:
    GPM_PATT = r"^2A\.GPM\.DPR\.V\d-\d+\.(\d{8})-S(\d{6})-E(\d{6})\.\d+\..+$"
    GMI_PATT = r"^1B\.GPM\.GMI\.TB2016\.(\d{8})-S(\d{6})-E(\d{6})\.\d+\..+$"
    GPM_BASE = "~/pub/gpmdata/"
    GPM_LINK = "arthurhouftps.pps.eosdis.nasa.gov"
    IMG_PATH = "./combined_images/"
    IMG_2D_PATH = "./2d_images/"
    IMG_3D_PATH = "./3d_images/"
    CLOUDSAT_LINK = "ftp.cloudsat.cira.colostate.edu"

    
    class FileType(Enum):
        HDF5 = ".HDF5"
        NC = ".nc"
        HDF = ".HDF"

    def __init__(self, data_path: str, debug_mode=False):
        self.data_path = data_path
        self.u_gpm = self.p_gpm = self.u_cloudsat = self.p_cloudsat = None
        self.debug = debug_mode
        self.gpm_files = []
        self.gmi_files = []
        self.nc_files = []
        self.f_to_dt = {}

        os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.IMG_2D_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.IMG_3D_PATH), exist_ok=True)
    
    def set_account_cloudsat(self, username: str, password: str):
        self.u_cloudsat = username
        self.p_cloudsat = password
    
    def set_account_gpm(self, username: str, password: str):
        self.u_gpm = username
        self.p_gpm = password
    
    def set_data_path(self, data_path: str):
        self.data_path = data_path
    
    def set_debug(self, toggle: bool):
        self.debug = toggle

    # Get date-time of a file from its name
    def get_dt_from_name_gpm(self, filename: str) -> Tuple[dt.datetime, dt.datetime]:
        filename = splitext(filename)[0]
        dt_format = "%Y%m%d %H%M%S"
        grps = re.search(self.GPM_PATT, filename)
        f_start = dt.datetime.strptime(grps.group(1) + " " + grps.group(2), dt_format)
        f_end = dt.datetime.strptime(grps.group(1) + " " + grps.group(3), dt_format)
        
        return (f_start, f_end) 

    def get_dt_from_name_gmi(self, filename: str) -> Tuple[dt.datetime, dt.datetime]:
        filename = splitext(filename)[0]
        dt_format = "%Y%m%d %H%M%S"
        grps = re.search(self.GMI_PATT, filename)
        f_start = dt.datetime.strptime(grps.group(1) + " " + grps.group(2), dt_format)
        f_end = dt.datetime.strptime(grps.group(1) + " " + grps.group(3), dt_format)
        
        return (f_start, f_end) 

    def download_files_gpm(self, start: dt, end: dt, files: List[str]):
        """Gets hdf files from the GPM FTP server based on date

        Args:
            start (dt.datetime): Start date for data.
            end (dt.datetime): End date for data.
            files (List[str]): List of files we need to find.
        """

        if self.u_gpm is None or self.p_gpm is None:
            exit("Please set username/password for GPM using set_account_gpm")

        FTP_TLS.ssl_version = ssl.PROTOCOL_TLSv1_2 
        ftp = FTP_TLS() 
        ftp.debugging = 1 if self.debug else 0
        ftp.connect(self.GPM_LINK, 21) 
        ftp.login(self.u_gpm, self.p_gpm) 
        ftp.prot_p() 

        delta = end - start
        for i in range(delta.days + 1):
            curr_day = start + dt.timedelta(days=i)
            curr_dir = "{}{}/{:02d}/{:02d}/radar".format(self.GPM_BASE, curr_day.year, curr_day.month, curr_day.day)
            ftp.cwd(curr_dir)
            file_names = ftp.nlst()

            for file_name in file_names:
                self.log("(Get) Checking: " + file_name)
                # ignore if we have it
                if file_name in files or splitext(file_name)[0] + ".nc" in files:
                    self.log("(Skip) Already have: " + file_name)
                    continue
                # ignore if not the correct type of file
                if re.match(self.GPM_PATT, file_name) is None:
                    self.log("(Skip) Does not fit pattern: " + file_name)
                    continue
                # ignore if not within timeframe
                (s_dt, e_dt) = self.get_dt_from_name_gpm(file_name)
                s_dt = s_dt.replace(minute=0, second=0)
                e_dt = e_dt.replace(minute=0, second=0)
                if (s_dt < start and e_dt < start) or (s_dt > end and e_dt > end): continue
                local_filename = os.path.join(self.data_path, file_name)
                files.append(file_name)
                self.gpm_files.append(file_name)
                self.f_to_dt[file_name] = (s_dt, e_dt)
                file = open(local_filename, 'wb')
                ftp.retrbinary('RETR '+ file_name, file.write)
                self.log("(Download) Saving: " + file_name)
                file.close()
            
        ftp.quit()

    def download_files_gmi(self, start: dt, end: dt, files: List[str]):
            """Gets hdf files from the GMI FTP server based on date

            Args:
                start (dt.datetime): Start date for data.
                end (dt.datetime): End date for data.
                files (List[str]): List of files we need to find.
            """

            if self.u_gpm is None or self.p_gpm is None:
                exit("Please set username/password for GPM using set_account_gpm")

            FTP_TLS.ssl_version = ssl.PROTOCOL_TLSv1_2 
            ftp = FTP_TLS() 
            ftp.debugging = 1 if self.debug else 0
            ftp.connect(self.GPM_LINK, 21) 
            ftp.login(self.u_gpm, self.p_gpm) 
            ftp.prot_p() 

            delta = end - start
            for i in range(delta.days + 1):
                curr_day = start + dt.timedelta(days=i)
                curr_dir = "{}{}/{:02d}/{:02d}/1B".format(self.GPM_BASE, curr_day.year, curr_day.month, curr_day.day)
                ftp.cwd(curr_dir)
                file_names = ftp.nlst()

                for file_name in file_names:
                    self.log("(Get) Checking: " + file_name)
                    # ignore if we have it
                    if file_name in files or splitext(file_name)[0] + ".nc" in files:
                        self.log("(Skip) Already have: " + file_name)
                        continue
                    # ignore if not the correct type of file
                    if re.match(self.GMI_PATT, file_name) is None:
                        self.log("(Skip) Does not fit pattern: " + file_name)
                        continue
                    # ignore if not within timeframe
                    (s_dt, e_dt) = self.get_dt_from_name_gmi(file_name)
                    s_dt = s_dt.replace(minute=0, second=0)
                    e_dt = e_dt.replace(minute=0, second=0)
                    if (s_dt < start and e_dt < start) or (s_dt > end and e_dt > end): continue
                    local_filename = os.path.join(self.data_path, file_name)
                    files.append(file_name)
                    self.gpm_files.append(file_name)
                    self.f_to_dt[file_name] = (s_dt, e_dt)
                    file = open(local_filename, 'wb')
                    ftp.retrbinary('RETR '+ file_name, file.write)
                    self.log("(Download) Saving: " + file_name)
                    file.close()
                
            ftp.quit()

    def convert_multi_gpm(self, files, del_old_files=False, thread_cnt=1):
        bad_ones = ['flagSurfaceSnow']

        def convert_one_gpm(filename):
            tmp = drp.GPMDPR(filename=filename, outer_swath=True)
            fixed_gpm = tmp.xrds.drop_vars(bad_ones)
            fixed_gpm = fixed_gpm.rename({"NSKu":"zFactorMeasured", "alt":"height"})
            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in fixed_gpm.data_vars}
            new_name = splitext(filename)[0] + ".nc"
            fixed_gpm.to_netcdf(new_name, encoding=encoding, engine='netcdf4')
            if del_old_files:
                remove(filename)

            return new_name

        pool = ThreadPool(thread_cnt)
        all_new_files = pool.map(convert_one_gpm, files)
        pool.close()
        pool.join()

        return all_new_files

    def get_files_by_dt_gpm(self, start: dt, end: dt) -> List[str]:

        directory = listdir(fsencode(self.data_path))
        # Looks for existing GPM DPR files
        self.nc_files = []
        for f in directory:
            dec = fsdecode(f)
            if  re.match(self.GPM_PATT, dec):
                if self.FileType.HDF5.value in dec:
                    self.gpm_files.append(dec)
                elif self.FileType.NC.value in dec:
                    self.nc_files.append(dec)
                
        for f_path_extr in self.gpm_files:
            f_start, f_end = self.get_dt_from_name_gpm(f_path_extr)
            self.f_to_dt[f_path_extr] = (f_start, f_end)

        for f_path_extr in self.nc_files:
            f_start, f_end = self.get_dt_from_name_gpm(f_path_extr)
            self.f_to_dt[f_path_extr] = (f_start, f_end)

        files = []

        for f_path, f_dts in self.f_to_dt.items():
            # ignore hdf5 if appropriate nc file exists
            split_name = splitext(f_path)
            if split_name[1] == self.FileType.HDF5.value and split_name[0] + ".nc" in self.f_to_dt:
                continue
            f_st, f_ed = f_dts
            # ignore minutes and seconds
            f_st = f_st.replace(minute=0, second=0)
            f_ed = f_ed.replace(minute=0, second=0)
            if (f_st >= start and f_ed <= end) or (f_ed >= start and f_ed <= end) \
                                               or (f_st >= start and f_st <= end):
                files.append(f_path)
        
        self.download_files_gpm(start, end, files)

        nc_files = [f for f in files if self.FileType.NC.value in f]
        hdf5_files = [self.data_path + f for f in files if self.FileType.HDF5.value in f]
        all_new_nc_files = self.convert_multi_gpm(hdf5_files, del_old_files=True)

        return list(set(all_new_nc_files + nc_files))
    
    def get_files_by_dt_gmi(self, start: dt, end: dt) -> List[str]:
        directory = listdir(fsencode(self.data_path))
        # Looks for existing GPM GMI files
        for f in directory:
            dec = fsdecode(f)
            if  re.match(self.GMI_PATT, dec) and self.FileType.HDF5.value in dec:
                self.gmi_files.append(dec)
                
        for f_path_extr in self.gmi_files:
            f_start, f_end = self.get_dt_from_name_gmi(f_path_extr)
            self.f_to_dt[f_path_extr] = (f_start, f_end)

        files = []

        for f_path, f_dts in self.f_to_dt.items():
            # ignore hdf5 if appropriate nc file exists
            split_name = splitext(f_path)
            if split_name[1] == self.FileType.HDF5.value:
                continue
            f_st, f_ed = f_dts
            # ignore minutes and seconds
            f_st = f_st.replace(minute=0, second=0)
            f_ed = f_ed.replace(minute=0, second=0)
            if (f_st >= start and f_ed <= end) or (f_ed >= start and f_ed <= end) \
                                               or (f_st >= start and f_st <= end):
                files.append(f_path)
        
        self.download_files_gmi(start, end, files)

        hdf5_files = [self.data_path + f for f in files if self.FileType.HDF5.value in f]

        return list(set(hdf5_files))

    def get_files_by_dt_cloudsat(self, start: dt.datetime, end: dt.datetime):
        """Gets hdf files from the CloudSat FTP server based on date

        Args:
            start (dt.datetime): Start date for data.
            end (dt.datetime): End date for data.
            username (str): CloudSat username
            password (str): CloudSat password
        """


        dt_range = pd.date_range(start, end, freq="1H")
        self.log("Attempting to access CloudSat ftp server.")

        if self.u_cloudsat is None or self.p_cloudsat is None:
            print("Please set username/password for CloudSat using set_account_cloudsat")
            return

        ftp = FTP(self.CLOUDSAT_LINK, user=self.u_cloudsat, passwd=self.p_cloudsat)
        ftp.set_pasv(True)
        ftp.login(user=self.u_cloudsat, passwd=self.p_cloudsat)
        for idt in dt_range:
            # Making base file names we want to get
            doy = idt.timetuple().tm_yday
            target_path = "~/2B-GEOPROF.P1_R05/{}/{:03d}".format(idt.year, doy)
            
            fstr = '%s*_R05_*.hdf'%idt.strftime('%Y%j%H')
            ftp.cwd(target_path)
            file_names = ftp.nlst(fstr)
            for file_name in file_names:
                local_filename = os.path.join(self.data_path, file_name)
                if os.path.exists(local_filename):
                    self.log("Skipping " + file_name)
                    continue
                file = open(local_filename, 'wb')
                ftp.retrbinary('RETR '+ file_name, file.write)
                self.log("Downloaded " + file_name)
                file.close()
        ftp.quit()

        return rc.read_cloudsat(self.data_path, start, end)


    def combine_imgs_by_path(self, images1:str, images2:str, out_files, ind: int, cloud_sat=None):
        for i in range(len(images1)):
            if cloud_sat:
                images = [Image.open(x) for x in [images2[i], images1[i], cloud_sat]]
            else:
                images = [Image.open(x) for x in [images2[i], images1[i]]]

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
        """Creates a video (combined 3d plot and 2d front plot) using GPM files given a start and end date

        Args:
            start (dt.datetime): Start date for data.
            end (dt.datetime): End date for data.
            frames (int): Number of frames in animation
            fps (int): The fps of the video
        """
        files = self.get_files_by_dt_gpm(start, end)
        files.sort()
        imgs1 = []
        imgs2 = []
        finalimgs = []
        i = 0
        print(files)
        for file in files:
            print(file)
            data = xr.open_dataset(self.data_path + file)
            title = "{} To {}".format(str(start), str(end))
            imgs1 = self.plot_front_2d(data, start, end, frames, title, i, save_path=self.IMG_PATH)
            if imgs1 is None: continue
            plt.clf()
            imgs2 = self.plot_3d(data, start, end, frames, i, save_path=self.IMG_PATH)
            plt.clf()
            data.close()
            mlab.close()
            self.combine_imgs_by_path(imgs1, imgs2, finalimgs, i)
            i += frames + 1

        self.combine_video(finalimgs, fps)

    def plot_combined_with_cloudsat(self, start: dt, end: dt, frames: int, fps: int):
        files = self.get_files_by_dt_gpm(start, end)
        cs_data = self.get_files_by_dt_cloudsat(start, end, self.date_path, self.u_gpm, self.p_gpm)
        cs_title = "{}/cs_img.png".format(self.IMG_2D_PATH)
        self.plot_side_2d(self, cs_data, start, end, cs_title)
        files.sort()
        imgs1 = []
        imgs2 = []
        finalimgs = []
        i = 0
        for file in files:
            data = netCDF4.Dataset(self.data_path + file, diskless=True, persist=False)
            title = "{} To {}".format(str(start), str(end))
            imgs1 = self.plot_side_2d(data, start, end, i, title, save_path=self.IMG_PATH)
            if imgs1 is None: continue
            plt.clf()
            imgs2 = self.plot_3d(data, start, end, frames, i, save_path=self.IMG_PATH)
            plt.clf()
            data.close()
            mlab.close()
            self.combine_imgs_by_path(imgs1, imgs2, finalimgs, i, cloud_sat=cs_title)
            i += frames + 1

        self.combine_video(finalimgs, fps)
        
    def plot_side_2d(self, data, start:dt.datetime, end: dt.datetime, ind:int, title:str, save_path:str = "2d_images/"):
        """Plots a contour plot of Latitude x Altitude x Reflectivity, given CloudSat data.

        Args:
            data (XArray): CloudSat data.
            start (dt.datetime): Range of dt the function should plot graphs for.
            end (dt.datetime): Range of dt the function should plot graphs for.
            title (str): Title for plot.
        """
        my_cmap = [(57/255, 78/255, 157/255), (0, 159/255, 60/255), (248/255, 244/255, 0),(1, 0, 0), (1, 1, 1)]
        my_cmap = colors.LinearSegmentedColormap.from_list("Reflectivity", my_cmap, N=26)

        all_dt = data.time.values
        start = np.datetime64(start)
        end = np.datetime64(end)
        good_ind = np.where((all_dt >= start) & (all_dt <= end))[0]

        lat = data.lat[good_ind].values
        lat = np.tile(lat, (125, 1))
        alt = data.height[good_ind, :].values.T / 1000.0
        obs = data.obs[good_ind, :].values.T
        obs[obs <= -24] = np.nan

        print(lat)
        print(alt)
        print(obs)

        vmin, vmax = -28.5, 47.5
        plt.figure(figsize=(20, 12))
        plt.contourf(lat, alt, obs, vmin=vmin, vmax=vmax, cmap=my_cmap)
        plt.title(title)
        plt.xlabel("Latitude, deg")
        plt.ylabel("Altitude, km")
        plt.colorbar(label="Reflectivity")
        plt.ylim((0, 17))
        plt.savefig("{}/{}{}.png".format(save_path, title, ind), facecolor='white', transparent=False)

    def plot_front_2d(self, data, start:dt.datetime, end: dt.datetime, frames: int, title:str, ind:int,
                      save_path:str="2d_images/"):
        """Plots a contour plot of Footprint x Altitude x Reflectivity, given GPM data.

        Args:
            data (xarray Dataset): 2A.GPM.DPRX.V8 or V9 data from GPM.
            start (dt.datetime): Range of dt the function should plot graphs for.
            end (dt.datetime): Range of dt the function should plot graphs for.
            frames (int): Number of frames to plot.
            title (str): Title for plot.
            ind (int): Label for the plots
            save_path (str): Where the screenshot should be saved to.
        """
        my_cmap = [(57/255, 78/255, 157/255), (0, 159/255, 60/255), (248/255, 244/255, 0),(1, 0, 0), (1, 1, 1)]
        my_cmap = colors.LinearSegmentedColormap.from_list("Reflectivity", my_cmap, N=26)

        start = np.datetime64(start)
        end = np.datetime64(end)
        # We need to get the range based on time
        data = data.where((data.time >= start) & (data.time <= end)).dropna("along_track", how="all")
        # Stop if times did not fit range
        if len(data.time) == 0: return None
        # Limit data to be between the min and max range
        # Shape is (range, footprint, height)
        obs = data["zFactorMeasured"].values
        alt = data["height"].values / 1000
        step = len(obs) // frames
        obs = obs[::step, :, :].T
        alt = alt[::step, :, :].T
        alt[alt > 10] = np.nan
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
            plt.ylim((0, 11))
            plt.xlim((0, 245))
            full_path = save_path + f"/2d{ind+layer_i:05}.png"
            files.append(full_path)
            plt.savefig(full_path, facecolor='white', transparent=False)
            plt.clf()

        return files


    def plot_3d(self, data, start:dt.datetime, end: dt.datetime, frames: int, ind:int, 
                      save_path:str="3d_images/"):
        """Given GPM dataset in HDF5 format and a figure to plot it on, it plots the reflectivity data in step steps, and saves each step to save_location.

        Args:
            data (netCDF4 Dataset): 2A.GPM.DPRX.V8 data from GPM.
            fig (mlab figure): The figure the data should be plotted on.
            step (int): Number of steps it should take to plot the whole figure, by default equals 1 which means the entire data will be plotted in one step.
            save_location (str): Where the screenshot in each step should be saved to.
            file_type (str): Raw ".HDF5" file or compressed ".nc" file. 
        """
        
        fig = self.plot_earth_wrapped()
        start = np.datetime64(start)
        end = np.datetime64(end)
        # We need to get the range based on time
        data = data.where((data.time >= start) & (data.time <= end)).dropna("along_track", how="all")
        # Stop if times did not fit range
        if len(data.time) == 0: return None
        # Reflectivity data
        obs = data["zFactorMeasured"].values.T
        # Latitude data
        lat = data["lats"].values
        # Longitude data
        lon = data["lons"].values
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