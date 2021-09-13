import GPMPy as gp
import netCDF4
import pathlib

filename = "./data/2A.GPM.DPRX.V8-20200326.20210828-S000741-E014014.042599.V06X.nc"
data2A = netCDF4.Dataset(filename, diskless=True, persist=False)
file_type = pathlib.Path(filename).suffix
gp.plot_lonlatref_2d(data2A, 10, 20, save_path="./2d_images/", file_type=file_type)
data2A.close()