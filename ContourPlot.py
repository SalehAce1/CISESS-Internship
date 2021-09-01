import netCDF4
import GPMPy as gp

data2A = netCDF4.Dataset("./data/2A.GPM.DPRX.V8-20200326.20210828-S000741-E014014.042599.V06X.HDF5", diskless=True, persist=False)
gp.plot_lonlatref_2d(data2A, -40, 40, save_path="./2d_images/")