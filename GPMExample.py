import GPMPy as gm
import datetime as dt

# Time for the files we want to download
st = dt.datetime(2022, 1, 1, 6, 0, 0)
ed = dt.datetime(2022, 1, 1, 12, 0, 0)
# Initialize GPMpy
# data/ is the folder we want the data files sent to
gp = gm.RadarDisplay("data/", debug_mode=False)
# Set the username and password
gp.set_account_gpm("TMP_USR", "TMP_PASS")
# Download the files by date (does not download if the files already exist)
# After downloading, this will replace the HDF5 file with a compressed xarray version (.nc)
files = gp.get_files_by_dt_gpm(st, ed)

st = dt.datetime(2022, 1, 1, 7, 30, 0)
ed = dt.datetime(2022, 1, 1, 8, 0, 0)
# Creates a video of the 3D weather plot and the front 2d plot using the GPM data and saves it into vid_out 
gp.plot_combined(st, ed, 100, 15)