import GPMPy as gm
import datetime as dt

# Time for the files we want to download
st = dt.datetime(2008, 9, 13, 1, 20, 7)
ed = dt.datetime(2008, 9, 13, 13, 25, 53)
# Initialize GPMpy
# data/ is the folder we want the data files sent to
gp = gm.RadarDisplay("data/", debug_mode=False)
# Set the username and password
gp.set_account_cloudsat("TMP_USR", "TMP_PASS")
# Download the files by date (does not download if the files already exist) and load them in
fs = gp.get_files_by_dt_cloudsat(st, ed)
# Prints a side plot (Lat x Alt x Reflectivity) of the loaded data
gp.plot_side_2d(fs, st, ed, 0, "Output")