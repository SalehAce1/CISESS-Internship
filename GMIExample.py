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
files = gp.get_files_by_dt_gmi(st, ed)

# Currently can't use the GMI data in any way