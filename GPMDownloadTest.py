import GPMPy as gm
import datetime as dt

st = dt.datetime(2020, 10, 5, 6, 0, 0)
ed = dt.datetime(2020, 10, 5, 10, 0, 0)
gp = gm.RadarDisplay("./data/", debug_mode=True)
gp.set_account_gpm("sghaemi1@umd.edu", "sghaemi1@umd.edu")
lst = gp.get_files_by_dt_gpm(st, ed)
print(lst)