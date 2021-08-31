import GPMPy as gp
import netCDF4
from mayavi import mlab
import cv2
from os import fsencode, fsdecode, listdir
from os.path import splitext
import datetime

s1 = 1
s2 = 50
fps = 10
datadone_path = "./datadone/"
data_path = "./data/"
image_path = "./images/"

fig = gp.plot_earth_wrapped()
# Loop through all files in datadone folder, get them using netCDF4, and plot them all in one step
directory = fsencode(datadone_path)
fig.scene.disable_render = True
for file in listdir(directory):
    target = datadone_path + fsdecode(file)
    print("Doing", target)
    data2A = netCDF4.Dataset(target, diskless=True, persist=False)
    gp.plot_gpm_data(data2A, fig, step=s1)
    data2A.close()

# Loop through all files in data folder, get them using netCDF4, and plot/save them step by step
directory = fsencode(data_path)
for file in listdir(directory):
    target = data_path + fsdecode(file)
    print("Doing", target)
    data2A = netCDF4.Dataset(target, diskless=True, persist=False)
    gp.plot_gpm_data(data2A, fig, step=s2)
    data2A.close()

mlab.close()

# Sort files by date-time
files = listdir(image_path)
format = "%m-%d-%H-%M-%S"
def extract_dt(name):
    noextname = splitext(name)[0]
    return datetime.datetime.strptime(noextname, format)
sorted(files, key=extract_dt)

# Turn images into video
img_array = []
for filename in files:
    img = cv2.imread(image_path + filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
out = cv2.VideoWriter('./project.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()