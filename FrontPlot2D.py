import GPMPy as gp
import netCDF4
import pathlib
from os import listdir
from os.path import splitext
import cv2

image_path = "./2d_images/"
filename = "./data/2A.GPM.DPRX.V8-20200326.20210829-S100409-E113642.042621.V06X.HDF5"
data2A = netCDF4.Dataset(filename, diskless=True, persist=False)
file_type = pathlib.Path(filename).suffix
gp.plot_front_2d(data2A, 0, 50, save_path="./2d_images/", file_type=file_type)
data2A.close()

# Sort files by date-time
files = listdir(image_path)
def extract_dt(name):
    return int(splitext(name)[0])
files = sorted(files, key=extract_dt)

# Turn images into videos
fps = 15
img_array = []
for filename in files:
    img = cv2.imread(image_path + filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
out = cv2.VideoWriter('./vid_out/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()