from pyBBarolo import Task, FitMod3D, Search
import numpy as np

raw_path = "/data/songzihao/data_ZY/raw"

f3d = FitMod3D(raw_path + "/NGC2403/NGC_2403.fits")
f3d.init(radii=np.arange(15,1200,30), xpos=77, ypos=77, vsys=132.8, vrot=120, 
         vdisp=8,vrad=0,z0=10,inc=60,phi=123.7)
f3d.set_options(mask="SMOOTH&SEARCH", free="VROT VDISP", outfolder=f'{raw_path}/NGC2403/fit')
f3d.set_beam(bmaj=180, bmin=180, bpa=0)
bfrings, bestmod = f3d.compute(threads=16)

