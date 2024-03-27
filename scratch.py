import matplotlib.pyplot as plt
import xarray as xr
from proj_utils import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import cmocean as cmo
import seaborn as sns
import scipy
sns.set()
pth = '/Users/lillienders/Downloads/'

ds_bathy = xr.open_dataset(pth+'gebco_2022_n45.0_s33.0_w-77.0_e-67.0.nc')
ds_bathy = ds_bathy.coarsen(lon =10,lat=10).mean()

bbox = [282.5, 287.5, 33.5, 37.5]
crs = ccrs.PlateCarree()
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(1, 1, 1, projection=crs)
ax.set_extent(bbox, crs)
# Add Filled Coastline
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='k', zorder=10)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
gl.top_labels = gl.right_labels = False
gl.xformatter = LongitudeFormatter(degree_symbol='')
gl.yformatter = LatitudeFormatter(degree_symbol='')
gl.xlocator = mticker.FixedLocator([-77,-77, -76, -75, -74,-73])
gl.ylocator = mticker.FixedLocator([33,34, 35, 36, 37,38])
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
plt.contour(ds_bathy.lon,ds_bathy.lat,abs(ds_bathy.elevation),levels = [20,60,200,1000],colors='k',linestyles='dotted')
plt.show()