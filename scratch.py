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
sv_pth = '/Users/lillienders/Desktop/'

ds = xr.open_dataset('/Users/lillienders/Desktop/First Generals/Data/Observations/altimetric_ssh_clean.nc')

ds_bathy = xr.open_dataset(pth+'gebco_2022_n45.0_s33.0_w-77.0_e-67.0.nc')
ds_bathy = ds_bathy.coarsen(lon =10,lat=10).mean()

import pandas as pd
txy = pd.read_csv('/Users/lillienders/Desktop/Second Generals/Scripts/txy_ll2.csv',names=['t','x','y'])
x_ll2 = txy['x']
y_ll2 = txy['y']
txy = pd.read_csv('/Users/lillienders/Desktop/Second Generals/Scripts/txy_grid2.csv',names=['t','x','y'])
x_g2 = txy['x']
y_g2 = txy['y']
txy = pd.read_csv('/Users/lillienders/Desktop/Second Generals/Scripts/txy_ll1.csv',names=['t','x','y'])
x_ll1 = txy['x']
y_ll1 = txy['y']

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
plt.plot(x_ll1,y_ll1,color='k',linewidth=4, label = 'LL1')
plt.plot(x_ll2,y_ll2,color='#ffb000',linewidth=3,linestyle = '--', label = 'LL2')
plt.plot(x_g2,y_g2,color='#785ef0',linewidth=4,linestyle = 'dotted', label = 'G2')
plt.legend(fontsize=18)
levels = np.linspace(-100,100,51)
colorplot = plt.contourf(ds.longitude,ds.latitude,ds.sla[232,:,:]+ds.adt, cmap='cmo.balance',zorder = 0,levels=levels)
cbar = plt.colorbar(colorplot,fraction=0.15, pad=0.02)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('May 2012 Mean SSH [cm]', size='25', labelpad=25)
plt.contour(ds_bathy.lon,ds_bathy.lat,abs(ds_bathy.elevation),levels = [20,60,200,1000],colors='k',linestyles='dotted')
plt.savefig(sv_pth+'CH_BG_transects.png',format='png', bbox_inches="tight", dpi=500)
plt.show()

sst = xr.open_dataarray('/Users/lillienders/Desktop/First Generals/may_sst_anom.nc')
bbox  =[281, 292, 35, 45] # North Atlantic
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
gl.xlocator = mticker.FixedLocator([-78,-76, -74, -72, -70,-68,-66])
gl.ylocator = mticker.FixedLocator([36,37,38,39,40,41,42,43,44])
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
plt.plot(x_ll1,y_ll1,color='k',linewidth=2, label = 'LL1')
plt.plot(x_ll2,y_ll2,color='k',linewidth=2, label = 'LL2')
plt.plot(x_g2,y_g2,color='k',linewidth=2, label = 'G2')

levels = np.arange(-6,6.1,0.1)
colorplot = plt.contourf(sst.lon,sst.lat,sst, cmap='cmo.balance',zorder = 0,levels=levels)
cbar = plt.colorbar(colorplot,fraction=0.15, pad=0.02)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('May 2012 SST Anomaly [˚C]', size='25', labelpad=25)
colorplot = plt.contour(ds.longitude,ds.latitude,ds.adt, colors='k',zorder = 0,levels=[33])
colorplot = plt.contour(ds.longitude,ds.latitude,ds.adt, colors='k',zorder = 0,linestyles='dashed',levels=[0])
plt.contour(ds_bathy.lon,ds_bathy.lat,abs(ds_bathy.elevation),levels = [200],colors='k',linestyles='dotted')
plt.savefig(sv_pth+'NA-SST-ANOM.png',format='png', bbox_inches="tight", dpi=500)
plt.show()
