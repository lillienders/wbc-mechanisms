import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from proj_utils import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean as cmo
import seaborn as sns
sns.set()

pth = '/Users/lillienders/Desktop/First Generals/'

ds_eta = xr.open_dataset(pth + '/ATL-2Layer/twolayer_79_19.nc')
climatology = ds_eta.groupby('time.month').mean('time')
anomalies   = ds_eta.groupby("time.month") - climatology
ds_eta['sla'] = anomalies.sla
ds_eta = linear_detrend(ds_eta)

ds_h   = xr.open_dataset(pth + '/ATL-2Layer/twolayer_h2_79_19_fixed.nc')
time = np.arange(np.datetime64(str(1979) + '-01-01'), np.datetime64(str(2019 + 1) + '-01-01'), np.timedelta64(1, 'M'), dtype='datetime64[M]')
adt = np.nanmean(ds_h['h2'].data,axis=0)
ha  = np.zeros(ds_h['h2'].shape)
for t in range(len(ds_h.time)):
    ha[t,:,:] = ds_h.h2[t,:,:] - adt
ds_h['h2']  = (ds_h['h2'].dims,ha)
ds_h['time'] = time
climatology = ds_h.groupby('time.month').mean('time')
anomalies   = ds_h.groupby("time.month") - climatology
ds_h['h2']  = anomalies.h2
ds_h        = linear_detrend(ds_h,var='h2')

density_term = 1.5/1030
baroclinic = -ds_h.h2*density_term
barotropic = ds_eta.sla+ds_h.h2*density_term

t_step = 263
bbox  =[260, 360, 0, 65] # North Atlantic
#bbox  =[279, 308, 33, 46] # Gulf Stream

crs = ccrs.PlateCarree()
ax=None
fig = plt.figure(figsize=(15,10))
if ax is None:
    ax = plt.subplot(1, 1, 1, projection=crs)
ax.set_extent(bbox, crs)
# Add Filled Coastline
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='k', zorder=10)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
gl.top_labels = gl.right_labels = False
gl.xformatter = LongitudeFormatter(degree_symbol='')
gl.yformatter = LatitudeFormatter(degree_symbol='')
levels = np.arange(-30,31,1)
colorplot = plt.contourf(ds_eta.longitude-360,ds_eta.latitude,ds_eta.sla[t_step,:,:]*100,cmap = 'cmo.balance',levels=levels)
cbar = plt.colorbar(colorplot,fraction=0.025, pad=0.04)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('SSH [cm]', size='25', labelpad=25)
plt.title('Total SSH: ' + str(time[t_step]),fontsize=25)
gl.xlabel_style, gl.ylabel_style = {'fontsize': 25}, {'fontsize': 25}
plt.savefig(pth + 'Figures/SSH-BT-BC/totalssh_tstep' + str(t_step) + '.png',dpi=500)
plt.show()

ax=None
fig = plt.figure(figsize=(15,10))
if ax is None:
    ax = plt.subplot(1, 1, 1, projection=crs)
ax.set_extent(bbox, crs)
# Add Filled Coastline
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='k', zorder=10)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
gl.top_labels = gl.right_labels = False
gl.xformatter = LongitudeFormatter(degree_symbol='')
gl.yformatter = LatitudeFormatter(degree_symbol='')
levels = np.arange(-0.3,0.31,0.01)
colorplot = plt.contourf(ds_h.longitude-360,ds_h.latitude,baroclinic[t_step,:,:],cmap = 'cmo.balance',levels=levels)
cbar = plt.colorbar(colorplot,fraction=0.025, pad=0.04)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('SSH [cm]', size='25', labelpad=25)
plt.title('Baroclinic SSH: ' + str(time[t_step]),fontsize=25)

gl.xlabel_style, gl.ylabel_style = {'fontsize': 25}, {'fontsize': 25}
plt.savefig(pth + 'Figures/SSH-BT-BC/baroclinicssh_tstep' + str(t_step) + '.png',dpi=500)

plt.show()

ax=None
fig = plt.figure(figsize=(15,10))
if ax is None:
    ax = plt.subplot(1, 1, 1, projection=crs)
ax.set_extent(bbox, crs)
# Add Filled Coastline
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='k', zorder=10)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
gl.top_labels = gl.right_labels = False
gl.xformatter = LongitudeFormatter(degree_symbol='')
gl.yformatter = LatitudeFormatter(degree_symbol='')
levels = np.arange(-0.3,0.31,0.01)
colorplot = plt.contourf(ds_h.longitude-360,ds_h.latitude,barotropic[t_step,:,:],cmap = 'cmo.balance',levels=levels)
cbar = plt.colorbar(colorplot,fraction=0.025, pad=0.04)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('SSH [cm]', size='25', labelpad=25)
plt.title('Barotropic SSH: ' + str(time[t_step]),fontsize=25)

gl.xlabel_style, gl.ylabel_style = {'fontsize': 25}, {'fontsize': 25}
plt.savefig(pth + 'Figures/SSH-BT-BC/barotropicssh_tstep' + str(t_step) + '.png',dpi=500)

plt.show()

