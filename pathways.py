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
import scipy
sns.set()

w = scipy.signal.firwin(numtaps = 12, cutoff = 1/(12*11), window = 'lanczos')

path      = '/Users/lillienders/Desktop/First Generals/Data/Observations/'
f_gsi      = 'gsi_array.nc'
gsi        = xr.open_dataarray(path + f_gsi)
gsi_filt   = scipy.signal.lfilter(w, 1.0, gsi)
gsi_filt   = (gsi_filt - np.nanmean(gsi_filt))/np.nanstd(gsi_filt)

f_ws    = 'wind_stress_anomaly_93_22.nc'
ws      = xr.open_dataarray(path+f_ws)
ws_filt = np.zeros(ws.shape)
for lat in range(len(ws.latitude)):
    for lon in range(len(ws.longitude)):
        ws_filt[:,lat,lon]  = scipy.signal.lfilter(w, 1.0, ws[:,lat,lon])
ws_filt = (ws_filt - np.nanmean(ws_filt))/np.nanstd(ws_filt)

ds_ws_cli = xr.open_dataset(path+'wind_stress_climatology_93_22.nc')
ds_cli = ds_ws_cli.mean(dim='month')

# WS Max Lag is when NAO Leads by 7 months
lag = 7
ntime = len(gsi_filt)
wslags = np.zeros((len(ws.latitude),len(ws.longitude)))
wsleads = np.zeros((len(ws.latitude),len(ws.longitude)))
for lat in range(len(ws.latitude)):
    for lon in range(len(ws.longitude)):
        gsi_in  = gsi_filt[lag:]
        ws_in  = ws_filt[:(ntime-lag),lat,lon]
        ws_in[np.isnan(ws_in)] = 0
        gsi_in[np.isnan(gsi_in)] = 0
        corr = np.corrcoef(ws_in,gsi_in)[0,1]
        wsleads[lat,lon] = (corr)

ci_lower,ci_upper = get_cis(ws[:,10,10],gsi,n_lags=60,jfm = False)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean as cmo
import matplotlib.ticker as tkr

bbox = [270, 360, 10, 70]
crs = ccrs.PlateCarree()
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(1, 1, 1, projection=crs)
ax.set_extent(bbox, crs)
# Add Filled Coastline
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='k', zorder=1)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
gl.top_labels = gl.right_labels = False
gl.xformatter = LongitudeFormatter(degree_symbol='')
gl.yformatter = LatitudeFormatter(degree_symbol='')
levels = np.linspace(-0.65, 0.65, 66)
colorplot = plt.contourf(ws.longitude, ws.latitude, wsleads, cmap='cmo.balance', zorder=0, levels=levels)
sig_contours = plt.contour(ws.longitude,ws.latitude,wsleads, colors='k',levels=[-np.nanquantile(wsleads.flatten(),0.95),np.nanquantile(wsleads.flatten(),0.95)])

cbar = plt.colorbar(colorplot, fraction=0.020, pad=0.02,
                    ticks=[-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
cbar.ax.set_yticklabels(['-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Correlation Coefficient', size='25', labelpad=25)
gl.xlabel_style, gl.ylabel_style = {'fontsize': 25}, {'fontsize': 25}

plt.savefig('/Users/lillienders/Desktop/First Generals/Figures/' + 'gsi_ws_max_corr.png',
            format='png', bbox_inches="tight", dpi=500)
