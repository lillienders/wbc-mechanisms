import numpy as np
def linear_detrend(dataset, var = 'sla'):
    """
    Remove a linear trend from each grid point
    Inputs:
    - dataset: xarray dataset, read in from read_file and formated in (time,lat,lon) dimensions
    Outputs:
     - dataset: xarray dataset, formated in (time,lat,lon) dimensions with linear trend removed
    """
    ds_temp = dataset
    time_idx = np.linspace(0, len(ds_temp.time) - 1, len(ds_temp.time))
    ds_temp['time'] = time_idx
    ds_poly = ds_temp.polyfit(dim='time', deg=1)
    indices = np.arange(len(ds_temp.time))
    fit_string = var + '_polyfit_coefficients'
    slope = np.array(ds_poly[fit_string][0]).flatten()
    intercept = np.array(ds_poly[fit_string][1]).flatten()
    lin_fit = np.zeros((len(ds_temp.time), len(slope)))
    for loc in range(len(slope)):
        lin_fit[:, loc] = slope[loc] * indices + intercept[loc]
    lin_fit = np.reshape(lin_fit, (len(ds_temp.time), len(ds_temp.latitude), len(ds_temp.longitude)))
    detrended_series = ds_temp[var] - lin_fit
    dataset[var] = detrended_series
    return (dataset)

def get_cis(forcing_pc,gsi_ts,n_lags = 31,jfm = False):
    n_sim  = 1000
    uci = np.zeros(n_lags+1)
    lci = np.zeros(n_lags+1)
    for lag in range(n_lags+1):
        len_ts = len(forcing_pc) - lag
        x_scram = phase_scramble_bs(forcing_pc[lag:-1],n_sim=n_sim)
        y_scram = phase_scramble_bs(gsi_ts[0:(-(lag+1))],n_sim=n_sim)
        if jfm == True:
            y_scram = phase_scramble_bs(gsi_ts[0:(30-(lag+1))],n_sim=n_sim)
        coefs = np.zeros(n_sim)
        for n in range(n_sim):
            coefs[n] = np.corrcoef(x_scram[:,n],y_scram[:,n])[0,1]
        uci[lag] = np.quantile(coefs,0.975)
        lci[lag] = np.quantile(coefs,0.025)

    uci_r = np.array(uci[::-1][:-1])
    lci_r = np.array(lci[::-1][:-1])
    ci_upper = np.concatenate((uci_r,uci))
    ci_lower = np.concatenate((lci_r,lci))
    if jfm == True:
        ci_upper = np.interp(x,xf,ci_upper)/1.5
        ci_lower = np.interp(x,xf,ci_lower)/1.5
    return(ci_lower,ci_upper)


def phase_scramble_bs(x, n_sim=1000):
    n_frms = len(x)
    if n_frms % 2 == 0:
        n_frms = n_frms - 1
        x = x[0:n_frms]

    blk_sz = int((n_frms - 1) / 2)
    blk_one = np.arange(1, blk_sz + 1)
    blk_two = np.arange(blk_sz + 1, n_frms)

    fft_x = np.fft.fft(x)
    ph_rnd = np.random.random((blk_sz, n_sim))

    ph_blk_one = np.exp(2 * np.pi * 1j * ph_rnd)
    ph_blk_two = np.conj(np.flipud(ph_blk_one))

    fft_x_surr = np.tile(fft_x[:, None], (1, n_sim))
    fft_x_surr[blk_one, :] = fft_x_surr[blk_one] * ph_blk_one
    fft_x_surr[blk_two, :] = fft_x_surr[blk_two] * ph_blk_two

    scrambled = np.real(np.fft.ifft(fft_x_surr, axis=0))
    return scrambled