# -*- coding: utf-8 -*-
#Imports
from scipy.stats.stats import pearsonr
from scipy.stats import kurtosis, skew
from novainstrumentation.peaks import bigPeaks
from novainstrumentation.tools import plotfft
import novainstrumentation as ni
import numpy as np
########################################################################################################################
# ####################################  TEMPORAL DOMAIN  ############################################################# #
########################################################################################################################


# Autocorrelation
def autocorr(sig):
    """Compute autocorrelation along the specified axis.

    Parameters
    ----------
    sig: ndarray
        input from which autocorrelation is computed.

    Returns
    -------
    corr: float
        Cross correlation of 1-dimensional sequence.
    """
    return float(np.correlate(sig, sig))


def zero_cross(sig):
    """Compute Zero-crossing rate along the specified axis.
         total number of times that the signal changes from positive to negative or vice versa, normalized by the window length.
    Parameters
    ----------
    sig: ndarray
        input from which the zero-crossing rate are computed.

    Returns
    -------
    count_vector: int
        number of times that signal value cross the zero axe.
    """
    #return np.where(ny.diff(ny.sign(sig)))[0]

    return len(np.where(np.diff(np.sign(sig)))[0])

def correlation(signal1, signal2):
    """Compute the Pearson Correlation coefficient along the specified axes.

    Parameters
    ----------
    signal1,signal2: ndarrays
        inputs from which correlation is computed.

    Returns
    -------
    pearson_coeff: int
        measures the linear relationship between tow datasets.
    """
    return pearsonr(signal1, signal2)[0]

def interq_range(sig):
    """Compute interquartile range along the specified axis.

        Parameters
        ----------
        sig: ndarray
            input from which interquartile range is computed.

        Returns
        -------
        corr: float
            Interquartile range of 1-dimensional sequence.
        """
    #ny.percentile(sig, 75) - ny.percentile(sig, 25)
    return np.percentile(sig, 75) - np.percentile(sig, 25)



########################################################################################################################
# ############################################ SPECTRAL DOMAIN ####################################################### #
########################################################################################################################

# Compute Fundamental Frequency
def fundamental_frequency(s, FS):
    # TODO: review fundamental frequency to guarantee that f0 exists
    # suggestion peak level should be bigger
    # TODO: explain code
    """Compute fundamental frequency along the specified axes.
    Parameters
    ----------
    s: ndarray
        input from which fundamental frequency is computed.
    FS: int
        sampling frequency
    Returns
    -------
    f0: int
       its integer multiple best explain the content of the signal spectrum.
    """

    s = s - np.mean(s)
    f, fs = plotfft(s, FS, doplot=False)

    fs = fs[1:len(fs) / 2]
    f = f[1:len(f) / 2]

    cond = find(f > 0.5)[0]
    bp = bigPeaks(fs[cond:], 0)

    if not bp:
        f0 = 0
    else:
        bp = bp + cond
        f0 = f[min(bp)]

    return f0


def max_frequency(sig, FS):
    """Compute max frequency along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which max frequency is computed.
    FS: int
        sampling frequency
    Returns
    -------
    f_max: int
       0.95 of max_frequency using cumsum.
    """

    f, fs = plotfft(sig, FS, doplot=False)
    t = np.cumsum(fs)

    try:
        ind_mag = np.where(t > t[-1]*0.95)[0][0]
    except:
        ind_mag = np.argmax(t)
    f_max = f[ind_mag]

    return f_max


def median_frequency(sig, FS):
    """Compute median frequency along the specified axes.
    Parameters
    ----------
    sig: ndarray
        input from which median frequency is computed.
    FS: int
        sampling frequency
    Returns
    -------
    f_max: int
       0.50 of max_frequency using cumsum.
    """

    f, fs = plotfft(sig, FS, doplot=False)
    t = np.cumsum(fs)
    ind_mag = np.where(t > t[-1] * 0.50)[0][0]
    f_median = f[ind_mag]

    return f_median


def ceps_coeff(sig,coefNumber):
    """Compute cepstral coefficients along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which cepstral coefficients are computed.
    coefNumber:
    Returns
    -------
    cc: ndarray

    """

    est=lpc_coef(sig,coefNumber)
    cc=lpcar2cc(est)
    if len(cc)==1:
        cc=float(cc)
    else:
        cc=tuple(cc)

    return cc


# Power Spectrum Density
def power_spectrum(sig, FS):
    """Compute power spectrum density along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which cepstral coefficients are computed.
    FS: scalar
        sampling frequency
    Returns
    -------
    max_power: ndarray
        max value of the power spectrum.
    peak_freq: ndarray
        max frequency corresponding to the elements in power spectrum.

    """
    if np.std(sig) == 0:
        return float(max(psd(sig, int(FS))[0]))
    else:
        return float(max(psd(sig/np.std(sig), int(FS))[0]))


def power_bandwidth(sig, FS, samples):
    """Compute power spectrum density bandwidth along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which cepstral coefficients are computed.
    FS: scalar
        sampling frequency
    samples: int
        number of bands
    Returns
    -------
    bandwidth: ndarray
        power in bandwidth
    """
    bd = []
    bdd = []
    power, freq = psd(sig/np.std(sig), FS)

    for i in range(len(power)):
        bd += [float(power[i])]

    bdd += bd[:samples]

    return tuple(bdd)


def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral domain
    freqs = np.zeros(nfilt+2) #modified
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)

    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))

    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1] #modified
        hi = freqs[i+2] #modified

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs


########################################################################################################################
####################################### STATISTICAL DOMAIN #############################################################
########################################################################################################################

# Kurtosis
def calc_kurtosis(sig):
     """Compute kurtosis along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which kurtosis is computed.

    Returns
    -------
    k: int
       kurtosis result.
    """
     return kurtosis(sig)


# Skewness
def calc_skewness(sig):
     """Compute skewness along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which skewness is computed.

    Returns
    -------
    s: int
       skewness result.
    """
     return skew(sig)


# Mean
def calc_mean(sig):
     """Compute mean value along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which mean is computed.

    Returns
    -------
    m: int
       mean result.
    """
     # m = mean(sig)
     return np.mean(sig)


# Standard Deviation
def calc_std(sig):
     """Compute standard deviation (std) along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which std is computed.

    Returns
    -------
    std_value: int
       std result.
    """
     return np.std(sig)


# Interquartile Range
def calc_iqr(sig):
     """Compute interquartile range along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which interquartile range is computed.

    Returns
    -------
    iqr: int
       interquartile range result.
    """
     # iqr = subtract(*percentile(sig, [75, 25]))
     return np.percentile(sig, 75) - np.percentile(sig, 25)


# Mean Absolute Deviation
def calc_mad(sig):
     """Compute mean absolute deviation along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which mean absolute deviation is computed.

    Returns
    -------
    mad: int
       mean absolute deviation result.
    """
     m = np.median(sig)
     diff = [abs(x-m) for x in sig]

     return np.mean(diff)


def calc_madiff(sig):
    """Compute mean absolute differences along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which mean absolute deviation is computed.

   Returns
   -------
   mad: int
      mean absolute difference result.
   """

    return np.mean(abs(np.diff(sig)))

def calc_sadiff(sig):
    """Compute sum of absolute differences along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which sum absolute diff is computed.

   Returns
   -------
   mad: int
      sum absolute difference result.
   """

    return np.sum(abs(np.diff(sig)))

def calc_mdiff(sig):
    """Compute mean of differences along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which mean absolute deviation is computed.

   Returns
   -------
   mad: int
      mean absolute difference result.
   """

    return np.mean(np.diff(sig))

# Root Mean Square
def rms(sig):
     """Compute root mean square along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which root mean square is computed.

    Returns
    -------
    rms: int
       square root of the arithmetic mean (average) of the squares of the original values.
    """

     return np.sqrt(np.sum(np.array(sig)**2)/len(sig))

# Histogram
def hist(sig,nbins,r):
    """Compute histogram along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    nbins: int
     the number of equal-width bins in the givel range.
    rang: [float,float]
        the lower and upper range of the bins.
    Returns
    -------
    histsig: ndarray
        the values of the histogram

    bin_edges: ndarray
        the bin_edges, 'len(hist)+1'

    """

    histsig, bin_edges = np.histogram(sig[::10], bins=nbins, range = r, density=False) #TODO:subsampling parameter

    bin_edges = bin_edges[:-1]
    bin_edges += (bin_edges[1]-bin_edges[0])/2.

    return tuple(histsig)


# Histogram for json format
def hist_json(sig, nbins, r):
    """Compute histogram along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    nbins: int
     the number of equal-width bins in the givel range.
    rang: float
        the lower(-r) and upper(r) range of the bins.
    Returns
    -------
    histsig: ndarray
        the values of the histogram

    bin_edges: ndarray
        the bin_edges, 'len(hist)+1'

    """

    histsig, bin_edges = np.histogram(sig, bins=nbins, range=[-r, r], density=False) #TODO:subsampling parameter

    # bin_edges = bin_edges[:-1]
    # bin_edges += (bin_edges[1]-bin_edges[0])/2.

    return tuple(histsig)

def minpeaks(sig):
    """Compute number of minimum peaks along the specified axes.

    Parameters
    ----------
    sig: ndarray

    Returns
    -------
     float
        min number of peaks

    """
    diff_sig = np.diff(sig)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if (diff_sig[nd]<0 and diff_sig[nd + 1]>0)])


def maxpeaks(sig):
    """Compute number of peaks along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    type: string
        can be 'all', 'max', and 'min', and expresses which peaks are going to be accounted
    Returns
    -------
    num_p: float
        total number of peaks

    """
    diff_sig = np.diff(sig)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if (diff_sig[nd+1]<0 and diff_sig[nd]>0)])

def all_pk(sig):
    """Compute number of peaks along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    type: string
        can be 'all', 'max', and 'min', and expresses which peaks are going to be accounted
    Returns
    -------
    num_p: float
        total number of peaks

    """

    return max_pk(sig) + min_pk(sig)
########################################################################################################################
# ######################################### CUIDADO FEATURES ######################################################### #
########################################################################################################################

#create time
def compute_time(sign, FS):
    """Creates the signal correspondent time array.
    """
    time = range(len(sign))
    time = [float(x)/FS for x in time]
    return time


def signal_energy(sign, time):
    """Computes the energy of the signal. For that, first is made the segmentation of the signal in 10 windows
    and after it's considered that the energy of the signal is the sum of all calculated points in each window.
    Parameters
    ----------
    sign: ndarray
        input from which max frequency is computed.
    Returns
    -------
    energy: float list
       signal energy.
    time_energy: float list
        signal time energy
    """

    window_len = len(sign)

    # window for energy calculation
    window = window_len/10 # each window of the total signal will have 10 windows

    energy = np.zeros(int(window_len/window))
    time_energy = np.zeros(int(window_len/window))

    i = 0
    for a in range(0, len(sign) - window, window):
        energy[i] = np.sum(np.array(sign[a:a+window])**2)
        interval_time = time[int(a+(window/2))]
        time_energy[i] = interval_time
        i += 1

    return list(energy), list(time_energy)


# Temporal Centroid
def centroid(sign, FS):
    """Computes the centroid along the time axis.
    ----------
    sign: ndarray
        input from which max frequency is computed.
    fs: int
        signal sampling frequency.
    Returns
    -------
    centroid: float
        temporal centroid
    """

    time = compute_time(sign, FS)

    energy, time_energy=signal_energy(sign, time)

    total_energy = np.dot(np.array(time_energy),np.array(energy))
    energy_sum = np.sum(energy)

    if (energy_sum == 0 or total_energy == 0):
        centroid = 0
    else:
        centroid = total_energy / energy_sum
    return centroid


# Total Energy
def total_energy_(sign, FS):
    """
    Compute the acc_total power, using the given windowSize and value time in samples

    """
    time = compute_time(sign, FS)

    return np.sum(np.array(sign)**2)/(time[-1]-time[0])


# Spectral Centroid
def spectral_centroid(sign, fs): #enter the portion of the signal
    f, ff = ni.plotfft(sign, fs)
    return np.dot(f,ff/np.sum(ff))


# Spectral Spread
def spectral_spread(sign, fs):
    f, ff = ni.plotfft(sign, fs)
    spect_centr = spectral_centroid(sign, fs)
    return np.dot(((f-spect_centr)**2),(ff / np.sum(ff)))


# Spectral Skewness
def spectral_skewness(sign, fs):
    f, ff = ni.plotfft(sign, fs)
    spect_centr = spectral_centroid(sign, fs)
    skew = ((f-spect_centr)**3)*(ff / np.sum(ff))
    spect_skew = np.sum(skew)
    return spect_skew/(spectral_spread(sign, fs)**(3/2))


# Spectral Kurtosis
def spectral_kurtosis(sign, fs):
    f, ff = ni.plotfft(sign, fs)
    spect_kurt = ((f-spectral_centroid(sign, fs))**4)*(ff / np.sum(ff))
    skew = np.sum(spect_kurt)
    return skew/(spectral_spread(sign, fs)**2)


# Spectral Slope
def spectral_slope(sign, fs):
    """Computes the constants m and b of the function aFFT = mf + b, obtained by linear regression of the
    spectral amplitude.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    m: float
        slope
    b: float
        y-intercept
    """
    f, ff = ni.plotfft(sign, fs)
    return (len(f) * np.dot(f, ff) - np.sum(f) * np.sum(ff)) / (len(f) * np.dot(f, f) - np.sum(f) ** 2)


# # Spectral Decrease
# def spectral_decrease(sign, fs):
#
#     f, ff = ni.plotfft(sign, fs)
#     energy, freq = signal_energy(ff, f)
#
#     soma_den = 0
#     soma_num = 0
#     k = len(energy)
#
#     for a in range(2, k):
#         soma_den = soma_den+energy[a]
#         soma_num = soma_num+((energy[a]-energy[0])/(a-1))
#
#     if soma_den == 0:
#         spect_dec = 0
#     else:
#         spect_dec = (1/soma_den)*soma_num
#
#     return spect_dec

# Spectral Decrease
def spectral_decrease(sign, fs):

    f, ff = ni.plotfft(sign, fs)

    k = len(ff)
    soma_num = 0
    for a in range(2, k):
        soma_num = soma_num + ((ff[a]-ff[1])/(a-1))

    ff2 = ff[2:]
    soma_den = 1/np.sum(ff2)

    if soma_den == 0:
        return 0
    else:
        return soma_den * soma_num

def spectral_roll_on(sign, fs):

    output = None
    f, ff = ni.plotfft(sign, fs)
    cum_ff = np.cumsum(ff)
    value = 0.05*(sum(ff))

    for i in range(len(ff)):
        if cum_ff[i] >= value:
            output = f[i]
            break
    return output


def spectral_roll_off(sign, fs):
    """Compute the spectral roll-off of the signal, i.e., the frequency where 95% of the signal energy is contained
    below of this value.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    roll_off: float
        spectral roll-off
    """
    output = None
    f, ff = ni.plotfft(sign, fs)
    cum_ff = np.cumsum(ff)
    value = 0.95*(sum(ff))

    for i in range(len(ff)):
        if cum_ff[i] >= value:
            output = f[i]
            break
    return output


def curve_distance(sign, fs):

    f, ff = ni.plotfft(sign, fs)
    cum_ff = np.cumsum(ff)
    points_y = np.linspace(0, cum_ff[-1], len(cum_ff))

    return np.sum(points_y-cum_ff)


# Temporal Variation of Spectrum
def spect_variation(sign, fs):
    '''
    returns the temporal variation
    '''

    f, ff = ni.plotfft(sign, fs)
    energy, freq = signal_energy(ff, f)

    sum1 = 0
    sum2 = 0
    sum3 = 0
    for a in range(len(energy)-1):
        sum1 = sum1+(energy[a-1]*energy[a])
        sum2 = sum2+(energy[a-1]**2)
        sum3 = sum3+(energy[a]**2)

    if (sum2 == 0 or sum3 == 0):
        variation = 1
    else:
        variation = 1-((sum1)/((sum2**0.5)*(sum3**0.5)))

    return variation


# Variance
def variance(sign, FS):
    '''
    searches and returns the signal variance
    '''
    time = compute_time(sign, FS)
    soma_den = 0
    soma_num = 0
    for z in range(0, len(sign)):
        soma_num = soma_num + (time[z]*((sign[z]*np.mean(sign))**2))
        soma_den = soma_den + time[z]

    return soma_num/soma_den


def covariance(sign1, sign2):
    '''
    searches and returns the signal covariance
    '''
    return np.cov(sign1, sign2)[0][1]


#Deviation
def deviation(sign, FS):

    time = compute_time(sign, FS)
    soma_den = 0
    soma_num = 0
    for z in range(0, len(sign)-1):
        soma_num = soma_num+(time[z+1]*(sign[z+1]-sign[z]))
        soma_den = soma_den+time[z+1]

    return soma_num/soma_den


def linear_regression(sign):

    t = np.linspace(0, 5, len(sign))

    return np.polyfit(t, sign, 1)[0]


def spectral_maxpeaks(sign, FS):
    """Compute number of peaks along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    type: string
        can be 'all', 'max', and 'min', and expresses which peaks are going to be accounted
    Returns
    -------
    num_p: float
        total number of peaks

    """
    f, ff = ni.plotfft(sign, FS)
    diff_sig = np.diff(ff)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if (diff_sig[nd+1]<0 and diff_sig[nd]>0)])
