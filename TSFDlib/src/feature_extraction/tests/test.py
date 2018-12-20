from numpy.testing import assert_array_equal, run_module_suite
import numpy as np
import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)
import matplotlib.pylab as plt
import novainstrumentation as ni
import time


def test_mean():
    a = [0,0,0,0,0]
    b = [1,2,3,4,5]
    c = [-1,-2,-3,-4,-5]
    np.testing.assert_equal(np.mean(a), 0)
    np.testing.assert_equal(np.mean(b), 3)
    np.testing.assert_equal(np.mean(c), -3)


def test_max():
    a = [0,0,0,0,0]
    b = [1,2,3,4,5]
    c = [-1,-2,-3,-4,-5]
    np.testing.assert_equal(np.max(a), 0)
    np.testing.assert_equal(np.max(b), 5)
    np.testing.assert_equal(np.max(c), -1)


def test_min():
    a = [0,0,0,0,0]
    b = [1,2,3,4,5]
    c = [-1,-2,-3,-4,-5]
    np.testing.assert_equal(np.min(a), 0)
    np.testing.assert_equal(np.min(b), 1)
    np.testing.assert_equal(np.min(c), -5)


def test_calc_mad():
    # output = np.sum(np.abs(b-np.median(b)))/len(b)
    a = [0,0,0,0,0]
    b = [1,2,3,4,5]
    np.testing.assert_equal(calc_mad(a), 0)
    np.testing.assert_equal(calc_mad(b), 1.2)


def test_variance():
    # np.sqrt(np.mean(abs(b - np.mean(b))**2))**2
    # pvariance
    a = [0,0,0,0,0]
    b = [1,2,3,4,5]
    np.testing.assert_equal(np.var(a), 0)
    np.testing.assert_almost_equal(np.var(b), 2, decimal=5)

# def test_centroid():
#     a = np.arange(0,20)
#     fs = len(a) / a[-1]
#     t = np.arange(len(a))
#     output = np.sum(rms(a)*t)/np.sum(rms(a))
#     np.testing.assert_equal(centroid(a,fs), output)


def test_std(): #population standard deviation
    # np.sqrt(np.mean(abs(b - np.mean(b))**2)
    a = [0,0,0,0,0]
    b = [1,2,3,4,5]
    np.testing.assert_equal(np.std(a), 0)
    np.testing.assert_almost_equal(np.std(b), 1.4142135623730951, decimal=5)


def test_rms():
    # np.sqrt(np.mean(b**2))
    a = [0,0,0,0,0]
    b = np.array([1,2,3,4,5])
    np.testing.assert_equal(np.std(a), 0)
    np.testing.assert_almost_equal(rms(b), 3.3166247903553998, decimal=1)


def test_int_range():
    # np.percentile(b, 75) - np.percentile(b, 25)
    a = [0,0,0,0,0]
    b = np.array([1,2,3,4,5])
    np.testing.assert_equal(np.std(a), 0)
    np.testing.assert_almost_equal(interq_range(b), 2, decimal=5)


def test_zeroCross():
    a = np.zeros(5)
    b = np.array([-1,2,-3,4,5])
    np.testing.assert_equal(zero_cross(a), 0)
    np.testing.assert_equal(zero_cross(b), 3)


def test_corr():
    # corrcoef(a, b)[0, 1]
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    c = [-5, 0, -2, 6, -3]
    d = [-1, -2, -3, -4, -5]

    np.testing.assert_equal(correlation(a, b), 1)
    np.testing.assert_almost_equal(correlation(b, c), 0.37582, decimal=5)
    np.testing.assert_equal(correlation(a, d), -1)


def test_autocorr():
    a = np.array([0, 0, 0, 0, 0])
    b = np.array([1, 2, 3, 4, 5])

    np.testing.assert_equal(autocorr(a), 0)
    np.testing.assert_equal(autocorr(b), 55)


def test_skew():
    # corrcoef(a, b)[0, 1]
    a = [0, 0, 0, 0, 0]
    b = [1, 1, 1, 7, 7, 7, 7, 8, 9, 6]
    c = [1, 1, 1, 1, 2, 2, 2, 3, 3, 7, 8]
    np.testing.assert_equal(skew(a), 0)
    np.testing.assert_equal(skew(b), -0.6647589964198858)
    np.testing.assert_equal(skew(c), 1.3437847545430435)


def test_kurtosis():
    np.random.seed(seed=23)
    x = np.random.normal(0, 2, 1000000)
    np.testing.assert_almost_equal(kurtosis(x), 0, decimal=1)
    mu, sigma = 0, 0.1
    x = mu + sigma * np.random.randn(100)
    np.testing.assert_almost_equal(kurtosis(x),  0.7301126158288906, decimal=1)


def test_max_fre():
    f = 0.2
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f * x / Fs)
    np.testing.assert_almost_equal(max_frequency(y, Fs), 0.2, decimal=0)
    f2 = 0.5
    y = np.cos(2 * np.pi * f * x / Fs) + np.cos(2 * np.pi * f2 * x / Fs)
    np.testing.assert_almost_equal(max_frequency(y, Fs), 0.5, decimal=0)
    a = np.ones(1000)
    np.testing.assert_almost_equal(max_frequency(a, 1), 0, decimal=0)


def test_med_fre():
    f = 0.2
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f * x / Fs)
    np.testing.assert_almost_equal(median_frequency(y, Fs), 0.1, decimal=0)
    f2 = 1
    y = np.cos(2 * np.pi * f * x / Fs) + np.cos(2 * np.pi * f2 * x / Fs) # ???
    np.testing.assert_almost_equal(median_frequency(y, Fs), 0.010000300007000151, decimal=0)
    a = np.ones(1000)
    np.testing.assert_almost_equal(median_frequency(a, 1), 0, decimal=0)
    # f, fs = ni.plotfft(y, 100, doplot=False)
    # ny.percentile(fs, 50)
    # fs_sorted = np.sort(fs)
    # f_med = f[fs.tolist().index(fs_sorted[len(fs) // 2])]


def fondamentals(frames0, samplerate):
    mid = 16
    sample = mid*2+1
    res = []
    for first in xrange(sample):
        last = first-sample
        frames = frames0[first:last]
        res.append(_fondamentals(frames, samplerate))
    res = sorted(res)
    return res[mid] # We use the medium value


def _fondamentals(frames, samplerate):
    frames2=frames*hamming(len(frames));
    frameSize=len(frames);
    ceps=np.fft.ifft(np.log(np.abs(fft(frames2))))
    nceps=ceps.shape[-1]*2/3
    peaks = []
    k=3
    while(k < nceps - 1):
        y1 = (ceps[k - 1])
        y2 = (ceps[k])
        y3 = (ceps[k + 1])
        if (y2 > y1 and y2 >= y3): peaks.append([float(samplerate)/(k+2),abs(y2), k, nceps])
        k=k+1
    maxi=max(peaks, key=lambda x: x[1])
    return maxi[0]


def test_fund_fre():
    f = 20
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x)/x[-1]
    y = np.sin(2 * np.pi * f * x / Fs)
    # call fondamentals(y, Fs)
    np.testing.assert_almost_equal(fundamental_frequency(y, Fs), 0, decimal=0)

def test_hist():
    x = np.ones(10)
    np.testing.assert_almost_equal(hist_json(x, 10, 5), (0, 0, 0, 0, 0, 0, 10, 0, 0, 0), decimal=0)


def test_power_spec():
    f = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f * x / Fs)
    # max(plt.psd(y/np.std(y), int(Fs))[0])
    np.testing.assert_almost_equal(power_spectrum(y, Fs), 32.99582867944011, decimal=5)


def test_total_energy():
    f = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f * x / Fs)
    time = compute_time(y, Fs)
    # sum(abs(y)**2.0)/(time[-1]-time[0])
    np.testing.assert_almost_equal(total_energy_(y, Fs), 50.001500029605715, decimal=5)
    y = [0,0,0,0,0,0,0]
    np.testing.assert_equal(total_energy_(y, Fs), 0)


def centroid(spectrum, f):
    s = spectrum / np.sum(spectrum)
    return np.dot(s, f)


def test_spectral_centroid():
    f1 = 0.1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    f, ff = ni.plotfft(y, Fs)
    # centroid(ff, f)
    np.testing.assert_almost_equal(spectral_centroid(y, Fs), 0.0021658004162115937, decimal=2)


def test_spectral_spread():
    f1 = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    f, ff = ni.plotfft(y, Fs)
    p = ff / np.sum(ff)
    # np.dot(((f-np.mean(centroid(ff, f)))**2),p)
    np.testing.assert_almost_equal(spectral_spread(y, Fs), 0.31264360946306424, decimal=5)


def test_spectral_skewness():
    f1 = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    np.testing.assert_almost_equal(spectral_skewness(y, Fs), 34.789505786824407, decimal=5)
    np.random.seed(seed=23)
    x = np.random.normal(0, 2, 1000000)
    np.testing.assert_almost_equal(spectral_skewness(x, Fs), 0, decimal=2)


def _kurtosis(spectrum, f, spr, c):
    s = spectrum / np.sum(spectrum)
    return np.dot(s, (f - c)**4) / spr**2


def test_spectral_kurtosis():
    # np.random.seed(seed=1)
    # x = np.random.normal(0, 2, 10000)
    # np.testing.assert_almost_equal(spectral_kurtosis(x, 100), 0, decimal=3)
    # mu, sigma = 20, 0.1
    # x = mu + sigma * np.random.randn(100)
    #_kurtosis(ff,f,sp,c)
    f1 = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    f, ff = ni.plotfft(y, Fs)
    sp = spectral_spread(y, Fs)
    c = spectral_centroid(y, Fs)
    np.testing.assert_almost_equal(spectral_kurtosis(y, Fs), 4293.9932884381524, decimal=0)


def slope(ff, f):
    c = np.vstack([f, np.ones(len(f))]).T
    return np.linalg.lstsq(c, ff)[0][0]


def test_spectral_slope():
    f1 = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    f, ff = ni.plotfft(y, Fs)
    # slope(ff, f)
    np.testing.assert_almost_equal(spectral_slope(y, Fs), -0.1201628830466239, decimal=5)


def test_spectral_decrease():
    f1 = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    np.testing.assert_almost_equal(spectral_decrease(y, Fs), 0.12476839555000206, decimal=5)


def roll_on(spectrum, f):
    sqr = np.square(spectrum)
    total = np.sum(sqr)
    s = 0
    output = 0
    for i in range(0, len(f)):
        s += sqr[i]
        if s >= 0.05 * total:
            output = f[i]
            break
    return output


def test_spectral_roll_on():
    f1 = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    f, ff = ni.plotfft(y, Fs)
    #roll_on(ff, f)
    np.testing.assert_almost_equal(spectral_roll_on(y, Fs), 0.010000300007000151, decimal=5)


def roll_off(spectrum, f):
    sqr = np.square(spectrum)
    total = np.sum(sqr)
    s = 0
    output = 0
    for i in range(0, len(f)):
        s += sqr[i]
        if s >= 0.95 * total:
            output = f[i]
            break
    return output


def test_spectral_roll_off():
    f1 = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    f, ff = ni.plotfft(y, Fs)
    #roll_off(ff, f)
    np.testing.assert_almost_equal(spectral_roll_off(y, Fs), 0.010000300007000151, decimal=5)


def test_spectral_curve_distance():
    f1 = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    f, ff = ni.plotfft(y, Fs)
    np.testing.assert_almost_equal(curve_distance(y, Fs), -1251684201.9742942, decimal=5)


def test_spect_variation():
    f1 = 1
    sample = 1000
    x = np.arange(0, sample, 0.01)
    Fs = len(x) / x[-1]
    y = np.sin(2 * np.pi * f1 * x / Fs)
    f, ff = ni.plotfft(y, Fs)
    np.testing.assert_almost_equal(spect_variation(y, Fs), 0.9999999999957343, decimal=5)


if __name__ == "__main__":
    run_module_suite()

