import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import generic_filter

def dp_pcnt_swept(data, method=None):
    # dead pixel corrections for PCnt/Swept mode
    method = method or 'mean'  # or median to keep it integer
    width = 5
    vn = 1.0
    z = (width-1)
    vz = -vn/z
    kernel = np.array([[vz, vz, vn, vz, vz]])
    activation = convolve(data, kernel)

    correction = getattr(np, method)(activation, axis=0)
    correction[correction < 1.] = 0
    correction = correction[np.newaxis, :]

    corrected = data - correction
    corrected[corrected < 0] = 0
    return corrected


def dp_pcnt_dither(data, dithsteps):
    width = 5
    n = dithsteps  # dither steps
    vn = 1.0
    z = (width-1)*n + 4*width
    vz = -vn*n/z
    kernel = np.array([[vz]*width]*2 + [[vz, vz, vn, vz, vz]]*n + [[vz]*width]*2)

    test = convolve(data, kernel)
    threshold = 0.01 * test.max()

    def det(buffer):
        if buffer[18] > threshold and buffer.argmax() == 18:
            return buffer[18]
        else:
            return 0

    testmax = generic_filter(test, det, size=(n, 1), mode='constant', cval=0)

    dp = np.argwhere(testmax != 0)
    dp_vals = testmax[testmax != 0]
    dp_t = sorted(zip(dp_vals, dp), key=lambda x: x[0], reverse=True)

    fixed2 = np.array(data, dtype=float)

    remove_threshold = 0.25*np.percentile(data, 50)
    n_remove = (dp_vals/n > remove_threshold).sum()
    #print('to remove', n_remove, 'treshold', remove_threshold)
    #n_remove = 60

    for dp_i, (val, (i, j)) in enumerate(dp_t):
        i = i+1

        #print(i, j, ":", val, val/n)
        val = val/n
        start, end = i-n//2, i+n//2
        start = max(start, 0)

        #fixedmin = np.percentile(side, 50)
        #print(fixed2.dtype, val.dtype, fixedmin.dtype)
        fixed2[start:end, j] = fixed2[start:end, j] - val

        if dp_i > n_remove:
            break

    fixed2[fixed2 < 0] = 0

    for dp_i, (val, (i, j)) in enumerate(dp_t):
        i = i+1
        val = val/n
        start, end = i-n//2, i+n//2
        start = max(start, 0)

        fixed2[start:end, j] = fixed2[start:end, j-2:j+2].mean(axis=1)
        if dp_i > n_remove:
            break

    return fixed2