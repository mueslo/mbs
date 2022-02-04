import numpy as np
from scipy.ndimage import convolve


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