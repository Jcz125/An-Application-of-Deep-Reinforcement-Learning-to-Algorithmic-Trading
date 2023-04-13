import numpy as np

def nanHelper(y):
    """
    Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolateNans(arr):
    nans, x = nanHelper(arr)
    arr[nans]= np.interp(x(nans), x(~nans), arr[~nans])
    return arr


def fillNansWith(arr, fill=0):
    return np.nan_to_num(arr, nan=fill)