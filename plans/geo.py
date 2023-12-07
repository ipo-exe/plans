"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all native geoprocessing functions of PLANS.

Copyright (C) 2022 Iporã Brito Possantti
"""
import numpy as np

def slope(dem, cellsize, degree=True):
    """Slope algorithm based on gradient built in functions of numpy
    :param dem: 2d numpy array of dem
    :type dem: :class:`numpy.ndarray`
    :param cellsize: float value of cellsize (delta x = delta y)
    :type cellsize: float
    :param degree: boolean to control output units. Default = True. If False output units are in radians
    :type degree: bool
    :return: 2d numpy array of slope
    :rtype: :class:`numpy.ndarray`
    """
    grad = np.gradient(dem)
    gradx = grad[0] / cellsize
    grady = grad[1] / cellsize
    gradv = np.sqrt((gradx * gradx) + (grady * grady))
    slope_array = np.arctan(gradv)
    if degree:
        slope_array = slope_array * 360 / (2 * np.pi)
    return slope_array

def euclidean_distance(grd_input):
    """Calculate the euclidean distance from pixels=1

    :param grd_input: pseudo-boolean 2d numpy array
    :type grd_input: :class:`numpy.ndarray`
    :return: 2d numpy array of euclidean distance
    :rtype: :class:`numpy.ndarray`
    """
    from scipy.ndimage import distance_transform_edt
    grd_input = 1 * (grd_input == 0) # reverse foreground values
    # Calculate the distance map
    return distance_transform_edt(grd_input)

def twi(slope, flowacc, cellsize):
    # +0.01 is a hack for non-nan values
    return np.log((flowacc / cellsize)/ (np.tan((slope * np.pi / 180) + 0.01)))

def rivers_wedge(grd_rivers, w=3, h=3):
    """Get a wedge-like trench along the river lines

    :param grd_rivers: pseudo-boolean grid of rivers
    :type grd_rivers: :class:`numpy.ndarray`
    :param w: width (single sided) in pixels
    :type w: int
    :param h: height in meters
    :type h: float
    :return: grid of wedge (positive)
    :rtype: :class:`numpy.ndarray`
    """
    grd_dist = euclidean_distance(grd_input=grd_rivers)
    return (((-h/w) * grd_dist) + h) * (grd_dist <= w)


def burn_dem(grd_dem, grd_rivers, w=3, h=10):
    """burn a dem map with rivers lines

    :param grd_dem: dem map
    :type grd_dem: :class:`numpy.ndarray`
    :param grd_rivers: rivers maps (pseudo-boolean)
    :type grd_rivers: :class:`numpy.ndarray`
    :param w: width parameter in pixels
    :type w: int
    :param h: heigth parameter
    :type h: float
    :return: burned dem
    :rtype: :class:`numpy.ndarray`
    """
    grd_wedge = rivers_wedge(grd_rivers, w=w, h=h)
    return (grd_dem + h) - grd_wedge


def downstream_coordinates(n_dir, i, j, s_convention='ldd'):
    """Compute i and j donwstream cell coordinates based on cell flow direction

    d8 - Direction convention:

    4   3   2
    5   0   1
    6   7   8

    ldd - Direction convention:

    7   8   9
    4   5   6
    1   2   3

    di is VERTICAL, DESCENDING direction
    dj is HORIZONTAL, ASCENDING direction

    :param n_dir: int flow direction code
    :param i: int i (row) array index
    :param j: int j (column) array index
    :param s_convention: string of flow dir covention. Options: 'ldd' and 'd8'
    :return: dict of downstream i, j and distance factor
    """
    # directions dictionaries
    dct_dirs = {
        'ldd': {
            '1': {'di': 1, 'dj': -1},
            '2': {'di': 1, 'dj':  0},
            '3': {'di': 1, 'dj':  1},
            '4': {'di': 0, 'dj': -1},
            '5': {'di': 0, 'dj':  0},
            '6': {'di': 0, 'dj':  1},
            '7': {'di':-1, 'dj': -1},
            '8': {'di':-1, 'dj':  0},
            '9': {'di':-1, 'dj':  1}
        },
        'd8': {
            '0': {'di': 0, 'dj': 0},
            '1': {'di': 0, 'dj': 1},
            '2': {'di':-1, 'dj': 1},
            '3': {'di':-1, 'dj': 0},
            '4': {'di':-1, 'dj':-1},
            '5': {'di': 0, 'dj':-1},
            '6': {'di': 1, 'dj':-1},
            '7': {'di': 1, 'dj': 0},
            '8': {'di': 1, 'dj': 1}
        }
    }
    # set dir dict
    dct_dir = dct_dirs[s_convention]
    di = dct_dir[str(n_dir)]['di']
    dj = dct_dir[str(n_dir)]['dj']
    dist = np.sqrt(np.power(di, 2) + np.power(dj, 2))
    dct_output = {
        'i': i + di,
        'j': j + dj,
        'distance': dist
    }
    return dct_output

def outlet_distance(grd_ldd, n_res=30, s_convention='ldd', b_tui=True):
    """Compute the outlet distance raster of a given basin

    Note: distance is set to 0 outside the basin area

    :param grd_ldd: 2d numpy array of flow direction LDD
    :param s_convention: string of flow dir covention. Options: 'ldd' and 'd8'
    :param b_tui: boolean for tui display
    :return: 2d numpy array distance
    """

    def is_offgrid(i, j):
        if i < 0:
            return True
        elif i >= n_rows:
            return True
        elif j < 0:
            return True
        elif j >= n_cols:
            return True
        else:
            return False

    def is_center(ldd):
        # halt
        dict_halt = {
            "ldd": 5,
            "d8": 0
        }
        if ldd == dict_halt[s_convention]:
            return True
        else:
            return False

    # compute total number of cells:
    n_total = grd_ldd.size
    # deploy out distance raster
    grd_outdist = -1 * np.ones(shape=np.shape(grd_ldd), dtype='float32')
    # set loop parameters
    n_rows = np.shape(grd_ldd)[0]
    n_cols = np.shape(grd_ldd)[1]
    n_counter = 0
    for i in range(n_rows):
        for j in range(n_cols):
            # initiate cell loop:
            i_current = i
            j_current = j
            # first check: if outdist is already set
            lcl_outdist = grd_outdist[i_current][j_current]
            # not set:
            if lcl_outdist == -1:
                lcl_ldd = grd_ldd[i_current][j_current]
                # set queues
                list_accdist = list()
                list_traced_is = list()
                list_traced_js = list()
                # trace loop
                n_trace_count = 0
                while True:
                    # get downstream position
                    dct_out = downstream_coordinates(
                        n_dir=lcl_ldd,
                        i=i_current,
                        j=j_current,
                        s_convention=s_convention)
                    # print(dct_out)
                    i_next = dct_out['i']
                    j_next = dct_out['j']
                    n_dist = dct_out['distance']
                    # append to queue
                    if n_trace_count == 0:
                        list_accdist.append(n_dist)
                    else:
                        list_accdist.append(n_dist + list_accdist[n_trace_count - 1])
                    list_traced_is.append(i_current)
                    list_traced_js.append(j_current)
                    # halting options
                    b_off = is_offgrid(i=i_next, j=j_next)
                    b_center = is_center(ldd=lcl_ldd)
                    if b_off or b_center:
                        # QUIT
                        n_traced_outdist = 0
                        break
                    # check if next cell is a mapped path
                    n_next_outdist = grd_outdist[i_next][j_next]
                    if n_next_outdist == -1:
                        # MOVE
                        i_current = i_next
                        j_current = j_next
                        # update flowdir
                        lcl_ldd = grd_ldd[i_current][j_current]
                        n_trace_count = n_trace_count + 1
                    else:
                        # QUIT
                        n_traced_outdist = n_next_outdist
                        break
                # now manipulate queues
                vct_accdist = np.array(list_accdist)
                vct_accdist_inv = vct_accdist[::-1]
                vct_is = np.array(list_traced_is)
                vct_js = np.array(list_traced_js)
                # get outdist path
                vct_outdist = vct_accdist_inv + n_traced_outdist
                # burn loop
                for k in range(len(vct_outdist)):
                    grd_outdist[vct_is[k]][vct_js[k]] = vct_outdist[k]
            else:
                pass
    return n_res * grd_outdist

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    grd = np.zeros(shape=(100, 100))
    grd[50][50] = 1

    grd_d = rivers_wedge(grd_rivers=grd)
    plt.imshow(grd_d)
    plt.show()