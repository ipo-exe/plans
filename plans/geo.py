"""
Custom standalone geoprocessing routines with minimal dependencies.

Description:
    The ``geo`` module provides custom geoprocessing routines.

License:
    This software is released under the GNU General Public License v3.0 (GPL-3.0).
    For details, see: https://www.gnu.org/licenses/gpl-3.0.html

Author:
    Iporã Possantti

Contact:
    possantti@gmail.com


Overview
--------

todo overview
Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

>>> from plans import geo

Class aptent taciti sociosqu ad litora torquent per
conubia nostra, per inceptos himenaeos. Nulla facilisi. Mauris eget nisl
eu eros euismod sodales. Cras pulvinar tincidunt enim nec semper.

Example
-------

todo examples
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis. Mauris gravida ex quam,
in porttitor lacus lobortis vitae. In a lacinia nisl.

.. code-block:: python

    import numpy as np
    from plans import analyst

    # get data to a vector
    data_vector = np.random.rand(1000)

    # instantiate the Univar object
    uni = analyst.Univar(data=data_vector, name="my_data")

    # view data
    uni.view()

Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl.
"""
import numpy as np


# VECTOR FUNCTIONS
def extents_to_wkt_box(xmin, ymin, xmax, ymax):
    """Returns a WKT box from the given extents (xmin, ymin, xmax, ymax).

    :param xmin: Minimum x-coordinate (left)
    :type xmin: float
    :param ymin: Minimum y-coordinate (bottom)
    :type xmin: float
    :param xmax: Maximum x-coordinate (right)
    :type xmax: float
    :param ymax: Maximum y-coordinate (top)
    :type ymax: float
    :return: A string representing the box in WKT format
    :rtype: str
    """
    return f"POLYGON(({xmin} {ymin}, {xmin} {ymax}, {xmax} {ymax}, {xmax} {ymin}, {xmin} {ymin}))"


# RASTER FUNCTIONS

# docs ok
def slope(dem, cellsize, degree=True):
    """Calculate slope using gradient-based algorithms on a 2D numpy array.

    :param dem: :class:`numpy.ndarray`
        2D numpy array representing the Digital Elevation Model (``DEM``).
    :type dem: :class:`numpy.ndarray`

    :param cellsize: float
        The size of a grid cell in the ``DEM`` (both in x and y directions).
    :type cellsize: float

    :param degree: bool, optional
        If True (default), the output slope values are in degrees. If False, output units are in radians.
    :type degree: bool

    :return: :class:`numpy.ndarray`
        2D numpy array representing the slope values.
    :rtype: :class:`numpy.ndarray`

    **Notes:**

    - The slope is calculated based on the gradient using the built-in functions of numpy.

    **Examples:**

    >>> dem_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> slope_result = slope(dem=dem_array, cellsize=1.0, degree=True)

    """
    grad = np.gradient(dem)
    gradx = grad[0] / cellsize
    grady = grad[1] / cellsize
    gradv = np.sqrt((gradx * gradx) + (grady * grady))
    slope_array = np.arctan(gradv)
    if degree:
        slope_array = slope_array * 360 / (2 * np.pi)
    return slope_array

# docs ok
def euclidean_distance(grd_input):
    """Calculate the Euclidean distance from pixels with value 1.

    :param grd_input: :class:`numpy.ndarray`
        Pseudo-boolean 2D numpy array where pixels with value 1 represent the foreground.
    :type grd_input: :class:`numpy.ndarray`

    :return: :class:`numpy.ndarray`
        2D numpy array representing the Euclidean distance.
    :rtype: :class:`numpy.ndarray`

    **Notes:**

    - The function uses the `distance_transform_edt` from `scipy.ndimage` to compute the Euclidean distance.
    - The input array is treated as a binary mask, and the distance is calculated from foreground pixels (value 1).

    **Examples:**

    >>> binary_mask = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    >>> distance_result = euclidean_distance(grd_input=binary_mask)

    """
    from scipy.ndimage import distance_transform_edt
    grd_input = 1 * (grd_input == 0) # reverse foreground values
    # Calculate the distance map
    return distance_transform_edt(grd_input)

# docs ok
def twi(slope, flowacc, cellsize):
    """Calculate the Topographic Wetness Index (``TWI``).

    :param slope: :class:`numpy.ndarray`
        2D numpy array representing the slope.
    :type slope: :class:`numpy.ndarray`

    :param flowacc: :class:`numpy.ndarray`
        2D numpy array representing the flow accumulation.
    :type flowacc: :class:`numpy.ndarray`

    :param cellsize: float
        The size of a grid cell (delta x = delta y).
    :type cellsize: float

    :return: :class:`numpy.ndarray`
        2D numpy array representing the Topographic Wetness Index.
    :rtype: :class:`numpy.ndarray`

    **Notes:**

    - The function uses the formula: TWI = ln( A / hc_colors tan(S)), where A is flow accumulation, hc_colors is the cell resolution and S is slope in radians.
    - The input arrays `slope` and `flowacc` should have the same dimensions.
    - The formula includes a small value (0.01) to prevent issues with tangent calculations for non-NaN values.

    **Examples:**

    >>> slope_data = np.array([[10, 15, 20], [8, 12, 18], [5, 10, 15]])
    >>> flowacc_data = np.array([[100, 150, 200], [80, 120, 180], [50, 100, 150]])
    >>> cell_size = 10.0
    >>> twi_result = twi(slope=slope_data, flowacc=flowacc_data, cellsize=cell_size)

    """
    # +0.01 is a hack for non-nan values
    return np.log((flowacc / cellsize)/ (np.tan((slope * np.pi / 180) + 0.01)))

# docs ok
def rivers_wedge(grd_rivers, w=3, h=3):
    """Generate a wedge-like trench along the river lines.

    :param grd_rivers: :class:`numpy.ndarray`
        Pseudo-boolean grid indicating the presence of rivers.
    :type grd_rivers: :class:`numpy.ndarray`

    :param w: int, optional
        Width (single-sided) in pixels. Default is 3.
    :type w: int

    :param h: float, optional
        Height in meters. Default is 3.
    :type h: float

    :return: :class:`numpy.ndarray`
        Grid of the wedge (positive).
    :rtype: :class:`numpy.ndarray`

    **Notes:**

    - The function generates a wedge-like trench along the river lines based on distance transform.
    - The input array `grd_rivers` should be a pseudo-boolean grid where rivers are represented as 1 and others as 0.
    - The width `w` controls the width of the trench, and `h` controls its height.

    **Examples:**

    >>> rivers_grid = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> wedge_result = rivers_wedge(grd_rivers=rivers_grid, w=3, h=5)

    """
    grd_dist = euclidean_distance(grd_input=grd_rivers)
    return (((-h/w) * grd_dist) + h) * (grd_dist <= w)

# docs ok
def burn_dem(grd_dem, grd_rivers, w=3, h=10):
    """Burn a ``DEM`` map with river lines.

    :param grd_dem: :class:`numpy.ndarray`
        DEM map.
    :type grd_dem: :class:`numpy.ndarray`

    :param grd_rivers: :class:`numpy.ndarray`
        River map (pseudo-boolean).

    :param w: int, optional
        Width parameter in pixels. Default is 3.
    :type w: int

    :param h: float, optional
        Height parameter. Default is 10.
    :type h: float

    :return: :class:`numpy.ndarray`
        Burned ``DEM``.
    :rtype: :class:`numpy.ndarray`

    **Notes:**

    - The function burns a ``DEM`` map with river lines, creating a wedge-like trench along the rivers.
    - The input array `grd_rivers` should be a pseudo-boolean grid where rivers are represented as 1 and others as 0.
    - The width `w` controls the width of the trench, and `h` controls its height.

    **Examples:**

    >>> dem = np.array([[10, 20, 30], [15, 25, 35], [5, 15, 25]])
    >>> rivers_grid = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> burned_dem = burn_dem(grd_dem=dem, grd_rivers=rivers_grid, w=3, h=10)

    """
    grd_wedge = rivers_wedge(grd_rivers, w=w, h=h)
    return (grd_dem + h) - grd_wedge

# docs ok
def downstream_coordinates(n_dir, i, j, s_convention='ldd'):
    """Compute i and j downstream cell coordinates based on cell flow direction.

    D8 - Direction convention:

    .. code-block:: text

        4   3   2
        5   0   1
        6   7   8

    LDD - Direction convention:

    .. code-block:: text

        7   8   9
        4   5   6
        1   2   3

    :param n_dir: int
        Flow direction code.

    :param i: int
        i (row) array index.

    :param j: int
        j (column) array index.

    :param s_convention: str, optional
        String of flow direction convention. Options: 'ldd' and 'd8'. Default is 'ldd'.

    :return: dict
        Dictionary of downstream i, j, and distance factor.

    **Notes:**

    - Assumes a specific flow direction convention ('ldd' or 'd8').
    - The output dictionary contains keys 'i', 'j', and 'distance'.
    - The 'i' and 'j' values represent downstream cell coordinates.
    - The 'distance' value is the Euclidean distance to the downstream cell.

    **Examples:**

    >>> downstream_coordinates(n_dir=2, i=3, j=4, s_convention='ldd')
    {'i': 4, 'j': 5, 'distance': 1.4142135623730951}

    >>> downstream_coordinates(n_dir=5, i=10, j=15, s_convention='d8')
    {'i': 10, 'j': 14, 'distance': 1.0}
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

# docs ok
def outlet_distance(grd_ldd, n_res=30, s_convention='ldd'):
    """Compute the distance to outlet ``DTO`` raster of a given basin.

    :param grd_ldd: 2d numpy array of flow direction LDD
    :type grd_ldd: :class:`numpy.ndarray`

    :param n_res: int, optional
        Resolution factor for the output distance raster. Default is 30.
    :type n_res: int

    :param s_convention: str, optional
        String of flow direction convention. Options: 'ldd' and 'd8'. Default is 'ldd'.
    :type s_convention: str

    :return: 2d numpy array distance
    :rtype: :class:`numpy.ndarray`

    **Notes:**

    - The distance is set to 0 outside the basin area.

    **Examples:**

    >>> out_distance = outlet_distance(grd_ldd, n_res=30, s_convention='ldd')

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