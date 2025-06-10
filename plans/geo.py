"""
Custom standalone geoprocessing routines with minimal dependencies.

Description:
    The ``geo`` module provides custom geoprocessing routines.

License:
    This software is released under the GNU General Public License v3.0 (GPL-3.0).
    For details, see: https://www.gnu.org/licenses/gpl-3.0.html

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


def find_array_bbox(image):
    """Finds the bounding box for the content (1s) in a 2D pseudo-boolean array.

    Parameters:
    - image: 2D numpy array with values 1 (content) and 0 (background)

    Returns:
    - dict: Dictionary with keys "i_min", "i_max", "j_min", "j_max" representing
            the row and column bounds of the content
    """
    # Find rows and columns containing 1's
    rows_with_content = np.where(image.sum(axis=1) > 0)[0]
    cols_with_content = np.where(image.sum(axis=0) > 0)[0]

    # If there's content, determine bounds; otherwise, return None for each bound
    if rows_with_content.size > 0 and cols_with_content.size > 0:
        i_min, i_max = rows_with_content[0], rows_with_content[-1]
        j_min, j_max = cols_with_content[0], cols_with_content[-1]
    else:
        i_min = i_max = j_min = j_max = None

    return {"i_min": i_min, "i_max": i_max, "j_min": j_min, "j_max": j_max}


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


def convert_values(array, old_values, new_values):
    """Convert values

    :param array: 2d numpy array to convert values
    :type array: :class:`numpy.ndarray`
    :param old_values: iterable of old values
    :type old_values: :class:`numpy.ndarray`
    :param new_values: iterable of new values
    :type new_values: :class:`numpy.ndarray`
    :return: converted map
    :rtype: :class:`numpy.ndarray`
    """
    new = array * 0.0
    for i in range(len(old_values)):
        _old = old_values[i]
        _new = new_values[i]
        new = new + (_new * (array == _old))
    return new


def reclassify(array, upvalues, classes):
    """Reclassify array based on list of upper values and list of classes values

    :param array: 2d numpy array to reclassify
    :param upvalues: 1d numpy array of upper values
    :param classes: 1d array of classes values
    :return: 2d numpy array reclassified
    """
    new = array * 0.0
    for i in range(len(upvalues)):
        if i == 0:
            new = new + ((array <= upvalues[i]) * classes[i])
        else:
            new = new + (
                (array > upvalues[i - 1]) * (array <= upvalues[i]) * classes[i]
            )
    return new


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

    grd_input = 1 * (grd_input == 0)  # reverse foreground values
    # Calculate the distance map
    return distance_transform_edt(grd_input)


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
    return np.log((flowacc / cellsize) / (np.tan((slope * np.pi / 180) + 0.01)))


def shalstab_wetness(
    flowacc,
    slope,
    cellsize,
    soil_phi,
    soil_z,
    soil_c,
    soil_p,
    water_p=997,
    g=9.8,
    degree=True,
    kPa=True,
):
    """Calculate the SHALSTAB wetness model

    :param flowacc: flow accumulation map (square meters)
    :type flowacc: :class:`numpy.ndarray`
    :param slope: slope map (degrees or radians)
    :type slope: :class:`numpy.ndarray`
    :param cellsize: grid cell size (meters)
    :type cellsize: float
    :param soil_phi: soil angle of internal friction (degrees or radians)
    :type soil_phi: :class:`numpy.ndarray` or float
    :param soil_z: soil depth (meters)
    :type soil_z: :class:`numpy.ndarray` or float
    :param soil_c: soil cohesion (Pa or N/m0² or kg/ms²)
    :type soil_c: :class:`numpy.ndarray` or float
    :param soil_p: soil density (kg/m0³)
    :type soil_p: :class:`numpy.ndarray` or float
    :param water_p: water density (kg/m0³)
    :type water_p: float
    :param g: gravity acceleration (m0 / s²)
    :type g: float
    :param degree: flag to note if slope and soil_phi are in degrees
    :type degree: bool
    :param kPa: flag to note if soil_c in kPa
    :type kPa: bool
    :return: map of q/T ratio
    :rtype: :class:`numpy.ndarray`
    """

    if degree:
        # convert to radians
        slope = np.radians(slope)
        soil_phi = np.radians(soil_phi)

    if kPa:
        # convert to Pa
        soil_c = soil_c * 1000

    # get density ratio
    density_r = soil_p / water_p

    # get tangent ratio
    tan_r = 1 - (np.tan(slope) / np.tan(soil_phi))

    slope_term = density_r * tan_r

    # get force term
    cohesion_term = soil_c / (
        np.square(np.cos(slope)) * np.tan(soil_phi) * water_p * g * soil_z
    )

    topo_term = cellsize * np.sin(slope) / flowacc

    # apply full equation
    q_t = topo_term * (slope_term + cohesion_term)

    # force non-negative
    q_t = q_t * (q_t > 0) + 0.0001

    shalstab_classes = reclassify(
        array=np.log10(q_t),
        upvalues=np.array([-3.1, -2.8, -2.5, -2.2, 10]),
        classes=np.array([6, 5, 4, 3, 2]),
    )

    # improve id 1 and id 7
    mask1 = 1 * (np.tan(slope) <= np.tan(soil_phi) * (1 - (1 / density_r)))
    mask7 = 1 * (np.tan(soil_phi) > np.tan(slope))

    # shalstab_classes = (shalstab_classes * (mask1 != 1)) + mask1
    # shalstab_classes = (shalstab_classes * (mask7 != 1)) + (mask7 * 7)

    return q_t, shalstab_classes


def usle_l(slope, cellsize):
    """Wischmeier & Smith (1978) L factor

    L = (x / 22.13) ^ m0

    where:

    m0 = 0.2 when sinθ < 0.01;
    m0 = 0.3 when 0.01 ≤ sinθ ≤ 0.03;
    m0 = 0.4 when 0.03 < sinθ < 0.05;
    m0 = 0.5 when sinθ ≥ 0.05

    x is the plot lenght taken as 1.4142 * cellsize  (diagonal length of cell)

    :param slope: slope in degrees of terrain 2d array
    :param cellsize: cell size in meters
    :return: Wischmeier & Smith (1978) L factor 2d array
    """
    slope_rad = np.pi * 2 * slope / 360
    lcl_grad = np.sin(slope_rad)
    m = reclassify(
        lcl_grad,
        upvalues=(0.01, 0.03, 0.05, np.max(lcl_grad)),
        classes=(0.2, 0.3, 0.4, 0.5),
    )
    return np.power(np.sqrt(2) * cellsize / 22.13, m)


def usle_s(slope):
    """Wischmeier & Smith (1978) S factor

    S = 65.41(sinθ)^2 + 4.56sinθ + 0.065

    :param slope: slope in degrees of terrain 2d array
    :return:
    """
    slope_rad = np.pi * 2 * slope / 360
    lcl_grad = np.sin(slope_rad)
    return (65.41 * np.power(lcl_grad, 2)) + (4.56 * lcl_grad) + 0.065


def usle_m_a(q, prec, r, k, l, s, c, p, cellsize=30):
    """USLE-M Annual Soil Loss (Kinnell & Risse, 1998)

    :param q: 2d numpy array of annual runoff in mm / year
    :param prec: 2d numpy array or float of annual precipitation in mm / year
    :param r: 2d numpy array or float of rain erosivity in MJ mm h-1 ha-1 year-1
    :param k: 2d numpy array or float of erodibility K factor in ton h MJ-1 mm-1
    :param l: 2d numpy array or float of USLE/RUSLE L factor
    :param s: 2d numpy array or float of USLE/RUSLE S factor
    :param c: 2d numpy array or float of C_UM factor
    :param p: 2d numpy array or float of USLE/RUSLE P factor
    :param cellsize: float of grid cell size in meters
    :return: 2d numpy array of Annual Soil Loss in ton / year
    """
    return (q / prec) * r * k * l * s * c * p * (cellsize * cellsize / (100 * 100))


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
    return (((-h / w) * grd_dist) + h) * (grd_dist <= w)


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


def downstream_coordinates(n_dir, i, j, s_convention="ldd"):
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
        "ldd": {
            "1": {"di": 1, "dj": -1},
            "2": {"di": 1, "dj": 0},
            "3": {"di": 1, "dj": 1},
            "4": {"di": 0, "dj": -1},
            "5": {"di": 0, "dj": 0},
            "6": {"di": 0, "dj": 1},
            "7": {"di": -1, "dj": -1},
            "8": {"di": -1, "dj": 0},
            "9": {"di": -1, "dj": 1},
        },
        "d8": {
            "0": {"di": 0, "dj": 0},
            "1": {"di": 0, "dj": 1},
            "2": {"di": -1, "dj": 1},
            "3": {"di": -1, "dj": 0},
            "4": {"di": -1, "dj": -1},
            "5": {"di": 0, "dj": -1},
            "6": {"di": 1, "dj": -1},
            "7": {"di": 1, "dj": 0},
            "8": {"di": 1, "dj": 1},
        },
    }
    # set dir dict
    dct_dir = dct_dirs[s_convention]
    di = dct_dir[str(n_dir)]["di"]
    dj = dct_dir[str(n_dir)]["dj"]
    dist = np.sqrt(np.power(di, 2) + np.power(dj, 2))
    dct_output = {"i": i + di, "j": j + dj, "distance": dist}
    return dct_output


def outlet_distance(grd_ldd, n_res=30, s_convention="ldd"):
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
        dict_halt = {"ldd": 5, "d8": 0}
        if ldd == dict_halt[s_convention]:
            return True
        else:
            return False

    # compute total number of cells:
    n_total = grd_ldd.size
    # deploy out distance raster
    grd_outdist = -1 * np.ones(shape=np.shape(grd_ldd), dtype="float32")
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
                        s_convention=s_convention,
                    )
                    i_next = dct_out["i"]
                    j_next = dct_out["j"]
                    n_dist = dct_out["distance"]
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
