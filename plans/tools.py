"""
Input-output processes.

Description:
    The ``tools`` module provides objects to handle all ``plans`` input-output processes.

License:
    This software is released under the GNU General Public License v3.0 (GPL-3.0).
    For details, see: https://www.gnu.org/licenses/gpl-3.0.html

Overview
--------

todo
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis.
Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

Class aptent taciti sociosqu ad litora torquent per
conubia nostra, per inceptos himenaeos. Nulla facilisi. Mauris eget nisl
eu eros euismod sodales. Cras pulvinar tincidunt enim nec semper.

Example
-------

todo
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
import glob
import logging
import os, time, shutil
import numpy as np
from plans.tui import logger_setup
import plans.datasets.spatial as sp

# ------------------------------ UTILS ------------------------------
# docs ok
def nowsep(sep="-"):
    """Generate a timestamp string with the current date and time separated by the specified separator.

    :param sep: str, optional
        Separator to use between date and time components, default is '-'.
    :type sep: str

    :return: str
        Timestamp string formatted as 'YYYY-MM-DD-HH-MM-SS'.
    :rtype: str

    **Notes:**

    - Uses the current date and time to generate a timestamp string.
    - The default separator is '-'.

    **Examples:**

    >>> nowsep()
    '2023-12-23-14-30-45'

    >>> nowsep(sep='_')
    '2023_12_23_14_30_45'

    """
    import datetime
    def_now = datetime.datetime.now()
    yr = def_now.strftime("%Y")
    mth = def_now.strftime("%m")
    dy = def_now.strftime("%d")
    hr = def_now.strftime("%H")
    mn = def_now.strftime("%M")
    sg = def_now.strftime("%S")
    def_lst = [yr, mth, dy, hr, mn, sg]
    def_s = str(sep.join(def_lst))
    return def_s

# docs ok
def create_rundir(workplace, label="", suffix=None, b_time=True):
    """Create a directory for a run with an optional label, suffix, and timestamp.

    :param workplace: str
        The base directory where the run directory will be created.
    :type workplace: str

    :param label: str, optional
        The label to be included in the run directory name.
    :type label: str

    :param suffix: str, optional
        The suffix to be added to the label for further customization.
    :type suffix: str

    :param b_time: bool, optional
        Whether to include a timestamp in the run directory name, default is True.
    :type b_time: bool

    :return: str
        Path to the created run directory.
    :rtype: str

    **Notes:**

    - Creates a directory with the specified label, suffix, and timestamp (if enabled).

    **Examples:**

    >>> create_rundir(workplace='/path/to/base', label='experiment', suffix='run1', b_time=True)
    '/path/to/base/experiment_run1_2023-12-23-14-30-45'

    >>> create_rundir(workplace='/path/to/base', label='test', suffix=None, b_time=False)
    '/path/to/base/test'

    """
    # define label
    if suffix is None:
        pass
    else:
        label = label + "_" + suffix
    # define dir path
    dir_path = workplace + "/" + label
    if b_time:
        dir_path = dir_path + "_" + nowsep()
    # make
    if os.path.isdir(dir_path):
        pass
    else:
        os.mkdir(dir_path)
    return dir_path

# ------------------------------ UTILS - DRY TOOLS ------------------------------

def _get_dict_basins(folder_basins):
    m_basins = sp.Basins(name="Basins")
    m_basins.load(
        asc_file=os.path.join(folder_basins, "basins.asc"),
        prj_file=os.path.join(folder_basins, "basins.prj"),
        table_file=os.path.join(folder_basins, "basins_info.csv")
    )
    basins_dct = {}
    for i in range(len(m_basins.table)):
        basins_id = m_basins.table["Id"].values[i]
        basins_dct[basins_id] = m_basins.get_basin_aoi(basin_id=basins_id)

    return basins_dct


# ------------------------------ TOOLS ------------------------------

def DEMO(
    project_name,
    inputfile1,
    outdir,
    b_param1,
    workplace=True,
    talk=False,
    export_inputs=False,
    export_views=False,
):
    # ---------------------- START ----------------------
    start_start = time.time()
    # define label
    toolname = DEMO.__name__
    prompt = "{}@{}: [{}]".format("plans", project_name, toolname)

    # ---------------------- RUN DIR ----------------------
    if workplace:
        outdir = create_rundir(workplace=outdir, label=toolname, suffix=project_name)

    # ---------------------- LOGGER ----------------------
    logger = logger_setup(
        logger_name=toolname,
        streamhandler=talk,
        filehandler=True,
        logfile="{}/logs.log".format(outdir),
    )
    logger.info("{} start".format(prompt))
    logger.info("{} run folder at {}".format(prompt, outdir))

    # ---------------------- LOAD ----------------------
    s_step = "loading"
    logger.info("{} {} data ...".format(prompt, s_step))
    start_time = time.time()

    time.sleep(1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- PROCESSING ----------------------
    s_step = "processing"
    logger.info("{} {} data ...".format(prompt, s_step))
    start_time = time.time()

    time.sleep(1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- EXPORTING - INPUTS ----------------------
    if export_inputs:
        s_step = "exporting input"
        logger.info("{} {} data ...".format(prompt, s_step))
        start_time = time.time()

        # make dir
        inpdir = "{}/inputs".format(outdir)
        os.mkdir(path=inpdir)

        # copy files or export
        time.sleep(1)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            "{} {} elapsed time: {} seconds".format(
                prompt, s_step, round(elapsed_time, 3)
            )
        )

    # ---------------------- EXPORTING - OUTPUTS ----------------------
    s_step = "exporting output"
    logger.info("{} {} data ...".format(prompt, s_step))
    start_time = time.time()

    time.sleep(1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- END ----------------------
    end_end = time.time()
    elapsed_time = end_end - start_start
    logger.info("{} end".format(prompt))
    logger.info(
        "{} total elapsed time: {} seconds".format(prompt, round(elapsed_time, 3))
    )
    return 0

def VTOPO(
    project_name,
    datasets_dir,
    outdir,
    by_basins=True,
    workplace=True,
    talk=False,
):
    """ todo docstring

    :param project_name:
    :type project_name:
    :param datasets_dir:
    :type datasets_dir:
    :param outdir:
    :type outdir:
    :param by_basins:
    :type by_basins:
    :param workplace:
    :type workplace:
    :param talk:
    :type talk:
    :return:
    :rtype:
    """
    # ---------------------- START ----------------------
    start_start = time.time()
    # define label
    toolname = VTOPO.__name__
    prompt = "{}@{}: [{}]".format("plans", project_name, toolname)

    # ---------------------- RUN DIR ----------------------
    if workplace:
        outdir = create_rundir(workplace=outdir, label=toolname, suffix=project_name)

    # ---------------------- LOGGER ----------------------
    logger = logger_setup(
        logger_name=toolname,
        streamhandler=talk,
        filehandler=True,
        logfile="{}/logs.log".format(outdir),
    )
    logger.info("{} start".format(prompt))
    logger.info("{} run folder at {}".format(prompt, outdir))

    # ---------------------- LOAD ----------------------
    s_step = "loading"
    logger.info("{} {} data ...".format(prompt, s_step))
    start_time = time.time()

    # folder setup
    folder_topo = os.path.join(datasets_dir, "topo")
    folder_basins = os.path.join(datasets_dir, "basins")

    # get basins aois
    if by_basins:
        basins_dct = _get_dict_basins(folder_basins=folder_basins)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- PROCESSING ----------------------
    s_step = "processing"
    logger.info("{} {} data ...".format(prompt, s_step))
    start_time = time.time()

    # dem
    dct_topo = {
        "dem": sp.Elevation,
        "twi": sp.TWI,
        "hand": sp.HAND,
        "slope": sp.Slope,
        "ldd": sp.LDD,
        "accflux": sp.AccFlux

    }

    # full folder setup
    folder_full = outdir
    if by_basins:
        folder_full = os.path.join(outdir, "full")
        if not os.path.isdir(folder_full):
            os.mkdir(folder_full)

    for topo in dct_topo:
        logger.info("{} {:<12} {:<10} -- full ...".format(prompt, "processing", topo))
        m = dct_topo[topo](name=project_name)
        m.load(
            asc_file=os.path.join(folder_topo, "{}.asc".format(topo)),
            prj_file=os.path.join(folder_topo, "{}.prj".format(topo))
        )
        # run main view
        logger.info("{} {:<12} {:<10} -- full ...".format(prompt, "exporting", topo))

        _filename = topo
        if by_basins:
            _filename = "{}-full".format(topo)
        m.view(
            show=False,
            folder=folder_full,
            filename=_filename
        )

        # run basins loop
        if by_basins:
            for b in basins_dct:
                # basin folder setup
                folder_basin = os.path.join(outdir, "basin_{}".format(b))
                if not os.path.isdir(folder_basin):
                    os.mkdir(folder_basin)

                logger.info("{} {:<12} {:<10} -- basin_{} ...".format(prompt, "processing", topo, b))
                m.apply_aoi_mask(grid_aoi=basins_dct[b].grid)

                # run sub views
                logger.info("{} {:<12} {:<10} -- basin_{} ...".format(prompt, "exporting", topo, b))
                m.view(
                    show=False,
                    folder=folder_basin,
                    filename="{}-basin_{}".format(topo, b)
                )
                m.release_aoi_mask()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- END ----------------------
    end_end = time.time()
    elapsed_time = end_end - start_start
    logger.info("{} end".format(prompt))
    logger.info(
        "{} total elapsed time: {} seconds".format(prompt, round(elapsed_time, 3))
    )
    return 0


def VLULC(
    project_name,
    datasets_dir,
    outdir,
    scenario="obs",
    by_basins=True,
    workplace=True,
    talk=False,
):
    """ todo docstring

    :param project_name:
    :type project_name:
    :param datasets_dir:
    :type datasets_dir:
    :param outdir:
    :type outdir:
    :param scenario:
    :type scenario:
    :param by_basins:
    :type by_basins:
    :param workplace:
    :type workplace:
    :param talk:
    :type talk:
    :return:
    :rtype:
    """
    # ---------------------- START ----------------------
    start_start = time.time()
    # define label
    toolname = VLULC.__name__
    prompt = "{}@{}: [{}]".format("plans", project_name, toolname)

    # ---------------------- RUN DIR ----------------------
    if workplace:
        outdir = create_rundir(workplace=outdir, label=toolname, suffix=project_name)

    # ---------------------- LOGGER ----------------------
    logger = logger_setup(
        logger_name=toolname,
        streamhandler=talk,
        filehandler=True,
        logfile="{}/logs.log".format(outdir),
    )
    logger.info("{} start".format(prompt))
    logger.info("{} run folder at {}".format(prompt, outdir))

    # ---------------------- LOAD ----------------------
    s_step = "loading"
    logger.info("{} {} data ...".format(prompt, s_step))
    start_time = time.time()

    # folder setup
    folder_lulc = os.path.join(datasets_dir, "lulc")
    folder_lulc = os.path.join(folder_lulc, scenario)
    folder_basins = os.path.join(datasets_dir, "basins")

    # get basins aois
    if by_basins:
        basins_dct = _get_dict_basins(folder_basins=folder_basins)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- PROCESSING ----------------------
    s_step = "processing"
    logger.info("{} {} data ...".format(prompt, s_step))
    start_time = time.time()

    # full folder setup
    folder_full = outdir
    if by_basins:
        folder_full = os.path.join(outdir, "full")
        if not os.path.isdir(folder_full):
            os.mkdir(folder_full)

    lulc_series = sp.LULCSeries(name=project_name)
    logger.info("{} loading LULC series ...".format(prompt))
    lulc_series.load_folder(
        folder=folder_lulc,
        table_file=os.path.join(folder_lulc, "lulc_info.csv"),
        name_pattern="lulc_*",
        talk=False,
        use_parallel=False,
    )
    logger.info("{} exporting LULC series ...".format(prompt))
    lulc_series.get_views(
        show=False,
        export_areas=False,
        folder=folder_full
    )
    logger.info("{} exporting LULC areas ...".format(prompt))
    lulc_series.view_series_areas(
        show=False,
        export_areas=True,
        folder=folder_full,
        filename="lulc_areas"
    )

    # proceed by basins
    if by_basins:
        for b in basins_dct:
            folder_basin = os.path.join(outdir, "basin_{}".format(b))
            if not os.path.isdir(folder_basin):
                os.mkdir(folder_basin)

            logger.info("{} {:<12} -- basin_{} ...".format(prompt, "processing", b))
            lulc_series.apply_aoi_masks(grid_aoi=basins_dct[b].grid)
            # run sub views
            logger.info("{} {:<12} -- basin_{} ...".format(prompt, "exporting views", b))
            # views
            lulc_series.get_views(
                show=False,
                export_areas=False,
                folder=folder_basin
            )
            logger.info("{} {:<12} -- basin_{} ...".format(prompt, "exporting areas", b))
            # areas
            lulc_series.view_series_areas(
                show=False,
                export_areas=True,
                folder=folder_basin,
                filename="lulc_areas_{}".format(b)
            )
            lulc_series.release_aoi_masks()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- END ----------------------
    end_end = time.time()
    elapsed_time = end_end - start_start
    logger.info("{} end".format(prompt))
    logger.info(
        "{} total elapsed time: {} seconds".format(prompt, round(elapsed_time, 3))
    )
    return 0


def TSC(
    kind,
    project_name,
    file_infotable,
    outdir,
    workplace=True,
    talk=False,
    clear_outliers=True,
    datarange_min=None,
    datarange_max=None,
    regionalize=False,
    export_inputs=False,
    export_views=False,
    filter_date_start=None,
    filter_date_end=None,

):
    """Process and analyze ``TSC`` (time series collection) data.

    **Notes:**

    - Performs data loading, processing, and exporting based on specified parameters.
    - Handles various types of time series data, such as rain, stage, and temperature.
    - Supports options for outlier cleaning, regionalization, and exporting inputs and views.

    **Examples:**

    .. code-block:: python

        from plans import tools

        tools.TSC(
            kind="rain",
            project_name="myproject",
            file_infotable="path/to/infotable.csv",
            outdir="path/to/output",
            workplace=True,
            talk=True,
            clear_outliers=True,
            datarange_min=0,
            datarange_max=120,
            regionalize=False
            export_inputs=True,
            export_views=True,
            filter_date_start="2018-01-01 00:00:00",
            filter_date_end="2023-01-01 00:00:00"
        )


    :param kind: str
        Type of time series data ('rain', 'stage', 'temperature', etc.).
    :type kind: str

    :param project_name: str
        Name of the project.
    :type project_name: str

    :param file_infotable: str
        Path to the information table file.
    :type file_infotable: str

    :param outdir: str
        Output directory where results will be saved.
    :type outdir: str

    :param workplace: bool, optional
        If True, create a run directory in the provided 'outdir', default is True.
    :type workplace: bool

    :param talk: bool, optional
        If True, enable streamhandler for logging output, default is False.
    :type talk: bool

    :param clear_outliers: bool, optional
        If True, clean outliers in the time series data, default is True.
    :type clear_outliers: bool

    :param datarange_min: datetime or None, optional
        Minimum date range for filtering data, default is None.
    :type datarange_min: datetime or None

    :param datarange_max: datetime or None, optional
        Maximum date range for filtering data, default is None.
    :type datarange_max: datetime or None

    :param regionalize: bool, optional
        If True, perform regionalization, default is False.
    :type regionalize: bool

    :param export_inputs: bool, optional
        If True, export input data, default is False.
    :type export_inputs: bool

    :param export_views: bool, optional
        If True, export views and visualizations, default is False.
    :type export_views: bool

    :param filter_date_start: datetime or None, optional
        Start date for filtering data, default is None.
    :type filter_date_start: datetime or None

    :param filter_date_end: datetime or None, optional
        End date for filtering data, default is None.
    :type filter_date_end: datetime or None

    :return: int
        Returns 0 upon successful completion.
    :rtype: int



    """
    from plans import ds
    from plans.geo import outlet_distance

    # ---------------------- START ----------------------
    start_start = time.time()

    # define label
    if regionalize:
        level = 2
    else:
        level = 1

    toolname = TSC.__name__ + "_T{}_".format(level) + kind
    prompt = "{}@{}: [{}]".format("plans", project_name, toolname)

    # ---------------------- RUN DIR ----------------------
    if workplace:
        outdir = create_rundir(workplace=outdir, label=toolname, suffix=project_name)

    # ---------------------- LOGGER ----------------------
    logger = logger_setup(
        logger_name=toolname,
        streamhandler=talk,
        filehandler=True,
        logfile="{}/logs.log".format(outdir),
    )
    logger.info("{} start".format(prompt))
    logger.info("{} run folder at {}".format(prompt, outdir))

    # ---------------------- LOAD ----------------------
    s_step = "loading"
    logger.info("{} {} ...".format(prompt, s_step))
    start_time = time.time()

    # Naming Setup
    if kind == "rain":
        tsc = datasets.RainSeriesSamples(name=kind + "_" + project_name)
        method = "idw"
    elif kind == "stage":
        tsc = datasets.StageSeriesCollection(name=kind + "_" + project_name)
    elif kind == "temperature":
        tsc = datasets.StageSeriesCollection(name=kind + "_" + project_name)
        method = "average"
    else:
        tsc = datasets.RainSeriesSamples(name=kind + "_" + project_name)
        method = "idw"

    # load from info table
    tsc.load_data(
        table_file=file_infotable,
        filter_dates=[filter_date_start, filter_date_end]
    )
    tsc.datarange_min = datarange_min
    tsc.datarange_max = datarange_max

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} --- elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- PROCESSING ----------------------
    s_step = "processing"
    logger.info("{} {} ...".format(prompt, s_step))
    start_time = time.time()

    if clear_outliers:
        logger.info("{} {} --- clean outliers".format(prompt, s_step))
        tsc.clear_outliers()

    logger.info("{} {} --- standardize".format(prompt, s_step))
    tsc.standardize()

    if regionalize:
        logger.info("{} {} --- regionalize".format(prompt, s_step))
        tsc.regionalize(method=method)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} --- elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- EXPORTING - INPUTS ----------------------
    if export_inputs:
        s_step = "exporting input"
        logger.info("{} {} ...".format(prompt, s_step))
        start_time = time.time()

        # make dir
        inpdir = "{}/inputs".format(outdir)
        os.mkdir(path=inpdir)
        logger.info("{} {} --- input folder at ./input".format(prompt, s_step))

        # export input main data
        logger.info("{} {} --- info table".format(prompt, s_step))
        # copy info table
        src = file_infotable
        shutil.copy(src=src, dst="{}/{}".format(inpdir, os.path.basename(src)))

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            "{} {} --- elapsed time: {} seconds".format(
                prompt, s_step, round(elapsed_time, 3)
            )
        )

    # ---------------------- EXPORTING - OUTPUTS ----------------------
    s_step = "exporting output"
    logger.info("{} {} ...".format(prompt, s_step))
    start_time = time.time()

    # export main data
    logger.info("{} {} --- time series collection".format(prompt, s_step))
    # export the catalog
    df = tsc.catalog
    df["VarField"] = df["VarField"] + "_" + df["Alias"]
    df["File"] = "{}/{}.csv".format(outdir, tsc.name)
    df.to_csv("{}/{}_info.csv".format(outdir, tsc.name), sep=";", index=False)
    # export merged data
    tsc.export_data(folder=outdir, merged=True)

    if export_views:
        # export view
        logger.info("{} {} --- exporting views".format(prompt, s_step))
        # only raw
        tsc.export_views(folder=outdir, skip_main=True, raw=True)
        # epochs
        tsc.export_views(folder=outdir, skip_main=False, raw=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} --- elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- END ----------------------
    end_end = time.time()
    elapsed_time = end_end - start_start
    logger.info(
        "{} total elapsed time: {} seconds".format(prompt, round(elapsed_time, 3))
    )
    logger.info("{} end".format(prompt))
    return 0


# docs ok
def DTO(
    project_name,
    file_ldd,
    outdir,
    workplace=True,
    talk=False,
    export_inputs=False,
    export_views=False,
):
    """Generate ``DTO`` (Distance To Outlet) raster map using a given ``LDD`` (Local Drain Direction) map.

    :param project_name: str
        Name of the project.
    :type project_name: str

    :param file_ldd: str
        Path to the ``LDD`` (Local Drain Direction) raster file.
    :type file_ldd: str

    :param outdir: str
        Output directory where results will be saved.
    :type outdir: str

    :param workplace: bool, optional
        If True, create a run directory in the provided 'outdir', default is True.
    :type workplace: bool

    :param talk: bool, optional
        If True, enable streamhandler for logging output, default is False.
    :type talk: bool

    :param export_inputs: bool, optional
        If True, export input data, default is False.
    :type export_inputs: bool

    :param export_views: bool, optional
        If True, export views and visualizations, default is False.
    :type export_views: bool

    :return: int
        Returns 0 upon successful completion.
    :rtype: int

    **Notes:**

    - Generates a Distance to Outlet (``DTO``) raster map from Local Drain Direction (``LDD``).
    - Calculates the hydrological distance from each grid cell to the nearest outlet.
    - Supports export of input ``LDD`` map, ``LDD`` views, DTO map, and DTO views.


    **Examples:**

    .. code-block:: python

        from plans import tools

        tools.DTO(
            project_name='MyProject',
            file_ldd='path/to/ldd.asc',
            outdir='output_directory',
            workplace=True,
            talk=False,
            export_inputs=True,
            export_views=True,
        )


    """
    from plans.ds import Raster, LDD, DTO
    from plans.geo import outlet_distance

    # ---------------------- START ----------------------
    start_start = time.time()
    # define label
    toolname = DTO.__name__
    prompt = "{}@{}: [{}]".format("plans", project_name, toolname)

    # ---------------------- RUN DIR ----------------------
    if workplace:
        outdir = create_rundir(workplace=outdir, label=toolname, suffix=project_name)

    # ---------------------- LOGGER ----------------------
    logger = logger_setup(
        logger_name=toolname,
        streamhandler=talk,
        filehandler=True,
        logfile="{}/logs.log".format(outdir),
    )
    logger.info("{} start".format(prompt))
    logger.info("{} run folder at {}".format(prompt, outdir))

    # ---------------------- LOAD ----------------------
    s_step = "loading"
    logger.info("{} {} ...".format(prompt, s_step))
    start_time = time.time()

    ldd = LDD(name=project_name)
    ldd.load(asc_file=file_ldd)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} --- elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- PROCESSING ----------------------
    s_step = "processing"
    logger.info("{} {} ...".format(prompt, s_step))
    start_time = time.time()

    grd_outdist = outlet_distance(grd_ldd=ldd.grid, n_res=ldd.cellsize, b_tui=talk)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} --- elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- EXPORTING - INPUTS ----------------------
    if export_inputs:
        s_step = "exporting input"
        logger.info("{} {} data ...".format(prompt, s_step))
        start_time = time.time()

        # make dir
        inpdir = "{}/inputs".format(outdir)
        os.mkdir(path=inpdir)
        logger.info("{} input folder at {}".format(prompt, inpdir))

        filename = "ldd_{}".format(project_name)

        # export map
        logger.info("{} exporting LDD raster map".format(prompt, inpdir))
        ldd.export(folder=inpdir, filename=filename)
        if export_views:
            # export view
            logger.info("{} exporting LDD raster view".format(prompt, inpdir))
            ldd.view(show=False, filename=filename, folder=inpdir)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            "{} {} --- elapsed time: {} seconds".format(
                prompt, s_step, round(elapsed_time, 3)
            )
        )

    # ---------------------- EXPORTING - OUTPUTS ----------------------
    s_step = "exporting output"
    logger.info("{} {} ...".format(prompt, s_step))
    start_time = time.time()

    dto = DTO(name=project_name)
    # set metadata
    logger.info("{} {} --- DTO setup".format(prompt, s_step))
    _asc_metadata = ldd.asc_metadata.copy()
    _asc_metadata["NODATA_value"] = -1
    dto.set_asc_metadata(metadata=_asc_metadata)
    dto.prj = ldd.prj
    # set data
    dto.set_grid(grid=grd_outdist)

    filename = "dto_{}".format(project_name)

    # export map
    logger.info("{} {} --- DTO raster map".format(prompt, s_step))
    dto.export(folder=outdir, filename=filename)
    if export_views:
        # export view
        logger.info("{} {} --- DTO raster view".format(prompt, s_step))
        dto.view(show=False, folder=outdir, filename=filename)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        "{} {} --- elapsed time: {} seconds".format(prompt, s_step, round(elapsed_time, 3))
    )

    # ---------------------- END ----------------------
    end_end = time.time()
    elapsed_time = end_end - start_start
    logger.info(
        "{} total elapsed time: {} seconds".format(prompt, round(elapsed_time, 3))
    )
    logger.info("{} end".format(prompt))
    return 0


if __name__ == "__main__":
    print("HI")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8")






