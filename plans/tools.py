"""
Input-output processes.

Description:
    The ``tools`` module provides objects to handle all ``plans`` input-output processes.

License:
    This software is released under the GNU General Public License v3.0 (GPL-3.0).
    For details, see: https://www.gnu.org/licenses/gpl-3.0.html

Author:
    Iporã Possantti

Contact:
    possantti@gmail.com


Overview
--------

todo
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis.
Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

>>> from plans import datasets

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
import logging
import os, time, shutil
import numpy as np
from plans.tui import logger_setup

# ------------------------------ UTILS ------------------------------
# docs todo
def nowsep(sep="-"):
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

# docs todo
def create_rundir(workplace, label="", suffix=None, b_time=True):
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

# docs ok
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
    """Process and analyze time series data.

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


    """
    from plans import datasets
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
        tsc = datasets.RainSeriesCollection(name=kind + "_" + project_name)
        method = "idw"
    elif kind == "stage":
        tsc = datasets.StageSeriesCollection(name=kind + "_" + project_name)
    elif kind == "temperature":
        tsc = datasets.StageSeriesCollection(name=kind + "_" + project_name)
        method = "average"
    else:
        tsc = datasets.RainSeriesCollection(name=kind + "_" + project_name)
        method = "idw"

    # load from info table
    tsc.load_data(
        input_file=file_infotable,
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
    from plans.datasets import Raster, LDD, DTO
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

    TSCT1(
        kind="rain",
        project_name="potiribu",
        file_infotable="C:/plans/potiribu/datasets/rain/obs/rain_info.csv",
        outdir="C:/output",
        workplace=True,
        talk=True,
        export_inputs=True,
        export_views=True,
        filter_date_start=None,
        filter_date_end=None,
        datarange_max=120,
        datarange_min=0
    )

    '''
    DTO(
        file_ldd="C:/plans/potiribu/datasets/topo/ldd.asc",
        outdir="C:/output",
        workplace=True,
        project_name="potiribu",
        talk=True,
        export_inputs=True,
    )
    
    '''


