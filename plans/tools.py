"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all tool functions of PLANS.

Copyright (C) 2022 Iporã Brito Possantti
"""
import os, time
import numpy as np
from plans.tui import ok, warning, done

def talker(talk, kind, message):
    dict_options = {
        "ok": ok,
        "warning": warning,
        "done": done
    }
    if talk:
        dict_options[kind](message=message)

def nowsep(sep=''):
    import datetime
    def_now = datetime.datetime.now()
    yr = def_now.strftime('%Y')
    mth = def_now.strftime('%m')
    dy = def_now.strftime('%d')
    hr = def_now.strftime('%H')
    mn = def_now.strftime('%M')
    sg = def_now.strftime('%S')
    def_lst = [yr, mth, dy, hr, mn, sg]
    def_s = str(sep.join(def_lst))
    return def_s

def create_rundir(workplace, label='', suffix=None, b_time=True):
    # define label
    if suffix is None:
        pass
    else:
        label = label + "_" + suffix
    # define dir path
    dir_path = workplace + '/' + label
    if b_time:
        dir_path = dir_path + '_' + nowsep()
    # make
    if os.path.isdir(dir_path):
        pass
    else:
        os.mkdir(dir_path)
    return dir_path

def DEMO(path_inputfile1, path_inputfile2, workplace, b_paramter1, suffix=None, talk=False):
    if talk:
        from tui import ok, warning, done
    # define label
    label = DEMO.__name__
    # get rundir
    dir_out = create_rundir(workplace=workplace, label=label, suffix=suffix)
    print()


def RSCT2(path_infofile, workplace, suffix, talk=False):
    from plans.datasets import RainSeriesCollection
    start_start = time.time()
    # define label
    label = RSCT2.__name__
    # get rundir
    dir_out = create_rundir(workplace=workplace, label=label, suffix=suffix)

    # Record the start time
    print("loading...")
    # LOAD
    start_time = time.time()
    rsc = RainSeriesCollection(name=suffix)
    rsc.load_data(
        input_file=path_infofile,
        #filter_dates=["2005-01-01 00:00:00", "2012-01-01 00:00:00"],
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds\n")

    # PROCESS DATA
    print("processing...")
    start_time = time.time()
    rsc.standardize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds\n")



    # EXPORT DATA
    print("exporting data...")
    start_time = time.time()

    for name in rsc.collection:
        alias = rsc.collection[name].alias
        rsc.collection[name].data.to_csv(
            "{}/P_{}.csv".format(dir_out, alias),
            sep=";",
            index=False
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds\n")

    # EXPORT views
    print("exporting views...")
    start_time = time.time()
    rsc.export_views(folder=dir_out)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds\n")

    end_end = time.time()
    elapsed_time = end_end - start_start
    print(f"TOTAL Elapsed Time: {elapsed_time} seconds\n")
    return 0


def DTO(path_ldd, workplace, suffix=None, talk=False):
    from datasets import Raster, LDD, DTO
    from geo import outlet_distance

    # define label
    label = DTO.__name__
    # get rundir
    dir_out = create_rundir(workplace=workplace, label=label, suffix=suffix)

    # Record the start time
    print("loading...")
    # LOAD
    start_time = time.time()
    ldd = LDD(name="ldd")
    ldd.load(asc_file=path_ldd)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds\n")

    # PROCESS DATA
    print("processing...")
    start_time = time.time()
    grd_outdist = outlet_distance(
        grd_ldd=ldd.grid,
        n_res=ldd.cellsize,
        b_tui=talk
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds\n")

    # EXPORT
    print("exporting data...")
    start_time = time.time()
    filename = "dto"
    dto = DTO(name=filename)
    _asc_metadata = ldd.asc_metadata.copy()
    _asc_metadata["NODATA_value"] = -1
    dto.set_asc_metadata(metadata=_asc_metadata)
    dto.prj = ldd.prj
    dto.set_grid(grid=grd_outdist)
    nd_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds\n")

    if suffix is None:
        pass
    else:
        filename = filename + "_" + suffix
    dto.export(
        folder=dir_out,
        filename=filename
    )
    return 0


if __name__ == "__main__":
    print("HI")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8")

    f1 = "C:/plans/nimes/datasets/topo/ldd.asc"
    f2 = "C:/data/ldd_m.asc"
    DTO(
        path_ldd=f1,
        workplace="C:/data",
        talk=False
    )


