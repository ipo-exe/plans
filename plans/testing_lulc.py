import numpy as np
import matplotlib.pyplot as plt
import datasets as ds
import glob
import warnings

warnings.filterwarnings('ignore')

full = True
toy = False

if full:
    s_name = "Potiribu"

    # instanciar
    s_date = "1990-01-01"
    map_lulc = ds.LULC(name="Potiribu LULC", date=s_date)
    # carregar mapa
    map_lulc.load_asc_raster(file="../samples/lulc/map_lulc_{}.asc".format(s_date))
    # carregar dados de projecao
    map_lulc.load_prj_file(file="../samples/lulc/map_lulc_{}.prj".format(s_date))
    # carregar tabela de atributos
    map_lulc.load_table(file="../samples/table_lulc.txt")
    bbox_lulc = map_lulc.get_bbox()
    # visualizar mapa qualitativo
    map_lulc.view_qualiraster(show=True)

    # instanciar
    map_ndvi = ds.NDVI(name="Potiribu NDVI 2017-12-13")
    # carregar mapa
    map_ndvi.load_asc_raster(file="../samples/ndvi/map_ndvi_2017-12-13.asc")
    # carregar dados de projecao
    map_ndvi.load_prj_file(file="../samples/ndvi/map_ndvi_2017-12-13.prj")
    bbox_ndvi = map_ndvi.get_bbox()
    print(map_ndvi.grid.shape)
    # visualizar mapa
    map_ndvi.view_raster(show=True)


    # evaluate bboxes
    r_collection = ds.RasterCollection()
    r_collection.append_raster(raster=map_lulc)
    r_collection.append_raster(raster=map_ndvi)
    print(r_collection.catalog.to_string())
    r_collection.view_bboxes(show=True, datapoints=False, colors=["red", "blue"])

    # rebase
    map_lulc.rebase_grid(base_raster=map_ndvi, inplace=True)
    map_lulc.view_qualiraster(show=True)

    # lulc series
    # instanciar
    series_lulc = ds.LULCSeries(name=s_name)
    # laço para carregar mapas
    lst_maps = glob.glob(r"..\samples\lulc\*.asc")
    lst_prjs = glob.glob(r"..\samples\lulc\*.prj")
    print("loading...")
    for i in range(len(lst_maps)):
        lcl_asc_file = lst_maps[i]
        lcl_prj_file = lst_prjs[i]
        s_date = lcl_asc_file.split("_")[-1].split(".")[0]
        # load lulc
        series_lulc.load_raster(
            name="Potiribu_{}".format(s_date),
            date=s_date,
            asc_file=lcl_asc_file,
            prj_file=lcl_prj_file,
            table_file="../samples/table_lulc.txt"
        )
    print(series_lulc.catalog.to_string())
    print("rebase grids...")
    series_lulc.rebase_grids(base_raster=map_ndvi)
    print()
    print(series_lulc.catalog.to_string())
    print("exporting views")
    series_lulc.export_views(
        folder="C:/bin", dpi=120
    )



if toy:
    s_name = "Potiribu"

    # instanciar
    s_date = "1990-01-01"
    map_lulc = ds.LULC(name="Potiribu LULC", date=s_date)
    # carregar mapa
    map_lulc.load_asc_raster(file="C:/output/lulc.asc")
    # carregar dados de projecao
    map_lulc.load_prj_file(file="../samples/lulc/map_lulc_{}.prj".format(s_date))
    # carregar tabela de atributos
    map_lulc.load_table(file="../samples/table_lulc.txt")
    bbox_lulc = map_lulc.get_bbox()
    # visualizar mapa qualitativo
    #map_lulc.view_qualiraster(show=True)

    # instanciar
    map_dem = ds.Elevation(name="Potiribu DEM")
    # carregar mapa
    map_dem.load_asc_raster(file="C:/output/dem.asc")
    # carregar dados de projecao
    map_dem.load_prj_file(file="../samples/map_dem.prj")
    bbox_ndvi = map_dem.get_bbox()
    print(map_dem.grid.shape)
    # visualizar mapa
    #map_dem.view_raster(show=True)


    r_collection = ds.RasterCollection()
    r_collection.append_raster(raster=map_lulc)
    r_collection.append_raster(raster=map_dem)
    print(r_collection.catalog.to_string())

    r_collection.view_bboxes(show=True, datapoints=True, colors=["red", "blue"])

    df_pts = map_lulc.get_grid_datapoints(drop_nan=False)
    print(df_pts.to_string())
    grd = np.reshape(df_pts["z"].values, newshape=map_lulc.grid.shape)
    plt.imshow(grd, cmap="jet")
    plt.show()


    grd = map_lulc.rebase_grid(base_raster=map_dem)
    map_new_lulc = ds.LULC(name="Potiribu New LULC", date=s_date)
    map_new_lulc.set_grid(grid=grd)
    map_new_lulc.set_asc_metadata(metadata=map_dem.asc_metadata)
    map_new_lulc.set_table(dataframe=map_lulc.table)
    map_new_lulc.prj = map_dem.prj
    #print(map_new_lulc.table)
    map_new_lulc.view_qualiraster(show=True)



