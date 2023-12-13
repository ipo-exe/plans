"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores pre-processing scripts for running in QGIS 3.x python console.
Required dependencies:

- pcraster
- saga
- gdal
- processing
- geopandas

Copyright (C) 2022 Iporã Brito Possantti
"""
import processing
from osgeo import gdal
from qgis.core import QgsCoordinateReferenceSystem
import numpy as np
from plans import geo
import os, shutil


def reproject_layer(input_db, layer_name, target_crs):
    """Reproject a vector layer

    :param input_db: path to geopackage database
    :type input_db: str
    :param layer_name: name of the layer in geopackage database
    :type layer_name: str
    :param target_crs: EPSG code (number only) for the target CRS
    :type target_crs: str
    :return: new layer name
    :rtype: str
    """
    dict_operations = {
        "31983": {"zone": 23, "hem": "south"},
        "31982": {"zone": 22, "hem": "south"},
        "31981": {"zone": 21, "hem": "south"},
    }
    new_layer_name = "{}_EPSG{}".format(layer_name, target_crs)
    processing.run(
        "native:reprojectlayer",
        {
            "INPUT": "{}|layername={}".format(input_db, layer_name),
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:{}".format(target_crs)),
            "OPERATION": "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone={} +{} +ellps=GRS80".format(
                dict_operations[target_crs]["zone"], dict_operations[target_crs]["hem"]
            ),
            "OUTPUT": "ogr:dbname='{}' table=\"{}\" (geom)".format(
                input_db, new_layer_name
            ),
        },
    )
    return new_layer_name


def get_extent_raster(file_input):
    """Get the extent from a raster layer

    :param file_input: file path to raster layer
    :type file_input: str
    :return: dictionary of bouding box: xmin, xmax, ymin, ymax
    :rtype: dict
    """
    from qgis.core import QgsRasterLayer

    # Create a raster layer object
    raster_layer = QgsRasterLayer(file_input, "Raster Layer")
    layer_extent = raster_layer.extent()

    return {
        "xmin": layer_extent.xMinimum(),
        "xmax": layer_extent.xMaximum(),
        "ymin": layer_extent.yMinimum(),
        "ymax": layer_extent.yMaximum(),
    }


def get_extent_vector(input_db, layer_name):
    """Get the extent from a vector layer

    :param input_db: path to geopackage database
    :type input_db: str
    :param layer_name: name of the layer in geopackage database
    :type layer_name: str
    :return: dictionary of bouding box: xmin, xmax, ymin, ymax
    :rtype: dict
    """
    from qgis.core import QgsVectorLayer

    # Create the vector layer from the GeoPackage
    layer = QgsVectorLayer(
        "{}|layername={}".format(input_db, layer_name), layer_name, "ogr"
    )
    layer_extent = layer.extent()
    return {
        "xmin": layer_extent.xMinimum(),
        "xmax": layer_extent.xMaximum(),
        "ymin": layer_extent.yMinimum(),
        "ymax": layer_extent.yMaximum(),
    }


def get_cellsize(file_input):
    """Get the cell size in map units (degrees or meters)

    :param file_input: path to raster file
    :type file_input: str
    :return: number of cell resolution in map units
    :rtype: float
    """
    # Open the raster file using gdal
    raster_input = gdal.Open(file_input)
    raster_geotransform = raster_input.GetGeoTransform()
    cellsize = raster_geotransform[1]
    # -- Close the raster
    raster_input = None
    return cellsize


def get_blank_raster(file_input, file_output, blank_value=0):
    """get a blank raster copy from other raster

    :param file_input: path to input raster file
    :type file_input: str
    :param file_output: path to output raster file
    :type file_output: str
    :param blank_value: value for constant blank raster
    :type blank_value: int
    :return: file path of output (echo)
    :rtype: str
    """
    # -------------------------------------------------------------------------
    # LOAD

    # Open the raster file using gdal
    raster_input = gdal.Open(file_input)

    # Get the raster band
    band_input = raster_input.GetRasterBand(1)

    # Read the raster data as a numpy array
    grid_input = band_input.ReadAsArray()

    # truncate to byte integer
    grid_input = grid_input.astype(np.uint8)

    # -- Collect useful metadata

    raster_x_size = raster_input.RasterXSize
    raster_y_size = raster_input.RasterYSize
    raster_projection = raster_input.GetProjection()
    raster_geotransform = raster_input.GetGeoTransform()

    # -- Close the raster
    raster_input = None

    # -------------------------------------------------------------------------
    # PROCESS
    grid_output = grid_input * blank_value

    # -------------------------------------------------------------------------
    # EXPORT RASTER FILE
    # Get the driver to create the new raster
    driver = gdal.GetDriverByName("GTiff")

    # Create a new raster with the same dimensions as the original
    raster_output = driver.Create(
        file_output, raster_x_size, raster_y_size, 1, gdal.GDT_Byte
    )

    # Set the projection and geotransform of the new raster to match the original
    raster_output.SetProjection(raster_projection)
    raster_output.SetGeoTransform(raster_geotransform)

    # Write the new data to the new raster
    raster_output.GetRasterBand(1).WriteArray(grid_output)

    # Close
    raster_output = None
    return file_output


def get_dem(
    file_main_dem,
    file_output,
    target_crs,
    target_extent_dict,
    target_cellsize=30,
    source_crs="4326",
):
    """Get a reprojected DEM raster from a larger DEM dataset

    :param file_main_dem: file path to main DEM dataset raster
    :type file_main_dem: str
    :param file_output: file path to output raster
    :type file_output: str
    :param target_crs: EPSG code (number only) for the target CRS
    :type target_crs: str
    :param target_extent_dict: dictionary of bouding box: xmin, xmax, ymin, ymax
    :type target_extent_dict: dict
    :param target_cellsize: target resolution of cellsize in map units (degrees or meters)
    :type target_cellsize: float
    :param source_crs: EPSG code (number only) for the source CRS. default is WGS-84 (4326)
    :type source_crs: str
    :return: file path of output (echo)
    :rtype: str
    """
    # get extent string
    target_extent_str = "{},{},{},{} [EPSG:{}]".format(
        target_extent_dict["xmin"],
        target_extent_dict["xmax"],
        target_extent_dict["ymin"],
        target_extent_dict["ymax"],
        target_crs,
    )
    # run warp
    processing.run(
        "gdal:warpreproject",
        {
            "INPUT": file_main_dem,
            "SOURCE_CRS": QgsCoordinateReferenceSystem("EPSG:{}".format(source_crs)),
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:{}".format(target_crs)),
            "RESAMPLING": 5,  # average=5; nearest=0
            "NODATA": -1,
            "TARGET_RESOLUTION": target_cellsize,  # in meters
            "OPTIONS": "",
            "DATA_TYPE": 6,  # float32=6, byte=1
            "TARGET_EXTENT": target_extent_str,
            "TARGET_EXTENT_CRS": QgsCoordinateReferenceSystem(
                "EPSG:{}".format(target_crs)
            ),
            "MULTITHREADING": False,
            "EXTRA": "",
            "OUTPUT": file_output,
        },
    )
    return file_output


def get_carved_dem(file_dem, file_rivers, file_output, w=3, h=10):
    """Apply a carving method to DEM raster

    :param file_dem: file path to DEM raster
    :type file_dem: str
    :param file_rivers: file path to main rivers pseudo-boolean raster
    :type file_rivers: str
    :param file_output: file path to output raster
    :type file_output: str
    :param w: width parameter in unit cells
    :type w: int
    :param h: height parameter in dem units (e.g., meters)
    :type h: float
    :return: file path of output (echo)
    :rtype: str
    """
    # -------------------------------------------------------------------------
    # LOAD DEM

    # Open the raster file using gdal
    raster_dem = gdal.Open(file_dem)

    # Get the raster band
    band_dem = raster_dem.GetRasterBand(1)

    # Read the raster data as a numpy array
    grd_dem = band_dem.ReadAsArray()

    # -- Collect useful metadata

    raster_x_size = raster_dem.RasterXSize
    raster_y_size = raster_dem.RasterYSize
    raster_projection = raster_dem.GetProjection()
    raster_geotransform = raster_dem.GetGeoTransform()

    # -- Close the raster
    raster_dem = None

    # -------------------------------------------------------------------------
    # LOAD RIVERs

    # Open the raster file using gdal
    raster_rivers = gdal.Open(file_rivers)

    # Get the raster band
    band_rivers = raster_rivers.GetRasterBand(1)

    # Read the raster data as a numpy array
    grd_rivers = band_rivers.ReadAsArray()

    # truncate to byte integer
    grd_rivers = grd_rivers.astype(np.uint8)

    # -- Close the raster
    raster_rivers = None

    # -------------------------------------------------------------------------
    # PROCESS
    grid_output = geo.burn_dem(grd_dem=grd_dem, grd_rivers=grd_rivers, w=w, h=h)

    # -------------------------------------------------------------------------
    # EXPORT RASTER FILE
    # Get the driver to create the new raster
    driver = gdal.GetDriverByName("GTiff")

    # Create a new raster with the same dimensions as the original
    raster_output = driver.Create(
        file_output, raster_x_size, raster_y_size, 1, gdal.GDT_Float32
    )

    # Set the projection and geotransform of the new raster to match the original
    raster_output.SetProjection(raster_projection)
    raster_output.SetGeoTransform(raster_geotransform)

    # Write the new data to the new raster
    raster_output.GetRasterBand(1).WriteArray(grid_output)

    # Close
    raster_output = None
    return file_output


def get_slope(file_dem, file_output):
    """Get the Slope map in degrees from a DEM raster

    :param file_dem: file path to DEM raster
    :type file_dem: str
    :param file_output: file path to output raster
    :type file_output: str
    :return: file path of output (echo)
    :rtype: str
    """
    # -------------------------------------------------------------------------
    # LOAD DEM

    # Open the raster file using gdal
    raster_dem = gdal.Open(file_dem)

    # Get the raster band
    band_dem = raster_dem.GetRasterBand(1)

    # Read the raster data as a numpy array
    grd_dem = band_dem.ReadAsArray()

    # -- Collect useful metadata

    raster_x_size = raster_dem.RasterXSize
    raster_y_size = raster_dem.RasterYSize
    raster_projection = raster_dem.GetProjection()
    raster_geotransform = raster_dem.GetGeoTransform()
    cellsize = raster_geotransform[1]
    # -- Close the raster
    raster_dem = None

    # -------------------------------------------------------------------------
    # PROCESS
    grid_output = geo.slope(dem=grd_dem, cellsize=cellsize, degree=True)

    # -------------------------------------------------------------------------
    # EXPORT RASTER FILE
    # Get the driver to create the new raster
    driver = gdal.GetDriverByName("GTiff")

    # Create a new raster with the same dimensions as the original
    raster_output = driver.Create(
        file_output, raster_x_size, raster_y_size, 1, gdal.GDT_Float32
    )

    # Set the projection and geotransform of the new raster to match the original
    raster_output.SetProjection(raster_projection)
    raster_output.SetGeoTransform(raster_geotransform)

    # Write the new data to the new raster
    raster_output.GetRasterBand(1).WriteArray(grid_output)

    # Close
    raster_output = None
    return file_output


def get_twi(file_slope, file_flowacc, file_output):
    """Get the TWI map from Slope and Flow Accumulation (SAGA)

    :param file_slope: file path to Slope raster map
    :type file_slope: str
    :param file_flowacc: file path to Flow Accumulation raster map
    :type file_flowacc: str
    :param file_output: file path to output raster
    :type file_output: str
    :return: file path of output (echo)
    :rtype: str
    """
    # -------------------------------------------------------------------------
    # LOAD SLOPE

    # Open the raster file using gdal
    raster_slope = gdal.Open(file_slope)

    # Get the raster band
    band_slope = raster_slope.GetRasterBand(1)

    # Read the raster data as a numpy array
    grd_slope = band_slope.ReadAsArray()

    # -- Collect useful metadata

    raster_x_size = raster_slope.RasterXSize
    raster_y_size = raster_slope.RasterYSize
    raster_projection = raster_slope.GetProjection()
    raster_geotransform = raster_slope.GetGeoTransform()
    cellsize = raster_geotransform[1]
    # -- Close the raster
    raster_slope = None

    # -------------------------------------------------------------------------
    # LOAD RIVERs

    # Open the raster file using gdal
    raster_flowacc = gdal.Open(file_flowacc)

    # Get the raster band
    band_flowacc = raster_flowacc.GetRasterBand(1)

    # Read the raster data as a numpy array
    grd_flowacc = band_flowacc.ReadAsArray()

    # -- Close the raster
    raster_flowacc = None

    # -------------------------------------------------------------------------
    # PROCESS
    grid_output = geo.twi(slope=grd_slope, flowacc=grd_flowacc, cellsize=cellsize)

    # -------------------------------------------------------------------------
    # EXPORT RASTER FILE
    # Get the driver to create the new raster
    driver = gdal.GetDriverByName("GTiff")

    # Create a new raster with the same dimensions as the original
    raster_output = driver.Create(
        file_output, raster_x_size, raster_y_size, 1, gdal.GDT_Float32
    )

    # Set the projection and geotransform of the new raster to match the original
    raster_output.SetProjection(raster_projection)
    raster_output.SetGeoTransform(raster_geotransform)

    # Write the new data to the new raster
    raster_output.GetRasterBand(1).WriteArray(grid_output)

    # Close
    raster_output = None
    return file_output


def get_hand(file_dem_filled, output_folder, hand_cells=100, file_dem_sample=None):
    """Get the HAND raster map from a DEM. The DEM must be hydrologically consistent (no sinks)

    :param file_dem_filled: file path to filled DEM (no sinks)
    :type file_dem_filled: str
    :param output_folder: path to output folder
    :type output_folder: str
    :param hand_cells: threshhold for HAND in number of maximum hillslope cells
    :type hand_cells: int
    :param file_dem_sample: [optional] file path to raw DEM for sampling the HAND.
    :type file_dem_sample: str
    :return: dictionary with raster output paths. keys:
    "ldd", "material", "accflux", "threshold", "drainage", "hs_outlets_scalar",
    "hs_outlets_nominal", "hillslopes", "hs_zmin", "hand"
    :rtype: dict
    """
    # -- PCRaster - get LDD map
    file_ldd = "{}/ldd.map".format(output_folder)
    processing.run(
        "pcraster:lddcreate",
        {
            "INPUT": file_dem_filled,
            "INPUT0": 0,
            "INPUT1": 0,
            "INPUT2": 9999999,
            "INPUT4": 9999999,
            "INPUT3": 9999999,
            "INPUT5": 9999999,
            "OUTPUT": file_ldd,
        },
    )
    # -- PCRaster - create material layer with cell x cell (area)
    cellsize = get_cellsize(file_input=file_dem_filled)
    file_material = "{}/material.map".format(output_folder)
    processing.run(
        "pcraster:spatial",
        {
            "INPUT": cellsize * cellsize,
            "INPUT1": 3,
            "INPUT2": file_dem_filled,
            "OUTPUT": file_material,
        },
    )
    # -- PCRaster - accumulate flux
    file_accflux = "{}/accflux.map".format(output_folder)
    processing.run(
        "pcraster:accuflux",
        {
            "INPUT": file_ldd,
            "INPUT2": file_material,
            "OUTPUT": file_accflux,
        },
    )
    # -- PCRaster - create drainage threshold layer with
    file_threshold = "{}/threshold.map".format(output_folder)
    processing.run(
        "pcraster:spatial",
        {
            "INPUT": hand_cells * cellsize * cellsize,
            "INPUT1": 3,
            "INPUT2": file_dem_filled,
            "OUTPUT": file_threshold,
        },
    )
    # -- PCRaster - create drainage boolean
    file_drainage = "{}/drainage.map".format(output_folder)
    processing.run(
        "pcraster:comparisonoperators",
        {
            "INPUT": file_accflux,
            "INPUT1": 1,
            "INPUT2": file_threshold,
            "OUTPUT": file_drainage,
        },
    )
    # -- PCRaster - create unique ids Map
    file_outlets_1 = "{}/hs_outlets_scalar.map".format(output_folder)
    processing.run(
        "pcraster:uniqueid",
        {"INPUT": file_drainage, "OUTPUT": file_outlets_1},
    )

    # -- PCRaster - convert unique ids to nominal
    file_outlets_2 = "{}/hs_outlets_nominal.map".format(output_folder)
    processing.run(
        "pcraster:convertdatatype",
        {
            "INPUT": file_outlets_1,
            "INPUT1": 1,
            "OUTPUT": file_outlets_2,
        },
    )
    # -- PCRaster - create subcatchments
    file_hillslopes = "{}/hillslopes.map".format(output_folder)
    processing.run(
        "pcraster:subcatchment",
        {
            "INPUT1": file_ldd,
            "INPUT2": file_outlets_2,
            "OUTPUT": file_hillslopes,
        },
    )

    # -- PCRaster - sample z min by catchment
    file_zmin = "{}/hs_zmin.map".format(output_folder)
    if file_dem_sample is None:
        file_dem_sample = file_dem_filled
    processing.run(
        "pcraster:areaminimum",
        {
            "INPUT": file_hillslopes,
            "INPUT2": file_dem_sample,
            "OUTPUT": file_zmin,
        },
    )

    # -- GDAL- compute HAND
    file_hand = "{}/hand.tif".format(output_folder)
    processing.run(
        "gdal:rastercalculator",
        {
            "INPUT_A": file_dem_sample,
            "BAND_A": 1,
            "INPUT_B": file_zmin,
            "BAND_B": 1,
            "INPUT_C": None,
            "BAND_C": None,
            "INPUT_D": None,
            "BAND_D": None,
            "INPUT_E": None,
            "BAND_E": None,
            "INPUT_F": None,
            "BAND_F": None,
            "FORMULA": "A - B",
            "NO_DATA": None,
            "PROJWIN": None,
            "RTYPE": 1,
            "OPTIONS": "",
            "EXTRA": "",
            "OUTPUT": file_hand,
        },
    )
    return {
        "ldd": file_ldd,
        "material": file_material,
        "accflux": file_accflux,
        "threshold": file_threshold,
        "drainage": file_drainage,
        "hs_outlets_scalar": file_outlets_1,
        "hs_outlets_nominal": file_outlets_2,
        "hillslopes": file_hillslopes,
        "hs_zmin": file_zmin,
        "hand": file_hand,
    }


def get_topo(
    output_folder,
    file_main_dem,
    target_crs,
    input_db,
    layer_aoi="aoi",
    layer_rivers="rivers",
    source_crs="4326",
    w=3,
    h=10,
    hand_cells=100,
):
    """Get all [topo] datasets for runnins PLANS

    :param output_folder: path to output folder to export datasets
    :type output_folder: str
    :param file_main_dem: file path to larger DEM dataset raster -- expected to be in WGS-84
    :type file_main_dem: str
    :param target_crs: EPSG code (number only) for the target CRS
    :type target_crs: str
    :param input_db: path to geopackage database
    :type input_db: str
    :param layer_aoi: name of Area Of Interest layer (polygon). Default is "aoi"
    :type layer_aoi: str
    :param layer_rivers: name of main rivers layers (lines). Default is "rivers"
    :type layer_rivers: str
    :param source_crs: EPSG code (number only) for the source CRS. default is WGS-84 (4326)
    :type source_crs: str
    :param w: [burning dem parameter] width parameter in unit cells
    :type w: int
    :param h: [burning dem parameter] height parameter in dem units (e.g., meters)
    :type h: float
    :param hand_cells: [hand parameter] threshhold for HAND in number of maximum hillslope cells
    :type hand_cells: int
    :return: None
    :rtype: None
    """
    # folders and file setup
    output_folder_interm = "{}/intermediate".format(output_folder)
    if os.path.isdir(output_folder_interm):
        pass
    else:
        os.mkdir(output_folder_interm)

    # GDAL folder
    output_folder_gdal = "{}/gdal".format(output_folder_interm)
    if os.path.isdir(output_folder_gdal):
        pass
    else:
        os.mkdir(output_folder_gdal)

    # SAGA folder
    output_folder_saga = "{}/saga".format(output_folder_interm)
    if os.path.isdir(output_folder_saga):
        pass
    else:
        os.mkdir(output_folder_saga)

    # PCRASTER folder
    output_folder_pcraster = "{}/pcraster".format(output_folder_interm)
    if os.path.isdir(output_folder_pcraster):
        pass
    else:
        os.mkdir(output_folder_pcraster)

    dict_files = {
        "dem": "{}/dem.tif".format(output_folder_gdal),
        "main_rivers": "{}/main_rivers.tif".format(output_folder_gdal),
        "dem_b": "{}/dem_b.tif".format(output_folder_gdal),
        "slope": "{}/slope.tif".format(output_folder_gdal),
        "fill": "{}/fill.sdat".format(output_folder_saga),
        "flowacc_mfd_sg": "{}/flowacc_mfd.sdat".format(output_folder_saga),
        "flowacc_mfd": "{}/flowacc_mfd.tif".format(output_folder_gdal),
        "twi": "{}/twi.tif".format(output_folder_gdal),
        "dem_bf": "{}/dem_bf.tif".format(output_folder_gdal),
        "dem_bfb": "{}/dem_bfb.tif".format(output_folder_gdal),
        "dem_pc": "{}/dem.map".format(output_folder_pcraster),
        "dem_bf_pc": "{}/dem_bf.map".format(output_folder_pcraster),
    }

    #
    print("reproject layers...")
    # reproject aoi
    new_layer_aoi = reproject_layer(
        input_db=input_db, layer_name=layer_aoi, target_crs=target_crs
    )
    # reproject rivers
    new_layer_rivers = reproject_layer(
        input_db=input_db, layer_name=layer_rivers, target_crs=target_crs
    )
    # 2)
    print("get aoi extent...")
    dict_bbox = get_extent_vector(input_db=input_db, layer_name=new_layer_aoi)
    print(dict_bbox)
    # 3)
    print("clip and warp dem...")
    file_dem = get_dem(
        file_main_dem=file_main_dem,
        target_crs=target_crs,
        target_extent_dict=dict_bbox,
        file_output=dict_files["dem"],
        source_crs=source_crs,
    )
    cellsize = get_cellsize(file_input=dict_files["dem"])
    # get rivers blank
    # 4)
    print("get main rivers...")
    file_rivers = get_blank_raster(
        file_input=file_dem, file_output=dict_files["main_rivers"], blank_value=0
    )
    print("rasterize...")
    # 5) rasterize rivers
    processing.run(
        "gdal:rasterize_over_fixed_value",
        {
            "INPUT": "{}|layername={}".format(input_db, new_layer_rivers),
            "INPUT_RASTER": dict_files["main_rivers"],
            "BURN": 1,
            "ADD": False,
            "EXTRA": "",
        },
    )
    print("burn dem...")
    # 6) get burned
    file_burn = get_carved_dem(
        file_dem=dict_files["dem"],
        file_rivers=dict_files["main_rivers"],
        file_output=dict_files["dem_b"],
        w=w,
        h=h,
    )
    # get slope
    print("get slope....")
    get_slope(file_dem=dict_files["dem"], file_output=dict_files["slope"])

    print("fill sinks...")
    processing.run(
        "saga:fillsinksplanchondarboux2001",
        {"DEM": dict_files["dem_b"], "RESULT": dict_files["fill"], "MINSLOPE": 0.01},
    )
    print("translate fill...")
    processing.run(
        "gdal:translate",
        {
            "INPUT": dict_files["fill"],
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:{}".format(target_crs)),
            "NODATA": None,
            "COPY_SUBDATASETS": False,
            "OPTIONS": "",
            "EXTRA": "",
            "DATA_TYPE": 6,
            "OUTPUT": dict_files["dem_bf"],
        },
    )
    print("get flowacc mfd ...")
    processing.run(
        "saga:flowaccumulationparallelizable",
        {
            "DEM": dict_files["dem_bf"],
            "FLOW": dict_files["flowacc_mfd_sg"],
            "UPDATE": 0,
            "METHOD": 2,
            "CONVERGENCE": 1.1,
        },
    )
    print("translate flowacc mfd...")
    processing.run(
        "gdal:translate",
        {
            "INPUT": dict_files["flowacc_mfd_sg"],
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:{}".format(target_crs)),
            "NODATA": None,
            "COPY_SUBDATASETS": False,
            "OPTIONS": "",
            "EXTRA": "",
            "DATA_TYPE": 6,
            "OUTPUT": dict_files["flowacc_mfd"],
        },
    )
    print("get TWI...")
    get_twi(
        file_slope=dict_files["slope"],
        file_flowacc=dict_files["flowacc_mfd"],
        file_output=dict_files["twi"],
    )
    print("convert dem to pcraster...")
    processing.run(
        "pcraster:converttopcrasterformat",
        {"INPUT": dict_files["dem"], "INPUT2": 3, "OUTPUT": dict_files["dem_pc"]},
    )
    print("convert dem_bf to pcraster...")
    processing.run(
        "pcraster:converttopcrasterformat",
        {"INPUT": dict_files["dem_bf"], "INPUT2": 3, "OUTPUT": dict_files["dem_bf_pc"]},
    )

    print("get HAND...")
    dict_hand = get_hand(
        file_dem_filled=dict_files["dem_bf_pc"],
        output_folder=output_folder_pcraster,
        hand_cells=hand_cells,
        file_dem_sample=dict_files["dem_pc"],
    )
    # append to dict
    for f in dict_hand:
        dict_files[f] = dict_hand[f]

    # translate all
    dict_raster = {"nodata": -1, "data_type": 6}
    dict_qualiraster = {"nodata": 0, "data_type": 1}
    dict_translate = {
        "dem": dict_raster,
        "slope": dict_raster,
        "twi": dict_raster,
        "accflux": dict_raster,
        "hand": dict_raster,
        "ldd": dict_qualiraster,
    }
    # loop
    print("translating outputs...")
    for k in dict_translate:
        processing.run(
            "gdal:translate",
            {
                "INPUT": dict_files[k],
                "TARGET_CRS": QgsCoordinateReferenceSystem(
                    "EPSG:{}".format(target_crs)
                ),
                "NODATA": dict_translate[k]["nodata"],
                "COPY_SUBDATASETS": False,
                "OPTIONS": "",
                "EXTRA": "",
                "DATA_TYPE": dict_translate[k]["data_type"],
                "OUTPUT": "{}/{}.asc".format(output_folder, k),
            },
        )

    return None


def get_lulc(
    list_main_files,
    list_dates,
    output_folder,
    target_file,
    target_crs,
    input_db=None,
    layer_roads_dirty="roads_dirty",
    layer_roads_paved="roads_paved",
    source_crs="4326",
    qml_file=None,
):
    """Get LULC maps from a larger main LULC dataset. Script example (for QGIS):
    .. code-block:: python

        # plans source code must be pasted to QGIS plugins directory
        from plans import iamlazy
        # set files
        lst_files = [
            "path/to/lulc_2015.tif",
            "path/to/lulc_2015.tif",
            "path/to/lulc_2016.tif",
        ]
        # set dates
        lst_dates = [
            "2014-01-01",
            "2015-01-01",
            "2016-01-01",
        ]
        # run function
        iamlazy.get_lulc(
            list_main_files=lst_files,
            list_dates=lst_dates,
            output_folder="path/to/output",
            target_file="/path/to/raster.tif",
            target_crs="31982",
            input_db="path/to/input_db.gpkg",
            layer_roads_dirty='roads_dirty',
            layer_roads_paved='roads_paved',
            source_crs='4326',
            qml_file="path/to/style.qml"
        )


    :param list_main_files: list of file paths to main lulc rasters
    :type list_main_files: list
    :param list_dates: list of dates (YYYY-MM-DD) for each main raster
    :type list_dates: list
    :param output_folder: path to output folder
    :type output_folder: str
    :param target_file: file path to target raster (e.g., the DEM raster)
    :type target_file: str
    :param target_crs: EPSG code (number only) for the target CRS
    :type target_crs: str
    :param input_db: [optional] path to geopackage database with road layers
    :type input_db: str
    :param layer_roads_dirty: layer name of dirty roads
    :type layer_roads_dirty: str
    :param layer_roads_paved: layer name of paved roads
    :type layer_roads_paved: str
    :param source_crs: EPSG code (number only) for the source CRS. default is WGS-84 (4326)
    :type source_crs: str
    :param qml_file: [optional] file path to QML style file
    :type qml_file: str
    :return: None
    :rtype: None
    """
    # folders and file setup
    output_folder_interm = "{}/intermediate".format(output_folder)
    if os.path.isdir(output_folder_interm):
        pass
    else:
        os.mkdir(output_folder_interm)
    # ---------- warp lulc -----------
    target_extent_dict = get_extent_raster(file_input=target_file)
    # get extent string
    target_extent_str = "{},{},{},{} [EPSG:{}]".format(
        target_extent_dict["xmin"],
        target_extent_dict["xmax"],
        target_extent_dict["ymin"],
        target_extent_dict["ymax"],
        target_crs,
    )
    cellsize = get_cellsize(file_input=target_file)
    list_output_files = list()
    list_filenames = list()
    print("clip and warp lulc ... ")
    for i in range(len(list_main_files)):
        file_main_lulc = list_main_files[i]
        date = list_dates[i]
        str_filename = "lulc_{}".format(date)
        file_output = "{}/{}.tif".format(output_folder_interm, str_filename)
        list_output_files.append(file_output[:])
        list_filenames.append(str_filename[:])
        # run warp
        processing.run(
            "gdal:warpreproject",
            {
                "INPUT": file_main_lulc,
                "SOURCE_CRS": QgsCoordinateReferenceSystem(
                    "EPSG:{}".format(source_crs)
                ),
                "TARGET_CRS": QgsCoordinateReferenceSystem(
                    "EPSG:{}".format(target_crs)
                ),
                "RESAMPLING": 0,
                "NODATA": 0,
                "TARGET_RESOLUTION": cellsize,
                "OPTIONS": "",
                "DATA_TYPE": 1,
                "TARGET_EXTENT": target_extent_str,
                "TARGET_EXTENT_CRS": QgsCoordinateReferenceSystem(
                    "EPSG:{}".format(target_crs)
                ),
                "MULTITHREADING": False,
                "EXTRA": "",
                "OUTPUT": file_output,
            },
        )

    #
    # ---------- handle roads -----------
    if input_db is None:
        pass
    else:
        print("reproject road layers...")
        # reproject dirty
        new_layer_dirty = reproject_layer(
            input_db=input_db, layer_name=layer_roads_dirty, target_crs=target_crs
        )
        # reproject paved
        new_layer_paved = reproject_layer(
            input_db=input_db, layer_name=layer_roads_paved, target_crs=target_crs
        )

        # rasterize loop
        print("rasterizing road layers...")
        for i in range(len(list_output_files)):
            input_raster = list_output_files[i]
            # dirty
            processing.run(
                "gdal:rasterize_over_fixed_value",
                {
                    "INPUT": "{}|layername={}".format(input_db, new_layer_dirty),
                    "INPUT_RASTER": input_raster,
                    "BURN": 255,
                    "ADD": False,
                    "EXTRA": "",
                },
            )
            # paved
            processing.run(
                "gdal:rasterize_over_fixed_value",
                {
                    "INPUT": "{}|layername={}".format(input_db, new_layer_paved),
                    "INPUT_RASTER": input_raster,
                    "BURN": 254,
                    "ADD": False,
                    "EXTRA": "",
                },
            )
    # ---------- translate all -----------
    print("translate...")
    for i in range(len(list_output_files)):
        input_file = list_output_files[i]
        filename = list_filenames[i]
        processing.run(
            "gdal:translate",
            {
                'INPUT': input_file,
                'TARGET_CRS': QgsCoordinateReferenceSystem("EPSG:{}".format(target_crs)),
                'NODATA': 0,
                'COPY_SUBDATASETS': False,
                'OPTIONS': '',
                'EXTRA': '',
                'DATA_TYPE': 1,
                'OUTPUT': '{}/{}.asc'.format(output_folder, filename)
            }
        )
        # Handle style
        if qml_file is None:
            pass
        else:
            shutil.copy(
                src=qml_file, dst="{}/{}.qml".format(output_folder, filename)
            )

    return None
