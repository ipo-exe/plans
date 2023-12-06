"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores pre-processing scripts for running in QGIS python console.

Copyright (C) 2022 Iporã Brito Possantti
"""
import processing
from osgeo import gdal
from qgis.core import QgsCoordinateReferenceSystem
import numpy as np
from plans import geo
import os


def reproject_layer(input_gpkg, layer_name, target_crs):
    dict_operations = {
        "31982": {
            "zone": 22, "hem": "south"
        }
    }
    new_layer_name = "{}_EPSG{}".format(layer_name, target_crs)
    processing.run(
        "native:reprojectlayer",
        {
            'INPUT': '{}|layername={}'.format(input_gpkg, layer_name),
            'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:{}'.format(target_crs)),
            'OPERATION': '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone={} +{} +ellps=GRS80'.format(
                dict_operations[target_crs]["zone"],
                dict_operations[target_crs]["hem"]
            ),
            'OUTPUT': 'ogr:dbname=\'{}\' table="{}" (geom)'.format(input_gpkg, new_layer_name)
        }
    )
    return new_layer_name

def get_extent(input_db, layer_name):
    from qgis.core import QgsVectorLayer

    # Create the vector layer from the GeoPackage
    layer = QgsVectorLayer('{}|layername={}'.format(
        input_db,
        layer_name
    ), layer_name, 'ogr')

    layer_extent = layer.extent()

    return {
        "xmin": layer_extent.xMinimum(),
        "xmax": layer_extent.xMaximum(),
        "ymin": layer_extent.yMinimum(),
        "ymax": layer_extent.yMaximum()
    }

def get_cellsize(file_input):
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
    :return: none
    :rtype:
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

def clip_and_warp(file_input, ):
    print("hello")


def get_dem(file_main_dem, target_crs, target_extent_dict, file_output):
    # 1) clip and warp
    processing.run(
        "gdal:warpreproject",
        {
            'INPUT': file_main_dem,
            'SOURCE_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
            'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:{}'.format(target_crs)),
            'RESAMPLING': 5,  # average=5; nearest=0
            'NODATA': -1,
            'TARGET_RESOLUTION': 30,
            'OPTIONS': '',
            'DATA_TYPE': 6,  # float32=6, byte=1
            'TARGET_EXTENT': '{},{},{},{} [EPSG:{}]'.format(
                target_extent_dict["xmin"],
                target_extent_dict["xmax"],
                target_extent_dict["ymin"],
                target_extent_dict["ymax"],
                target_crs
            ),
            'TARGET_EXTENT_CRS': QgsCoordinateReferenceSystem('EPSG:{}'.format(target_crs)),
            'MULTITHREADING': False,
            'EXTRA': '',
            'OUTPUT': file_output
        }
    )
    return file_output


def get_burned_dem(file_dem, file_rivers, file_output, w=3, h=10):
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


def topo_pipeline(file_main_dem, target_crs, input_db, layer_name_aoi, layer_name_rivers, output_folder, w=3, h=10, hand_cells=100):

    # folders and file setup
    output_folder_topo = "{}/topo".format(output_folder)
    if os.path.isdir(output_folder_topo):
        pass
    else:
        os.mkdir(output_folder_topo)

    output_folder_saga = "{}/saga".format(output_folder_topo)
    if os.path.isdir(output_folder_saga):
        pass
    else:
        os.mkdir(output_folder_saga)

    output_folder_pcraster = "{}/pcraster".format(output_folder_topo)
    if os.path.isdir(output_folder_pcraster):
        pass
    else:
        os.mkdir(output_folder_pcraster)

    output_folder_plans = "{}/plans".format(output_folder_topo)
    if os.path.isdir(output_folder_plans):
        pass
    else:
        os.mkdir(output_folder_plans)

    dict_files = {
        "dem": "{}/dem.tif".format(output_folder_topo),
        "main_rivers": "{}/main_rivers.tif".format(output_folder_topo),
        "dem_b": "{}/dem_b.tif".format(output_folder_topo),
        "slope": "{}/slope.tif".format(output_folder_topo),
        "fill": "{}/fill.sdat".format(output_folder_saga),
        "flowacc_mfd_sg": "{}/flowacc_mfd.sdat".format(output_folder_saga),
        "flowacc_mfd": "{}/flowacc_mfd.tif".format(output_folder_topo),
        "twi_bf": "{}/twi_bf.tif".format(output_folder_topo),
        "dem_bf": "{}/dem_bf.tif".format(output_folder_topo),
        "dem_bfb": "{}/dem_bfb.tif".format(output_folder_topo),
        "dem_pc": "{}/dem.map".format(output_folder_pcraster),
        "dem_bf_pc": "{}/dem_bf.map".format(output_folder_pcraster),
        "ldd_pc": "{}/ldd.map".format(output_folder_pcraster),
        "material_pc": "{}/mat.map".format(output_folder_pcraster),
        "accflux_pc": "{}/accflux.map".format(output_folder_pcraster),
        "rivers_pc": "{}/rivers.map".format(output_folder_pcraster),
        "outlets_hills_pc": "{}/outlets_hills.map".format(output_folder_pcraster),
        "outlets_hills2_pc": "{}/outlets_hills_nom.map".format(output_folder_pcraster),
        "hills_pc": "{}/hillslopes.map".format(output_folder_pcraster),
        "hand": "{}/hand_{}px.tif".format(output_folder_topo, hand_cells),
    }


    # 1)
    print("reproject layers...")
    # reproject aoi
    new_layer_aoi = reproject_layer(
        input_gpkg=input_db,
        layer_name=layer_name_aoi,
        target_crs=target_crs
    )
    # reproject rivers
    new_layer_rivers = reproject_layer(
        input_gpkg=input_db,
        layer_name=layer_name_rivers,
        target_crs=target_crs
    )
    # 2)
    print("get aoi extent...")
    dict_bbox = get_extent(input_db=input_db, layer_name=new_layer_aoi)
    print(dict_bbox)
    # 3)
    print("clip and warp dem...")
    file_dem = get_dem(
        file_main_dem=file_main_dem,
        target_crs=target_crs,
        target_extent_dict=dict_bbox,
        file_output=dict_files["dem"]
    )
    cellsize = get_cellsize(file_input=dict_files["dem"])
    # get rivers blank
    # 4)
    print("get main rivers...")
    file_rivers = get_blank_raster(
        file_input=file_dem,
        file_output=dict_files["main_rivers"],
        blank_value=0
    )
    print("rasterize...")
    # 5) rasterize rivers
    processing.run(
        "gdal:rasterize_over_fixed_value",
        {
            'INPUT':'{}|layername={}'.format(input_db, new_layer_rivers),
            'INPUT_RASTER':dict_files["main_rivers"],
            'BURN':1,
            'ADD':False,
            'EXTRA':''
        }
    )
    print("burn dem...")
    # 6) get burned
    file_burn = get_burned_dem(
        file_dem=dict_files["dem"],
        file_rivers=dict_files["main_rivers"],
        file_output=dict_files["dem_b"],
        w=w,
        h=h
    )
    print("slope....")
    # get slope
    get_slope(
        file_dem=dict_files["dem"],
        file_output=dict_files["slope"]
    )
    print("fill sinks...")
    processing.run(
        "saga:fillsinksplanchondarboux2001",
        {
            'DEM': dict_files["dem_b"],
            'RESULT': dict_files["fill"],
            'MINSLOPE': 0.01
        }
    )
    print("translate fill...")
    processing.run(
        "gdal:translate",
        {
            'INPUT': dict_files["fill"],
            'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:{}'.format(target_crs)),
            'NODATA': None,
            'COPY_SUBDATASETS': False,
            'OPTIONS': '',
            'EXTRA': '',
            'DATA_TYPE': 6,
            'OUTPUT': dict_files["dem_bf"]
        }
    )
    '''
    print("burn dem again...")
    # 6) get burned
    file_burn = get_burned_dem(
        file_dem=dict_files["dem_bf"],
        file_rivers=dict_files["rivers"],
        file_output=dict_files["dem_bfb"],
        w=w,
        h=h
    )
    '''

    print("get flowacc mfd ...")
    processing.run(
        "saga:flowaccumulationparallelizable",
        {
            'DEM': dict_files["dem_bf"],
            'FLOW': dict_files["flowacc_mfd_sg"],
            'UPDATE': 0,
            'METHOD': 2,
            'CONVERGENCE': 1.1
        }
    )
    print("translate flowacc mfd...")
    processing.run(
        "gdal:translate",
        {
            'INPUT': dict_files["flowacc_mfd_sg"],
            'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:{}'.format(target_crs)),
            'NODATA': None,
            'COPY_SUBDATASETS': False,
            'OPTIONS': '',
            'EXTRA': '',
            'DATA_TYPE': 6,
            'OUTPUT': dict_files["flowacc_mfd"]
        }
    )
    print("get TWI...")
    get_twi(
        file_slope=dict_files["slope"],
        file_flowacc=dict_files["flowacc_mfd"],
        file_output=dict_files["twi_bf"]
    )
    print("convert dem to pcraster...")
    processing.run(
        "pcraster:converttopcrasterformat",
        {
            'INPUT': dict_files["dem"],
            'INPUT2': 3,
            'OUTPUT': dict_files["dem_pc"]
        }
    )
    print("convert dem_bf to pcraster...")
    processing.run(
        "pcraster:converttopcrasterformat",
        {
            'INPUT': dict_files["dem_bf"],
            'INPUT2': 3,
            'OUTPUT': dict_files["dem_bf_pc"]
        }
    )
    print("get ldd.map...")
    processing.run("pcraster:lddcreate",
                   {'INPUT': dict_files["dem_bf_pc"], 'INPUT0': 0, 'INPUT1': 0, 'INPUT2': 9999999,
                    'INPUT4': 9999999, 'INPUT3': 9999999, 'INPUT5': 9999999,
                    'OUTPUT': dict_files["ldd_pc"]})

    # 4) -- PCRaster - create spatial layer with cell x cell
    processing.run("pcraster:spatial",
                   {'INPUT': cellsize * cellsize,
                    'INPUT1': 3,
                    'INPUT2': dict_files["dem_bf_pc"],
                    'OUTPUT': dict_files["material_pc"]
                    })
    print("get accflux")
    # 5) -- PCRaster - accumulate flux
    processing.run("pcraster:accuflux",
                   {'INPUT': dict_files["ldd_pc"],
                    'INPUT2': dict_files["material_pc"],
                    'OUTPUT': dict_files["accflux_pc"]
                    })
    print("get HAND...")
    # 6) -- PCRaster - create drainage threshold layer with
    dict_files["hand_threshold"] = "{}/hand_threshold_{}px.map".format(output_folder_pcraster, hand_cells)
    processing.run("pcraster:spatial",
                   {'INPUT': hand_cells * cellsize * cellsize,
                    'INPUT1': 3,
                    'INPUT2': dict_files["dem_bf_pc"],
                    'OUTPUT': dict_files["hand_threshold"]
                    })

    # 7) -- PCRaster - create drainage boolean
    processing.run("pcraster:comparisonoperators",
                   {'INPUT': dict_files["accflux_pc"],
                    'INPUT1': 1,
                    'INPUT2': dict_files["hand_threshold"],
                    'OUTPUT': dict_files["rivers_pc"]
                    })

    # 8) -- PCRaster - create unique ids Map
    processing.run("pcraster:uniqueid",
                   {'INPUT': dict_files["rivers_pc"],
                    'OUTPUT': dict_files["outlets_hills_pc"]
                    })

    # 9) -- PCRaster - convert unique ids to nominal
    processing.run("pcraster:convertdatatype",
                   {'INPUT': dict_files["outlets_hills_pc"],
                    'INPUT1': 1,
                    'OUTPUT': dict_files["outlets_hills2_pc"]
                    })

    # 10) -- PCRaster - create subcatchments
    processing.run("pcraster:subcatchment",
                   {'INPUT1': dict_files["ldd_pc"],
                    'INPUT2': dict_files["outlets_hills2_pc"],
                    'OUTPUT': dict_files["hills_pc"]
                    })

    # 11) -- PCRaster - sample z min by catchment
    dict_files["zmin"] = "{}/zmin.map".format(output_folder_pcraster)
    processing.run("pcraster:areaminimum",
                   {'INPUT': dict_files["hills_pc"],
                    'INPUT2': dict_files["dem_pc"], # check if other is best
                    'OUTPUT': dict_files["zmin"]
                    })

    # 12) -- GDAL- compute HAND
    processing.run(
        "gdal:rastercalculator",
        {'INPUT_A': dict_files["dem_pc"],
         'BAND_A': 1,
         'INPUT_B': dict_files["zmin"],
         'BAND_B': 1,
         'INPUT_C': None,
         'BAND_C': None,
         'INPUT_D': None,
         'BAND_D': None,
         'INPUT_E': None,
         'BAND_E': None,
         'INPUT_F': None,
         'BAND_F': None,
         'FORMULA': 'A - B',
         'NO_DATA': None,
         'PROJWIN': None,
         'RTYPE': 1,
         'OPTIONS': '',
         'EXTRA': '',
         'OUTPUT': dict_files["hand"]
         })





