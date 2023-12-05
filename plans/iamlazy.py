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


def reproject_layer(input_db, layer_name, target_crs):
    dict_operations = {
        "31982": {
            "zone": 22, "hem": "south"
        }
    }
    processing.run(
        "native:reprojectlayer",
        {
            'INPUT': '{}|layername={}'.format(input_db, layer_name),
            'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:{}'.format(target_crs)),
            'OPERATION': '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone={} +{} +ellps=GRS80'.format(
                dict_operations[target_crs]["zone"],
                dict_operations[target_crs]["zone"]
            ),
            'OUTPUT': 'ogr:dbname=\'{}\' table="{}_utm" (geom)'.format(input_db, layer_name)
        }
    )

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
    return None


def clip_and_warp(file_input, ):
    print("hello")


def get_dem(file_main_dem, target_crs, target_extent_dict, output_folder, output_name):
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
            'OUTPUT': '{}/{}.tif'.format(output_folder, output_name)
        }
    )

