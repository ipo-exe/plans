"""
Utility routines for QGIS 3.x Python console

Overview
--------

Required dependencies in QGIS:

- pcraster
- saga
- gdal
- processing
- geopandas

Example
-------

# todo [major docstring improvement] -- examples
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis. Mauris gravida ex quam,
in porttitor lacus lobortis vitae. In a lacinia nisl.

.. code-block:: python

    import numpy as np
    print("Hello World!")

Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl.
"""
# native imports
import glob, os, shutil
from pathlib import Path

import pandas as pd
# basic imports
import numpy as np
import geopandas as gpd
# qgis imports
import processing
from osgeo import gdal
from qgis.core import QgsCoordinateReferenceSystem
# plugin imports
from plans import geo

# ----------------- MODULE CONSTANTS ----------------- #

# provider base
DC_PROVIDERS = {
    "4326": "EPSG",
    "31983": "EPSG",
    "31982": "EPSG",
    "31981": "EPSG",
    "5641": "EPSG",
    "102033": "ESRI",
}
# operations base (hard-coded catalog)
DC_OPERATIONS = {
    "31983": "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone=23 +south +ellps=GRS80",
    "31982": "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone=22 +south +ellps=GRS80",
    "31981": "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone=21 +south +ellps=GRS80",
    "5641": "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=merc +lat_ts=-2 +lon_0=-43 +x_0=5000000 +y_0=10000000 +ellps=GRS80",
    "102033": "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=push +v_3 +step +proj=cart +ellps=WGS84 +step +proj=helmert +x=57 +y=-1 +z=41 +step +inv +proj=cart +ellps=aust_SA +step +proj=pop +v_3 +step +proj=aea +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA",
}

# ----------------- MODULE CONSTANTS ----------------- #


def clip_layer(db_input, layer_input, db_overlay, layer_overlay, db_output, layer_output):
    processing.run("native:clip",
       {
            'INPUT': f'{db_input}|layername={layer_input}',
            'OVERLAY': f'{db_overlay}|layername={layer_overlay}',
            'OUTPUT': "ogr:dbname='{}' table=\"{}\" (geom)".format(db_output, layer_output),
       }
    )
    return layer_output


def reproject_layer(db_input, layer_name, crs_target, db_output=None, layer_output=None):
    """Reproject a vector layer

    :param db_input: path to geopackage database
    :type db_input: str
    :param crs_target: EPSG code (number only) for the target CRS
    :type crs_target: str
    :param layer_name: name of the layer in database
    :type layer_name: str
    :return: new layer name
    :rtype: str
    """
    input_file = "{}|layername={}".format(db_input, layer_name)
    if layer_output is None:
        layer_output = "{}_{}_{}".format(
            layer_name, DC_PROVIDERS[crs_target], crs_target
        )
    # handle if output is the same
    if db_output is None:
        db_output = db_input[:]
    output_file = "ogr:dbname='{}' table=\"{}\" (geom)".format(
        db_output, layer_output
    )
    processing.run(
        "native:reprojectlayer",
        {
            "INPUT": input_file,
            "TARGET_CRS": QgsCoordinateReferenceSystem(
                "{}:{}".format(DC_PROVIDERS[crs_target], crs_target)
            ),
            "OPERATION": DC_OPERATIONS[crs_target],
            "OUTPUT": output_file,
        },
    )
    return layer_output


def is_bounded_by(extent_a, extent_b):
    """Check if extent B is bounded by extent A

    :param extent_a: extent A
    :type extent_a: dict
    :param extent_b: extent B
    :type extent_b: dict
    :return: Case if bounded
    :rtype: bool
    """
    return (
        extent_a["xmin"] <= extent_b["xmin"] <= extent_b["xmax"] <= extent_a["xmax"]
        and extent_a["ymin"] <= extent_b["ymin"] <= extent_b["ymax"] <= extent_a["ymax"]
    )


def get_extent_from_raster(file_input):
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


def get_extent_from_vector(db_input, layer_name):
    """Get the extent from a vector layer

    :param db_input: path to geopackage database
    :type db_input: str
    :param layer_name: name of the layer in geopackage database
    :type layer_name: str
    :return: dictionary of bouding box: xmin, xmax, ymin, ymax
    :rtype: dict
    """
    from qgis.core import QgsVectorLayer
    # Create the vector layer from the GeoPackage
    layer = QgsVectorLayer(
        "{}|layername={}".format(db_input, layer_name), layer_name, "ogr"
    )
    layer_extent = layer.extent()
    return {
        "xmin": layer_extent.xMinimum(),
        "xmax": layer_extent.xMaximum(),
        "ymin": layer_extent.yMinimum(),
        "ymax": layer_extent.yMaximum(),
    }


def count_vector_features(input_db, layer_name):
    """Get the number of features from a vector layer

    :param input_db: path to geopackage database
    :type input_db: str or Path
    :param layer_name: name of the layer in geopackage database
    :type layer_name: str
    :return: number of features
    :rtype: int
    """
    from qgis.core import QgsVectorLayer

    # Create the vector layer from the GeoPackage
    layer = QgsVectorLayer(
        "{}|layername={}".format(input_db, layer_name), layer_name, "ogr"
    )
    return int(layer.featureCount())


def get_basins_areas(file_basins, ids_basins):
    # todo [docstring]
    # -------------------------------------------------------------------------
    # LOAD BASINS

    # Open the raster file using gdal
    raster_input = gdal.Open(file_basins)

    # Get the raster band
    band_input = raster_input.GetRasterBand(1)

    # Read the raster data as a numpy array
    grid_basins = band_input.ReadAsArray()
    # truncate to byte integer
    grid_basins = grid_basins.astype(np.uint8)

    # get the cell sizes
    raster_geotransform = raster_input.GetGeoTransform()
    cellsize = raster_geotransform[1]

    # -- Close the raster
    raster_input = None

    # -------------------------------------------------------------------------
    # PROCESS
    basin_areas = []
    for i in ids_basins:
        grid_basin = 1 * (grid_basins == i)
        area_sqkm = cellsize * cellsize * np.sum(grid_basin) / (1000 * 1000)
        basin_areas.append(area_sqkm)

    # -------------------------------------------------------------------------
    # RETURN
    return {"Id": ids_basins, "UpstreamArea": basin_areas}


def get_cellsize(file_input):
    """Get the cell size in map units (degrees or meters)

    :param file_input: path to raster file
    :type file_input: str or Path
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


def get_blank_raster(file_input, file_output, blank_value=0, dtype="byte"):
    """Get a blank raster copy from other raster

    :param file_input: path to inputs raster file
    :type file_input: str
    :param file_output: path to output raster file
    :type file_output: str
    :param blank_value: value for constant blank raster
    :type blank_value: int
    :param dtype: output raster data type ("byte", "int" -- else defaults to float)
    :type dtype: str
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

    # -- Collect useful metadata

    raster_x_size = raster_input.RasterXSize
    raster_y_size = raster_input.RasterYSize
    raster_projection = raster_input.GetProjection()
    raster_geotransform = raster_input.GetGeoTransform()

    # -- Close the raster
    raster_input = None

    # -------------------------------------------------------------------------
    # PROCESS
    # get ones to byte integer
    grid_input = np.ones(shape=grid_input.shape, dtype=np.uint8)

    grid_output = grid_input * blank_value

    # -------------------------------------------------------------------------
    # EXPORT RASTER FILE
    # Get the driver to create the new raster
    driver = gdal.GetDriverByName("GTiff")

    if dtype == "byte":
        # Create a new raster with the same dimensions as the original
        raster_output = driver.Create(
            file_output, raster_x_size, raster_y_size, 1, gdal.GDT_Byte
        )
    elif dtype == "int":
        # Create a new raster with the same dimensions as the original
        raster_output = driver.Create(
            file_output, raster_x_size, raster_y_size, 1, gdal.GDT_Int16
        )
    else:
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

def retrieve_raster_tiles(
    db_aoi,
    layer_aoi,
    db_tiles,
    layer_tiles,
    raster_dir,
    output_dir,
    output_name,
):
    # todo [docstring]
    print("retrieving...")
    # extract tiles
    processing.run("native:extractbylocation",
        {
            'INPUT':f'{db_tiles}|layername={layer_tiles}',
            'PREDICATE':[0],
            'INTERSECT':f'{db_aoi}|layername={layer_aoi}',
            'OUTPUT':f'ogr:dbname=\'{output_dir}/{output_name}.gpkg\' table="tiles" (geom)'
        }
    )
    # list tiles
    gdf = gpd.read_file(f"{output_dir}/{output_name}.gpkg", layer="tiles")
    ls_codes = list(gdf["tile_code"])
    ls_rasters = []
    for code in ls_codes:
        ls_raster = glob.glob(f"{raster_dir}/*{code}*.tif")
        for raster_file in ls_raster:
            rfile = str(Path(raster_file))
            print(rfile)
            ls_rasters.append(rfile)
    print("merging...")
    output_file = f'{output_dir}/{output_name}.tif'
    processing.run("gdal:merge",
        {
            'INPUT': ls_rasters,
            'PCT': False,
            'SEPARATE': False,
            'NODATA_INPUT': None,
            'NODATA_OUTPUT': None,
            'OPTIONS': '',
            'EXTRA': '',
            'DATA_TYPE': 5,
            'OUTPUT': output_file,
        }
    )
    print("done")
    return output_file

def get_dem(
    file_dem_src,
    file_output,
    target_crs,
    target_extent,
    target_cellsize=30,
    source_crs="4326",
):
    """Get a reprojected DEM raster from a larger DEM dataset

    :param file_dem_src: file path to sourced (larger) DEM dataset raster
    :type file_dem_src: str
    :param file_output: file path to output raster
    :type file_output: str
    :param target_crs: EPSG code (number only) for the target CRS
    :type target_crs: str
    :param target_extent: dictionary of bouding box: xmin, xmax, ymin, ymax
    :type target_extent: dict
    :param target_cellsize: target resolution of cellsize in map units (degrees or meters)
    :type target_cellsize: float
    :param source_crs: EPSG code (number only) for the source CRS. default is 4326 (WGS-84)
    :type source_crs: str
    :return: file path of output (echo)
    :rtype: str
    """
    # get extent string
    target_extent_str = "{},{},{},{} [{}:{}]".format(
        target_extent["xmin"],
        target_extent["xmax"],
        target_extent["ymin"],
        target_extent["ymax"],
        DC_PROVIDERS[target_crs],
        target_crs,
    )
    # run warp
    processing.run(
        "gdal:warpreproject",
        {
            "INPUT": file_dem_src,
            "SOURCE_CRS": QgsCoordinateReferenceSystem(
                "{}:{}".format(DC_PROVIDERS[source_crs], source_crs)
            ),
            "TARGET_CRS": QgsCoordinateReferenceSystem(
                "{}:{}".format(DC_PROVIDERS[target_crs], target_crs)
            ),
            "RESAMPLING": 1,  # average=5; nearest=0, bilinear=1
            "NODATA": -1,
            "TARGET_RESOLUTION": target_cellsize,  # in meters
            "OPTIONS": "",
            "DATA_TYPE": 6,  # float32=6, byte=1
            "TARGET_EXTENT": target_extent_str,
            "TARGET_EXTENT_CRS": QgsCoordinateReferenceSystem(
                "{}:{}".format(DC_PROVIDERS[target_crs], target_crs)
            ),
            "MULTITHREADING": False,
            "EXTRA": "",
            "OUTPUT": file_output,
        },
    )
    return file_output


def get_downstream_ids(file_ldd, file_basins, file_outlets):
    """Get the basin Ids from downstream cells.

    ldd - Direction convention:

    .. code-block:: text

        7   8   9
        4   5   6
        1   2   3


    :param file_ldd: file path to LDD raster (map of LDD direction convention)
    :type file_ldd: str
    :param file_basins: file path to basins raster
    :type file_basins: str
    :param file_outlets: file path to outlets raster
    :type file_outlets: str
    :return: dictionaty with lists of "Id" and "Downstream_Id"
    :rtype: dict
    """

    # -------------------------------------------------------------------------
    # LOAD LDD

    # Open the raster file using gdal
    raster_input = gdal.Open(file_ldd)

    # Get the raster band
    band_input = raster_input.GetRasterBand(1)

    # Read the raster data as a numpy array
    grid_ldd = band_input.ReadAsArray()
    # truncate to byte integer
    grid_ldd = grid_ldd.astype(np.uint8)

    # -- Close the raster
    raster_input = None

    # -------------------------------------------------------------------------
    # LOAD OUTLETS

    # Open the raster file using gdal
    raster_input = gdal.Open(file_outlets)

    # Get the raster band
    band_input = raster_input.GetRasterBand(1)

    # Read the raster data as a numpy array
    grid_outlets = band_input.ReadAsArray()
    # truncate to byte integer
    grid_outlets = grid_outlets.astype(np.uint8)

    # -- Close the raster
    raster_input = None

    # -------------------------------------------------------------------------
    # LOAD BASINS

    # Open the raster file using gdal
    raster_input = gdal.Open(file_basins)

    # Get the raster band
    band_input = raster_input.GetRasterBand(1)

    # Read the raster data as a numpy array
    grid_basins = band_input.ReadAsArray()
    # truncate to byte integer
    grid_basins = grid_basins.astype(np.uint8)

    # -- Close the raster
    raster_input = None

    # -------------------------------------------------------------------------
    # PROCESS

    # Get the indices of non-zero elements
    nonzero_indices = np.nonzero(grid_outlets)
    # Create a list of tuples containing row and column indices
    list_positions = list(zip(nonzero_indices[0], nonzero_indices[1]))

    list_ids = list()
    list_ds_ids = list()
    for outlet in list_positions:
        i = outlet[0]
        j = outlet[1]
        # get the ID
        n_id = grid_outlets[i][j]
        # get the direction code (LDD)
        n_dir = grid_ldd[i][j]
        # get downstream coordinates
        dict_dc = geo.downstream_coordinates(n_dir=n_dir, i=i, j=j, s_convention="ldd")
        new_i = dict_dc["i"]
        new_j = dict_dc["j"]
        n_ds_id = grid_basins[new_i][new_j]
        list_ids.append(n_id)
        list_ds_ids.append(n_ds_id)

    return {"Id": list_ids, "Downstream_Id": list_ds_ids}


def get_carved_dem(file_dem, file_rivers, file_output, wedge_width=3, wedge_depth=10):
    """Apply a carving method to DEM raster

    :param file_dem: file path to DEM raster
    :type file_dem: str
    :param file_rivers: file path to main rivers pseudo-boolean raster
    :type file_rivers: str
    :param file_output: file path to output raster
    :type file_output: str
    :param wedge_width: width parameter in unit cells (pixels)
    :type wedge_width: int
    :param wedge_depth: depth parameter in dem units (e.g., meters)
    :type wedge_depth: float
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
    grid_output = geo.carve_dem(grd_dem=grd_dem, grd_rivers=grd_rivers, wedge_width=wedge_width, wedge_depth=wedge_depth)

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

def get_slope(file_dem, file_output, degree=True):
    """Get the Slope map in degrees from a DEM raster

    :param file_dem: file path to DEM raster
    :type file_dem: str
    :param file_output: file path to output raster
    :type file_output: str
    :param degree: If True (default), the output slope values are in degrees. If False, output units are in radians.
    :type degree: bool
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
    grid_output = geo.slope(dem=grd_dem, cellsize=cellsize, degree=degree)

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


def get_shalstab(
    file_slope,
    file_flowacc,
    output_folder,
    suffix,
    cellsize=30,
    soil_phi=15,
    soil_z=1,
    soil_p=1600,
    soil_c=1,
    water_p=997,
):
    # todo [docstring]
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
    # LOAD Flowacc

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
    w1, w2 = geo.shalstab_wetness(
        flowacc=grd_flowacc,
        slope=grd_slope,
        cellsize=cellsize,
        soil_phi=soil_phi,
        soil_z=soil_z,
        soil_p=soil_p,
        soil_c=soil_c,
        water_p=water_p,
        kPa=True,
    )

    # -------------------------------------------------------------------------
    # EXPORT RASTER FILE
    # Get the driver to create the new raster
    driver = gdal.GetDriverByName("GTiff")

    # Create a new raster with the same dimensions as the original
    file_output1 = f"{output_folder}/{suffix}_qt.tif"
    raster_output1 = driver.Create(
        file_output1, raster_x_size, raster_y_size, 1, gdal.GDT_Float32
    )

    # Set the projection and geotransform of the new raster to match the original
    raster_output1.SetProjection(raster_projection)
    raster_output1.SetGeoTransform(raster_geotransform)

    # Write the new data to the new raster
    raster_output1.GetRasterBand(1).WriteArray(w1)

    # Create a new raster with the same dimensions as the original
    file_output2 = f"{output_folder}/{suffix}_cls.tif"
    raster_output2 = driver.Create(
        file_output2, raster_x_size, raster_y_size, 1, gdal.GDT_Byte
    )

    # Set the projection and geotransform of the new raster to match the original
    raster_output2.SetProjection(raster_projection)
    raster_output2.SetGeoTransform(raster_geotransform)

    # Write the new data to the new raster
    raster_output2.GetRasterBand(1).WriteArray(w2)

    # Close
    raster_output1 = None
    raster_output2 = None
    return file_output1, file_output2


def get_twi(file_slope, file_upa, file_output):
    """Get the TWI map from Slope and Upslope Area

    :param file_slope: file path to Slope raster map
    :type file_slope: str
    :param file_upa: file path to Upslope Area raster map
    :type file_upa: str
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
    # LOAD Flowacc

    # Open the raster file using gdal
    raster_upa = gdal.Open(file_upa)

    # Get the raster band
    band_upa = raster_upa.GetRasterBand(1)

    # Read the raster data as a numpy array
    grd_upa = band_upa.ReadAsArray()

    # -- Close the raster
    raster_upa = None

    # -------------------------------------------------------------------------
    # PROCESS
    grid_output = geo.twi(slope=grd_slope, flowacc=grd_upa, cellsize=cellsize)

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


def setup_hand(file_dem_filled, output_folder):
    """Get basic files for HAND raster analysis. The DEM must be hydrologically consistent (no sinks).

    :param file_dem_filled: file path to filled DEM (no sinks)
    :type file_dem_filled: str
    :param output_folder: path to output folder
    :type output_folder: str
    :return: dictionary with raster output paths and cellsize. keys:
        "ldd", "material", "accflux", "cellsize"

    :rtype: dict
    """
    # ----------- Setup -----------

    # -- PCRaster - get LDD map
    file_ldd = "{}/ldd.map".format(output_folder)
    print("[pc-raster] get LDD...")
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
    print("[pc-raster] get material (constant)...")
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
    print("[pc-raster] get flux...")
    file_accflux = "{}/accflux.map".format(output_folder)
    processing.run(
        "pcraster:accuflux",
        {
            "INPUT": file_ldd,
            "INPUT2": file_material,
            "OUTPUT": file_accflux,
        },
    )
    return {
        "ldd": file_ldd,
        "material": file_material,
        "accflux": file_accflux,
        "cellsize": cellsize,
    }


def get_hand(
    file_dem_filled, output_folder, hand_threshholds=[0.1, 1, 2], file_dem_sample=None
):
    """Get the HAND raster map from a DEM.
    The DEM must be hydrologically consistent (no sinks).

    :param file_dem_filled: file path to filled DEM (no sinks)
    :type file_dem_filled: str
    :param output_folder: path to output folder
    :type output_folder: str
    :param hand_threshholds: list of area threshhold for HAND in number drainage area in sq km (0.1 km2 = 10 ha)
    :type hand_threshhold: list
    :param file_dem_sample: [optional] file path to raw DEM for sampling the HAND.
    :type file_dem_sample: str
    :return: dictionary with raster output paths. keys:
        "ldd", "material", "accflux", "threshold", "drainage", "hs_outlets_scalar",
        "hs_outlets_nominal", "hillslopes", "hs_zmin", "hand"

    :rtype: dict
    """

    # ----------- Setup -----------
    dict_setup = setup_hand(file_dem_filled, output_folder)
    file_ldd = dict_setup["ldd"]
    file_material = dict_setup["material"]
    file_accflux = dict_setup["accflux"]
    cellsize = dict_setup["cellsize"]

    # ----------- Hand -----------
    d_files = {}
    for t in hand_threshholds:
        if "." in str(t):
            str_t = str(t).replace(".", "P")
        else:
            str_t = str(t) + "P0"
        # create a local folder
        lcl_output = "{}/hand_{}".format(output_folder, str_t)
        if not os.path.isdir(lcl_output):
            os.mkdir(lcl_output)

        # -- PCRaster - create drainage threshold layer with
        file_threshold = "{}/threshold.map".format(lcl_output)
        processing.run(
            "pcraster:spatial",
            {
                "INPUT": int(t * 1000 * 1000),
                "INPUT1": 3,
                "INPUT2": file_dem_filled,
                "OUTPUT": file_threshold,
            },
        )
        # -- PCRaster - create drainage boolean
        file_drainage = "{}/drainage.map".format(lcl_output)
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
        file_outlets_1 = "{}/hs_outlets_scalar.map".format(lcl_output)
        processing.run(
            "pcraster:uniqueid",
            {"INPUT": file_drainage, "OUTPUT": file_outlets_1},
        )

        # -- PCRaster - convert unique ids to nominal
        file_outlets_2 = "{}/hs_outlets_nominal.map".format(lcl_output)
        processing.run(
            "pcraster:convertdatatype",
            {
                "INPUT": file_outlets_1,
                "INPUT1": 1,
                "OUTPUT": file_outlets_2,
            },
        )
        # -- PCRaster - create subcatchments
        file_hillslopes = "{}/hillslopes.map".format(lcl_output)
        processing.run(
            "pcraster:subcatchment",
            {
                "INPUT1": file_ldd,
                "INPUT2": file_outlets_2,
                "OUTPUT": file_hillslopes,
            },
        )

        # -- PCRaster - sample z min by catchment
        file_zmin = "{}/hs_zmin.map".format(lcl_output)
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
        file_hand = "{}/hand_{}.tif".format(lcl_output, str_t)
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
        d_files[str_t] = file_hand

    d_files["ldd"] = file_ldd
    d_files["accflux"] = file_accflux

    return d_files



def setup_topo(
    project_folder,
    file_src_dem,
    crs_target,
    db_aoi,
    layer_aoi,
    db_rivers,
    layer_river_lines,
    layer_river_polygons=None,
    crs_source="4326",
    target_cellsize=30,
    translate_to_ascii=False,
):
    """Setup protocols for getting ``topo`` datasets

    :param project_folder: path to main folder to export datasets
    :type project_folder: str
    :param file_src_dem: file path to larger source DEM raster -- expected to be in WGS-84 (4326)
    :type file_src_dem: str
    :param crs_target: QGIS code (number only) for the target CRS
    :type crs_target: str
    :param db_aoi: path to AOI layer geopackage database
    :type db_aoi: str
    :param layer_aoi: name of Area Of Interest layer (polygon).
    :type layer_aoi: str
    :param db_rivers: path to rivers layer geopackage database
    :type db_rivers: str
    :param layer_river_lines: name of main rivers lines layer.
    :type layer_river_lines: str
    :param layer_river_polygons: name of main rivers polygon layer (optional).
    :type layer_river_polygons: str
    :param crs_source: QGIS code (number only) for the source CRS. default is WGS-84 (4326)
    :type crs_source: str
    :param translate_to_ascii: flag to convert to asc format
    :type translate_to_ascii: bool
    :return: dictionary of output resources
    :rtype: dict

    Script example (for QGIS Python Console):

    .. code-block:: python

        # plans source code must be pasted to QGIS plugins directory
        from plans import qutils

        # call function
        qutils.setup_topo(
            project_folder="path/to/project/folder",
            file_src_dem="path/to/src_dem.tif",
            crs_target="5641",
            db_aoi="path/to/src_aoi.gpkg",
            layer_aoi="src_aoi",
            db_rivers="path/to/src_rivers.gpkg",
            layer_river_lines="src_rivers",
            layer_river_polygons="src_water",
            crs_source="4326",
        )

    """
    # folders and file setup
    topo_folder = os.path.join(project_folder, "topo")
    if os.path.isdir(topo_folder):
        pass
    else:
        os.mkdir(topo_folder)

    output_folder_interm = "{}/_intermediate".format(topo_folder)
    if os.path.isdir(output_folder_interm):
        pass
    else:
        os.mkdir(output_folder_interm)

    # vector ops
    db_project = f"{project_folder}/vectors.gpkg"

    # reproject aoi to project database
    print("reproject aoi layer...")
    new_layer_aoi = reproject_layer(
        db_input=db_aoi,
        layer_name=layer_aoi,
        crs_target=crs_target,
        db_output=db_project,
        layer_output="aoi"
    )
    # reproject rivers to project database
    print("clip river lines layer...")
    clipped_layer_rivers = clip_layer(
        db_input=db_rivers,
        layer_input=layer_river_lines,
        db_overlay=db_aoi,
        layer_overlay=layer_aoi,
        db_output=db_project,
        layer_output="rivers_src",
    )
    print("reproject river lines layer...")
    # save to same database
    new_layer_rivers = reproject_layer(
        db_input=db_project,
        layer_name=clipped_layer_rivers,
        crs_target=crs_target,
        db_output=db_project,
        layer_output="rivers"
    )

    if layer_river_polygons is not None:
        # reproject rivers to project database
        print("clip rivers polygons layer...")
        clipped_layer_rivers2 = clip_layer(
            db_input=db_rivers,
            layer_input=layer_river_polygons,
            db_overlay=db_aoi,
            layer_overlay=layer_aoi,
            db_output=db_project,
            layer_output="rivers_src_polygons",
        )
        print("reproject river lines layer...")
        # save to same database
        new_layer_rivers_polygons = reproject_layer(
            db_input=db_project,
            layer_name=clipped_layer_rivers2,
            crs_target=crs_target,
            db_output=db_project,
            layer_output="rivers_polygons"
        )

    # get aoi extent
    print("get aoi extent...")
    dict_bbox = get_extent_from_vector(
        db_input=db_project,
        layer_name=new_layer_aoi
    )

    # clip and warp dem
    print(r"clip-n-warp dem...")
    file_dem = get_dem(
        file_dem_src=file_src_dem,
        target_crs=crs_target,
        target_extent=dict_bbox,
        file_output=f"{topo_folder}/dem.tif",
        source_crs=crs_source,
        target_cellsize=target_cellsize,
    )

    if translate_to_ascii:
        print("translate...")
        processing.run(
            "gdal:translate",
            {
                "INPUT": file_dem,
                "TARGET_CRS": QgsCoordinateReferenceSystem(
                    "{}:{}".format(DC_PROVIDERS[crs_target], crs_target)
                ),
                "NODATA": -9999,
                "COPY_SUBDATASETS": False,
                "OPTIONS": "",
                "EXTRA": "",
                "DATA_TYPE": 6,
                "OUTPUT": "{}/dem.asc".format(project_folder),
            },
        )
    return {
        "db": db_project,
        "aoi": new_layer_aoi,
        "rivers": new_layer_rivers,
        "file_dem": file_dem,
    }


def get_topo(
    output_folder,
    file_dem,
    input_db,
    layer_rivers,
    wedge_width=3,
    wedge_depth=10,
    hand_threshold=5,
    use_saga=False,
    fill_xxl=False,
    translate_to_ascii=False,
    style_folder=None
):
    """Get all ``topo`` datasets for running PLANS.

    .. warning::

        run ``setup_topo()`` first.

    :param output_folder: path to output folder to export datasets
    :type output_folder: str
    :param file_dem: file path to larger DEM dataset raster -- expected to be in WGS-84
    :type file_dem: str
    :param input_db: path to geopackage database
    :type input_db: str
    :param layer_rivers: name of main rivers layers (lines) in target CRS.
    :type layer_rivers: str
    :param wedge_width: carving dem parameter -- width parameter in unit cells
    :type wedge_width: int
    :param wedge_depth: carving dem parameter -- depth parameter in dem units (e.g., meters)
    :type wedge_depth: float
    :param hand_threshold: [hand parameter in sqkm] threshhold for HAND in number of maximum hillslope area
    :type hand_threshold: int
    :param translate_to_ascii: flag to convert to asc format
    :type translate_to_ascii: bool
    :param style_folder: path to folder containing style files
    :type style_folder: str
    :return: None
    :rtype: None

    Script example (for QGIS Python Console):

    .. code-block:: python

        # plans source code must be pasted to QGIS plugins directory
        from plans import qutils

        # call function
        qutils.get_topo(
            output_folder="path/to/folder",
            file_dem="path/to/main_large_dem.tif",
            input_db="path/to/project_db.gpkg",
            layer_rivers='rivers',
            wedge_width=3,
            wedge_depth=10,
            hand_cells=100,
            translate_to_ascii=False,
            style_folder="path/to/style_folder"
        )

    """

    # ----- SETUP -----
    # folders and file setup
    output_folder_interm = "{}/_intermediate".format(output_folder)
    if os.path.isdir(output_folder_interm):
        pass
    else:
        os.mkdir(output_folder_interm)


    dict_files = {
        "dem": file_dem,
        "main_rivers": "{}/main_rivers.tif".format(output_folder_interm),
        "dem_b": "{}/dem-carved.tif".format(output_folder_interm),
        "slope": "{}/slp.tif".format(output_folder),
        "hsd": "{}/hsd.tif".format(output_folder),
        "fill_saga": "{}/dem-filled.sdat".format(output_folder_interm),
        "fac_mfd_saga": "{}/fac_mfd.sdat".format(output_folder_interm),
        "fac": "{}/fac.tif".format(output_folder),
        "upa": "{}/upa.tif".format(output_folder),
        "twi": "{}/twi.tif".format(output_folder),
        "ldd": "{}/ldd.tif".format(output_folder),
        "hand": "{}/hnd.tif".format(output_folder),
        "dem_bf": "{}/dem-filled.tif".format(output_folder_interm),
        # "dem_bfb": "{}/dem_bfb.tif".format(output_folder_interm),
        "dem_pc": "{}/dem.map".format(output_folder_interm),
        "dem_bf_pc": "{}/dem_bf.map".format(output_folder_interm),
    }

    # get cellsize
    cellsize = get_cellsize(file_input=dict_files["dem"])

    # 4) get rivers blank
    print("get main rivers...")
    file_rivers = get_blank_raster(
        file_input=file_dem, file_output=dict_files["main_rivers"], blank_value=0
    )

    # rasterize rivers
    print("rasterize rivers...")
    processing.run(
        "gdal:rasterize_over_fixed_value",
        {
            "INPUT": "{}|layername={}".format(input_db, layer_rivers),
            "INPUT_RASTER": dict_files["main_rivers"],
            "BURN": 1,
            "ADD": False,
            "EXTRA": "",
        },
    )

    # get carved
    print("carve dem...")
    file_burn = get_carved_dem(
        file_dem=dict_files["dem"],
        file_rivers=dict_files["main_rivers"],
        file_output=dict_files["dem_b"],
        wedge_width=wedge_width,
        wedge_depth=wedge_depth,
    )

    # get slope
    print("get slope...")
    # use gdal
    processing.run("gdal:slope",
        {
            'INPUT': dict_files["dem"],
            'BAND': 1,
            'SCALE': 1,
            'AS_PERCENT': False,
            'COMPUTE_EDGES': True,
            'ZEVENBERGEN': False,
            'OPTIONS': '',
            'EXTRA': '',
            'OUTPUT': dict_files["slope"]
        }
    )
    #get_slope(file_dem=dict_files["dem"], file_output=dict_files["slope"])

    processing.run("gdal:hillshade",
        {
            'INPUT': dict_files["dem"],
            'BAND': 1,
            'Z_FACTOR': 2,
            'SCALE': 1,
            'AZIMUTH': 315,
            'ALTITUDE': 45,
            'COMPUTE_EDGES': True,
            'ZEVENBERGEN': False,
            'COMBINED': False,
            'MULTIDIRECTIONAL': False,
            'OPTIONS': '',
            'EXTRA': '',
            'OUTPUT': dict_files["hsd"]
        }
    )

    # SAGA processes
    if use_saga:
        # fill sinks
        if fill_xxl:
            print("[saga] fill sinks XXL...")
            processing.run(
                "sagang:fillsinksxxlwangliu",
                {
                    "ELEV": dict_files["dem_b"],
                    "FILLED": dict_files["fill_saga"],
                    "MINSLOPE": 0.01,
                },
            )
        else:
            print("[saga] fill sinks ...")
            processing.run(
                "sagang:fillsinksplanchondarboux2001",
                {
                    "DEM": dict_files["dem_b"],
                    "RESULT": dict_files["fill_saga"],
                    "MINSLOPE": 0.01,
                },
            )
        print("translate fill...")
        processing.run(
            "gdal:translate",
            {
                "INPUT": dict_files["fill_saga"],
                "TARGET_CRS": QgsCoordinateReferenceSystem(
                    "{}:{}".format(DC_PROVIDERS[target_crs], target_crs)
                ),
                "NODATA": None,
                "COPY_SUBDATASETS": False,
                "OPTIONS": "",
                "EXTRA": "",
                "DATA_TYPE": 6,
                "OUTPUT": dict_files["dem_bf"],
            },
        )
        # get flow acc
        print("[saga] get upslope area...")
        processing.run(
            "sagang:flowaccumulationparallelizable",
            {
                "DEM": dict_files["dem_bf"],
                "FLOW": dict_files["flowacc_mfd_saga"],
                "UPDATE": 0,
                "METHOD": 2,
                "CONVERGENCE": 1.1,
            },
        )
        # translate flow acc
        print("[saga] translate upslope area...")
        processing.run(
            "gdal:translate",
            {
                "INPUT": dict_files["flowacc_mfd_saga"],
                "TARGET_CRS": QgsCoordinateReferenceSystem(
                    "{}:{}".format(DC_PROVIDERS[target_crs], target_crs)
                ),
                "NODATA": None,
                "COPY_SUBDATASETS": False,
                "OPTIONS": "",
                "EXTRA": "",
                "DATA_TYPE": 6,
                "OUTPUT": dict_files["fac"],
            },
        )
    else:
        print("[wbt] fill sinks ...")
        # fill sinks
        processing.run("wbt:FillDepressionsWangAndLiu",
           {
               'dem': dict_files["dem_b"],
               'fix_flats': True,
               'flat_increment': None,
               'output': dict_files["dem_bf"],
           }
        )
        # get flow acc
        print("[wbt] flow accumulation...")
        processing.run("wbt:DInfFlowAccumulation",
            {
            'input': dict_files["dem_bf"],
            'out_type': 2,
                'threshold': None,
                'log': False,
                'clip': False,
                'pntr': False,
                'output': dict_files["fac"],
            }
        )
    print("[wbt] get upslope area...")
    # get upslope area
    processing.run("wbt:D8FlowAccumulation",
        {
            'input': dict_files["dem_bf"],
            'out_type': 1,
            'log': False,
            'clip': False,
            'pntr': False,
            'esri_pntr': False,
            'output': dict_files["upa"]
        }
    )
    # get TWI
    print("get TWI...")
    get_twi(
        file_slope=dict_files["slope"],
        file_upa=dict_files["fac"],
        file_output=dict_files["twi"],
    )

    # HAND
    # DEM PC-raster
    print("[pc-raster] convert dem to pcraster...")
    processing.run(
        "pcraster:converttopcrasterformat",
        {
            "INPUT": dict_files["dem"],
            "INPUT2": 3,
            "OUTPUT": dict_files["dem_pc"]
        },
    )
    # DEM_bf PC-raster
    print("[pc-raster] convert dem_bf to pcraster...")
    processing.run("pcraster:converttopcrasterformat",
        {
            "INPUT": dict_files["dem_bf"],
            "INPUT2": 3,
            "OUTPUT": dict_files["dem_bf_pc"]
        }
    )
    print("[pc-raster] get HAND...")
    dict_hand = get_hand(
        file_dem_filled=dict_files["dem_bf_pc"],
        output_folder=output_folder_interm,
        hand_threshholds=[hand_threshold],
        file_dem_sample=dict_files["dem_pc"],
    )
    processing.run("gdal:translate",
        {
            'INPUT': dict_hand["ldd"],
            'TARGET_CRS':None,
            'NODATA':10,
            'COPY_SUBDATASETS':False,
            'OPTIONS':'',
            'EXTRA':'',
            'DATA_TYPE':1,
            'OUTPUT':dict_files["ldd"],
        }
    )
    shutil.copy(
        src=dict_hand["{}P0".format(hand_threshold)],
        dst=dict_files["hand"],
    )

    if translate_to_ascii:
        # append to dict
        for f in dict_hand:
            dict_files[f] = dict_hand[f]
        # translate all
        dict_raster = {"nodata": -9999, "data_type": 6}
        dict_qualiraster = {"nodata": 0, "data_type": 1}
        dict_translate = {
            "dem": dict_raster,
            "slope": dict_raster,
            "twi": dict_raster,
            "accflux": dict_raster,
            "ldd": dict_qualiraster,
        }
        # loop
        print("translating outputs...")
        for k in dict_translate:
            input_file = dict_files[k]
            processing.run(
                "gdal:translate",
                {
                    "INPUT": input_file,
                    "TARGET_CRS": None,
                    "NODATA": dict_translate[k]["nodata"],
                    "COPY_SUBDATASETS": False,
                    "OPTIONS": "",
                    "EXTRA": "",
                    "DATA_TYPE": dict_translate[k]["data_type"],
                    "OUTPUT": "{}/{}.asc".format(output_folder, k),
                },
            )

    # todo [improvements] -- include a style handler
    if style_folder is not None:
        ls_styles = ["dem", "hand", "slope", "twi", "fac", "upa", "ldd"]


    print("get topo --DONE")
    return None

# todo [RESUME HERE] evaluate to include HTWI derivation in a folder
# todo -- DEM -- problem of retrieving a DEM from a tile system
# todo -- LULC -- problem with conversion between mapbiomas and plans
def get_lulc(
    list_main_files,
    list_dates,
    lulc_table,
    output_folder,
    target_file,
    crs_target,
    db_roads=None,
    layer_roads_dirty="roads_dirty",
    layer_roads_paved="roads_paved",
    source_crs="4326",
    qml_file=None,
    translate=True,
):
    """Get LULC maps from a larger main LULC dataset.

    :param list_main_files: list of file paths to main lulc rasters
    :type list_main_files: list
    :param list_dates: list of dates (YYYY-MM-DD) for each main raster
    :type list_dates: list
    :param lulc_table: path to parameter table file. Expected columns: Id, Name, Alias and Color
    :type lulc_table: str
    :param output_folder: path to output folder
    :type output_folder: str
    :param target_file: file path to target raster (e.g., the DEM raster)
    :type target_file: str
    :param crs_target: EPSG code (number only) for the target CRS
    :type crs_target: str
    :param db_roads: [optional] path to geopackage database with road layers
    :type db_roads: str
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

    Script example (for QGIS):

    .. code-block:: python

        # plans source code must be pasted to QGIS plugins directory
        from plans import qutils
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
        qutils.get_lulc(
            list_main_files=lst_files,
            list_dates=lst_dates,
            output_folder="path/to/output",
            target_file="/path/to/raster.tif",
            crs_target="5641",
            db_roads="path/to/input_db.gpkg",
            layer_roads_dirty='roads_dirty',
            layer_roads_paved='roads_paved',
            source_crs='4326',
            qml_file="path/to/style.qml"
        )

    """
    # folders and file setup
    if translate:
        output_folder_interm = "{}/intermediate".format(output_folder)
        if os.path.isdir(output_folder_interm):
            pass
        else:
            os.mkdir(output_folder_interm)
    else:
        output_folder_interm = output_folder
    # ---------- warp lulc -----------
    target_extent_dict = get_extent_from_raster(file_input=target_file)
    # get extent string
    target_extent_str = "{},{},{},{} [{}:{}]".format(
        target_extent_dict["xmin"],
        target_extent_dict["xmax"],
        target_extent_dict["ymin"],
        target_extent_dict["ymax"],
        DC_PROVIDERS[crs_target],
        crs_target,
    )
    cellsize = get_cellsize(file_input=target_file)
    list_output_files = list()
    list_filenames = list()
    print("clip-n-warp lulc ... ")
    for i in range(len(list_main_files)):
        file_main_lulc = list_main_files[i]
        date = list_dates[i]
        str_filename = "lulc_{}".format(date)
        file_output1 = "{}/_{}.tif".format(output_folder_interm, str_filename)

        # run warp
        processing.run(
            "gdal:warpreproject",
            {
                "INPUT": file_main_lulc,
                "SOURCE_CRS": QgsCoordinateReferenceSystem(
                    "{}:{}".format(DC_PROVIDERS[source_crs], source_crs)
                ),
                "TARGET_CRS": QgsCoordinateReferenceSystem(
                    "{}:{}".format(DC_PROVIDERS[crs_target], crs_target)
                ),
                "RESAMPLING": 0,  # nearest
                "NODATA": 0,
                "TARGET_RESOLUTION": cellsize,
                "OPTIONS": "",
                "DATA_TYPE": 1,
                "TARGET_EXTENT": target_extent_str,
                "TARGET_EXTENT_CRS": QgsCoordinateReferenceSystem(
                    "{}:{}".format(DC_PROVIDERS[crs_target], crs_target)
                ),
                "MULTITHREADING": False,
                "EXTRA": "",
                "OUTPUT": file_output1,
            },
        )
        file_output = "{}/{}.tif".format(output_folder_interm, str_filename)
        list_output_files.append(file_output[:])
        list_filenames.append(str_filename[:])
        # fill no data
        processing.run(
            "gdal:fillnodata",
            {
                "INPUT": file_output1,
                "BAND": 1,
                "DISTANCE": 10,
                "ITERATIONS": 0,
                "MASK_LAYER": None,
                "OPTIONS": "",
                "EXTRA": "-interp nearest",
                "OUTPUT": file_output,
            },
        )
        # delete warp
        os.remove(file_output1)
    #
    # ---------- handle roads -----------
    if db_roads is None:
        pass
    else:
        print("reproject road layers...")
        # reproject dirty

        new_layer_dirty = reproject_layer(
            db_input=db_roads, layer_name=layer_roads_dirty, crs_target=crs_target
        )
        # reproject paved
        new_layer_paved = reproject_layer(
            db_input=db_roads, layer_name=layer_roads_paved, crs_target=crs_target
        )

        # rasterize loop
        print("rasterizing road layers...")
        for i in range(len(list_output_files)):
            input_raster = list_output_files[i]
            # dirty
            processing.run(
                "gdal:rasterize_over_fixed_value",
                {
                    "INPUT": "{}|layername={}".format(db_roads, new_layer_dirty),
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
                    "INPUT": "{}|layername={}".format(db_roads, new_layer_paved),
                    "INPUT_RASTER": input_raster,
                    "BURN": 254,
                    "ADD": False,
                    "EXTRA": "",
                },
            )
    # ---------- translate all -----------
    for i in range(len(list_output_files)):
        input_file = list_output_files[i]
        filename = list_filenames[i]
        if translate:
            print("translate...")
            processing.run(
                "gdal:translate",
                {
                    "INPUT": input_file,
                    "TARGET_CRS": QgsCoordinateReferenceSystem(
                        "EPSG:{}".format(crs_target)
                    ),
                    "NODATA": 0,
                    "COPY_SUBDATASETS": False,
                    "OPTIONS": "",
                    "EXTRA": "",
                    "DATA_TYPE": 1,
                    "OUTPUT": "{}/{}.asc".format(output_folder, filename),
                },
            )

        # Handle style
        if qml_file is None:
            pass
        else:
            shutil.copy(src=qml_file, dst="{}/{}.qml".format(output_folder, filename))

    # Handle table
    print("table...")
    df_table = pd.read_csv(lulc_table, sep=";")
    df_roads = pd.DataFrame(
        {
            "Id": [254, 255],
            "Name": ["Paved roads", "Dirty roads"],
            "Alias": ["RdP", "RdD"],
            "Color": ["#611d1d", "#976d3f"],
        }
    )
    df_table = pd.concat([df_table, df_roads], ignore_index=True)
    df_table.to_csv(os.path.join(output_folder, "lulc_info.csv"), sep=";", index=False)

    return None


def get_rain(
    output_folder,
    src_folder,
    input_db,
    target_file,
    target_crs,
    layer_aoi="aoi",
    layer_rain_gauges="rain",
):
    """Get ``rain`` datasets for running PLANS.

    :param output_folder: path to output folder
    :type output_folder: str
    :param src_folder: path to source folder for all csv series files. Files must be named by rain_<alias>.csv
    :type src_folder: str
    :param input_db: path to geopackage database with rain gauge layer
    :type input_db: str
    :param target_file: file path to target raster (e.g., the ``DEM`` raster)
    :type target_file: str
    :param target_crs: EPSG code (number only) for the target CRS
    :type target_crs: str
    :param layer_aoi: name of Area Of Interest layer (polygon). Default is "aoi"
    :type layer_aoi: str
    :param layer_rain_gauges: layer name of rain gauges layer.
        Required fields in layer:

            - ``Id``: int, required. Unique number id.
            - ``Name``: str, required. Simple name.
            - ``Alias``: str, required. Short nickname.
            - ``Rain_File``: str, required. Path to data time series ``csv`` source file.
            - ``X``: float, optional. Recomputed from layer. Longitude in WGS 84 Datum (EPSG4326).
            - ``Y``: float, optional. Recomputed from layer. Latitude in WGS 84 Datum (EPSG4326).
            - ``Code``: str, op
            - ``Source``: str, required
            - ``Description``: str, required
            - ``Color``: str, optional

    :type layer_rain_gauges: str
    :return: None
    :rtype: None

    Script example (for QGIS Python Console):

    .. code-block:: python

        # plans source code must be pasted to QGIS plugins directory
        from plans import qutils

        # call function
        qutils.get_rain(
            output_folder="path/to/rain",
            src_folder="path/to/src",
            target_file="path/to/dem.asc",
            target_crs="31982",
            input_db="path/to/my_db.gpkg",
            layer_aoi="aoi",
            layer_rain_gauges="rain"
        )

    """
    from plans import datasets

    def calculate_buffer_ratios(box_small, box_large):
        delta_x_small = abs(box_small["xmax"] - box_small["xmin"])
        lower_x = abs(box_small["xmin"] - box_large["xmin"])
        upper_x = abs(box_small["xmax"] - box_large["xmax"])
        x_ratio = max(lower_x / delta_x_small, upper_x / delta_x_small)

        delta_y_small = abs(box_small["ymax"] - box_small["ymin"])
        lower_y = abs(box_small["ymin"] - box_large["ymin"])
        upper_y = abs(box_small["ymax"] - box_large["ymax"])
        y_ratio = max(lower_y / delta_y_small, upper_y / delta_y_small)

        return (x_ratio, y_ratio)

    # folders and file setup
    print("folder setup...")
    output_folder_interm = "{}/intermediate".format(output_folder)
    if os.path.isdir(output_folder_interm):
        pass
    else:
        os.mkdir(output_folder_interm)
    print("get info...")

    # ---------------- RAIN SERIES ----------------
    # load table
    rain_gdf = gpd.read_file(input_db, layer=layer_rain_gauges)

    # overwrite
    rain_gdf["Rain_Units"] = "mm"
    rain_gdf["Rain_DtField"] = "DateTime"
    rain_gdf["Rain_VarField"] = "P"

    # compute X and Y
    rain_gdf["X"] = rain_gdf.geometry.x
    rain_gdf["Y"] = rain_gdf.geometry.y
    if "Color" not in rain_gdf.columns:
        rain_gdf["Color"] = datasets.get_colors(size=len(rain_gdf))

    # rename files and copy
    lst_files_src = rain_gdf["Rain_File"].values
    lst_files_dst = [
        "{}/rain_{}.csv".format(output_folder, a) for a in rain_gdf["Alias"].values
    ]
    lst_new_files = []
    for i in range(len(rain_gdf)):
        if os.path.isfile(lst_files_src[i]):
            shutil.copy(src=lst_files_src[i], dst=lst_files_dst[i])
            new_file_name = "rain_{}.csv".format(rain_gdf["Alias"].values[i])
        else:
            new_file_name = ""
        lst_new_files.append(new_file_name)
    # reset File
    rain_gdf["Rain_File"] = lst_new_files

    # Drop the geometry column
    rain_gdf = rain_gdf.drop(columns=["geometry"])

    # organize columns
    rain_gdf = rain_gdf[
        [
            "Id",
            "Name",
            "Alias",
            "X",
            "Y",
            "Source",
            "Code",
            "Description",
            "Color",
            "Rain_Units",
            "Rain_VarField",
            "Rain_DtField",
            "Rain_File",
        ]
    ]

    # export csv file
    rain_info_file = "{}/rain_info.csv".format(output_folder)
    rain_gdf.to_csv(rain_info_file, sep=";", index=False)
    # print(rain_gdf.to_string())

    # ---------------- RAIN ZONES ----------------

    # first check number of features
    n_rain_gauges = count_vector_features(input_db=input_db, layer_name=layer_rain_gauges)
    dict_blank = {"constant": 0, "voronoi": 0}
    s_type = "voronoi"
    if n_rain_gauges < 2:
        print(" -- only one rain gauge found")
        s_type = "constant"
        # create constant raster layer based on target and Id

        # >> find rain id
        n_id = rain_gdf["Id"].values[0]
        # update dict
        dict_blank["constant"] = n_id
    else:
        # get voronoi zones

        # first validade extents
        dict_extent_aoi = get_extent_from_vector(db_input=input_db, layer_name=layer_aoi)
        dict_extent_rain = get_extent_from_vector(
            db_input=input_db, layer_name=layer_rain_gauges
        )

        # check is is bounded:
        if is_bounded_by(extent_a=dict_extent_rain, extent_b=dict_extent_aoi):
            n_buffer = 20
        else:
            n_buffer = (
                100
                * 1.2
                * max(
                    calculate_buffer_ratios(
                        box_small=dict_extent_rain, box_large=dict_extent_aoi
                    )
                )
            )

        # reproject rain
        layer_rain_gauges_2 = reproject_layer(
            db_input=input_db, layer_name=layer_rain_gauges, crs_target=target_crs
        )
        print("get voronoi polygons...")
        # run function
        zones_shp = "{}/rain_zones.shp".format(output_folder_interm)
        processing.run(
            "qgis:voronoipolygons",
            {
                "INPUT": "{}|layername={}".format(input_db, layer_rain_gauges_2),
                "BUFFER": n_buffer,
                "OUTPUT": zones_shp,
            },
        )
    print("get zones raster...")
    zones_raster = "{}/rain_zones.tif".format(output_folder_interm)
    # >> create constant raster layer
    get_blank_raster(
        file_input=target_file, file_output=zones_raster, blank_value=dict_blank[s_type]
    )
    # rasterize target raster
    if s_type == "voronoi":
        # rasterize voronoi
        processing.run(
            "gdal:rasterize_over",
            {
                "INPUT": zones_shp,
                "INPUT_RASTER": zones_raster,
                "FIELD": "Id",
                "ADD": False,
                "EXTRA": "",
            },
        )
    print("translate...")
    # translante raster layer
    processing.run(
        "gdal:translate",
        {
            "INPUT": zones_raster,
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:{}".format(target_crs)),
            "NODATA": 0,
            "COPY_SUBDATASETS": False,
            "OPTIONS": "",
            "EXTRA": "",
            "DATA_TYPE": 1,
            "OUTPUT": "{}/rain_zones.asc".format(output_folder),
        },
    )
    print("end")
    return None


def get_basins(
    output_folder,
    input_db,
    ldd_file,
    target_crs,
    layer_stream_gauges="stream",
):
    """Get all ``basins`` datasets for running PLANS.

    :param output_folder: path to output folder
    :type output_folder: str
    :param input_db: path to geopackage database with streamflow gauges layer
    :type input_db: str
    :param ldd_file: file path to LDD raster
    :type ldd_file: str
    :param target_crs: EPSG code (number only) for the target CRS
    :type target_crs: str
    :param layer_stream_gauges: layer name of streamflow gauges layer
        Required fields in layer:

            - ``Id``: int, required. Unique number id.
            - ``Name``: str, required. Simple name.
            - ``Alias``: str, required. Short nickname.
            - ``X``: float, optional. Recomputed from layer. Longitude in WGS 84 Datum (EPSG4326).
            - ``Y``: float, optional. Recomputed from layer. Latitude in WGS 84 Datum (EPSG4326).
            - ``Code``: str, op
            - ``Source``: str, required
            - ``Description``: str, required
            - ``Color``: str, optional
            - ``Stage_File``: str, required. Path to data time series ``csv`` source file.
            - ``Flow_File``: str, required. Path to data time series ``csv`` source file.

    :type layer_stream_gauges: str
    :return: None
    :rtype: None

    Script example (for QGIS Python Console):

    .. code-block:: python

        # plans source code must be pasted to QGIS plugins directory
        from plans import qutils

        # call function
        qutils.get_basins(
            output_folder="path/to/basins",
            ldd_file="path/to/ldd.asc",
            target_crs="31982",
            input_db="path/to/my_db.gpkg",
            layer_stream_gauges="stream"
        )

    """
    from plans import datasets

    print("folder setup...")
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

    # ------------------- BASIN MAPS ------------------- #

    # PCRASTER folder
    output_folder_pcraster = "{}/pcraster".format(output_folder_interm)
    if os.path.isdir(output_folder_pcraster):
        pass
    else:
        os.mkdir(output_folder_pcraster)
    print("reproject...")
    # reproject streams
    layer_stream_gauges_new = reproject_layer(
        db_input=input_db, crs_target=target_crs, layer_name=layer_stream_gauges
    )
    print("blank raster...")
    gauge_raster = "{}/gauges.tif".format(output_folder_gdal)
    # get blanks
    get_blank_raster(
        file_input=ldd_file, file_output=gauge_raster, blank_value=0, dtype="float"
    )
    print("rasterize...")
    # rasterize
    processing.run(
        "gdal:rasterize_over",
        {
            "INPUT": "{}|layername={}".format(input_db, layer_stream_gauges_new),
            "INPUT_RASTER": gauge_raster,
            "FIELD": "Id",
            "ADD": False,
            "EXTRA": "",
        },
    )
    # todo validate gauge positions

    # convert to pc raster
    print("get outlets...")
    outlets_raster_pc = "{}/outlets.map".format(output_folder_pcraster)
    processing.run(
        "pcraster:converttopcrasterformat",
        {"INPUT": gauge_raster, "INPUT2": 1, "OUTPUT": outlets_raster_pc},
    )
    # convert ldd
    print("get ldd...")
    ldd_file_pc = "{}/ldd.map".format(output_folder_pcraster)
    processing.run(
        "pcraster:converttopcrasterformat",
        {"INPUT": ldd_file, "INPUT2": 5, "OUTPUT": ldd_file_pc},
    )
    # get basins
    basins_file_pc = "{}/basins.map".format(output_folder_pcraster)
    print("get basins...")
    processing.run(
        "pcraster:subcatchment",
        {"INPUT1": ldd_file_pc, "INPUT2": outlets_raster_pc, "OUTPUT": basins_file_pc},
    )
    # translante raster layers
    outlets_raster = "{}/outlets.asc".format(output_folder)
    processing.run(
        "gdal:translate",
        {
            "INPUT": outlets_raster_pc,
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:{}".format(target_crs)),
            "NODATA": 0,
            "COPY_SUBDATASETS": False,
            "OPTIONS": "",
            "EXTRA": "",
            "DATA_TYPE": 1,
            "OUTPUT": outlets_raster,
        },
    )
    basins_raster = "{}/basins.asc".format(output_folder)
    processing.run(
        "gdal:translate",
        {
            "INPUT": basins_file_pc,
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:{}".format(target_crs)),
            "NODATA": 0,
            "COPY_SUBDATASETS": False,
            "OPTIONS": "",
            "EXTRA": "",
            "DATA_TYPE": 1,
            "OUTPUT": basins_raster,
        },
    )

    # ------------------- BASINS INFO ------------------- #
    # load table
    basins_gdf = gpd.read_file(input_db, layer=layer_stream_gauges)

    print("handle file series")

    # compute X and Y
    basins_gdf["X"] = basins_gdf.geometry.x
    basins_gdf["Y"] = basins_gdf.geometry.y

    if "Color" not in basins_gdf.columns:
        basins_gdf["Color"] = datasets.get_colors(size=len(basins_gdf))

    # fill attributes
    lst_aux = ["Units", "VarField", "DtField"]
    dct_vals = {
        "Stage": ["cm", "H", "DateTime"],
        "Flow": ["m3/s", "Q", "DateTime"],
    }
    # Iterate over the items in dct_vals
    for key, values in dct_vals.items():
        # Create the column name by combining the key and the first item in values
        for i in range(len(lst_aux)):
            column_name = f"{key}_{lst_aux[i]}"
            # Set the values in the GeoDataFrame
            basins_gdf[column_name] = values[i]

    lst_aux = ["Stage", "Flow"]
    for l in lst_aux:
        # rename files and copy
        print("{} files...".format(l))
        lst_files_src = basins_gdf["{}_File".format(l)].values
        lst_files_dst = [
            "{}/{}_{}.csv".format(output_folder, l.lower(), a)
            for a in basins_gdf["Alias"].values
        ]
        lst_new_files = []
        for i in range(len(basins_gdf)):
            if os.path.isfile(lst_files_src[i]):
                shutil.copy(src=lst_files_src[i], dst=lst_files_dst[i])
                new_file_name = "{}_{}.csv".format(
                    l.lower(), basins_gdf["Alias"].values[i]
                )
            else:
                new_file_name = ""
            lst_new_files.append(new_file_name)
        # reset File
        basins_gdf["{}_File".format(l)] = lst_new_files

    # Drop the geometry column
    basins_gdf = basins_gdf.drop(columns=["geometry"])

    print("compute basin topology")

    # This appends a "Downstream_Id" field
    dict_aux = get_downstream_ids(
        file_ldd=ldd_file, file_basins=basins_raster, file_outlets=outlets_raster
    )
    aux_gdf = gpd.GeoDataFrame.from_dict(dict_aux, geometry=None)
    basins_gdf = basins_gdf.merge(aux_gdf, on="Id")

    # GET UPSTREAM AREAS
    dict_aux = get_basins_areas(
        file_basins=basins_raster, ids_basins=list(basins_gdf["Id"])
    )
    aux_gdf = gpd.GeoDataFrame.from_dict(dict_aux, geometry=None)
    basins_gdf = basins_gdf.merge(aux_gdf, on="Id")

    # HOW to organize columns? Base attribute first, Stage and Flow last
    column_order = [
        # Base attributes
        "Id",
        "Name",
        "Alias",
        # Geo Attributes
        "X",
        "Y",
        "Downstream_Id",
        "UpstreamArea",
        # Extra Attributes
        "Code",
        "Source",
        "Description",
        "Color",
        # Stage attributes
        "Stage_Units",
        "Stage_VarField",
        "Stage_DtField",
        "Stage_File",
        # Flow attributes
        "Flow_Units",
        "Flow_VarField",
        "Flow_DtField",
        "Flow_File",
    ]

    # Reorder the columns
    basins_gdf = basins_gdf[column_order]

    # export csv file
    basins_gdf.to_csv("{}/basins_info.csv".format(output_folder), sep=";", index=False)

    print("end")
    return None


def testing():
    print("testing")