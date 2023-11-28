"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all project-related objects and routines of PLANS.

Copyright (C) 2022 Iporã Brito Possantti
"""
import os
import glob
import pandas as pd
import datasets

class Project:

    def __init__(self, name, root):
        """
        Initiate a project
        :param name: name of project
        :type name: str
        :param root: path to root folder
        :type root: str
        """
        self.name = name
        self.root = root
        self.path_main = "{}/{}".format(root, name)
        self.path_ds = "{}/datasets".format(self.path_main)
        self.path_ds_topo = "{}/topo".format(self.path_ds)
        self.path_ds_soil = "{}/soil".format(self.path_ds)
        self.path_ds_lulc = "{}/lulc".format(self.path_ds)
        self.path_ds_lulc_obs = "{}/lulc/obs".format(self.path_ds)
        self.path_ds_topo = "{}/topo".format(self.path_ds)
        self.path_out = "{}/outputs".format(self.path_main)
        self.path_out_sim = "{}/simulation".format(self.path_out)
        self.path_out_asm = "{}/assessment".format(self.path_out)
        self.path_out_unc = "{}/uncertainty".format(self.path_out)
        self.path_out_sal = "{}/sensitivity".format(self.path_out)
        self.dict_paths_obs = dict()
        #

        self.dict_ds_structure = {
            "topo": {
                "dem": datasets.Elevation,
                "slope": datasets.Slope,
                "twi": datasets.TWI,
                "hand": datasets.HAND,
                # todo create flowacc and ldd objects
                "flowacc": datasets.Raster,
                "ldd": datasets.QualiRaster
            },
            "soil": {
                "soils": datasets.Soils,
                "litology": datasets.Lithology
            },
            "land": {
              "obs": {

              }
            },
            "lulc" : datasets.LULCSeries,
            "ndvi": datasets.NDVISeries,
            "et": datasets.ETSeries,
            "series": ["stage", "rain", "temp"],
            "model": ['---']
        }
        # fill folders
        self.fill()
        self.fill_ds_obs()
        #
        #
        self.geo = None
        self.soil = None
        self.lulc = None
        self.et = None
        self.ndvi = None

    def fill(self):
        list_folders = [
            self.path_main,
            self.path_ds,
            self.path_out,
            self.path_out_unc,
            self.path_out_asm,
            self.path_out_sal,
            self.path_out_sim
        ]
        for d in list_folders:
            if os.path.isdir(d):
                pass
            else:
                os.mkdir(d)

    def fill_ds_obs(self):
        for k in self.dict_ds_structure:
            d = self.path_ds + "/" + k
            self.dict_paths_obs[k] = d[:]
            if os.path.isdir(d):
                pass
            else:
                os.mkdir(d)

    def teste(self):
        print("********** hey *************")

    def get_geo_collection(self):
        str_label = "geo"
        self.geo = datasets.RasterCollection(name="Geomorphology")
        for k in self.dict_ds_structure[str_label]:
            new_raster = self.dict_ds_structure[str_label][k](name=k)
            new_raster.load(
                asc_file=self.dict_paths_obs[str_label] + "/{}.asc".format(k),
                prj_file=self.dict_paths_obs[str_label] + "/{}.prj".format(k)
            )
            self.geo.append(new_object=new_raster)
            del new_raster

    def get_soil_collection(self):
        str_label = "soil"
        self.soil = datasets.QualiRasterCollection(name="Soil")
        for k in self.dict_ds_structure[str_label]:
            new_raster = self.dict_ds_structure[str_label][k](name=k)
            new_raster.load(
                asc_file=self.dict_paths_obs[str_label] + "/{}.asc".format(k),
                prj_file=self.dict_paths_obs[str_label] + "/{}.prj".format(k),
                table_file=self.dict_paths_obs[str_label] + "/{}.csv".format(k),
            )
            self.soil.append(new_object=new_raster)
            del new_raster

    def get_lulc_collection(self, name_pattern="map_lulc_*"):
        str_label = "lulc"
        self.lulc = datasets.LULCSeries(name="LULC")
        self.lulc.load_folder(
            folder=self.dict_paths_obs[str_label],
            table_file=self.dict_paths_obs[str_label] + "/lulc.csv",
            name_pattern=name_pattern
        )


    def download_datasets(self, zip_url):
        """
        Download datasets from a URL. The download is expected to be a ZIP file. Note: requests library is required
        :param zip_url: url to dataset ZIP file
        :type zip_url: str
        :return: None
        :rtype: None
        """
        import requests
        # 1) download to main folder
        print("Downloading datasets from URL...")
        response = requests.get(zip_url)
        # just in case of any problem
        response.raise_for_status()
        _s_zipfile = "{}/samples.zip".format(self.path_main)
        with open(_s_zipfile, 'wb') as file:
            file.write(response.content)
        # 2) extract to datasets folder
        self.extract_datasets(zip_file=_s_zipfile, remove=True)
        return None

    def download_default_datasets(self):
        """
        Download the default datasets for PLANS
        :return: None
        :rtype: None
        """
        zip_url = "https://zenodo.org/record/8217681/files/default_samples.zip?download=1"
        self.download_datasets(zip_url=zip_url)
        return None

    def extract_datasets(self, zip_file, remove=False):
        """
        Extract from ZIP file to datasets folder
        :param zip_file: path to zip file
        :type zip_file: str
        :param remove: option for deleting the zip file after extraction
        :type remove: bool
        :return: None
        :rtype: None
        """
        import zipfile
        print("Unzipping dataset files...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.path_ds)
        # 3) delete zip file
        if remove:
            os.remove(zip_file)
        return None


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")

    p = Project(name="teste", root="C:/plans")
    p.get_lulc_collection()

    print(p.lulc.catalog.to_string())
    p.lulc.collection["map_lulc_1985-01-01"].view(show=True)
    p.lulc.get_views(
        show=False,
        folder=p.dict_paths_obs["lulc"]
    )
    '''
    p.soil.collection["soils"].view(show=True)
    p.soil.get_views(
        show=False,
        folder=p.dict_paths_obs["soil"]
    )
    '''
