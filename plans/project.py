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


def make_dir(str_path):
    if os.path.isdir(str_path):
        pass
    else:
        os.mkdir(str_path)


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
        # make main dir
        make_dir(str_path=self.path_main)

        #
        # struture of the project
        self.structure = {
            "datasets": {
                "topo":{
                    "dem": datasets.Elevation,
                    "slope": datasets.Slope,
                    "twi": datasets.TWI,
                    "hand": datasets.HAND,
                    "flowacc": datasets.Raster,
                    "ldd": datasets.Raster
                },
                "lulc":{
                    "obs":{
                        "lulc_*": datasets.LULC,
                    },
                    "bau":{
                        "lulc_*": datasets.LULC,
                    },
                    "base": {
                        "lulc_*": datasets.LULC,
                    },
                    "nbs": {
                        "lulc_*": datasets.LULC,
                    },
                },
                "soil":{
                    "soils": datasets.Soils,
                    "lito": datasets.Lithology
                },
                "et":{
                    "et_*": datasets.ET24h
                },
                "ndvi":{
                    "ndvi_*": datasets.NDVI
                },
                "flu":{
                    "flu_*": datasets.Streamflow
                },
                "plu": {
                    "obs": {
                        "plu_*": None
                    },
                    "bau": {
                        "plu_*": None
                    }
                },
                "temp": {
                    "obs":{
                        "temp_*": None
                    }
                },
                "model":{
                    "hist": None
                }
            },
            "outputs": {
                "simulation": {
                    "sim_*": None
                },
                "assessment": {
                    "asm_*": None
                },
                "uncertainty": {
                    "unc_*": None
                },
                "sensitivity": {
                    "sal_*": None
                }
            }
        }

        # fill folders
        self.fill(
            dict_struct=self.structure,
            local_root=self.path_main
        )
        #
        #
        self.geo = None
        self.soil = None
        self.lulc = None
        self.et = None
        self.ndvi = None

    def fill(self, dict_struct, local_root):
        for k in dict_struct:
            if isinstance(dict_struct[k], dict):
                # make a dir
                _d = local_root + "/" + k
                make_dir(str_path=_d)
                # now move down:
                self.fill(
                    dict_struct = dict_struct[k],
                    local_root=_d
                )
            else: # is file
                pass


    def teste(self):
        print("********** hey *************")
    '''
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
    '''


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

    print("hi")