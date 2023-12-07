"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all project-related objects and routines of PLANS.

Copyright (C) 2022 Iporã Brito Possantti
"""
import os, shutil, glob
import pandas as pd
from plans import datasets

def get_file_size_mb(file_path):
    """Util for getting the file size in MB

    :param file_path: path to file
    :type file_path: str
    :return: file size in MB
    :rtype: float
    """
    # Get the file size in bytes
    file_size_bytes = os.path.getsize(file_path)
    # Convert bytes to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)
    return file_size_mb

def make_dir(str_path):
    """Util function for making a diretory

    :param str_path: path to dir
    :type str_path: str
    :return: None
    :rtype: None
    """
    if os.path.isdir(str_path):
        pass
    else:
        os.mkdir(str_path)
    return None

def fill_dir_strucuture(dict_struct, local_root):
    """Recursive function for filling a directory structure

    :param dict_struct: dicitonary of directory structure
    :type dict_struct: dict
    :param local_root: path to local folder
    :type local_root: str
    :return: None
    :rtype: None
    """
    for k in dict_struct:
        if isinstance(dict_struct[k], dict):
            # make a dir
            _d = local_root + "/" + k
            make_dir(str_path=_d)
            # now move down:
            fill_dir_strucuture(
                dict_struct=dict_struct[k],
                local_root=_d
            )
        else:  # is file
            pass
    return None

class Project:

    def __init__(self, name, root):
        """Initiate a project

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
                    "accflux": datasets.Raster,
                    "ldd": datasets.LDD
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
                "stream":{
                    "stream_*": datasets.Streamflow
                },
                "rain": {
                    "obs": {
                        "rain_*": None
                    },
                    "bau": {
                        "rain_*": None
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
        self.fill()
        #
        #
        self.topo_status = None
        self.update_status_topo()

        self.topo = None
        self.soil = None
        self.lulc = None
        self.et = None
        self.ndvi = None

    def fill(self):
        fill_dir_strucuture(
            dict_struct=self.structure,
            local_root=self.path_main
        )
        return None

    def update_status_topo(self):
        str_path = self.path_main + "/datasets/topo"
        list_names = list(self.structure["datasets"]["topo"].keys())
        dict_lists = {
            "Name": list_names,
            "File": ["{}.asc".format(f) for f in list_names],
        }
        dict_lists["Path"] = ["{}/{}".format(str_path, f) for f in dict_lists["File"]]
        dict_lists["Status"] = ["missing" for i in dict_lists["Name"]]
        dict_lists["Size"] = ["" for i in dict_lists["Name"]]
        dict_lists["Shape"] = ["" for i in dict_lists["Name"]]
        dict_lists["Cellsize"] = ["" for i in dict_lists["Name"]]
        dict_lists["Nodata"] = ["" for i in dict_lists["Name"]]
        dict_lists["Origin"] = ["" for i in dict_lists["Name"]]

        # Set up
        for i in range(len(dict_lists["Path"])):
            f = dict_lists["Path"][i]
            name = dict_lists["Name"][i]
            if os.path.isfile(f):
                dict_lists["Status"][i] = "available"
                dict_lists["Size"][i] = "{:.1f} MB".format(get_file_size_mb(f))
                rst_aux = self.structure["datasets"]["topo"][name](name=name)
                rst_aux.load_asc_metadata(file=f)
                dict_meta = rst_aux.asc_metadata
                dict_lists["Shape"][i] = "{} x {}".format(
                    dict_meta["nrows"],
                    dict_meta["ncols"]
                )
                dict_lists["Cellsize"][i] = "{} m".format(dict_meta["cellsize"])
                dict_lists["Nodata"][i] = "{}".format(dict_meta["NODATA_value"])
                dict_lists["Origin"][i] = "({}, {})".format(
                    dict_meta["xllcorner"],
                    dict_meta["yllcorner"]
                )

        str_liner = "="
        for k in dict_lists:
            list_maxs = [len(dict_lists[k][e]) + 2 for e in range(len(dict_lists[k]))]
            n_max = max(list_maxs)
            dict_lists[k].insert(0, str_liner*n_max)


        # Set attribute
        self.topo_status = pd.DataFrame(dict_lists)
        return None

    def download_datasets(self, zip_url):
        """Download datasets from a URL. The download is expected to be a ZIP file. Note: requests library is required

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
        """Download the default datasets for PLANS

        :return: None
        :rtype: None
        """
        zip_url = "https://zenodo.org/record/8217681/files/default_samples.zip?download=1"
        self.download_datasets(zip_url=zip_url)
        return None

    def extract_datasets(self, zip_file, remove=False):
        """Extract from ZIP file to datasets folder

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
    #import matplotlib.pyplot as plt
    #plt.style.use("seaborn-v0_8")

    p = Project(
        name="nimes",
        root="C:/plans"
    )

    print(p.topo_status.to_string())