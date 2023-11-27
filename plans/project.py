"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all project-related objects and routines of PLANS.

Copyright (C) 2022 Iporã Brito Possantti
"""
import os
import glob
import pandas as pd

class Workplace:
    def __init__(self, root):

        # root setup
        self.root = root
        if os.path.isdir(self.root):
            pass
        else:
            os.mkdir(self.root)

        self.catalogfile = "{}/catalog.csv".format(self.root)
        if os.path.isfile(self.catalogfile):
            pass
        else:
            df = pd.DataFrame(
                {
                    "Project_Name": [""],
                    "Project_Alias": [""]

                }
            )

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
        self.path_ds_obs = "{}/observed".format(self.path_ds)
        self.path_ds_scn = "{}/scenarios".format(self.path_ds)
        self.path_out = "{}/outputs".format(self.path_main)
        self.path_out_sim = "{}/simulation".format(self.path_out)
        self.path_out_asm = "{}/assessment".format(self.path_out)
        self.path_out_unc = "{}/uncertainty".format(self.path_out)
        self.path_out_sal = "{}/sensitivity".format(self.path_out)
        #
        self.ds_structure = {
            "geo": ['dem.asc', 'slope.asc', 'twi.asc', 'hand.asc', 'soils.asc', 'lito.asc'],
            "lulc" : ["lulc_*.asc", "lulc_table.csv"],
            "ndvi": ["ndvi_*.asc"],
            "et": ["et_*.asc"],
            "series": ["stage", "rain", "temp"],
            "model": ['---']
        }

        self.fill()
        self.fill_ds_obs()

    def fill(self):
        list_folders = [
            self.path_main,
            self.path_ds,
            self.path_ds_obs,
            self.path_ds_scn,
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
        for k in self.ds_structure:
            d = self.path_ds_obs + "/" + k
            if os.path.isdir(d):
                pass
            else:
                os.mkdir(d)

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