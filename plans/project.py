"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all project-related objects and routines of PLANS.

Copyright (C) 2022 Iporã Brito Possantti

************ GNU GENERAL PUBLIC LICENSE ************

https://www.gnu.org/licenses/gpl-3.0.en.html

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import os
import glob


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

        # set up folders
        if os.path.isdir(self.path_main):
            # handle datasets folder
            if os.path.isdir(self.path_ds):
                pass
            else:
                os.mkdir(self.path_ds)
        else:
            os.mkdir(self.path_main)
            os.mkdir(self.path_ds)

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
        print("Downloading...")
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
        print("Unzipping files...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.path_ds)
        # 3) delete zip file
        if remove:
            os.remove(zip_file)
        return None