"""
Project-related objects and routines

Description:
    The ``project`` module provides project-related objects and routines of ``plans``.

License:
    This software is released under the GNU General Public License v3.0 (GPL-3.0).
    For details, see: https://www.gnu.org/licenses/gpl-3.0.html

Author:
    Iporã Possantti

Contact:
    possantti@gmail.com


Overview
--------

todo
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis.
Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

>>> from plans import ds

Class aptent taciti sociosqu ad litora torquent per
conubia nostra, per inceptos himenaeos. Nulla facilisi. Mauris eget nisl
eu eros euismod sodales. Cras pulvinar tincidunt enim nec semper.

Example
-------

todo
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis. Mauris gravida ex quam,
in porttitor lacus lobortis vitae. In a lacinia nisl.

.. code-block:: python

    import numpy as np
    from plans import analyst

    # get data to a vector
    data_vector = np.random.rand(1000)

    # instantiate the Univar object
    uni = analyst.Univar(data=data_vector, name="my_data")

    # view data
    uni.view()

Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl.
"""
import os, shutil, glob
import pandas as pd
from plans import ds
from plans.root import FileSys

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


class Project(FileSys):

    def __init__(self, root, name):
        super().__init__(folder_base=root, name=name, alias=None)
        # load standard data
        self.load_data()

    def load_data(self):
        """Load data from file. Expected to overwrite superior methods.

        :param file_data: file path to data.
        :type file_data: str
        :return: None
        :rtype: None
        """
        # -------------- overwrite relative path input -------------- #
        file_data = os.path.abspath("./plans/iofiles.csv")

        # -------------- implement loading logic -------------- #

        # -------------- call loading function -------------- #
        self.data = pd.read_csv(
            file_data,
            sep=self.file_data_sep
        )

        # -------------- post-loading logic -------------- #
        self.data = self.data[["Folder","File","Format","File_Source",'Folder_Source']].copy()

        return None

class Project_(FileSys):

    def __init__(self, name, folder_base, alias=None):
        """Initiate a project

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from ``name``
        :type alias: str

        :param folder_base: path to base folder
        :type name: str
        """
        # ---------- call super -----------#
        super().__init__(name=name, folder_base=folder_base, alias=alias)

        # overwrite struture of the project
        self.structure = self.get_structure()

        self.topo_status = None
        #self.update_status_topo()

        self.topo = None
        self.soil = None
        self.lulc = None
        self.et = None
        self.ndvi = None


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


    p = Project(name="arles", folder_base="C:/data")

    print(p)
    p.setup()
