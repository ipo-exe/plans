"""
Project-related classes and routines

Overview
--------

# todo [major docstring improvement] -- overview
Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

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

import os
import pandas as pd
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


class Project(FileSys):

    def __init__(self, root, name):
        super().__init__(folder_base=root, name=name, alias=None)

        self.structure = {
            "inputs": {
                "topo": {
                    "_aux": {0: 0},
                    "htwi": {0: 0},
                },
                "lulc": {
                    "obs": {
                        "_aux": {0: 0},
                    },
                    "bsl": {
                        "_aux": {0: 0},
                    },
                    "bau": {
                        "_aux": {0: 0},
                    },
                },
                "clim": {
                    "obs": {
                        "_aux": {0: 0},
                    },
                    "bsl": {
                        "_aux": {0: 0},
                    },
                    "bau": {
                        "_aux": {0: 0},
                    },
                },
                "basins": {
                    "_aux": {0: 0},
                },
                "soils": {
                    "_aux": {0: 0},
                },
            },
            "outputs": {0: 0},
        }
        self.make_dir(str_path=self.folder_main)
        self.fill(
            dict_struct=self.structure, folder=self.folder_main, handle_files=False
        )

        # load standard data
        # self.load_data()

    def load_data(self):
        """Load data from file. Expected to overwrite superior methods.

        :param file_data: file path to data.
        :type file_data: str
        :return: None
        :rtype: None
        """
        # -------------- overwrite relative path inputs -------------- #
        file_data = os.path.abspath("./plans/iofiles.csv")

        # -------------- implement loading logic -------------- #

        # -------------- call loading function -------------- #
        self.data = pd.read_csv(file_data, sep=self.file_csv_sep)

        # -------------- post-loading logic -------------- #
        self.data = self.data[
            ["Folder", "File", "Format", "File_Source", "Folder_Source"]
        ].copy()

        return None


# todo [refactor] -- here we got some very interesting stuff
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
        # self.update_status_topo()

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
        with open(_s_zipfile, "wb") as file:
            file.write(response.content)
        # 2) extract to datasets folder
        self.extract_datasets(zip_file=_s_zipfile, remove=True)
        return None

    def download_default_datasets(self):
        """Download the default datasets for PLANS

        :return: None
        :rtype: None
        """
        zip_url = (
            "https://zenodo.org/record/8217681/files/default_samples.zip?download=1"
        )
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
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(self.path_ds)
        # 3) delete zip file
        if remove:
            os.remove(zip_file)
        return None


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # plt.style.use("seaborn-v0_8")

    p = Project(name="arles", folder_base="C:/data")

    print(p)
    p.setup()
