import glob
import os, copy, shutil
import pandas as pd


class MbaE:
    """
    **Mba'e** in Guarani means **Thing**.

    .. important::

        **Mba'e is the origin**. The the very-basic almost-zero level object.
        Deeper than here is only the Python builtin ``object`` class.


    """

    def __init__(self, name="MyMbaE", alias=None):
        """Initialize the ``MbaE`` object.

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from ``name``
        :type alias: str

        """
        # ------------ pseudo-static ----------- #

        #
        self.object_name = self.__class__.__name__
        self.object_alias = "mbae"

        # name
        self.name = name

        # alias
        self.alias = alias

        # handle None alias
        if self.alias is None:
            self._create_alias()

        # fields
        self._set_fields()

        # ------------ set mutables ----------- #
        self.file_table_info = None
        self.folder_src = "./"  # start in the local place

        # ... continues in downstream objects ... #

    def __str__(self):
        """The ``MbaE`` string"""
        str_type = str(type(self))
        str_df_metadata = self.get_metadata_df().to_string(index=False)
        str_out = "[{} ({})]\n{} ({}):\t{}\n{}".format(
            self.name,
            self.alias,
            self.object_name,
            self.object_alias,
            str_type,
            str_df_metadata,
        )
        return str_out

    def _create_alias(self):
        """If ``alias`` is ``None``, it takes the first and last characters from ``name``"""
        if len(self.name) >= 2:
            self.alias = self.name[0] + self.name[len(self.name) - 1]
        else:
            self.alias = self.name[:]

    def _set_fields(self):
        """Set fields names"""

        # Attribute fields
        self.name_field = "Name"
        self.alias_field = "Alias"

        # Metadata fields
        self.mdata_attr_field = "Attribute"
        self.mdata_val_field = "Value"
        # ... continues in downstream objects ... #

    def get_metadata(self):
        """Get a dictionary with object metadata.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        dict_meta = {
            self.name_field: self.name,
            self.alias_field: self.alias,
        }
        return dict_meta

    def get_metadata_df(self):
        """Get a :class:`pandas.DataFrame` created from the metadata dictionary

        :return: :class:`pandas.DataFrame` with ``Attribute`` and ``Value``
        :rtype: :class:`pandas.DataFrame`
        """
        dict_metadata = self.get_metadata()
        df_metadata = pd.DataFrame(
            {
                self.mdata_attr_field: [k for k in dict_metadata],
                self.mdata_val_field: [dict_metadata[k] for k in dict_metadata],
            }
        )
        return df_metadata

    def set(self, dict_setter):
        """Set selected attributes based on an incoming dictionary

        :param dict_setter: incoming dictionary with attribute values
        :type dict_setter: dict
        """
        # ---------- set basic attributes --------- #
        self.name = dict_setter[self.name_field]
        self.alias = dict_setter[self.alias_field]

        # ... continues in downstream objects ... #

    def load(self, file_table_info):
        """Load attributes from ``csv`` table.

        :param file_table_info: file path to ``csv`` table with metadata information.
            Expected format:

            .. code-block:: text

                Attribute;Value
                Name;ResTia
                Alias;Ra

        :type file_table_info: str

        :return:
        :rtype: str
        """
        # ---------- update file attributes ---------- #
        self.file_table_info = file_table_info[:]
        self.folder_src = os.path.dirname(file_table_info)

        # get expected fields
        list_columns = [self.mdata_attr_field, self.mdata_val_field]

        # read info table from ``csv`` file. metadata keys are the expected fields
        df_info_table = pd.read_csv(file_table_info, sep=";", usecols=list_columns)

        # setter loop
        dict_setter = {}
        for i in range(len(df_info_table)):
            # build setter from row
            dict_setter[df_info_table[self.mdata_attr_field].values[i]] = df_info_table[
                self.mdata_val_field
            ].values[i]

        # pass setter to set() method
        self.set(dict_setter=dict_setter)

        return None


class Collection(MbaE):
    """
    A collection of primitive ``MbaE`` objects with associated metadata.

    Attributes:

    - ``catalog`` (:class:`pandas.DataFrame`): A catalog containing metadata of the objects in the test_collection.
    - ``collection`` (dict): A dictionary containing the objects in the ``Collection``.
    - name (str): The name of the ``Collection``.
    - alias (str): The name of the ``Collection``.
    - baseobject: The class of the base object used to initialize the ``Collection``.

    Methods:

    - __init__(self, base_object, name="myCatalog"): Initializes a new ``Collection`` with a base object.
    - update(self, details=False): Updates the ``Collection`` catalog.
    - append(self, new_object): Appends a new object to the ``Collection``.
    - remove(self, name): Removes an object from the ``Collection``.


    **Examples:**
    Here's how to use the Collection class:

    1. Initializing a Collection

    >>> base_obj = YourBaseObject()
    >>> test_collection = Collection(base_object=base_obj, name="myCatalog")

    2. Appending a New Object

    >>> new_obj = YourNewObject()
    >>> test_collection.append(new_object=new_obj)

    3. Removing an Object

    >>> test_collection.remove(name="ObjectToRemove")

    4. Updating the Catalog

    >>> test_collection.update(details=True)

    """

    def __init__(self, base_object, name="MyCollection", alias="Col0"):
        """Initialize the ``Collection`` object.

        :param base_object: ``MbaE``-based object for collection
        :type base_object: :class:`MbaE`

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from name
        :type alias: str

        """
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)
        # ------------ set pseudo-static ----------- #
        self.object_alias = "COL"
        # Set the name and baseobject attributes
        self.baseobject = base_object
        self.baseobject_name = base_object.__name__

        # Initialize the catalog with an empty DataFrame
        dict_metadata = self.baseobject().get_metadata()

        self.catalog = pd.DataFrame(columns=dict_metadata.keys())

        # Initialize the ``Collection`` as an empty dictionary
        self.collection = dict()

        # ------------ set mutables ----------- #
        self.size = 0

        self._set_fields()
        # ... continues in downstream objects ... #

    def __str__(self):
        """
        The ``Collection`` string.
        Expected to overwrite superior methods.
        """
        str_type = str(type(self))
        str_df_metadata = self.get_metadata_df().to_string(index=False)
        str_df_data = self.catalog.to_string(index=False)
        str_out = "{}:\t{}\nMetadata:\n{}\nCatalog:\n{}\n".format(
            self.name, str_type, str_df_metadata, str_df_data
        )
        return str_out

    def _set_fields(self):
        """
        Set fields names.
        Expected to increment superior methods.
        """
        # ------------ call super ----------- #
        super()._set_fields()

        # Attribute fields
        self.size_field = "Size"
        self.baseobject_field = "Base_Object"  # self.baseobject().__name__

        # ... continues in downstream objects ... #

    def get_metadata(self):
        """Get a dictionary with object metadata.
        Expected to increment superior methods.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        # ------------ call super ----------- #
        dict_meta = super().get_metadata()

        # customize local metadata:
        dict_meta_local = {
            self.size_field: self.size,
            self.baseobject_field: self.baseobject_name,
        }

        # update
        dict_meta.update(dict_meta_local)
        return dict_meta

    def update(self, details=False):
        """Update the ``Collection`` catalog.

        :param details: Option to update catalog details, defaults to False.
        :type details: bool
        :return: None
        :rtype: None
        """

        # Update details if specified
        if details:
            # Create a new empty catalog
            df_new_catalog = pd.DataFrame(columns=self.catalog.columns)

            # retrieve details from collection
            for name in self.collection:
                # retrieve updated metadata from base object
                dct_meta = self.collection[name].get_metadata()

                # set up a single-row helper dataframe
                lst_keys = dct_meta.keys()
                _dct = dict()
                for k in lst_keys:
                    _dct[k] = [dct_meta[k]]

                # Set new information
                df_aux = pd.DataFrame(_dct)

                # Append to the new catalog
                df_new_catalog = pd.concat([df_new_catalog, df_aux], ignore_index=True)

            # consider if the name itself has changed in the
            old_key_names = list(self.collection.keys())[:]
            new_key_names = list(df_new_catalog[self.catalog.columns[0]].values)

            # loop for checking consistency in collection keys
            for i in range(len(old_key_names)):
                old_key = old_key_names[i]
                new_key = new_key_names[i]
                # name change condition
                if old_key != new_key:
                    # rename key in the collection dictionary
                    self.collection[new_key] = self.collection.pop(old_key)

            # Update the catalog with the new details
            self.catalog = df_new_catalog.copy()
            # clear
            del df_new_catalog

        # Basic updates
        # --- the first row is expected to be the Unique name
        str_unique_name = self.catalog.columns[0]
        self.catalog = self.catalog.drop_duplicates(subset=str_unique_name, keep="last")
        self.catalog = self.catalog.sort_values(by=str_unique_name).reset_index(
            drop=True
        )
        self.size = len(self.catalog)
        return None

    # review ok
    def append(self, new_object):
        """Append a new object to the ``Collection``.

        The object is expected to have a ``.get_metadata()`` method that
        returns a dictionary with metadata keys and values

        :param new_object: Object to append.
        :type new_object: object

        :return: None
        :rtype: None
        """
        # Append a copy of the object to the ``Collection``
        copied_object = copy.deepcopy(new_object)
        self.collection[new_object.name] = copied_object

        # Update the catalog with the new object's metadata
        dct_meta = new_object.get_metadata()
        dct_meta_df = dict()
        for k in dct_meta:
            dct_meta_df[k] = [dct_meta[k]]
        df_aux = pd.DataFrame(dct_meta_df)
        self.catalog = pd.concat([self.catalog, df_aux], ignore_index=True)

        self.update()
        return None

    def remove(self, name):
        """Remove an object from the ``Collection`` by the name.

        :param name: Name attribute of the object to remove.
        :type name: str

        :return: None
        :rtype: None
        """
        # Delete the object from the ``Collection``
        del self.collection[name]
        # Delete the object's entry from the catalog
        str_unique_name = self.catalog.columns[
            0
        ]  # assuming the first column is the unique name
        self.catalog = self.catalog.drop(
            self.catalog[self.catalog[str_unique_name] == name].index
        ).reset_index(drop=True)
        self.update()
        return None


class DataSet(MbaE):
    """
    The core ``DataSet`` base/demo object.
    Expected to hold one :class:`pandas.DataFrame`.

    """

    def __init__(self, name="MyDataSet", alias="DS0"):
        """Initialize the ``DataSet`` object.
        Expected to increment superior methods.

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from name
        :type alias: str

        """
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "DS"

        # ------------ set mutables ----------- #
        self.file_data = None
        self.data = None
        self.size = None

        # descriptors
        self.source_data = None
        self.descri_data = None

        # ------------ set defaults ----------- #
        self.color = "blue"
        self._set_view_specs()

        # ... continues in downstream objects ... #

    def __str__(self):
        """
        The ``DataSet`` string.
        Expected to overwrite superior methods.

        """
        str_super = super().__str__()
        if self.data is None:
            str_df_data = "None"
            str_out = "{}\nData:\n{}\n".format(str_super, str_df_data)
        else:
            # print the first 5 rows
            str_df_data_head = self.data.head().to_string(index=False)
            str_df_data_tail = self.data.tail().to_string(index=False)
            str_out = "{}\nData:\n{}\n ... \n{}\n".format(
                str_super, str_df_data_head, str_df_data_tail
            )
        return str_out

    def _set_fields(self):
        """
        Set fields names. Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()

        # Attribute fields
        self.filedata_field = "File_Data"
        self.size_field = "Size"
        self.color_field = "Color"
        self.source_data_field = "Source"
        self.descri_data_field = "Description"

        # ... continues in downstream objects ... #

    def _set_view_specs(self):
        """Set view specifications.
        Expected to overwrite superior methods.

        :return: None
        :rtype: None
        """
        self.view_specs = {
            "folder": self.folder_src,
            "filename": self.name,
            "fig_format": "jpg",
            "dpi": 300,
            "title": self.name,
            "width": 5 * 1.618,
            "height": 5 * 1.618,
            "xvar": "RM",
            "yvar": "TempDB",
            "xlabel": "RM",
            "ylabel": "TempDB",
            "color": self.color,
            "xmin": None,
            "xmax": None,
            "ymin": None,
            "ymax": None,
        }
        return None

    def get_metadata(self):
        """Get a dictionary with object metadata.
        Expected to increment superior methods.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        # ------------ call super ----------- #
        dict_meta = super().get_metadata()

        # customize local metadata:
        dict_meta_local = {
            self.size_field: self.size,
            self.color_field: self.color,
            self.source_data_field: self.source_data,
            self.descri_data_field: self.descri_data,
            self.filedata_field: self.file_data,
        }

        # update
        dict_meta.update(dict_meta_local)
        return dict_meta

    def set(self, dict_setter, load_data=True):
        """Set selected attributes based on an incoming dictionary.
        Expected to increment superior methods.

        :param dict_setter: incoming dictionary with attribute values
        :type dict_setter: dict

        :param load_data: option for loading data from incoming file. Default is True.
        :type load_data: bool

        """
        super().set(dict_setter=dict_setter)

        # ---------- settable attributes --------- #
        self.color = dict_setter[self.color_field]

        # handle filename only
        if os.path.isfile(dict_setter[self.filedata_field]):
            self.file_data = dict_setter[self.filedata_field][:]
        else:
            # assumes file is in the source folder
            self.file_data = os.path.join(
                self.folder_src, dict_setter[self.filedata_field][:]
            )

        # -------------- set data logic here -------------- #
        if load_data:
            self.load_data(file_data=self.file_data)

        # set view specs
        self._set_view_specs()
        # ... continues in downstream objects ... #

    def load_data(self, file_data):
        """Load data from file. Expected to overwrite superior methods.

        :param file_data: file path to data.
        :type file_data: str
        :return: None
        :rtype: None
        """

        # -------------- overwrite relative path input -------------- #
        file_data = os.path.abspath(file_data)

        # -------------- implement loading logic -------------- #
        default_columns = {
            #'DateTime': 'datetime64[1s]',
            "P": float,
            "RM": float,
            "TempDB": float,
        }
        # -------------- call loading function -------------- #
        self.data = pd.read_csv(
            file_data,
            sep=";",
            dtype=default_columns,
            usecols=list(default_columns.keys()),
        )

        # -------------- post-loading logic -------------- #
        self.data.dropna(inplace=True)

        # ------------ update related attributes ------------ #
        self.file_data = file_data[:]
        self.folder_src = os.path.dirname(self.file_data)
        # view specs -- folder setup
        self._set_view_specs()
        # data size (rows_
        self.size = len(self.data)

        return None

    def view(self, show=True):
        """Get a basic visualization.
        Expected to overwrite superior methods.

        :param show: option for showing instead of saving.
        :type show: bool

        :return: None or file path to figure
        :rtype: None or str

        **Notes:**

        - Uses values in the ``view_specs()`` attribute for plotting

        **Examples:**

        Simple visualization:

        >>> ds.view(show=True)

        Customize view specs:

        >>> ds.view_specs["title"] = "My Custom Title"
        >>> ds.view_specs["xlabel"] = "The X variable"
        >>> ds.view(show=True)

        Save the figure:

        >>> ds.view_specs["folder"] = "path/to/folder"
        >>> ds.view_specs["filename"] = "my_visual"
        >>> ds.view_specs["fig_format"] = "png"
        >>> ds.view(show=False)

        """
        # get specs
        specs = self.view_specs.copy()

        # --------------------- figure setup --------------------- #
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        # --------------------- plotting --------------------- #
        plt.scatter(
            self.data[specs["xvar"]],
            self.data[specs["yvar"]],
            marker=".",
            color=specs["color"],
        )

        # --------------------- post-plotting --------------------- #
        # set basic plotting stuff
        plt.title(specs["title"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["xlabel"])

        # handle min max
        if specs["xmin"] is None:
            specs["xmin"] = self.data[specs["xvar"]].min()
        if specs["ymin"] is None:
            specs["ymin"] = self.data[specs["yvar"]].min()
        if specs["xmax"] is None:
            specs["xmax"] = self.data[specs["xvar"]].max()
        if specs["ymax"] is None:
            specs["ymax"] = self.data[specs["yvar"]].max()

        plt.xlim(specs["xmin"], specs["xmax"])
        plt.ylim(specs["ymin"], 1.2 * specs["ymax"])

        # Adjust layout to prevent cutoff
        plt.tight_layout()

        # --------------------- end --------------------- #
        # show or save
        if show:
            plt.show()
            return None
        else:
            file_path = "{}/{}.{}".format(
                specs["folder"], specs["filename"], specs["fig_format"]
            )
            plt.savefig(file_path, dpi=specs["dpi"])
            plt.close(fig)
            return file_path


class FileSys(DataSet):
    """
    The core ``FileSys`` base/demo object. File System object.

    """

    def __init__(self, folder_base, name="MyFS", alias="FS0"):
        """Initialize the ``FileSys`` object.
        Expected to increment superior methods.

        :param folder_base: path to File System folder location
        :type folder_base: str

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from name
        :type alias: str

        """
        # prior attributes
        self.folder_base = folder_base

        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)

        # overwriters
        self.object_alias = "FS"

        # ------------ set mutables ----------- #
        self.folder_main = os.path.join(self.folder_base, self.name)
        self._set_view_specs()

        # ... continues in downstream objects ... #

    def _set_fields(self):
        """
        Set fields names. Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()

        # Attribute fields
        self.folder_base_field = "Folder_Base"

        # ... continues in downstream objects ... #

    def _set_view_specs(self):
        """Set view specifications.
        Expected to overwrite superior methods.

        :return: None
        :rtype: None
        """
        self.view_specs = {
            "folder": self.folder_src,
            "filename": self.name,
            "fig_format": "jpg",
            "dpi": 300,
            "title": self.name,
            "width": 5 * 1.618,
            "height": 5 * 1.618,
            # todo modify here
        }
        return None

    @staticmethod
    def get_extensions():
        list_basics = [
            "pdf",
            "docx",
            "xlsx",
            "bib",
            "tex",
            "svg",
            "png",
            "jpg",
            "txt",
            "csv",
            "qml",
            "tif",
            "gpkg",
        ]
        dict_extensions = {e: [e] for e in list_basics}
        dict_aliases = {
            "table": ["csv"],
            "raster": ["asc", "prj", "qml"],
            "qraster": ["asc", "prj", "csv", "qml"],
            "fig": ["jpg"],
            "vfig": ["svg"],
            "receipt": ["pdf", "jpg"],
        }
        dict_extensions.update(dict_aliases)
        return dict_extensions

    @staticmethod
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

    @staticmethod
    def copy_batch(dst_pattern, src_pattern):
        # todo docstring
        # handle destination variables
        dst_basename = os.path.basename(dst_pattern).split(".")[0].replace("*", "")  # k
        dst_folder = os.path.dirname(dst_pattern)  # folder

        # handle sourced variables
        src_folder = os.path.dirname(dst_folder)
        src_extension = os.path.basename(src_pattern).split(".")[1]
        src_prefix = os.path.basename(src_pattern).split(".")[0].replace("*", "")

        # get the list of sourced files
        list_files = glob.glob(src_pattern)
        # copy loop
        if len(list_files) != 0:
            for _f in list_files:
                _full = os.path.basename(_f).split(".")[0]
                _suffix = _full[len(src_prefix) :]
                _dst = os.path.join(
                    dst_folder, dst_basename + _suffix + "." + src_extension
                )
                shutil.copy(_f, _dst)
        return None

    @staticmethod
    def fill(dict_struct, folder, handle_files=True):
        """Recursive function for filling the ``FileSys`` structure

        :param dict_struct: dicitonary of directory structure
        :type dict_struct: dict

        :param folder: path to local folder
        :type folder: str

        :return: None
        :rtype: None
        """

        def handle_file(dst_name, lst_specs, dst_folder):
            # todo doctring
            dict_exts = FileSys.get_extensions()
            lst_exts = dict_exts[lst_specs[0]]
            src_name = lst_specs[1]
            src_dir = lst_specs[2]

            # there is a sourcing directory
            if os.path.isdir(src_dir):
                # extension loop:
                for extension in lst_exts:
                    # source
                    src_file = src_name + "." + extension
                    src_filepath = os.path.join(src_dir, src_file)
                    # destination
                    dst_file = dst_name + "." + extension
                    dst_filepath = os.path.join(dst_folder, dst_file)
                    #
                    # there might be a sourced file
                    if os.path.isfile(src_filepath):
                        shutil.copy(src=src_filepath, dst=dst_filepath)
                    elif "*" in src_name:
                        # is a pattern file
                        FileSys.copy_batch(
                            src_pattern=src_filepath, dst_pattern=dst_filepath
                        )
                    else:
                        pass

        # structure loop:
        for k in dict_struct:
            # get current folder or file
            _d = folder + "/" + k

            # [case 1] bottom is a folder
            if isinstance(dict_struct[k], dict):
                # make a dir
                FileSys.make_dir(str_path=_d)
                # now move down:
                FileSys.fill(dict_struct=dict_struct[k], folder=_d)

            # bottom is an expected file
            else:
                if handle_files:
                    handle_file(dst_name=k, lst_specs=dict_struct[k], dst_folder=folder)

        return None

    def get_metadata(self):
        """Get a dictionary with object metadata.
        Expected to increment superior methods.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        # ------------ call super ----------- #
        dict_meta = super().get_metadata()

        # customize local metadata:
        dict_meta_local = {self.folder_base_field: self.folder_base}

        # update
        dict_meta.update(dict_meta_local)
        # remove color
        dict_meta.pop(self.color_field)
        # remove source
        dict_meta.pop(self.source_data_field)

        return dict_meta

    def get_structure(self):
        """Get FileSys structure dictionary. Expected to overwrite superior methods

        :return: structure dictionary
        :rtype: dict
        """
        # get pandas DataFrame
        df = self.data.copy()

        # Initialize the nested dictionary
        dict_structure = {}

        # Iterate over the rows of the DataFrame
        for index, row in df.iterrows():
            current_dict = dict_structure

            # Split the folder path into individual folder names
            folders = row["Folder"].split("/")

            # Iterate over the folders to create the nested structure
            for folder in folders:
                current_dict = current_dict.setdefault(folder, {})

            # If a file is present, add it to the nested structure
            if pd.notna(row["File"]):
                current_dict[row["File"]] = [
                    row["Format"],
                    row["File_Source"] if pd.notna(row["File_Source"]) else "",
                    row["Folder_Source"] if pd.notna(row["Folder_Source"]) else "",
                ]
        return dict_structure

    def set(self, dict_setter, load_data=True):
        """Set selected attributes based on an incoming dictionary.
        Expected to increment superior methods.

        :param dict_setter: incoming dictionary with attribute values
        :type dict_setter: dict

        :param load_data: option for loading data from incoming file. Default is True.
        :type load_data: bool

        """
        # overwrite color
        dict_setter[self.color_field] = None

        super().set(dict_setter=dict_setter)

        # ---------- set basic attributes --------- #
        self.folder_base = dict_setter[self.folder_base_field]

        # set main folder
        self.folder_main = os.path.join(self.folder_base, self.name)

        # -------------- set data logic here -------------- #
        if load_data:
            self.load_data(file_data=self.file_data)

        # set view specs
        self._set_view_specs()
        # ... continues in downstream objects ... #

    def load_data(self, file_data):
        """Load data from file. Expected to overwrite superior methods.

        :param file_data: file path to data.
        :type file_data: str
        :return: None
        :rtype: None
        """
        # -------------- overwrite relative path input -------------- #
        file_data = os.path.abspath(file_data)

        # -------------- implement loading logic -------------- #

        # -------------- call loading function -------------- #
        self.data = pd.read_csv(file_data, sep=";")

        # -------------- post-loading logic -------------- #

        # ------------ update related attributes ------------ #
        self.file_data = file_data[:]
        self.folder_src = os.path.dirname(self.file_data)
        # view specs -- folder setup
        self._set_view_specs()
        # data size (rows)
        self.size = len(self.data)
        return None

    def setup(self):
        """
        This method sets up all the FileSys structure (default folders and files)

        .. danger::

            This method overwrites all existing default files.

        """
        # update structure
        self.structure = self.get_structure()

        # make main dir
        self.make_dir(str_path=self.folder_main)

        # fill structure
        FileSys.fill(dict_struct=self.structure, folder=self.folder_main)

    def view(self, show=True):
        """Get a basic visualization.
        Expected to overwrite superior methods.

        :param show: option for showing instead of saving.
        :type show: bool

        :return: None or file path to figure
        :rtype: None or str

        **Notes:**

        - Uses values in the ``view_specs()`` attribute for plotting

        **Examples:**

        Simple visualization:

        >>> ds.view(show=True)

        Customize view specs:

        >>> ds.view_specs["title"] = "My Custom Title"
        >>> ds.view(show=True)

        Save the figure:

        >>> ds.view_specs["folder"] = "path/to/folder"
        >>> ds.view_specs["filename"] = "my_visual"
        >>> ds.view_specs["fig_format"] = "png"
        >>> ds.view(show=False)

        """
        # todo implement
        # get specs
        specs = self.view_specs.copy()

        # --------------------- figure setup --------------------- #

        # fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        # --------------------- plotting --------------------- #

        # --------------------- post-plotting --------------------- #
        # set basic plotting stuff

        # Adjust layout to prevent cutoff
        # plt.tight_layout()

        # --------------------- end --------------------- #
        # show or save
        if show:
            # plt.show()
            return None
        else:
            file_path = "{}/{}.{}".format(
                specs["folder"], specs["filename"], specs["fig_format"]
            )
            plt.savefig(file_path, dpi=specs["dpi"])
            plt.close(fig)
            return file_path


if __name__ == "__main__":
    folder = "C:\data"

    fs = FileSys(folder_base=folder)
    fs.load(file_table_info="../tests/data/core2.csv")
    print(fs.data.to_string())
    fs.setup()
