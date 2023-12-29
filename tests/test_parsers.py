import os
import unittest
import pandas as pd
from plans.parsers import inmet
from tests import core

datafolder = "../docs/datasets/parsers"

# --------------------- REUSABLE PARSER TESTS ---------------------- #

def test_parse_data(parser):
    # Check if the 'data' attribute is a DataFrame
    assert isinstance(parser.data, pd.DataFrame)
    # Check if station metadata df is a DataFrame
    assert isinstance(parser.station_metadata_df, pd.DataFrame)
    # Check if the metadata is dict
    assert isinstance(parser.station_metadata, dict)

def test_export_data(parser):
    """Test the ``export_data()`` method of the parser object"""
    file_path = parser.export_data(folder=datafolder)
    core.assert_file_exists(file_path=file_path)

def test_export_metadata(parser):
    """Test the ``export_metadata()`` method of the parser object"""
    file_path = parser.export_metadata(folder=datafolder)
    core.assert_file_exists(file_path=file_path)

def test_export_glossary(parser):
    """Test the ``export_glossary()`` method of the parser object"""
    file_path = parser.export_glossary(folder=datafolder)
    core.assert_file_exists(file_path=file_path)

def test_export(parser):
    """Test the ``export()`` method of the parser object"""
    file_dict = parser.export(folder=datafolder)
    for k in file_dict:
        core.assert_file_exists(file_path=file_dict[k])

    test_export_data(parser=parser)
    test_export_metadata(parser=parser)
    test_export_glossary(parser=parser)

def test_exports(parser):
    """Test all exporting methods of the parser object"""
    test_export_data(parser=parser)
    test_export_metadata(parser=parser)
    test_export_glossary(parser=parser)
    test_export(parser=parser)


# --------------------- TEST OBJECTS ---------------------- #

class TestINMETConvDaily(unittest.TestCase):

    def setUp(self):
        # Initialize
        self.prefix = "inmet_convd"
        self.test_src_file_name = "inmet_convd_src.csv"
        self.testfile_path = "{}/{}".format(datafolder, self.test_src_file_name)

        # Set the expected columns
        numeric = ["float64", "int64"]
        columns_map = {
            'DateTime': [['datetime64[ns]']],
            'EvP': [numeric],
            'IsT': [numeric],
            'P': [numeric],
            'Temp_max': [numeric],
            'Temp_mean': [numeric],
            'Temp_min': [numeric],
            'RM_mean': [numeric],
            'RM_min': [numeric],
            'WS_mean': [numeric],
        }
        self.expected_columns = list(columns_map.keys())
        self.expected_dtypes = {key: value[0] for key, value in columns_map.items()}

        # set parser
        self.parser = inmet.ConventionalDaily(name="Testing")
        # load data
        self.parser.load_data(file_path=self.testfile_path)

    def test_initialization(self):
        # Check if glossary is a DataFrame
        df = self.parser.get_glossary()
        self.assertIsInstance(df, pd.DataFrame)

    def test_parse_data(self):
        test_parse_data(parser=self.parser)

    def test_columns_in_dataframe(self):
        core.assert_columns_in_dataframe(
            parser_data=self.parser.data,
            expected_columns=self.expected_columns
        )

    def test_dtypes_in_dataframe(self):
        core.assert_dtypes_in_dataframe(
            parser_data=self.parser.data,
            expected_dtypes=self.expected_dtypes
        )

    def test_export_methods(self):
        test_exports(parser=self.parser)

    def tearDown(self):
        # Clean up any resources created during the tests
        pass


class TestINMETAutomatic(TestINMETConvDaily):

    def setUp(self):
        # Initialize
        self.prefix = "inmet_auto"
        self.test_src_file_name = "inmet_auto_src.csv"
        self.testfile_path = "{}/{}".format(datafolder, self.test_src_file_name)

        # Set the expected columns
        numeric = ("float64", "int64")
        columns_map = {
            'DateTime': [['datetime64[ns]']],
            'P': [numeric],
            'PALoc': [numeric],
            'PASea': [numeric],
            'PALoc_premax': [numeric],
            'PALoc_premin': [numeric],
            'Rad': [numeric],
            'TempCPU': [numeric],
            'TempDB': [numeric],
            'TempDP': [numeric],
            'TempDB_premax': [numeric],
            'TempDB_premin': [numeric],
            'TempDP_premax': [numeric],
            'TempDP_premin': [numeric],
            'Volt': [numeric],
            'RM_premin': [numeric],
            'RM_premax': [numeric],
            'RM': [numeric],
            'WD': [numeric],
            'WG': [numeric],
            'WS': [numeric]
        }

        self.expected_columns = list(columns_map.keys())
        self.expected_dtypes = {key: value[0] for key, value in columns_map.items()}

        # set parser
        self.parser = inmet.Automatic(name="Testing")
        # load data
        self.parser.load_data(file_path=self.testfile_path)


class TestINMETAutomaticByYear(TestINMETConvDaily):

    def setUp(self):
        # Initialize
        self.prefix = "inmet_autoy"
        self.test_src_file_name = "inmet_autoy_src.csv"
        self.testfile_path = "{}/{}".format(datafolder, self.test_src_file_name)

        # Set the expected columns
        numeric = ["float64", "int64"]
        columns_map = {
            'DateTime': [['datetime64[ns]']],
            'P': [numeric],
            'PALoc': [numeric],
            'PALoc_premax': [numeric],
            'PALoc_premin': [numeric],
            'Rad': [numeric],
            'TempDB': [numeric],
            'TempDP': [numeric],
            'TempDB_premax': [numeric],
            'TempDB_premin': [numeric],
            'TempDP_premax': [numeric],
            'TempDP_premin': [numeric],
            'RM_premax': [numeric],
            'RM_premin': [numeric],
            'RM': [numeric],
            'WD': [numeric],
            'WG': [numeric],
            'WS': [numeric]
        }

        self.expected_columns = list(columns_map.keys())
        self.expected_dtypes = {key: value[0] for key, value in columns_map.items()}

        # set parser
        self.parser = inmet.AutomaticByYear(name="Testing")
        # load data
        self.parser.load_data(file_path=self.testfile_path)



if __name__ == '__main__':
    unittest.main()
