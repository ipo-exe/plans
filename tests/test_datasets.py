import unittest
import pandas as pd
from datetime import datetime
from plans.ds import TimeSeries, Collection


class TestObject:

    def __init__(self, name="TestObject", alias="TOb"):
        self.name = name
        self.alias = alias
        self.timestamp = datetime.now()

    def get_metadata(self):
        return {
            "Name": self.name,
            "Alias": self.alias,
            "Timestamp": self.timestamp
        }

class TestCollection(unittest.TestCase):

    def setUp(self):
        # Initialize the base object and test_collection for testing
        self.test_collection = Collection(base_object=TestObject, name="myCatalog")

    def test_initialization(self):
        # Check if the catalog is initially empty
        self.assertTrue(self.test_collection.catalog.empty, msg=f"Expected empty catalog DataFrame")

        # Check if the collection is dict
        self.assertIsInstance(self.test_collection.collection, dict, msg=f"Expected type 'dict', but got {type(self.test_collection.collection)}.")

        # Check if the dictionary is empty
        self.assertDictEqual(self.test_collection.collection, {}, msg="Dictionary should be empty.")

    def test_append_and_remove(self):
        # Create a new object
        new_obj = TestObject(name="NewObject", alias="NOb")

        # Append the new object to the test_collection
        self.test_collection.append(new_object=new_obj)

        # Check if the catalog has one entry after appending
        self.assertEqual(len(self.test_collection.catalog), 1)

        # Check if the test_collection has one entry after appending
        self.assertEqual(len(self.test_collection.collection), 1)

        # Remove the object from the test_collection
        self.test_collection.remove(name=new_obj.name)

        # Check if the catalog is empty after removing
        self.assertTrue(self.test_collection.catalog.empty)

        # Check if the test_collection is empty after removing
        self.assertTrue(not self.test_collection.collection)

    def test_update(self):
        # Create a new object
        detail_name = "Detail"
        detail_alias = "Dtl"
        detail_time = datetime.now()
        new_obj = TestObject(name=detail_name, alias=detail_alias)
        new_obj.timestamp = detail_time

        # Append the new object to the test_collection
        self.test_collection.append(new_object=new_obj)

        # Update the catalog with details
        self.test_collection.update(details=True)

        # Check if the catalog has details after updating
        self.assertIn(detail_name, list(self.test_collection.catalog["Name"]), msg=f"{detail_name} should be in the catalog.")
        self.assertIn(detail_alias, list(self.test_collection.catalog["Alias"]), msg=f"{detail_alias} should be in the catalog.")
        self.assertIn(detail_time, list(self.test_collection.catalog["Timestamp"]), msg=f"{detail_time} should be in the catalog.")

    def tearDown(self):
        # Clean up any resources created during the tests
        pass


class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        # Initialize any necessary objects or data for the tests
        # For example, create an instance of YourClassWithAggregateMethod with test data
        self.ts = TimeSeries(
            name="TestTS",
            alias="TTS",
            varfield="Temperature",
        )

    def test_load_data(self):
        # load data via method
        self.ts.load_data(
            input_file="./tests/data/timeseries_min.csv",  # expected to be in project level
            input_varfield="Temp",
            input_dtfield="Datetime",
            sep=";"
        )
        # Check if ts.data is a DataFrame
        self.assertIsInstance(self.ts.data, pd.DataFrame)

        # Check if ts.data has 2 columns
        self.assertEqual(len(self.ts.data.columns), 2)

        # Check if columns are DateTime and Temperature
        expected_columns = ["DateTime", "Temperature"]
        self.assertListEqual(list(self.ts.data.columns), expected_columns)

        # Check if DateTime column is a datetime field
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.ts.data["DateTime"]))

    def tearDown(self):
        # Clean up any resources created in the setUp method
        self.ts = None


# --------------------- TEST OBJECTS ---------------------- #


if __name__ == '__main__':
    unittest.main()


