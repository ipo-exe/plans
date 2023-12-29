import os
import unittest, warnings
import pandas as pd
from plans import datasets
from tests import core
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


# Ignore all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

talk = True
export_files = True
datafolder = "../docs/datasets"


# --------------------- REUSABLE TS TESTS ---------------------- #

def test_ts_talk(ts):
    if talk:
        print(ts.data.head().to_string())
        ts.view()
        ts.update_epochs_stats()
        ts.view_epochs()

def test_ts_data(ts):
    # Check if the 'data' attribute is a DataFrame
    assert isinstance(ts.data, pd.DataFrame)

def test_ts_standardize(ts):
    ts.standardize()
    # Check if the 'data' attribute is a DataFrame
    assert isinstance(ts.data, pd.DataFrame)

def test_ts_interpolate(ts, method):
    # standardize first
    ts.standardize()
    # update epochs
    ts.update_epochs_stats()
    # view prior
    if talk:
        print(ts.epochs_stats.to_string())
        ts.view_epochs()
    ts.interpolate_gaps(method=method, inplace=True)
    ts.update_epochs_stats()
    # view posterior
    if talk:
        print(ts.epochs_stats.to_string())
        ts.view_epochs()
    assert ts.smallgaps_n == 0

def test_ts_aggregate(ts, freq, bad_max, agg_funcs=None):
    if talk:
        ts.view()
    df = ts.aggregate(freq=freq, bad_max=bad_max, agg_funcs=agg_funcs)
    assert isinstance(ts.data, pd.DataFrame)
    if talk:
        print(df.head(15).to_string())
        ts2 = datasets.TimeSeries(
            name="{} - Agg {}".format(ts.name, freq),
            varname=ts.varname,
            varfield=ts.varfield,
            units=ts.units
        )
        ts2.set_data(input_df=df, input_varfield="{}_mean".format(ts.varfield), input_dtfield=ts.dtfield)
        ts2.dtfreq = freq
        ts2.view()

def test_ts_upscale(ts, freq, bad_max):
    if talk:
        ts.view()
    df = ts.upscale(freq=freq, bad_max=bad_max)
    assert isinstance(ts.data, pd.DataFrame)
    if talk:
        print(ts.data.head(5).to_string())
        print(df.head(5).to_string())
        ts2 = datasets.TimeSeries(
            name="{} - Upscale {}".format(ts.name, freq),
            varname=ts.varname,
            varfield=ts.varfield,
            units=ts.units
        )
        ts2.set_data(input_df=df, input_varfield="{}".format(ts.varfield), input_dtfield=ts.dtfield)
        ts2.dtfreq = freq
        ts2.view()

def test_ts_gapsize(ts, n_gap, n_epochs_expected):
    # set gap
    ts.gapsize = n_gap
    ts.update_epochs_stats()
    # view
    if talk:
        ts.view_epochs()
    n_epochs_actual = ts.epochs_n
    assert n_epochs_actual == n_epochs_expected


def test_ts_view(ts, folder, filename):
    if export_files:
        file_path = ts.view(
            show=False,
            folder=folder,
            filename=filename
        )
        core.assert_file_exists(file_path=file_path)


def test_ts_view_epochs(ts, folder, filename):
    if export_files:
        file_path = ts.view_epochs(
            show=False,
            folder=folder,
            filename=filename
        )
        core.assert_file_exists(file_path=file_path)



class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        # Initialize any necessary objects or data for the tests
        self.prefix = "wind_hourly"
        self.test_src_file = "clim_hourly_src.csv"
        self.dir_out = "{}/misc".format(datafolder,)
        self.testfile_path = "{}/{}".format(self.dir_out, self.test_src_file)
        self.method_interpolation = "cubic"

        # set --- expected parameters
        # raw gaps - epochs
        self.dict_gaps_epochs = {
            3: 5, # Gapsize -> Expected epochs
            12: 3
        }

        # create instance
        self.ts = datasets.TimeSeries(
            name="INMET Station",
            alias="Auto",
            varfield="Rad",
            units="kJ/m$^2$",
            varname="Radiation",
        )
        # set attributes
        self.ts.rawcolor = "tab:orange"
        self.ts.gapsize = 12

        # load data
        self.ts.load_data(
            input_file=self.testfile_path,
            input_varfield="Rad",
            input_dtfield="Date"
        )

    def test_talk(self):
        test_ts_talk(ts=self.ts)

    def test_load_data(self):
        test_ts_data(ts=self.ts)

    def test_export_view(self):
        test_ts_view(
            ts=self.ts,
            folder=self.dir_out,
            filename=self.prefix
        )

    def test_export_view_epochs(self):
        test_ts_view_epochs(
            ts=self.ts,
            folder=self.dir_out,
            filename=self.prefix + "_epochs"
        )

    def test_gaps(self):
        for k in self.dict_gaps_epochs:
            test_ts_gapsize(ts=self.ts, n_gap=k, n_epochs_expected=self.dict_gaps_epochs[k])

    def test_interpolation(self):
        test_ts_interpolate(ts=self.ts, method=self.method_interpolation)

    def test_aggregate_3h(self):
        test_ts_aggregate(
            ts=self.ts, freq="3H", bad_max=3, agg_funcs=None
        )

    def test_aggregate_1d(self):
        test_ts_aggregate(
            ts=self.ts, freq="D", bad_max=24, agg_funcs=None
        )

    def test_upscale_3h(self):
        test_ts_upscale(ts=self.ts, freq="3H", bad_max=3)

    def test_upscale_1d(self):
        test_ts_upscale(ts=self.ts, freq="D", bad_max=24)

    def tearDown(self):
        # Clean up any resources created in the setUp method
        self.ts = None


class TestRainSeries(TestTimeSeries):

    def setUp(self):
        # Initialize any necessary objects or data for the tests
        self.prefix = "rain_hourly"
        self.test_src_file = "rain_IPH41.csv"
        self.dir_out = "{}/rain".format(datafolder, )
        self.testfile_path = "{}/{}".format(self.dir_out, self.test_src_file)
        self.method_interpolation = "constant"

        # set --- expected parameters
        # raw gaps - epochs
        self.dict_gaps_epochs = {
            7 * 72 : 2,  # Gapsize -> Expected epochs
            3*12: 3
        }

        # create instance
        self.ts = datasets.RainSeries(
            name="IPH-PVG41",
            alias="IPH41",
        )
        # set attributes
        self.ts.gapsize = 12

        # load data
        self.ts.load_data(
            input_file=self.testfile_path,
            input_varfield="P_IPH41_mm",
            input_dtfield="DateTime"
        )

    def test_talk(self):
        test_ts_talk(ts=self.ts)

    def test_load_data(self):
        test_ts_data(ts=self.ts)

    def test_export_view(self):
        test_ts_view(
            ts=self.ts,
            folder=self.dir_out,
            filename=self.prefix
        )

    def test_export_view_epochs(self):
        test_ts_view_epochs(
            ts=self.ts,
            folder=self.dir_out,
            filename=self.prefix + "_epochs"
        )

    def test_gaps(self):
        for k in self.dict_gaps_epochs:
            test_ts_gapsize(ts=self.ts, n_gap=k, n_epochs_expected=self.dict_gaps_epochs[k])

    def test_interpolation(self):
        test_ts_interpolate(ts=self.ts, method=self.method_interpolation)

    def test_aggregate_3h(self):
        test_ts_aggregate(
            ts=self.ts, freq="3H", bad_max=3, agg_funcs=None
        )

    def test_aggregate_1d(self):
        test_ts_aggregate(
            ts=self.ts, freq="D", bad_max=24, agg_funcs=None
        )

    def test_upscale_3h(self):
        test_ts_upscale(ts=self.ts, freq="3H", bad_max=9)

    def test_upscale_1d(self):
        test_ts_upscale(ts=self.ts, freq="D", bad_max=24)

    def tearDown(self):
        # Clean up any resources created in the setUp method
        self.ts = None

if __name__ == '__main__':
    unittest.main()