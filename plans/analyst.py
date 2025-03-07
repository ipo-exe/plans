"""
Description:
    The ``analyst`` module provides objects to handle data analysis.

License:
    This software is released under the GNU General Public License v3.0 (GPL-3.0).
    For details, see: https://www.gnu.org/licenses/gpl-3.0.html

Overview
--------

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis.
Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

>>> from plans import analyst

Class aptent taciti sociosqu ad litora torquent per
conubia nostra, per inceptos himenaeos. Nulla facilisi. Mauris eget nisl
eu eros euismod sodales. Cras pulvinar tincidunt enim nec semper.

Example
--------

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis.

.. code-block:: python

    import numpy as np
    from plans import analyst

    # get data to a vector
    data_vector = np.random.rand(1000)

    # instantiate the Univar object
    uni = analyst.Univar(sample=data_vector, name="my_data")

    # view sample
    uni.view()

Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plans.root import DataSet
import os


# --------- Functions -----------
def linear(x, c0, c1):
    """Linear function f(x) = c0 + c1 * x

    :param x: function input
    :type x: float | :class:`numpy.ndarray`
    :param c0: translational parameter
    :type c0: float
    :param c1: scaling parameter
    :type c1: float
    :return: function output
    :rtype: float | :class:`numpy.ndarray`
    """
    return c0 + (x * c1)


def power(x, c0, c1, c2):
    """Power function f(x) =  c2 * ((x + c0)^c1)

    :param x: function input
    :type x: float | :class:`numpy.ndarray`
    :param c0: translational parameter
    :type c0: float
    :param c1: exponent parameter
    :type c1: float
    :param c2: scaling parameter
    :type c2: float
    :return: function output
    :rtype: float | :class:`numpy.ndarray`
    """
    return c2 * (np.power((x + c0), c1))


def power_zero(x, c0, c1):
    """Power function with root in zero f(x) =  c1 * ((x)^c0)

    :param x: function input
    :type x: float | :class:`numpy.ndarray`
    :param c0: exponent parameter
    :type c0: float
    :param c1: scaling parameter
    :type c1: float
    :return: function output
    :rtype: float | :class:`numpy.ndarray`
    """
    return c1 * (np.power((x), c0))


# --------- Objects -----------


class Univar(DataSet):
    """The Univariate object"""

    def __init__(self, name="MyUnivar", alias="Uv0"):
        # ------------ set defaults ----------- #
        self.varfield = "V"
        self.varalias = "Var"
        self.varname = "Variable"
        self.units = "units"

        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)

        # overwriters

        # attributes
        self.stats_df = None
        self.freq_df = None
        self.weibull_df = None

    def load_data(self, file_data):
        """Load data from file. Expected to overwrite superior methods.

        :param file_data: file path to data.
        :type file_data: str
        :return: None
        :rtype: None
        """

        # -------------- overwrite relative path input -------------- #
        self.file_data = os.path.abspath(file_data)

        # -------------- implement loading logic -------------- #
        default_columns = {
            #'DateTime': 'datetime64[1s]',
            self.varfield: float,
        }

        # -------------- call loading function -------------- #
        self.data = pd.read_csv(
            self.file_data,
            sep=self.file_data_sep,
            dtype=default_columns,
            usecols=list(default_columns.keys()),
        )

        # -------------- post-loading logic -------------- #
        self.data.dropna(inplace=True)

        # -------------- update other mutables -------------- #
        self.update()

        # ... continues in downstream objects ... #

        return None

    def update(self):
        # ------------ call super ----------- #
        super().update()

        # update stats
        if self.data is not None:
            self.stats_df = self.assess_basic_stats()
            self.freq_df = self.assess_frequency()
            self.weibull_df = self.assess_weibull_cdf()

        # ... continues in downstream objects ... #
        return None

    # Assessing methods

    def assess_normality(self, clevel=0.95):
        """Assessment on normality using standard tests

        :return: dataframe of assessment results
        :rtype: :class:`pandas.DataFrame`
        """

        df_clean = self.data.dropna()
        data = df_clean[self.varfield].values

        # run tests
        lst_tests = []
        lst_tests.append(Univar.test_normality_ks(data, clevel=clevel))
        lst_tests.append(Univar.test_normality_sw(data, clevel=clevel))
        lst_tests.append(Univar.test_normality_dp(data, clevel=clevel))
        # create dataframe
        lst_names = []
        lst_stats = []
        lst_p = []
        lst_clvl = []
        lst_is = []
        for e in lst_tests:
            lst_names.append(e["Test"])
            lst_stats.append(e["Statistic"])
            lst_p.append(e["p-value"])
            lst_clvl.append(e["Confidence"])
            lst_is.append(e["is_normal"])
        df_result = pd.DataFrame(
            {
                "Test": lst_names,
                "Statistic": lst_stats,
                "p-value": lst_p,
                "is_normal": lst_is,
                "Confidence": lst_clvl,
            }
        )
        return df_result

    def assess_frequency(self):
        """Assessment on data frequencies

        :return: result dataframe
        :rtype: :class:`pandas.DataFrame`
        """

        df_clean = self.data.dropna()
        data = df_clean[self.varfield].values

        # compute percentiles intevarls
        vct_percentiles = np.arange(0, 100)

        # get CFC values
        vct_cfc = np.percentile(data, vct_percentiles)

        # reverse to get exceedance
        vct_exeed = 100 - vct_percentiles

        # get count
        vct_count = np.histogram(data, bins=len(vct_percentiles))[0]

        # get empirical prob in the histogram
        vct_empprob = vct_count / np.sum(vct_count)

        df_result = pd.DataFrame(
            {
                "Percentiles": vct_percentiles,
                "Exceedance": vct_exeed,
                "Frequency": vct_count,
                "Empirical Probability": vct_empprob,
                "Values": vct_cfc,
            }
        )
        return df_result

    def assess_basic_stats(self):

        df_clean = self.data.dropna()
        data = df_clean[self.varfield].values

        dct = {
            "Count": len(data),
            "Sum": np.sum(data),
            "Mean": np.mean(data),
            "SD": np.std(data),
            "Min": np.min(data),
            "p01": np.percentile(data, 1),
            "p05": np.percentile(data, 5),
            "p25": np.percentile(data, 25),
            "p50": np.percentile(data, 50),
            "p75": np.percentile(data, 75),
            "p90": np.percentile(data, 90),
            "p95": np.percentile(data, 95),
            "p99": np.percentile(data, 99),
            "Max": np.max(data),
        }
        df_result = pd.DataFrame(
            {"Statistic": list(dct.keys()), "Value": [dct[key] for key in dct]}
        )
        return df_result

    def assess_weibull_cdf(self):
        """Get the Weibull model

        :param x: function input
        :type x: :class:`numpy.ndarray`
        :return: model dataframe
        :rtype: :class:`pandas.DataFrame`
        """
        df_clean = self.data.dropna()
        x = df_clean[self.varfield].values
        v = np.sort(x)
        vr = v[::-1]
        r = np.arange(0, len(x)) + 1
        px = Univar.weibull_px(ranks=r)
        result_df = pd.DataFrame({"Data": vr, "P(X)": px, "F(X)": 1 - px})
        return result_df

    def assess_gumbel_cdf(self):
        """Assess the Gumbel CDF for the data (it assumes that is a maxima dataset)

        :return: multiple results from the assessment
        :rtype: dict
        """
        from scipy import stats

        # clean data
        df = self.data.dropna()

        # get extra values
        df = df.sort_values(by=self.varfield, ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        df["P(X)_Empirical"] = Univar.empirical_px(ranks=df["Rank"].values)
        df["P(X)_Weibull"] = Univar.weibull_px(ranks=df["Rank"].values)
        df["T(X)_Weibull"] = 1 / df["P(X)_Weibull"]
        df["P(X)_Gringorten"] = Univar.gringorten_px(ranks=df["Rank"].values)

        # ---- model Gumbel using the method of moments ----

        # using scipy
        # Fit a Gumbel distribution to the data
        params = stats.gumbel_r.fit(df[self.varfield].values, method="MM")  # MLM or MM
        gumbel_a = params[0]
        gumbel_b = params[1]

        # compute fit gumbel values
        df["P(X)_Gumbel"] = 1 - Univar.gumbel_fx(
            x=df[self.varfield].values, a=gumbel_a, b=gumbel_b
        )
        # return period
        df["T(X)_Gumbel"] = 1 / df["P(X)_Gumbel"]

        # compute uncertainty 90% bands for T(X) in the Annual Series
        t = stats.t.ppf((1 + 0.9) / 2, df=len(df) - 1)
        gumbel_se = Univar.gumbel_se(
            std_sample=np.std(a=df[self.varfield].values),
            n_sample=len(df),
            tx=df["T(X)_Gumbel"],
        )
        # Standar error analysis
        df["T(X)_Gumbel_SE90"] = t * gumbel_se
        h_max = df[self.varfield] + df["T(X)_Gumbel_SE90"]
        h_min = df[self.varfield] - df["T(X)_Gumbel_SE90"]
        df["T(X)_Gumbel_P05"] = Univar.gumbel_tx(x=h_min, a=gumbel_a, b=gumbel_b)
        df["T(X)_Gumbel_P95"] = Univar.gumbel_tx(x=h_max, a=gumbel_a, b=gumbel_b)

        # -------- Return Period analysis

        # evely spaced
        txs = np.arange(2, 1001, step=1)
        k = Univar.gumbel_freqfactor(txs)

        # apply reverse formula
        stages = np.mean(a=df[self.varfield]) + k * np.std(a=df[self.varfield])

        # get standard error
        gumbel_se = Univar.gumbel_se(
            std_sample=np.std(a=df[self.varfield]), n_sample=len(df), tx=txs
        )
        # compute 50% uncertainty bands by Student's t distribution
        t_50 = stats.t.ppf((1 + 0.5) / 2, df=len(df) - 1)
        h_range_50 = t_50 * gumbel_se

        # compute 90% uncertainty bands by Student's t distribution
        t_90 = stats.t.ppf((1 + 0.9) / 2, df=len(df) - 1)
        h_range_90 = t_90 * gumbel_se

        # Set TX dataframe
        df_tx = pd.DataFrame(
            {
                "T(X)_Gumbel": txs,
                f"{self.varfield}": stages,
                f"{self.varfield}_P25": stages - h_range_50,
                f"{self.varfield}_P75": stages + h_range_50,
                f"{self.varfield}_P05": stages - h_range_90,
                f"{self.varfield}_P95": stages + h_range_90,
            }
        )

        # -------- QQ plots
        qq, qq_params = stats.probplot(
            x=df[self.varfield].values, dist="gumbel_r", sparams=params
        )
        # Set QQ dataframe
        df_qq = pd.DataFrame({self.varfield: qq[1], "T-Q": qq[0]})

        # -------- Goodness of fit tests
        ks_stat, ks_p_value = stats.kstest(
            df[self.varfield].values, cdf="gumbel_r", args=params
        )
        # accept Null Hypothesis
        is_gumbel = True
        if ks_p_value <= 0.05:
            is_gumbel = False

        # -------- Metadata dataframe
        df_meta = pd.DataFrame(
            {
                "Metadata": [
                    "N",
                    "Gumbel a",
                    "Gumbel b",
                    "KS-test s-value",
                    "KS-test p-value",
                    "KS-test Is Gumbel (NullH)",
                    "QQ-plot c0",
                    "QQ plot c1",
                    "QQ-plot r2",
                ],
                "Value": [
                    len(df),
                    gumbel_a,
                    gumbel_b,
                    ks_stat,
                    ks_p_value,
                    is_gumbel,
                    qq_params[1],
                    qq_params[0],
                    qq_params[2],
                ],
            }
        )

        return {"Data": df, "Data_T(X)": df_tx, "Data_QQ": df_qq, "Metadata": df_meta}

    # Plotting methods

    def plot_hist(
        self,
        bins=100,
        colored=False,
        annotated=False,
        rule=None,
        show=False,
        folder="C:/sample",
        filename="histogram",
        specs=None,
        dpi=300,
    ):
        """Plot histogram of sample

        :param bins: number of bins
        :type bins: int
        :param colored: Boolean to quantile-colored histogram
        :type colored: bool
        :param annotated: Boolean to plot stats texts over histogram
        :type annotated: bool
        :param rule: name of rule to compute bins
        :type rule: str
        :param show: Boolean to show instead of saving
        :type show: bool
        :param folder: output folder
        :type folder: str
        :param filename: image file name
        :type filename: str
        :param specs: specification dictionary
        :type specs: dict
        :param dpi: image resolution (default = 96)
        :type dpi: int
        """

        # get bins
        if rule is None:
            pass
        else:
            bins = self.nbins_by_rule(rule=rule)

        # get specs
        default_specs = {
            "color": "tab:grey",
            "title": "Histogram of {}".format(self.name),
            "width": 4 * 1.618,
            "height": 4,
            "xlabel": "value",
            "ylim": (0, 0.1),
            "xlim": (
                0.95 * np.min(self.data[self.varfield]),
                1.05 * np.max(self.data[self.varfield]),
            ),
            "subtitle": None,
            "cmap": "viridis",
            "colors": None,
            "n_classes": 5,
            "grid": False,
            "quantiles": True,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # start plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        ax = plt.gca()
        ax.set_position([0.15, 0.15, 0.75, 0.75])
        if specs["subtitle"] is None:
            plt.title(specs["title"])
        else:
            plt.title("{} | {}".format(specs["title"], specs["subtitle"]))

        # Calculate quantiles
        data = pd.Series(self.data[self.varfield])
        # filter data to xlim
        data = data[data.between(specs["xlim"][0], specs["xlim"][1])]
        if specs["quantiles"]:
            quantiles = data.quantile(np.linspace(0, 1, specs["n_classes"] + 1))
        else:
            quantiles = pd.Series(
                np.linspace(
                    start=data.min(), stop=data.max(), num=specs["n_classes"] + 1
                )
            )

        # Determine the overall bin width you desire
        bin_width = (
            data.max() - data.min()
        ) / bins  # for example, 40 equal width bins across the data range
        # handle colored plots
        if colored:
            if specs["colors"] is None:
                cmap = mpl.colormaps[specs["cmap"]]
                colors = [cmap(i) for i in np.linspace(0, 1, len(quantiles) - 1)]
            else:
                colors = specs["colors"]

            # Create the bins
            binsv = np.arange(start=data.min(), stop=data.max(), step=bin_width)
            binsv = np.append(
                binsv, data.max()
            )  # Ensure the last bin includes the max value

            # Plot the histogram
            n, bins, patches = plt.hist(
                self.data,
                bins=binsv,
                linewidth=0,
                weights=np.ones(len(self.data)) / len(self.data),
            )

            # Color the bars based on the quantile
            for patch, edge in zip(patches, bins[:-1]):
                if edge < quantiles.values[1]:
                    plt.setp(patch, "facecolor", colors[0])
                elif edge < quantiles.values[2]:
                    plt.setp(patch, "facecolor", colors[1])
                elif edge < quantiles.values[3]:
                    plt.setp(patch, "facecolor", colors[2])
                elif edge < quantiles.values[4]:
                    plt.setp(patch, "facecolor", colors[3])
                else:
                    plt.setp(patch, "facecolor", colors[4])
        else:
            plt.hist(
                self.data,
                bins=bins,
                weights=np.ones(len(self.data)) / len(self.data),
                color=specs["color"],
            )

        plt.xlabel(specs["xlabel"])
        plt.ylim(specs["ylim"])
        plt.xlim(specs["xlim"])
        plt.grid(specs["grid"])
        # Optionally, add vertical lines for each quantile and annotate them
        mu = np.mean(self.data)
        plt.axvline(mu, color="r", linestyle="dashed", linewidth=1)
        plt.text(
            mu,
            plt.gca().get_ylim()[1] * 0.92,
            f"$\mu$ = {mu:.1f}",
            ha="left",
            va="bottom",
            fontsize=10,
            rotation=0,
            color="red",
        )
        if annotated:
            aux = 0
            for q in quantiles:
                plt.axvline(q, color="k", linestyle="dashed", linewidth=1)
                plt.text(
                    q,
                    plt.gca().get_ylim()[1] * (0.9 - aux),
                    f"{q:.1f}",
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    rotation=0,
                )
                aux = aux + 0.05
        # plt.tight_layout()
        # show or save
        if show:
            plt.show()
        else:
            plt.savefig(
                "{}/{}_{}.png".format(folder, self.name, filename),
                dpi=dpi,
                bbox_inches="tight",
            )

    def _set_view_specs(self):
        """Set view specifications.
        Expected to overwrite superior methods.

        :return: None
        :rtype: None
        """
        self.view_specs = {
            "folder": self.folder_data,
            "filename": self.name,
            "fig_format": "jpg",
            "dpi": 300,
            "title": "View of {}".format(self.name),
            "width": 8,
            "height": 3,
            "xvar": "i",
            "yvar": self.varname,
            "xlabel": "i",
            "xlabel_b": "Count",
            "xlabel_c": "P(X)",
            "ylabel": self.units,
            "color": self.color,
            "color_b": "tab:grey",
            "color_c": "blue",
            "alpha": 0.9,
            "ylim": None,
            "xlim": None,
            "subtitle_a": None,
            "subtitle_b": None,
            "subtitle_c": None,
            "plot_grid": False,
            "hist_density": False,
            "gs_wspace": 0.4,
            "gs_hspace": 0.1,
            "gs_left": 0.08,
            "gs_right": 0.98,
            "gs_bottom": 0.2,
            "gs_top": 0.88,
        }
        return None

    def view(self, show=True, return_fig=False):
        """Get a basic visualization.
        Expected to overwrite superior methods.

        :param show: option for showing instead of saving.
        :type show: bool
        :param return_fig: option for returning the figure object itself.
        :type return_fig: bool
        :return: None or file path to figure
        :rtype: None or str
        """

        specs = self.view_specs.copy()

        # start plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        plt.suptitle(specs["title"])
        # grid
        gs = mpl.gridspec.GridSpec(
            1,
            5,
            wspace=specs["gs_wspace"],
            hspace=specs["gs_hspace"],
            left=specs["gs_left"],
            right=specs["gs_right"],
            bottom=specs["gs_bottom"],
            top=specs["gs_top"],
        )  # nrows, ncols

        # ------------ scatter plot ------------
        ax = fig.add_subplot(gs[0, :3])
        plt.scatter(
            np.arange(len(self.data)),
            self.data[self.varfield].values,
            marker=".",
            color="tab:grey",
            alpha=specs["alpha"],
        )
        if specs["subtitle_a"] is not None:
            plt.title(specs["subtitle_a"])
        if specs["ylim"] is not None:
            plt.ylim(specs["ylim"])
        plt.xlabel(specs["xlabel"])
        plt.ylabel(specs["ylabel"])
        plt.xlim(0, len(self.data))
        plt.grid(specs["plot_grid"])

        # ------------ hist ------------
        ax2 = fig.add_subplot(gs[0, 3], sharey=ax)
        # ensure clean data
        df_clean = self.data.dropna()
        data = df_clean[self.varfield].values
        plt.hist(
            data,
            bins=Univar.nbins_fd(data=data),
            color=specs["color_b"],
            alpha=1,
            orientation="horizontal",
            density=specs["hist_density"],
        )
        if specs["subtitle_b"] is not None:
            plt.title(specs["subtitle_b"])
        if specs["ylim"] is not None:
            plt.ylim(specs["ylim"])
        plt.xlabel(specs["xlabel_b"])
        plt.grid(specs["plot_grid"])

        # ------------ cdf ------------
        ax3 = fig.add_subplot(gs[0, 4], sharey=ax)
        plt.plot(
            self.weibull_df["P(X)"], self.weibull_df["Data"], color=specs["color_c"]
        )
        if specs["subtitle_c"] is not None:
            plt.title(specs["subtitle_c"])
        plt.xlabel(specs["xlabel_c"])
        plt.grid(specs["plot_grid"])

        # ------------ return object, show or save
        if return_fig:
            return fig
        elif show:  # default case
            plt.show()
            return None
        else:
            file_path = "{}/{}.{}".format(
                specs["folder"], specs["filename"], specs["fig_format"]
            )
            plt.savefig(file_path, dpi=specs["dpi"])
            plt.close(fig)
            return file_path

    def plot_qqplot(
        self, show=True, folder="C:/sample", filename="qqplot", specs=None, dpi=300
    ):
        """Plot Q-Q Plot on Normal distribution

        :param show: Boolean to show instead of saving
        :type show: bool
        :param folder: output folder
        :type folder: str
        :param filename: image file name
        :type filename: str
        :param specs: specification dictionary
        :type specs: dct
        :param dpi: image resolution (default = 300)
        :type dpi: int
        :return: None
        :rtype: None
        """
        df_clean = self.data.dropna()
        data = df_clean[self.varfield]
        # get specs
        default_specs = {
            "color": "tab:grey",
            "title": "Q-Q Plot of {}".format(self.name),
            "width": 4 * 1.618,
            "height": 4,
            "xlabel": "value",
            "ylim": (0.95 * np.min(data), 1.05 * np.max(data)),
            "xlim": (-3, 3),
            "subtitle": None,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # process quantiles
        _df = Univar.qqplot(data=data)
        # plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        # grid
        gs = mpl.gridspec.GridSpec(
            1, 1, wspace=0.4, hspace=0.5, left=0.1, bottom=0.15, top=0.9, right=0.95
        )  # nrows, ncols
        # scatter plot
        ax = fig.add_subplot(gs[0, 0])
        plt.title(specs["title"])
        plt.scatter(_df["T-Quantiles"], _df["Data"], marker=".", color="tab:grey")
        plt.ylim([_df["Data"].min(), _df["Data"].max()])
        plt.xlim(specs["xlim"])
        plt.xlabel("Normal Theoretical Quantiles")
        plt.ylabel("Data Empirical Quantiles")
        plt.gca().set_aspect(
            (specs["xlim"][1] - specs["xlim"][0])
            / (_df["Data"].max() - _df["Data"].min())
        )

        # show or save
        if show:
            plt.show()
        else:
            plt.savefig("{}/{}_{}.png".format(folder, self.name, filename), dpi=dpi)

    @staticmethod
    def test_distribution(test_name, stat, p, clevel=0.95, distr="normal"):
        """Util function for statistical testing

        :param test_name: name of test
        :type test_name: str
        :param stat: statistic
        :type stat: float
        :param p: p-value
        :type p: float
        :param clevel: confidence level
        :type clevel: float
        :param distr: name of distribution
        :type distr: str
        :return: summary of test
        :rtype: dict
        """
        # built output dict
        dct_out = {
            "Test": test_name,
            "Statistic": stat,
            "p-value": p,
            "Confidence": clevel,
        }

        if p > (1 - clevel):
            dct_out["is_{}".format(distr.lower())] = True
        else:
            dct_out["is_{}".format(distr.lower())] = False

        return dct_out

    @staticmethod
    def test_normality_ks(data, clevel=0.95):
        """Test for normality using the Kolmogorov-Smirnov test

        Kolmogorov-Smirnov Test: This test compares the observed distribution with
        the expected normal distribution using a test statistic and a p-value.
        A p-value less than 0.05 indicates that the null hypothesis should be rejected.

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :return: test result dictionary. Keys: Statistic, p-value and Is normal
        :rtype: dct
        """
        from scipy.stats import kstest

        vct_norm = (data - np.mean(data)) / np.std(data)

        # test
        result = kstest(vct_norm, "norm")
        return Univar.test_distribution(
            test_name="Kolmogorov-Smirnov",
            stat=result.statistic,
            p=result.pvalue,
            clevel=clevel,
            distr="normal",
        )

    @staticmethod
    def test_normality_sw(data, clevel=0.95):
        """Test for normality using the Shapiro-Wilk test.

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :return: test result dictionary. Keys: Statistic, p-value and Is normal
        :rtype: dct
        """
        from scipy.stats import shapiro

        # test
        stat, p = shapiro(data)

        return Univar.test_distribution(
            test_name="Shapiro-Wilk", stat=stat, p=p, clevel=clevel, distr="normal"
        )

    @staticmethod
    def test_normality_dp(data, clevel=0.95):
        """Test for normality using the D'Agostino-Pearson test.

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :return: test result dictionary. Keys: Statistic, p-value and Is normal
        :rtype: dct
        """
        from scipy.stats import normaltest

        # test
        stat, p = normaltest(data)

        return Univar.test_distribution(
            test_name="D'Agostino-Pearson",
            stat=stat,
            p=p,
            clevel=clevel,
            distr="normal",
        )

    @staticmethod
    def get_tx(fx):
        """Simple function for computing the Return Period from the CDF

        :param fx: CDF (FX)
        :type fx: float | :class:`numpy.ndarray`
        :return: Return Period
        :rtype: float | :class:`numpy.ndarray`
        """
        return 1 / (1 - fx)

    @staticmethod
    def gumbel_fx(x, a, b):
        """Gumbel probability distribution F(X)

        :param x: function input
        :type x: float | :class:`numpy.ndarray`
        :param a: distribution parameter a
        :type a: float
        :param b: distribution parameter b
        :type b: float
        :return: function output
        :rtype: float | :class:`numpy.ndarray`
        """
        aux_1 = (x - a) / b
        aux_2 = np.exp(-aux_1)
        g_fx = np.exp(-aux_2)
        return g_fx

    @staticmethod
    def gumbel_tx(x, a, b):
        """Gumbel return period distribution T(X)

        :param x: function input
        :type x: float | :class:`numpy.ndarray`
        :param a: distribution parameter a
        :type a: float
        :param b: distribution parameter b
        :type b: float
        :return: function output
        :rtype: float | :class:`numpy.ndarray`
        """
        g_fx = Univar.gumbel_fx(x, a, b)
        return Univar.get_tx(fx=g_fx)

    @staticmethod
    def gumbel_freqfactor(tx=2):
        """Gumbel Frequency Factor K(T)

        :param tx: return period T
        :type tx: float | :class:`numpy.ndarray`
        :return: function output
        :rtype: float | :class:`numpy.ndarray`
        """
        aux1 = tx / (tx - 1)
        aux2 = np.log(aux1)
        aux3 = np.log(aux2)
        aux4 = 0.5772 + aux3
        aux5 = -np.sqrt(6) * aux4 / np.pi
        return aux5

    @staticmethod
    def gumbel_se(std_sample, n_sample, tx):
        """Gumbel Standard Error for the MM fitted Gumbel function

        :param std_sample: sample standard deviation
        :type std_sample: float
        :param n_sample: sample size
        :type n_sample: int
        :param tx: return period T
        :type tx: float | :class:`numpy.ndarray`
        :return: function output
        :rtype: float | :class:`numpy.ndarray`
        """
        aux1 = std_sample / np.sqrt(n_sample)
        k = Univar.gumbel_freqfactor(tx=tx)
        aux2 = 1 + (1.14 * k) + (1.1 * np.square(k))
        aux3 = np.sqrt(aux2)
        return aux1 * aux3

    @staticmethod
    def empirical_px(ranks):
        """Get the empirical exceedance probability P(X)

        :param ranks: vector of ranks
        :type ranks: class:`numpy.array`
        :return: empirical exceedance probability P(X)
        :rtype: class:`numpy.array`
        """
        return ranks / len(ranks)

    @staticmethod
    def weibull_px(ranks):
        """Get the Weibull exceedance probability P(X)

        :param ranks: vector of ranks
        :type ranks: class:`numpy.array`
        :return: Weibull exceedance probability P(X)
        :rtype: class:`numpy.array`
        """
        return ranks / (len(ranks) + 1)

    @staticmethod
    def gringorten_px(ranks):
        """Get the Gringorten exceedance probability P(X)

        :param ranks: vector of ranks
        :type ranks: class:`numpy.array`
        :return: Gringorten exceedance probability P(X)
        :rtype: class:`numpy.array`
        """
        return (ranks - 0.44) / (len(ranks) + 0.12)

    @staticmethod
    def nbins_fd(data):
        """This function computes the number of bins for histograms using the Freedman-Diaconis rule, which takes into
        account the interquartile range (IQR) of the sample, in addition to its range.

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :return: number of bins for histogram using the Freedman-Diaconis rule
        :rtype: int
        """
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        binsize = 2 * iqr * len(data) ** (-1 / 3)
        # hack for non infinite values
        if binsize == 0:
            binsize = 100
        return int(np.ceil((max(data) - min(data)) / binsize))

    @staticmethod
    def nbins_sturges(data):
        """This function computes the number of bins using the Sturges rule, which assumes that the data follows a normal distribution and computes the number of bins based on its data runsize.

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :return: number of bins using the Sturges rule
        :rtype: int
        """
        return int(np.ceil(np.log2(len(data)) + 1))

    @staticmethod
    def nbins_scott(data):
        """This function computes the number of bins using the Scott rule,
        which is similar to the Freedman-Diaconis rule, but uses the standard deviation
        of the data to compute the bin runsize.

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :return: number of bins using the Scott rule
        :rtype: int
        """
        binsize = 3.5 * np.std(data) * len(data) ** (-1 / 3)
        return int(np.ceil((max(data) - min(data)) / binsize))

    @staticmethod
    def nbins_by_rule(data, rule=None):
        """Util function for rule-based nbins computation

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :param rule: rule code (sturges, fd, scott)
        :type rule: str
        :return: number of bins for histogram
        :rtype: int
        """
        bins = 10
        if rule is None:
            pass
        else:
            if rule.lower() == "sturges":
                bins = Univar.nbins_sturges()
            elif rule.lower() == "fd":
                bins = Univar.nbins_fd()
            elif rule.lower() == "scott":
                bins = Univar.nbins_scott()
            else:
                pass
        return bins

    @staticmethod
    def histogram(data, bins=100, rule=None):
        """Compute the histogram of the sample

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :param bins: number of bins
        :type bins: int
        :param rule: rule to define the number of bins. If 'none', uses bins parameters.
        :type rule: str
        :return: dataframe of histogram
        :rtype: :class:`pandas.DataFrame`
        """

        if rule is None:
            pass
        else:
            bins = Univar.nbins_by_rule(rule=rule)
        # compute histogram
        vct_hist, vct_bins = np.histogram(data, bins=bins)
        # get dataframe
        df_hist = pd.DataFrame({"Upper value": vct_bins[1:], "Count": vct_hist})
        return df_hist

    @staticmethod
    def qqplot(data):
        """Calculate the QQ-plot of data against normal distribution

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :return: dataframe of QQ plot
        :rtype: :class:`pandas.DataFrame`
        """
        from scipy.stats import norm

        # process quantiles
        _df = pd.DataFrame(
            {
                "Data": np.sort(data),
                "E-Quantiles": (np.arange(1, len(data) + 1) - 0.5) / len(data),
            }
        )
        # get theoretical
        _df["T-Quantiles"] = norm.ppf(_df["E-Quantiles"])
        return _df

    @staticmethod
    def trace_variance(data):
        """Trace the mean variance from sample

        :param data: vector of data without nan values
        :type data: :class:`numpy.ndarray`
        :return: dataframe of accumulated variance
        :rtype: :class:`pandas.DataFrame`
        """
        vct_variance_mean = np.zeros(len(data))
        for i in range(2, len(data)):
            vct_variance_mean[i] = np.var(data[:i])

        # process quantiles
        _df = pd.DataFrame(
            {
                "Data": data,
                "Variance": vct_variance_mean,
            }
        )

        return _df


class Bivar:
    """
    The Bivariate analyst base_object

    """

    def __init__(self, df_data, x_name="x", y_name="y", name="myvars"):

        # set input attributes
        self.xname = x_name
        self.yname = y_name
        self.name = name

        # set sorted data and reset index
        df_data = df_data[[x_name, y_name]]
        df_data = df_data.dropna()
        self.data = df_data.sort_values(by=self.xname).reset_index(drop=True)

        # models setup
        self.models = {
            "Linear": {
                "Function": linear,
                "Formula": "f(x) = c0 + c1 * x",
                "Setup": pd.DataFrame(
                    {"Parameters": ["c_0", "c_1"], "Mean": [0, 1], "SD": [0.1, 0.1]}
                ),
                "Data": None,
                "RMSE": None,
            },
            "Power": {
                "Function": power,
                "Formula": "f(x) =  c2 * ((x + c0)^c1)",
                "Setup": pd.DataFrame(
                    {
                        "Parameters": ["c_0", "c_1", "c_2"],
                        "Mean": [0, 1, 1],
                        "SD": [0.1, 0.1, 0.1],
                    }
                ),
                "Data": None,
                "RMSE": None,
            },
            "Power_zero": {
                "Function": power_zero,
                "Formula": "f(x) =  c1 * (x^c0)",
                "Setup": pd.DataFrame(
                    {"Parameters": ["c_0", "c_1"], "Mean": [1, 1], "SD": [0.1, 0.1]}
                ),
                "Data": None,
                "RMSE": None,
            },
            "Gumbel": {
                "Function": gumbel,
                "Formula": "f(x) =  exp(-exp(-((x-a)/b)))",
                "Setup": pd.DataFrame(
                    {"Parameters": ["a", "b"], "Mean": [1, 1], "SD": [0.1, 0.1]}
                ),
                "Data": None,
                "RMSE": None,
            },
        }

    def fit(self, model_type="Linear"):
        """Fit model to bivariate object

        :param model_type: model type. options: Linear, Power, Power_zero
        :type model_type: str
        :return: None
        :rtype: None
        """
        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(
            f=self.models[model_type]["Function"],
            xdata=self.data[self.xname],
            ydata=self.data[self.yname],
            p0=self.models[model_type]["Setup"]["Mean"].values,
        )
        pstd = np.sqrt(np.diag(pcov))
        # update model
        self.update_model(params_mean=popt, params_sd=pstd, model_type=model_type)
        return None

    def update_model(self, params_mean, params_sd=None, model_type="Linear"):
        """Update model based on parameters

        :param params_mean: list of mean of parameters
        :type params_mean: list
        :param params_sd: list of standard deviation of parameters
        :type params_sd: list
        :param model_type: model type. options: Linear, Power, Power_zero
        :type model_type: str
        :return: None
        :rtype: None
        """
        # Setup
        self.models[model_type]["Setup"]["Mean"] = params_mean
        if params_sd is None:
            pass
        else:
            self.models[model_type]["Setup"]["SD"] = params_sd
        # update sample
        self.updata_model_data(model_type=model_type)
        # update model metrics
        vct_e = self.models[model_type]["Data"]["e_Mean"].values
        self.models[model_type]["RMSE"] = np.square(np.mean(np.square(vct_e)))
        return None

    def updata_model_data(self, model_type="Linear"):
        """Update only model data output

        :param model_type: model type. options: Linear, Power, Power_zero
        :type model_type: str
        :return: None
        :rtype: None
        """
        popt = self.models[model_type]["Setup"]["Mean"].values
        _df = self.data.copy()
        s_ymodel = "{}_Mean".format(self.yname)
        # compute model on sample:
        _df[s_ymodel] = self.models[model_type]["Function"](
            self.data[self.xname], *popt
        )
        # compute error
        _df["e_Mean".format(model_type)] = _df[s_ymodel] - _df[self.yname]
        # set attribute
        self.models[model_type]["Data"] = _df.copy()
        del _df
        return None

    def view(
        self,
        show=True,
        folder="C:/sample",
        filename="view",
        specs=None,
        fig_format="jpg",
        dpi=300,
    ):
        """Plot basic view of Bivar base_object

        :param show: Boolean to show instead of saving
        :type show: bool
        :param folder: output folder
        :type folder: str
        :param filename: image file name
        :type filename: str
        :param specs: specification dictionary
        :type specs: dict
        :param dpi: image resolution (default = 300)
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """

        # get specs
        default_specs = {
            "color": "tab:grey",
            "color_scatter": "tab:blue",
            "title": "View of {}".format(self.name),
            "width": 6,
            "height": 6,
            "xlim": [self.data[self.xname].min(), self.data[self.xname].max()],
            "ylim": [self.data[self.yname].min(), self.data[self.yname].max()],
            "xlabel": self.xname,
            "ylabel": self.yname,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # start plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        plt.suptitle(specs["title"])
        # grid
        gs = mpl.gridspec.GridSpec(
            3, 3, wspace=0.3, hspace=0.3, left=0.12, bottom=0.1, top=0.90, right=0.95
        )  # nrows, ncols

        # scatter plot
        ax = fig.add_subplot(gs[1:, :2])
        plt.scatter(
            self.data[self.xname],
            self.data[self.yname],
            marker=".",
            color=specs["color_scatter"],
        )
        plt.xlabel(specs["xlabel"])
        plt.ylabel(specs["ylabel"])
        plt.xlim(specs["xlim"])
        plt.ylim(specs["ylim"])

        # x hist
        ax_histx = fig.add_subplot(gs[0, :2], sharex=ax)
        plt.ylabel("p({})".format(self.xname))
        xuni = Univar(data=self.data[self.xname].values)
        plt.hist(
            self.data[self.xname],
            bins=xuni.nbins_fd(),
            color=specs["color"],
            alpha=1,
            weights=np.ones(len(self.data)) / len(self.data),
        )
        plt.xlim(specs["xlim"])

        # y hist
        ay_histy = fig.add_subplot(gs[1:, 2], sharey=ax)
        plt.xlabel("p({})".format(self.yname))
        yuni = Univar(data=self.data[self.yname].values)
        plt.hist(
            yuni.data,
            bins=yuni.nbins_fd(),
            color=specs["color"],
            alpha=1,
            orientation="horizontal",
            weights=np.ones(len(self.data)) / len(self.data),
        )
        plt.ylim(specs["ylim"])

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = self.name
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
        plt.close(fig)

    def view_model(
        self,
        model_type="Power",
        show=True,
        folder="C:/sample",
        filename=None,
        specs=None,
        dpi=300,
        fig_format="jpg",
    ):
        """Plot pannel for model analysis

        :param show: Boolean to show instead of saving
        :type show: bool
        :param folder: output folder
        :type folder: str
        :param filename: image file name
        :type filename: str
        :param specs: specification dictionary
        :type specs: dict
        :param dpi: image resolution (default = 300)
        :type dpi: int
        :param fig_format: image fig_format (ex: png or jpg). Default jpg
        :type fig_format: str
        :return: None
        :rtype: None
        """

        # ensure data is update
        self.updata_model_data(model_type=model_type)

        # get specs
        default_specs = {
            "color": "tab:grey",
            "color_scatter": "tab:blue",
            "color_model": "black",
            "color_variance": "darkred",
            "title": "View of {} {} Model".format(self.name, model_type),
            "width": 8,
            "height": 8,
            "xlim": [self.data[self.xname].min(), self.data[self.xname].max()],
            "ylim": [self.data[self.yname].min(), self.data[self.yname].max()],
            "elim": [
                -1.5 * self.models[model_type]["Data"]["e_Mean"].max(),
                1.5 * self.models[model_type]["Data"]["e_Mean"].max(),
            ],
            "alpha_xy": 0.75,
            "alpha_e": 0.75,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # get some ranges
        e_range = specs["elim"][1] - specs["elim"][0]
        e_range_10 = e_range / 20
        x_range = specs["xlim"][1] - specs["xlim"][0]
        x_range_10 = x_range / 10

        # -------------- start plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        plt.suptitle(specs["title"])
        # grid
        gs = mpl.gridspec.GridSpec(
            4, 4, wspace=0.3, hspace=0.45, left=0.12, bottom=0.1, top=0.90, right=0.95
        )  # nrows, ncols

        # ----------------- XY plot -----------------
        ax = fig.add_subplot(gs[:2, :2])

        plt.scatter(
            self.data[self.xname],
            self.data[self.yname],
            marker=".",
            color=specs["color_scatter"],
            alpha=specs["alpha_xy"],
            zorder=1,
        )
        plt.plot(
            self.models[model_type]["Data"][self.xname],
            self.models[model_type]["Data"]["{}_Mean".format(self.yname)],
            color=specs["color_model"],
            zorder=2,
            linestyle="--",
        )
        plt.xlabel(self.xname)
        plt.ylabel(self.yname)
        plt.xlim(specs["xlim"])
        plt.ylim(specs["ylim"])

        # ----------------- error -----------------
        e_uni = Univar(data=self.models[model_type]["Data"]["e_Mean"].values)
        # get mean
        e_mean = np.mean(e_uni.data)

        ax = fig.add_subplot(gs[2, :2])
        plt.scatter(
            self.models[model_type]["Data"][self.xname],
            self.models[model_type]["Data"]["e_Mean"].values,
            marker=".",
            alpha=specs["alpha_e"],
            color=specs["color"],
        )
        # mean line
        plt.hlines(
            y=e_mean,
            xmin=specs["xlim"][0],
            xmax=specs["xlim"][1],
            color="tab:red",
            alpha=0.4,
        )
        plt.text(
            y=e_mean + e_range_10,
            x=specs["xlim"][1] - (3 * x_range_10),
            s="$\mu$: " + str(round(e_mean, 2)),
            color="tab:red",
        )
        plt.xlabel(self.xname)
        plt.ylabel("$\epsilon$")
        plt.xlim(specs["xlim"])
        plt.ylim(specs["elim"])

        # ----------------- error hist -----------------
        ax = fig.add_subplot(gs[2, 2])
        plt.hist(
            e_uni.data,
            bins=e_uni.nbins_fd(),
            color=specs["color"],
            alpha=1,
            orientation="horizontal",
            weights=np.ones(len(self.data)) / len(self.data),
        )
        # mean line
        plt.hlines(y=e_mean, xmin=0, xmax=0.15, color="tab:red", alpha=0.4)
        plt.ylim(specs["elim"])
        plt.xlabel("p($\epsilon$)")

        # ----------------- qq plot error -----------------
        df_qq = e_uni.qqplot()
        ax = fig.add_subplot(gs[2, 3])
        plt.scatter(
            x=df_qq["T-Quantiles"],
            y=df_qq["Data"],
            marker=".",
            alpha=specs["alpha_e"],
            color=specs["color"],
        )
        # mean line
        plt.hlines(
            y=e_mean,
            xmin=df_qq["T-Quantiles"].min(),
            xmax=df_qq["T-Quantiles"].max(),
            color="tab:red",
            alpha=0.4,
        )
        plt.ylim(specs["elim"])
        plt.xlabel("normal quantiles")

        # ----------------- variance tracing -----------------
        ax = fig.add_subplot(gs[3, :2])
        vct_var = e_uni.trace_variance()
        plt.plot(
            self.models[model_type]["Data"][self.xname],
            vct_var,
            color=specs["color_variance"],
        )

        plt.xlabel(self.xname)
        plt.ylabel("$\sigma^2$")
        plt.xlim(specs["xlim"])
        plt.ylim([0, 1.5 * np.max(vct_var)])

        # show or save
        if show:
            plt.show()
        else:
            if filename is None:
                filename = self.name
            plt.savefig("{}/{}.{}".format(folder, filename, fig_format), dpi=dpi)
        plt.close(fig)
        return None

    def correlation(self):
        """Compute the R correlation coefficient of the base_object

        :return: R correlation coefficient
        :rtype: float
        """
        corr_df = self.data.corr().loc[self.xname, self.yname]
        return corr_df

    def prediction_bands(
        self, lst_bounds=None, n_sim=100, n_grid=100, n_seed=None, p0=None
    ):
        """Run Monte Carlo Simulation to get prediciton bands

        :param lst_bounds: list of prediction bounds [min, max], if None, 3x the date range is used
        :type lst_bounds: list
        :param n_sim: number of simulation runs
        :type n_sim: int
        :param n_grid: number of prediction intervals
        :type n_grid: int
        :param n_seed: number of random seed for reproducibility. Default = None
        :type n_seed: int
        :param p0: list of initial values to search. Default: None
        :type p0: list
        :return: base_object with result dataframes
        :rtype: dict
        """
        from scipy.optimize import curve_fit

        # todo this seems to be deprecated
        # handle None bounds
        if lst_bounds is None:
            _min = self.data[self.xname].min()
            _max = self.data[self.xname].max()
            # extrapolation zone 2x the interpolation zone
            lst_bounds = [_min, _max + (2 * _min)]
        # handle None seed
        if n_seed is None:
            from datetime import datetime

            _seed = int(datetime.now().timestamp())
            n_seed = np.random.seed(_seed)

        # --- set dataframes
        # scale of simulations
        n_scale = int(np.log10(n_sim)) + 1

        # simulation labels
        lst_mcsim = ["MC{}".format(str(i).zfill(n_scale)) for i in range(n_sim + 1)]
        s_refsim = lst_mcsim[0]

        # parameter labels
        _lst_params = list(self.linear_model["Parameter"].values)
        _labels = ["Mean", "Std"]
        lst_params = list()
        for i in range(len(_labels)):
            _label = _labels[i]
            for j in range(len(_lst_params)):
                _param = _lst_params[j]
                lst_params.append("{}_{}".format(_param, _label))

        # models dataframe
        _cols = lst_params.copy()
        _cols.insert(0, "Model")
        _data = np.zeros(shape=(n_sim + 1, len(_cols)))
        df_models = pd.DataFrame(data=_data, columns=_cols)
        df_models["Model"] = lst_mcsim

        # prediction dataframe
        _cols = ["{}_{}".format(self.yname, s) for s in lst_mcsim]
        _cols.insert(0, self.xname)
        _data = np.zeros(shape=(n_grid, len(_cols)))
        df_preds = pd.DataFrame(data=_data, columns=_cols)
        _min = np.min(lst_bounds)
        _max = np.max(lst_bounds)
        df_preds[self.xname] = np.linspace(_min, _max, n_grid)

        # bands dataframe
        _cols = ["Min", "Max", "Mean", "p01", "p05", "p25", "p50", "p75", "p95", "p99"]
        _cols.insert(0, self.xname)
        _data = np.zeros(shape=(n_grid, len(_cols)))
        df_bands = pd.DataFrame(data=_data, columns=_cols)
        df_bands[self.xname] = df_preds[self.xname].values

        # set reference simulation parameters
        _lst = ["Mean", "Std"]
        for s in _lst:
            for i in range(len(self.linear_model)):
                _s_par = self.linear_model["Parameter"].values[i]
                _s_key = "{}_{}".format(_s_par, s)
                _value = self.linear_model["Fit_{}".format(s)].values[i]
                df_models[_s_key].values[0] = _value

        # set reference simulation prediction
        _x = df_preds[self.xname].values
        _params = self.linear_model["Fit_Mean"].values
        df_preds["y_{}".format(s_refsim)] = linear(_x, *_params)

        # get standard deviation from fit sample
        n_std_e = self.linear_model_data["e"].std()
        vct_yfit = self.linear_model_data["{}_fit".format(self.yname)].values

        # simulation loop:
        for n in range(1, n_sim + 1):
            _sim = lst_mcsim[n]

            # get new error
            vct_e = np.random.normal(0.0, n_std_e, len(self.linear_model_data))
            # get new y obs
            vct_yobs = vct_yfit + vct_e

            # fit new model
            if p0 is None:
                popt, pcov = curve_fit(
                    f=linear, xdata=self.linear_model_data[self.xname], ydata=vct_yobs
                )
            else:
                popt, pcov = curve_fit(
                    f=linear,
                    xdata=self.linear_model_data[self.xname],
                    ydata=vct_yobs,
                    p0=p0,
                )
            pstd = np.sqrt(np.diag(pcov))

            # store model values
            _dct = {"Mean": popt, "Std": pstd}
            for s in _dct:
                for i in range(len(self.linear_model)):
                    _s_par = self.linear_model["Parameter"].values[i]
                    _s_key = "{}_{}".format(_s_par, s)
                    _value = _dct[s][i]
                    df_models[_s_key].values[n] = _value

            # store prediction values
            _x = df_preds[self.xname].values
            df_preds["y_{}".format(_sim)] = linear(_x, *popt)

        # get bands
        for i in range(len(df_preds)):
            _values = df_preds.values[i, 1:]
            df_bands["Min"].values[i] = np.min(_values)
            df_bands["Max"].values[i] = np.max(_values)
            df_bands["Mean"].values[i] = np.mean(_values)
            df_bands["p01"].values[i] = np.quantile(_values, 0.01)
            df_bands["p05"].values[i] = np.quantile(_values, 0.05)
            df_bands["p25"].values[i] = np.quantile(_values, 0.25)
            df_bands["p50"].values[i] = np.quantile(_values, 0.5)
            df_bands["p75"].values[i] = np.quantile(_values, 0.75)
            df_bands["p95"].values[i] = np.quantile(_values, 0.95)
            df_bands["p99"].values[i] = np.quantile(_values, 0.99)

        # return base_object
        return {"Models": df_models, "Predictions": df_preds, "Bands": df_bands}

    def assess_error_normality(self, model_type="Linear", clevel=0.95):
        self.updata_model_data(model_type=model_type)
        vct_e = self.models[model_type]["Data"]["e_Mean"].values
        uni = Univar(data=vct_e, name=self.name)
        df = uni.assess_normality(clevel=clevel)
        return df

    @staticmethod
    def bias(pred, obs):
        """Compute the Bias between predicted and observated sample

        :param pred: :class:`numpy.ndarray`
            vector of prediction vales
        :type pred: :class:`numpy.ndarray`
        :param obs: :class:`numpy.ndarray`
            vector of observed values
        :type obs: :class:`numpy.ndarray`
        :return: Bias value
        :rtype: float
        """
        return 100 * np.sum(pred - obs) / np.sum(obs)

    @staticmethod
    def rmse(pred, obs):
        """Compute the RMSE metric between predicted and observated sample

        :param pred: :class:`numpy.ndarray`
            vector of prediction vales
        :type pred: :class:`numpy.ndarray`
        :param obs: :class:`numpy.ndarray`
            vector of observed values
        :type obs: :class:`numpy.ndarray`
        :return: RMSE value
        :rtype: float
        """
        return np.sqrt(np.sum(np.power(pred - obs, 2)) / len(pred))

    @staticmethod
    def mae(pred, obs):
        """Compute the Mean Absolute Error (MAE) between predicted and observated sample

        :param pred: :class:`numpy.ndarray`
            vector of prediction vales
        :type pred: :class:`numpy.ndarray`
        :param obs: :class:`numpy.ndarray`
            vector of observed values
        :type obs: :class:`numpy.ndarray`
        :return: MAE value
        :rtype: float
        """
        return np.sum(np.abs(pred - obs)) / len(pred)

    @staticmethod
    def rsq(pred, obs):
        """Compute the R-square between predicted and observated sample

        :param pred: :class:`numpy.ndarray`
            vector of prediction vales
        :type pred: :class:`numpy.ndarray`
        :param obs: :class:`numpy.ndarray`
            vector of observed values
        :type obs: :class:`numpy.ndarray`
        :return: R-square value
        :rtype: float
        """
        return 1 - (
            np.sum(np.power(pred - obs, 2)) / np.sum(np.power(obs - np.mean(obs), 2))
        )


class Bayes:
    """
    The Bayes Theorem Analyst Object
    """

    def __init__(self, df_hypotheses, name="myBayes", nomenclature=None, gridsize=100):
        """Deploy the Bayes Analyst
        :param df_hypotheses: dataframe listing all model hypotheses. Must contain a field Name (of parameter), Min and Max.
        :type df_hypotheses: :class:pandas.DataFrame
        :param name: name of analyst
        :type name: str
        :param nomenclature: dictionary for rename nomenclature
        :type nomenclature: dict
        :param gridsize: grid resolution in histrograms (bins)
        :type gridsize: int
        """
        self.hypotheses = df_hypotheses
        # list of hypotheses by step
        self.bands = list()
        self.bands.append(self.hypotheses.copy())
        # other parameters
        self.name = name
        self.gridsize = gridsize

        # zero
        self.zero = 1 / gridsize

        # set labels
        self.shyp = "H"
        self.sevid = "E"
        self.sprior = "P(H)"
        self.slike = "P(E | H)"
        self.spost = "P(H | E)"
        self.saux = self.sprior + self.slike
        if nomenclature is not None:
            self._reset_nomenclature(dct_names=nomenclature)

        # set omega
        self.steps = list()
        # create step 0
        self._insert_new_step()

    def __str__(self):
        s1 = self.hypotheses.to_string()
        return s1

    def _reset_nomenclature(self, dct_names):
        """Reset nomenclature
        :param dct_names: dictionary to rename nomenclatures
        :type dct_names: dict
        :return: None
        :rtype: None
        """
        for k in dct_names:
            self.sevid = self.sevid.replace(k, dct_names[k])
            self.shyp = self.shyp.replace(k, dct_names[k])
            self.sprior = self.sprior.replace(k, dct_names[k])
            self.slike = self.slike.replace(k, dct_names[k])
            self.spost = self.spost.replace(k, dct_names[k])
            self.saux = self.sprior + self.slike

    def _insert_new_step(self):
        """convenience void function for inserting new step objects
        :return: None
        :rtype:
        """
        self.steps.append(dict())
        self.steps[len(self.steps) - 1]["Omega"] = dict()
        self.steps[len(self.steps) - 1]["Evidence"] = dict()
        self.steps[len(self.steps) - 1]["Bands"] = dict()
        for h in self.hypotheses["Name"].values:
            n_min = self.hypotheses.loc[self.hypotheses["Name"] == h, "Min"].values[0]
            n_max = self.hypotheses.loc[self.hypotheses["Name"] == h, "Max"].values[0]
            self.steps[len(self.steps) - 1]["Omega"][h] = pd.DataFrame(
                {
                    self.shyp: np.linspace(n_min, n_max, self.gridsize),
                    self.sprior: np.ones(self.gridsize) / self.gridsize,
                    self.slike: np.zeros(self.gridsize),
                    self.saux: np.zeros(self.gridsize),
                    self.spost: np.zeros(self.gridsize),
                }
            )
        return None

    def _accumulate(self, n_step):
        """convenience void function for accumulate probability
        :param n_step: step number to accumulate
        :type n_step: int
        :return: None
        :rtype: none
        """
        lst_labels1 = [self.sprior, self.slike, self.spost]
        for h in self.hypotheses["Name"].values:
            for i in range(len(lst_labels1)):
                s_field0 = lst_labels1[i]
                s_field1 = "{}_acc".format(lst_labels1[i])
                self.steps[n_step]["Omega"][h][s_field1] = 0.0
                for j in range(len(self.steps[n_step]["Omega"][h])):
                    if j == 0:
                        self.steps[n_step]["Omega"][h][s_field1].values[j] = self.steps[
                            n_step
                        ]["Omega"][h][s_field0].values[j]
                    else:
                        self.steps[n_step]["Omega"][h][s_field1].values[j] = (
                            self.steps[n_step]["Omega"][h][s_field1].values[j - 1]
                            + self.steps[n_step]["Omega"][h][s_field0].values[j]
                        )
        return None

    def conditionalize(self, dct_evidence, s_varfield="E", s_weightfield="W"):
        """Conditionalize procedure of the Bayes Theorem
        :param dct_evidence: base_object of evidence dataframes
        :type dct_evidence: dict
        :param s_varfield: name of variable field in evidence dataframes
        :type s_varfield: str
        :param s_weightfield: name of weights field in evidence dataframes
        :type s_weightfield: str
        :return: None
        :rtype: none
        """
        # instantiate new step
        n_step = len(self.steps)
        self._insert_new_step()

        # set evidence dict for step
        self.steps[n_step]["Evidence"] = dct_evidence

        # main loop
        for h in self.hypotheses["Name"].values:
            # get prior from last step
            if n_step > 1:
                self.steps[n_step]["Omega"][h][self.sprior] = self.steps[n_step - 1][
                    "Omega"
                ][h][self.spost].values
            # get likelihood
            hist, edges = np.histogram(
                a=dct_evidence[h][s_varfield],
                bins=self.steps[n_step]["Omega"][h][self.shyp],
                weights=dct_evidence[h][s_weightfield],
            )
            hist = hist / np.sum(hist)
            hist = list(hist)
            hist.append(0)
            self.steps[n_step]["Omega"][h][self.slike] = hist
            # insert baseline and normalize
            v_like = self.steps[n_step]["Omega"][h][self.slike].values
            v_like = v_like + self.zero
            self.steps[n_step]["Omega"][h][self.slike] = v_like / np.sum(v_like)

            # get posterior with Bayes Theorem
            v_prior = self.steps[n_step]["Omega"][h][self.sprior].values
            v_like = self.steps[n_step]["Omega"][h][self.slike].values
            v_posterior = (v_prior * v_like) / np.sum(v_prior * v_like)
            self.steps[n_step]["Omega"][h][self.spost] = v_posterior

            # get accumulated values
            self._accumulate(n_step)

            self.steps[n_step]["Bands"][h] = dict()

            # get percentiles of accumulated
            lst_aux = [self.sprior, self.slike, self.spost]
            for s in lst_aux:
                self.steps[n_step]["Bands"][h][s] = dict()
                # 90% band
                s_query = "`{}` > 0.05 and `{}` <= 0.95".format(s + "_acc", s + "_acc")
                df_aux = self.steps[n_step]["Omega"][h].query(s_query)
                self.steps[n_step]["Bands"][h][s]["p05"] = df_aux[self.shyp].min()
                self.steps[n_step]["Bands"][h][s]["p95"] = df_aux[self.shyp].max()
                # 50% band
                s_query = "`{}` > 0.25 and `{}` <= 0.75".format(s + "_acc", s + "_acc")
                df_aux = self.steps[n_step]["Omega"][h].query(s_query)
                self.steps[n_step]["Bands"][h][s]["p25"] = df_aux[self.shyp].min()
                self.steps[n_step]["Bands"][h][s]["p75"] = df_aux[self.shyp].max()

        # return
        return None

    def plot_step(
        self,
        n_step,
        folder="C:/sample",
        filename="bayes",
        specs=None,
        dpi=300,
        show=False,
    ):
        """Void function for plot pannel of conditionalization step
        :param n_step: step number
        :type n_step: int
        :param folder: export folder
        :type folder: str
        :param filename: file name
        :type filename: str
        :param specs: plot specs base_object
        :type specs: dict
        :param dpi: plot resolution
        :type dpi: int
        :param show: control to show plot instead of saving
        :type show: bool
        :return: None
        :rtype: None
        """
        # get specs
        default_specs = {
            "color": "tab:grey",
            "title": "Conditionalization",
            "width": 10,
            "height": 5,
            "xlabel": "value",
            "ylim": (0, 1),
            "xlim": (0, 1),
            "subtitle": None,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # hunt some parameters
        lst_pmax = list()
        for h in self.hypotheses["Name"].values:  # get objects
            _df = self.steps[n_step]["Omega"][h]
            # get max p
            lst_pmax.append(
                np.max(
                    [
                        _df[self.sprior].max(),
                        _df[self.slike].max(),
                        _df[self.spost].max(),
                    ]
                )
            )
        n_pmax = 1.5 * np.max(lst_pmax)

        # main loop
        for h in self.hypotheses["Name"].values:
            specs["subtitle"] = "${}$".format(h)
            # get objects
            _df = self.steps[n_step]["Omega"][h]
            _bands = self.steps[n_step]["Bands"][h]

            # get width
            _wid = 0.7 * (_df[self.shyp].values[1] - _df[self.shyp].values[0])

            # get min max
            n_min = self.hypotheses.loc[self.hypotheses["Name"] == h, "Min"].values[0]
            n_max = self.hypotheses.loc[self.hypotheses["Name"] == h, "Max"].values[0]
            # start plot

            fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

            if specs["subtitle"] is None:
                plt.suptitle(specs["title"])
            else:
                plt.suptitle("{} | {}".format(specs["title"], specs["subtitle"]))
            # grid
            gs = mpl.gridspec.GridSpec(
                2, 3, wspace=0.4, hspace=0.5, left=0.1, bottom=0.1, top=0.85, right=0.95
            )  # nrows, ncols

            # grid loop
            lst_aux = [self.sprior, self.slike, self.spost]
            for i in range(len(lst_aux)):
                # Histogram
                ax = fig.add_subplot(gs[0, i])
                plt.title(lst_aux[i])
                plt.bar(
                    _df[self.shyp],
                    _df[lst_aux[i]],
                    width=_wid,
                    color=specs["color"],
                    align="edge",
                )
                plt.ylim(0, n_pmax)
                plt.xlim(n_min, n_max)
                plt.ylabel("$p({})$".format(h))
                plt.xlabel("${}$".format(h))
                # plt.gca().set_aspect(1 / (1.5))

                if i == 1:
                    # Accumulated
                    ax = fig.add_subplot(gs[1, i])
                    # plt.title(lst_aux[i])
                    plt.scatter(
                        self.steps[n_step]["Evidence"][h]["E"],
                        self.steps[n_step]["Evidence"][h]["W"],
                        marker=".",
                        c="k",
                        alpha=0.5,
                        linewidths=0.0,
                        edgecolors=None,
                    )
                    plt.xlim(n_min, n_max)
                    plt.ylabel("$W({})$".format(h))
                    plt.xlabel("${}$".format(h))
                    # plt.gca().set_aspect(1 / (1.5))
                else:
                    # Accumulated
                    ax = fig.add_subplot(gs[1, i])
                    # plt.title(lst_aux[i])
                    plt.plot(
                        _df[self.shyp],
                        _df["{}_acc".format(lst_aux[i])],
                        color="tab:blue",
                        zorder=3,
                    )
                    plt.fill_betweenx(
                        [-1, 2],
                        x1=self.steps[n_step]["Bands"][h][lst_aux[i]]["p25"],
                        x2=self.steps[n_step]["Bands"][h][lst_aux[i]]["p75"],
                        color="tab:grey",
                        alpha=0.5,
                        zorder=2,
                    )
                    plt.fill_betweenx(
                        [-1, 2],
                        x1=self.steps[n_step]["Bands"][h][lst_aux[i]]["p05"],
                        x2=self.steps[n_step]["Bands"][h][lst_aux[i]]["p95"],
                        color="tab:grey",
                        alpha=0.3,
                        zorder=1,
                    )
                    plt.ylim(-0.05, 1.05)
                    plt.xlim(n_min, n_max)
                    plt.ylabel("$P({})$".format(h))
                    plt.xlabel("${}$".format(h))
                    # plt.gca().set_aspect(1 / (1.5))

            # show or save
            if show:
                plt.show()
            else:
                plt.savefig(
                    "{}/{}_{}_step{}.png".format(folder, filename, h, n_step), dpi=dpi
                )

        # return
        return None


# ----- DEPRECATED CLASSES -----


class _Univar:
    """[Deprecated]
    The Univariate Analyst Object
    """

    def __init__(self, data, name="myvar"):
        """Deploy the Analyst

        :param data: n-D vector of sample
        :type data: :class:`numpy.ndarray`
        """
        self.data = data
        self.name = name

    def nbins_fd(self):
        """This function computes the number of bins for histograms using the Freedman-Diaconis rule, which takes into account the interquartile range (IQR) of the sample, in addition to its range.

        :return: number of bins for histogram using the Freedman-Diaconis rule
        :rtype: int
        """
        iqr = np.subtract(*np.percentile(self.data, [75, 25]))
        binsize = 2 * iqr * len(self.data) ** (-1 / 3)
        # hack for non infinite values
        if binsize == 0:
            binsize = 100
        return int(np.ceil((max(self.data) - min(self.data)) / binsize))

    def nbins_sturges(self):
        """This function computes the number of bins using the Sturges rule, which assumes that the data follows a normal distribution and computes the number of bins based on its data runsize.

        :return: number of bins using the Sturges rule
        :rtype: int
        """
        return int(np.ceil(np.log2(len(self.data)) + 1))

    def nbins_scott(self):
        """This function computes the number of bins using the Scott rule,
        which is similar to the Freedman-Diaconis rule, but uses the standard deviation
        of the data to compute the bin runsize.

        :return: number of bins using the Scott rule
        :rtype: int
        """
        binsize = 3.5 * np.std(self.data) * len(self.data) ** (-1 / 3)
        return int(np.ceil((max(self.data) - min(self.data)) / binsize))

    def nbins_by_rule(self, rule=None):
        """Util function for rule-based nbins computation
        :param rule: rule code (sturges, fd, scott)
        :type rule: str
        :return: number of bins for histogram
        :rtype: int
        """
        bins = 10
        if rule is None:
            pass
        else:
            if rule.lower() == "sturges":
                bins = self.nbins_sturges()
            elif rule.lower() == "fd":
                bins = self.nbins_fd()
            elif rule.lower() == "scott":
                bins = self.nbins_scott()
            else:
                pass
        return bins

    def histogram(self, bins=100, rule=None):
        """Compute the histogram of the sample

        :param bins: number of bins
        :type bins: int
        :param rule: rule to define the number of bins. If 'none', uses bins parameters.
        :type rule: str
        :return: dataframe of histogram
        :rtype: :class:`pandas.DataFrame`
        """

        if rule is None:
            pass
        else:
            bins = self.nbins_by_rule(rule=rule)
        # compute histogram
        vct_hist, vct_bins = np.histogram(self.data, bins=bins)
        # get dataframe
        df_hist = pd.DataFrame({"Upper value": vct_bins[1:], "Count": vct_hist})
        return df_hist

    def qqplot(self):
        """Calculate the QQ-plot of data against normal distribution

        :return: dataframe of QQ plot
        :rtype: :class:`pandas.DataFrame`
        """
        from scipy.stats import norm

        # process quantiles
        _df = pd.DataFrame(
            {
                "Data": np.sort(self.data),
                "E-Quantiles": (np.arange(1, len(self.data) + 1) - 0.5)
                / len(self.data),
            }
        )
        # get theoretical
        _df["T-Quantiles"] = norm.ppf(_df["E-Quantiles"])
        return _df

    def trace_variance(self):
        """Trace the mean variance from sample

        :return: vector of accumulated variance
        :rtype: :class:`numpy.ndarray`
        """
        vct_variance_mean = np.zeros(len(self.data))
        for i in range(2, len(self.data)):
            vct_variance_mean[i] = np.var(self.data[:i])
        return vct_variance_mean

    def plot_hist(
        self,
        bins=100,
        colored=False,
        annotated=False,
        rule=None,
        show=False,
        folder="C:/sample",
        filename="histogram",
        specs=None,
        dpi=300,
    ):
        """Plot histogram of sample

        :param bins: number of bins
        :type bins: int
        :param colored: Boolean to quantile-colored histogram
        :type colored: bool
        :param annotated: Boolean to plot stats texts over histogram
        :type annotated: bool
        :param rule: name of rule to compute bins
        :type rule: str
        :param show: Boolean to show instead of saving
        :type show: bool
        :param folder: output folder
        :type folder: str
        :param filename: image file name
        :type filename: str
        :param specs: specification dictionary
        :type specs: dict
        :param dpi: image resolution (default = 96)
        :type dpi: int
        """

        # get bins
        if rule is None:
            pass
        else:
            bins = self.nbins_by_rule(rule=rule)

        # get specs
        default_specs = {
            "color": "tab:grey",
            "title": "Histogram of {}".format(self.name),
            "width": 4 * 1.618,
            "height": 4,
            "xlabel": "value",
            "ylim": (0, 0.1),
            "xlim": (0.95 * np.min(self.data), 1.05 * np.max(self.data)),
            "subtitle": None,
            "cmap": "viridis",
            "colors": None,
            "n_classes": 5,
            "grid": False,
            "quantiles": True,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # start plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        ax = plt.gca()
        ax.set_position([0.15, 0.15, 0.75, 0.75])
        if specs["subtitle"] is None:
            plt.title(specs["title"])
        else:
            plt.title("{} | {}".format(specs["title"], specs["subtitle"]))

        # Calculate quantiles
        data = pd.Series(self.data)
        # filter data to xlim
        data = data[data.between(specs["xlim"][0], specs["xlim"][1])]
        if specs["quantiles"]:
            quantiles = data.quantile(np.linspace(0, 1, specs["n_classes"] + 1))
        else:
            quantiles = pd.Series(
                np.linspace(
                    start=data.min(), stop=data.max(), num=specs["n_classes"] + 1
                )
            )

        # Determine the overall bin width you desire
        bin_width = (
            data.max() - data.min()
        ) / bins  # for example, 40 equal width bins across the data range
        # handle colored plots
        if colored:
            if specs["colors"] is None:
                cmap = mpl.colormaps[specs["cmap"]]
                colors = [cmap(i) for i in np.linspace(0, 1, len(quantiles) - 1)]
            else:
                colors = specs["colors"]

            # Create the bins
            binsv = np.arange(start=data.min(), stop=data.max(), step=bin_width)
            binsv = np.append(
                binsv, data.max()
            )  # Ensure the last bin includes the max value

            # Plot the histogram
            n, bins, patches = plt.hist(
                self.data,
                bins=binsv,
                linewidth=0,
                weights=np.ones(len(self.data)) / len(self.data),
            )

            # Color the bars based on the quantile
            for patch, edge in zip(patches, bins[:-1]):
                if edge < quantiles.values[1]:
                    plt.setp(patch, "facecolor", colors[0])
                elif edge < quantiles.values[2]:
                    plt.setp(patch, "facecolor", colors[1])
                elif edge < quantiles.values[3]:
                    plt.setp(patch, "facecolor", colors[2])
                elif edge < quantiles.values[4]:
                    plt.setp(patch, "facecolor", colors[3])
                else:
                    plt.setp(patch, "facecolor", colors[4])
        else:
            plt.hist(
                self.data,
                bins=bins,
                weights=np.ones(len(self.data)) / len(self.data),
                color=specs["color"],
            )

        plt.xlabel(specs["xlabel"])
        plt.ylim(specs["ylim"])
        plt.xlim(specs["xlim"])
        plt.grid(specs["grid"])
        # Optionally, add vertical lines for each quantile and annotate them
        mu = np.mean(self.data)
        plt.axvline(mu, color="r", linestyle="dashed", linewidth=1)
        plt.text(
            mu,
            plt.gca().get_ylim()[1] * 0.92,
            f"$\mu$ = {mu:.1f}",
            ha="left",
            va="bottom",
            fontsize=10,
            rotation=0,
            color="red",
        )
        if annotated:
            aux = 0
            for q in quantiles:
                plt.axvline(q, color="k", linestyle="dashed", linewidth=1)
                plt.text(
                    q,
                    plt.gca().get_ylim()[1] * (0.9 - aux),
                    f"{q:.1f}",
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    rotation=0,
                )
                aux = aux + 0.05
        # plt.tight_layout()
        # show or save
        if show:
            plt.show()
        else:
            plt.savefig(
                "{}/{}_{}.png".format(folder, self.name, filename),
                dpi=dpi,
                bbox_inches="tight",
            )

    def view(self, show=True, folder="C:/sample", filename="view", specs=None, dpi=300):
        """Plot basic view of sample

        :param show: Boolean to show instead of saving
        :type show: bool
        :param folder: output folder
        :type folder: str
        :param filename: image file name
        :type filename: str
        :param specs: specification dictionary
        :type specs: dct
        :param dpi: image resolution (default = 300)
        :type dpi: int
        :return: None
        :rtype: None
        """

        # get specs
        default_specs = {
            "color": "tab:grey",
            "title": "View of {}".format(self.name),
            "width": 5 * 1.618,
            "height": 4,
            "xlabel": "i",
            "ylabel": "",
            "ylim": (0.95 * np.min(self.data), 1.05 * np.max(self.data)),
            "subtitle_1": "Scatter",
            "subtitle_2": "Histogram",
        }

        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # start plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        plt.suptitle(specs["title"])
        # grid
        gs = mpl.gridspec.GridSpec(
            1, 3, wspace=0.4, hspace=0.5, left=0.1, bottom=0.25, top=0.85, right=0.95
        )  # nrows, ncols

        # scatter plot
        ax = fig.add_subplot(gs[0, :2])
        plt.scatter(np.arange(len(self.data)), self.data, marker=".", color="tab:grey")
        plt.title(specs["subtitle_1"])
        plt.ylim(specs["ylim"])
        plt.xlabel(specs["xlabel"])
        plt.ylabel(specs["ylabel"])

        # hist
        ax2 = fig.add_subplot(gs[0, 2], sharey=ax)
        plt.hist(
            self.data,
            bins=self.nbins_fd(),
            color="tab:grey",
            alpha=1,
            orientation="horizontal",
        )
        plt.title(specs["subtitle_2"])
        plt.ylim(specs["ylim"])
        plt.xlabel("Count")
        plt.ylabel(specs["ylabel"])
        # show or save
        if show:
            plt.show()
        else:
            plt.savefig(
                "{}/{}_{}.png".format(folder, self.name, filename),
                dpi=dpi,
                bbox_inches="tight",
            )

    def plot_qqplot(
        self, show=True, folder="C:/sample", filename="qqplot", specs=None, dpi=300
    ):
        """Plot Q-Q Plot on Normal distribution

        :param show: Boolean to show instead of saving
        :type show: bool
        :param folder: output folder
        :type folder: str
        :param filename: image file name
        :type filename: str
        :param specs: specification dictionary
        :type specs: dct
        :param dpi: image resolution (default = 300)
        :type dpi: int
        :return: None
        :rtype: None
        """
        # get specs
        default_specs = {
            "color": "tab:grey",
            "title": "Q-Q Plot of {}".format(self.name),
            "width": 4 * 1.618,
            "height": 4,
            "xlabel": "value",
            "ylim": (0.95 * np.min(self.data), 1.05 * np.max(self.data)),
            "xlim": (-3, 3),
            "subtitle": None,
        }
        # handle input specs
        if specs is None:
            pass
        else:  # override default
            for k in specs:
                default_specs[k] = specs[k]
        specs = default_specs

        # process quantiles
        _df = self.qqplot()
        # plot
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height
        # grid
        gs = mpl.gridspec.GridSpec(
            1, 1, wspace=0.4, hspace=0.5, left=0.1, bottom=0.15, top=0.9, right=0.95
        )  # nrows, ncols
        # scatter plot
        ax = fig.add_subplot(gs[0, 0])
        plt.title(specs["title"])
        plt.scatter(_df["T-Quantiles"], _df["Data"], marker=".", color="tab:grey")
        plt.ylim([_df["Data"].min(), _df["Data"].max()])
        plt.xlim(specs["xlim"])
        plt.xlabel("Normal Theoretical Quantiles")
        plt.ylabel("Data Empirical Quantiles")
        plt.gca().set_aspect(
            (specs["xlim"][1] - specs["xlim"][0])
            / (_df["Data"].max() - _df["Data"].min())
        )

        # show or save
        if show:
            plt.show()
        else:
            plt.savefig("{}/{}_{}.png".format(folder, self.name, filename), dpi=dpi)

    def _distribution_test(self, test_name, stat, p, clevel=0.95, distr="normal"):
        """Util function

        :param test_name: name of test
        :type test_name: str
        :param stat: statistic
        :type stat: float
        :param p: p-value
        :type p: float
        :param clevel: confidence level
        :type clevel: float
        :param distr: name of distribution
        :type distr: str
        :return: summary of test
        :rtype: dict
        """
        # built output dict
        dct_out = {
            "Test": test_name,
            "Statistic": stat,
            "p-value": p,
            "Confidence": clevel,
        }

        if p > (1 - clevel):
            dct_out["is_{}".format(distr.lower())] = True
        else:
            dct_out["is_{}".format(distr.lower())] = False

        return dct_out

    def test_normal_ks(self, clevel=0.95):
        """Test for normality using the Kolmogorov-Smirnov test

        Kolmogorov-Smirnov Test: This test compares the observed distribution with
        the expected normal distribution using a test statistic and a p-value.
        A p-value less than 0.05 indicates that the null hypothesis should be rejected.

        :return: test result dictionary. Keys: Statistic, p-value and Is normal
        :rtype: dct
        """
        from scipy.stats import kstest

        vct_norm = (self.data - np.mean(self.data)) / np.std(self.data)

        # test
        result = kstest(vct_norm, "norm")
        return self._distribution_test(
            test_name="Kolmogorov-Smirnov",
            stat=result.statistic,
            p=result.pvalue,
            clevel=clevel,
            distr="normal",
        )

    def test_shapiro_wilk(self, clevel=0.95):
        """Test for normality using the Shapiro-Wilk test.

        :return: test result dictionary. Keys: Statistic, p-value and Is normal
        :rtype: dct
        """
        from scipy.stats import shapiro

        # test
        stat, p = shapiro(self.data)

        return self._distribution_test(
            test_name="Shapiro-Wilk", stat=stat, p=p, clevel=clevel, distr="normal"
        )

    def test_dagostino_pearson(self, clevel=0.95):
        """Test for normality using the D'Agostino-Pearson test.

        :return: test result dictionary. Keys: Statistic, p-value and Is normal
        :rtype: dct
        """
        from scipy.stats import normaltest

        # test
        stat, p = normaltest(self.data)

        return self._distribution_test(
            test_name="D'Agostino-Pearson",
            stat=stat,
            p=p,
            clevel=clevel,
            distr="normal",
        )

    def assess_normality(self, clevel=0.95):
        """Assessment on normality using standard tests
        :return: dataframe of assessment results
        :rtype: :class:`pandas.DataFrame`
        """
        # run tests
        lst_tests = []
        lst_tests.append(self.test_normal_ks(clevel=clevel))
        lst_tests.append(self.test_shapiro_wilk(clevel=clevel))
        lst_tests.append(self.test_dagostino_pearson(clevel=clevel))
        # create dataframe
        lst_names = []
        lst_stats = []
        lst_p = []
        lst_clvl = []
        lst_is = []
        for e in lst_tests:
            lst_names.append(e["Test"])
            lst_stats.append(e["Statistic"])
            lst_p.append(e["p-value"])
            lst_clvl.append(e["Confidence"])
            lst_is.append(e["is_normal"])
        df_result = pd.DataFrame(
            {
                "Test": lst_names,
                "Statistic": lst_stats,
                "p-value": lst_p,
                "is_normal": lst_is,
                "Confidence": lst_clvl,
            }
        )
        return df_result

    def assess_frequency(self):
        """Assessment on data frequency
        :return: result dataframe
        :rtype: :class:`pandas.DataFrame`
        """
        # compute percentiles
        vct_percentiles = np.arange(0, 100)
        # get CFC values
        vct_cfc = np.percentile(self.data, vct_percentiles)
        # reverse to get exceedance
        vct_exeed = 100 - vct_percentiles
        # get count
        vct_count = np.histogram(self.data, bins=len(vct_percentiles))[0]
        # get empirical prob
        vct_empprob = vct_count / np.sum(vct_count)

        df_result = pd.DataFrame(
            {
                "Percentiles": vct_percentiles,
                "Exceedance": vct_exeed,
                "Frequency": vct_count,
                "Empirical Probability": vct_empprob,
                "Values": vct_cfc,
            }
        )
        return df_result

    def assess_basic_stats(self):
        dct = {
            "Count": len(self.data),
            "Sum": np.sum(self.data),
            "Mean": np.mean(self.data),
            "SD": np.std(self.data),
            "Min": np.min(self.data),
            "p01": np.percentile(self.data, 1),
            "p05": np.percentile(self.data, 5),
            "p25": np.percentile(self.data, 25),
            "p50": np.percentile(self.data, 50),
            "p75": np.percentile(self.data, 75),
            "p90": np.percentile(self.data, 90),
            "p95": np.percentile(self.data, 95),
            "p99": np.percentile(self.data, 99),
            "Max": np.max(self.data),
        }
        df_result = pd.DataFrame(
            {"Statistic": list(dct.keys()), "Value": [dct[key] for key in dct]}
        )
        return df_result


if __name__ == "__main__":

    print("Hello world!")
