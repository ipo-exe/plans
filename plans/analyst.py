"""
PLANS - Planning Nature-based Solutions

Module description:
This module stores all analyst objects of PLANS.

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
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normal_curve(mu=0, sigma=5, vmin=-20, vmax=20, ngrid=100):
    """
    Return the normal curve
    :param mu: mean
    :type mu: float
    :param sigma: standard deviation
    :type sigma: float
    :param vmin: minimal value
    :type vmin: float
    :param vmax: maximal value
    :type vmax: float
    :param ngrid: number of data points
    :type ngrid: int
    :return: array of the computed normal curve
    :rtype: :class:`numpy.ndarray`
    """
    # Generate x values
    x = np.linspace(vmin, vmax, ngrid)

    # Calculate y values using Gaussian formula
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    return {"x": x, "y": y}


class Univar:
    """
    The Univariate Analyst Object

    """

    def __init__(self, data, name="myvar"):
        """
        Deploy the Analyst
        :param data: n-D vector of data
        :type data: :class:`numpy.ndarray`
        """
        self.data = data
        self.name = name

    def nbins_fd(self):
        """
        This function computes the number of bins for histograms using the Freedman-Diaconis rule,
        which takes into account the interquartile range (IQR) of the data, in addition to its range.
        :return: number of bins for histogram using the Freedman-Diaconis rule
        :rtype: int
        """
        iqr = np.subtract(*np.percentile(self.data, [75, 25]))
        binsize = 2 * iqr * len(self.data) ** (-1 / 3)
        return int(np.ceil((max(self.data) - min(self.data)) / binsize))

    def nbins_sturges(self):
        """
        This function computes the number of bins using the Sturges rule, which assumes that
        the data follows a normal distribution and computes the number of bins based on its sample size.
        :return: number of bins using the Sturges rule
        :rtype: int
        """
        return int(np.ceil(np.log2(len(self.data)) + 1))

    def nbins_scott(self):
        """
        This function computes the number of bins using the Scott rule, which is
        similar to the Freedman-Diaconis rule, but uses the standard deviation of the data to compute the bin size.
        :return: number of bins using the Scott rule
        :rtype: int
        """
        binsize = 3.5 * np.std(self.data) * len(self.data) ** (-1 / 3)
        return int(np.ceil((max(self.data) - min(self.data)) / binsize))

    def nbins_by_rule(self, rule=None):
        """
        Util function for rule-based nbins computation
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
        """
        Compute the histogram of the sample
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

    def plot_hist(
        self,
        bins=100,
        rule=None,
        show=False,
        folder="C:/data",
        filename="histogram",
        specs=None,
        dpi=96,
    ):
        """
        Plot histogram of data

        :param bins: number of bins
        :type bins: int
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
        import matplotlib.ticker as mtick
        plt.style.use("seaborn-v0_8")

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
        plt.hist(
            self.data,
            bins=bins,
            weights=np.ones(len(self.data)) / len(self.data),
            color=specs["color"])
        plt.xlabel(specs["xlabel"])
        plt.ylim(specs["ylim"])
        plt.xlim(specs["xlim"])

        # Set the y-axis formatter as percentages
        yticks = mtick.PercentFormatter(xmax=1, decimals=1, symbol='%', is_latex=False)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(yticks)

        # show or save
        if show:
            plt.show()
        else:
            plt.savefig("{}/{}.png".format(folder, filename), dpi=96)

    def _distribution_test(self, test_name, stat, p, clevel=0.05, distr="normal"):
        """
        Util function
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
            "Confidence": 1 - clevel,
        }

        if p > clevel:
            dct_out["Is {}".format(distr)] = True
        else:
            dct_out["Is {}".format(distr)] = False

        return dct_out

    def test_normal_ks(self):
        """
        Test for normality using the Kolmogorov-Smirnov test

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
            distr="normal",
        )

    def test_shapiro_wilk(self):
        """
        Test for normality using the Shapiro-Wilk test.

        :return: test result dictionary. Keys: Statistic, p-value and Is normal
        :rtype: dct
        """
        from scipy.stats import shapiro

        # test
        stat, p = shapiro(self.data)

        return self._distribution_test(
            test_name="Shapiro-Wilk", stat=stat, p=p, distr="normal"
        )

    def test_dagostino_pearson(self):
        """
        Test for normality using the D'Agostino-Pearson test.

        :return: test result dictionary. Keys: Statistic, p-value and Is normal
        :rtype: dct
        """
        from scipy.stats import normaltest

        # test
        stat, p = normaltest(self.data)

        return self._distribution_test(
            test_name="D'Agostino-Pearson", stat=stat, p=p, distr="normal"
        )

    def assess_normality(self):
        """
        Assessment on normality using standard tests
        :return: dataframe of assessment results
        :rtype: :class:`pandas.DataFrame`
        """
        # run tests
        lst_tests = []
        lst_tests.append(self.test_normal_ks())
        lst_tests.append(self.test_shapiro_wilk())
        lst_tests.append(self.test_dagostino_pearson())
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
            lst_is.append(e["Is normal"])
        df_result = pd.DataFrame(
            {
                "Test": lst_names,
                "Statistic": lst_stats,
                "P-value": lst_p,
                "Is Normal": lst_is,
                "Confidence": lst_clvl,
            }
        )
        return df_result

    def assess_frequency(self):
        """
        Assessment on data frequency
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
        df_aux = pd.DataFrame(
            {
                "Value": self.data
            }
        )
        df_stats = df_aux.describe()
        df_result = df_stats.reset_index().rename(columns={"index": "Statistic"})
        return df_result


if __name__ == "__main__":

    np.random.seed(6175)
    vct = np.random.normal(100, 10, 50)
    vct_random = np.random.randint(10, 100, 1000)

    vct_norm = normal_curve(mu=100, sigma=10, vmin=50, vmax=150, ngrid=1000)

    uni = Univar(data=vct, name="Random")


    df = uni.assess_basic_stats()
    print(df)