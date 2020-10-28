# general imports
import os
from os.path import join
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import h5py
import re
import seaborn as sns
import descartes
from pylab import text
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

folder = "testPRAS10.27"
data = join(os.environ["HOMEPATH"], "Desktop", folder)
sys.path.insert(0, data)
from MISO_data_utility_functions import LoadMISOData, NRELEFSprofiles

NREL = False
NREL_year, NREL_profile = 2012, ""

casename = "VRE0.5_wind_2012base160%_8760_"

miso_datapath = join(os.environ["HOMEPATH"], "Desktop", folder, "VREData")
hifld_datapath = join(os.environ["HOMEPATH"], "Desktop", folder, "HIFLD_shapefiles")
shp_path = os.environ["CONDA_PREFIX"] + r"\Library\share\gdal"
results = join(data, "results")

miso_data = LoadMISOData(data, miso_datapath, hifld_datapath, shp_path)
miso_data.convert_CRS()


class plotter(object):
    def __init__(self, data_folder, results_folder, casename, miso_data):
        print("loading plotting data...")
        self.miso_map = miso_data.miso_map
        self.iso_map = miso_data.iso_map
        self.states_map = miso_data.states_map
        self.utilities_map = miso_data.utilities_map
        self.casename = casename

        miso_data.utility_territory_mapping()
        self.map_dict = miso_data.map_dict
        assert (type(casename)) == str
        # loads data, mostly
        # data loads
        miso_map = pd.read_excel(
            join(data_folder, "NREL-Seams Model (MISO).xlsx"), sheet_name="Mapping"
        )
        miso_loads = pd.read_excel(
            join(data_folder, "NREL-Seams Model (MISO).xlsx"), sheet_name="Load"
        )

        # results loads
        region_lole = pd.read_csv(
            join(results_folder, casename + "regionlole.csv"), header=None
        )
        region_eue = pd.read_csv(
            join(results_folder, casename + "regioneue.csv"), header=None
        )
        region_period_eue = pd.read_csv(
            join(results_folder, casename + "regionperiodeue.csv"), header=None
        )
        period_eue = pd.read_csv(
            join(results_folder, casename + "periodeue.csv"), header=None
        )
        period_lolp = pd.read_csv(
            join(results_folder, casename + "periodlolp.csv"), header=None
        )

        # clean and reformat some of the loaded info
        region_lole.index, region_eue.index = (
            list(miso_map["CEP Bus ID"]),
            list(miso_map["CEP Bus ID"]),
        )
        region_lole.columns, region_eue.columns = ["LOLE"], ["EUE"]
        region_df = pd.concat([region_lole, region_eue], axis=1)
        tmps = len(region_period_eue.columns)
        region_df["load"] = list(miso_loads.iloc[:tmps, 1:].sum(axis=0))
        region_df["names"] = miso_map["CEP Bus Name"].values

        # create attributes of stuff we want later
        self.results_folder = results_folder
        self.miso_map = miso_map
        self.miso_loads = miso_loads
        self.region_df = region_df
        self.region_lole = region_lole
        self.region_eue = region_eue
        self.region_period_eue = region_period_eue
        self.period_eue = period_eue
        self.period_lolp = period_lolp
        print("...plotting data loaded")

    def data_check(self):
        # runs some checks on data formatting from uploads
        return self.region_df

    def format12x24(self, df, mean=False):
        # does some formatting for use with seaborn heatmaps
        df["Date"] = list(self.miso_loads["Date"])
        df["Month"] = [
            int(re.findall(r"-(\d+)-", str(d))[0]) for d in df["Date"].values
        ]
        df["HourBegin"] = df.index.values % 24
        np_12x24 = np.zeros((12, 24))
        for r in range(np_12x24.shape[0]):
            for c in range(np_12x24.shape[1]):
                if mean:
                    np_12x24[r, c] += df[0][
                        (df.Month == r + 1) & (df.HourBegin == c)
                    ].mean()
                else:
                    np_12x24[r, c] += df[0][
                        (df.Month == r + 1) & (df.HourBegin == c)
                    ].sum()
        insert_row = np.zeros((1, 24))
        np_12x24 = np.vstack((insert_row, np_12x24))  # helps to fix indexing
        return np_12x24

    def heatmap(self, attribute_string, show=False, mean=False):
        # plots heatmap
        assert (type(attribute_string)) == str
        attribute = getattr(self, attribute_string)
        fig, ax = plt.subplots(1, figsize=(8, 5))
        ax = sns.heatmap(
            self.format12x24(attribute, mean=mean), linewidth=0.5, cmap="Reds"
        )
        ax.set_ylim(1, 13)
        ax.set_ylabel("Month")
        ax.set_xlabel("Hour Beginning")
        ax.set_title(attribute_string)
        plt.savefig(
            join(self.results_folder, attribute_string + self.casename + "heatmap.jpg"),
            dpi=300,
        )
        if show:
            plt.show()
        return None

    def load_utility_df(self, attribute_string):

        col = attribute_string[attribute_string.find("_") + 1 :].upper()
        for k in self.map_dict.keys():
            self.map_dict[k].append(
                self.region_df[col][self.region_df["names"] == k].values[0]
            )

        index_list = []
        for k, v in self.map_dict.items():
            for utility in v[0]:
                index_list.append(
                    self.utilities_map.index[
                        self.utilities_map["NAME"] == utility
                    ].values[0]
                )
        utilities_plot_df = self.utilities_map[
            self.utilities_map.index.isin(index_list)
        ]

        l = []
        for utility in utilities_plot_df["NAME"]:
            for k, v in self.map_dict.items():
                for u in v[0]:
                    if u == utility:
                        l.append(v[2])
        pd.set_option("mode.chained_assignment", None)  # for now to suppress warnings
        utilities_plot_df[col] = l
        return (utilities_plot_df, col)

    def geography_plot(self, attribute_string):
        # plots onto map of MISO
        fig, ax = plt.subplots(1, figsize=(8, 8))
        myaxes = plt.axes()
        myaxes.set_ylim([23, 50])
        myaxes.set_xlim([-108, -82])
        self.iso_map[self.iso_map["NAME"] == self.iso_map.at[0, "NAME"]].plot(
            ax=myaxes, facecolor="b", edgecolor="r", alpha=0.1, linewidth=2
        )
        self.states_map.plot(ax=myaxes, edgecolor="k", facecolor="None")

        divider = make_axes_locatable(myaxes)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)

        utilities_plot_df, label = self.load_utility_df(attribute_string)
        utilities_plot_df.plot(
            ax=myaxes,
            column=label,
            cmap="Reds",
            legend=True,
            cax=cax,
            legend_kwds={"label": label + " (MWh/y)", "orientation": "horizontal"},
        )
        plt.savefig(join(self.results_folder, self.casename + label + ".jpg"), dpi=300)
        return None

    def aggregate_zones(self, df, tmps=24, monthly=False):
        if monthly == False:
            df = df.iloc[:, 1:]
        else:
            df["Month"] = [
                int(re.findall(r"-(\d+)-", str(d))[0]) for d in df["Date"].values
            ]
            df = df.iloc[:, 1:]
        df["hour"] = list(range(tmps)) * int(len(df.index) / tmps)
        if monthly == False:
            df = df.groupby("hour").mean()
        else:
            df = df.groupby(["Month", "hour"]).mean()
        return df

    def modify_load_year(self, year, new_load_magnitude, new_load_profile):
        scalar = new_load_magnitude / (self.miso_loads.iloc[:, 1:].sum(axis=1).mean())
        self.miso_loads.iloc[:, 1:] = self.miso_loads.iloc[:, 1:] * scalar
        seams_profile = (
            self.miso_loads.iloc[:, 1:].sum(axis=1)
            / self.miso_loads.iloc[:, 1:].sum(axis=1).sum()
        )
        self.miso_loads.iloc[:, 1:] = self.miso_loads.iloc[:, 1:].mul(
            (new_load_profile.LoadMW.reset_index().LoadMW / seams_profile), axis=0
        )

    def plot_zonal_loads(
        self, monthly=True, show=False, NREL=False, year_lab=2012, scenario_lab=""
    ):
        miso_loads_zones = self.aggregate_zones(self.miso_loads, monthly=monthly)
        color_list = [self.map_dict[k][1] for k in miso_loads_zones.columns]
        fig, ax = plt.subplots(1, figsize=(10, 6))
        miso_loads_zones.plot.area(ax=ax, color=color_list)

        ax.set_ylabel("MWh")
        if monthly:
            ax.set_xlabel("(Month-Hour) Average")
        else:
            ax.set_xlabel("Hour Beginning")
        for index, label in enumerate(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        ):
            plt.axvline(x=(index + 1) * 24, color="k")
            text(
                (index + 1) * 24 - 17,
                miso_loads_zones.sum(axis=1).max() * 1.01,
                label,
                fontsize=12,
            )
        # draw vertical line from (70,100) to (70, 250)
        # plt.plot([24, 0], [24, 100000], 'k-', lw=2)

        lgd = plt.legend(bbox_to_anchor=(1.2, 1.01), loc="upper right")
        if NREL:
            label = "NREL" + str(scenario_lab) + str(year_lab) + "load.jpg"
        else:
            label = "MISOload.jpg"
        plt.savefig(
            join(self.results_folder, label),
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
            dpi=300,
        )
        if show:
            plt.show()
        return None


test = plotter(data, results, casename, miso_data)

if NREL:
    nreltester = NRELEFSprofiles(data, NREL_profile)
    load, normprofile = nreltester.run_all(NREL_year)
    test.modify_load_year(NREL_year, load, normprofile)
    l = [m for m in re.finditer("_", NREL_profile)]
    scenario_label = (
        NREL_profile[l[0].end() : l[1].start()] + NREL_profile[l[1].end() :]
    )
else:
    scenario_label = ""


test.geography_plot("region_lole")
test.geography_plot("region_eue")
test.heatmap("period_eue")
# test.heatmap("period_lolp", mean=True)
test.plot_zonal_loads(NREL=NREL, year_lab=NREL_year, scenario_lab=scenario_label)

