# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 2020

@author: Luke
"""

# general imports
import os
from os.path import join
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerPatch
import descartes
from scipy.spatial.distance import cdist
import h5py
import requests
from zipfile import ZipFile
import subprocess, sys

# patch handlers
# https://stackoverflow.com/questions/40672088/matplotlib-customize-the-legend-to-show-squares-instead-of-rectangles

# --- handlers ---


class HandlerRect(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):

        x = width // 2
        y = 0
        w = h = 10

        # create
        p = patches.Rectangle(xy=(x, y), width=w, height=h)

        # update with data from oryginal object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]


class HandlerCircle(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):

        r = 5
        x = r + width // 2
        y = height // 2

        # create
        p = patches.Circle(xy=(x, y), radius=r)

        # update with data from oryginal object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]


class LoadMISOData(object):
    def __init__(self, folder_datapath, miso_datapath, hifld_datapath, shp_datapath):
        self.folder_datapath = folder_datapath
        print("loading data...")
        self.miso_map = pd.read_csv(
            os.path.join(miso_datapath, "Bus Mapping Extra Data.csv")
        )
        self.utilities_map = gpd.read_file(
            os.path.join(shp_datapath, "Retail_Service_Territories.shp")
        )
        self.iso_map = gpd.read_file(
            os.path.join(hifld_datapath, "Independent_System_Operators.shp")
        )
        self.states_map = gpd.read_file(os.path.join(hifld_datapath, "states.shp"))
        self.miso_gdf = gpd.GeoDataFrame(
            self.miso_map,
            geometry=gpd.points_from_xy(self.miso_map.Lat, self.miso_map.Lon),
        )
        print("...data loaded")

    def reverse_gdf_coordinates(self):
        miso_gdf = gpd.GeoDataFrame(
            self.miso_map,
            geometry=gpd.points_from_xy(self.miso_map.Lon, self.miso_map.Lat),
        )
        return miso_gdf

    def convert_CRS(self, CRS=4326):
        self.miso_gdf.crs = "EPSG:" + str(CRS)
        self.miso_gdf.to_crs(epsg=CRS, inplace=True)
        self.utilities_map.to_crs(epsg=CRS, inplace=True)
        self.iso_map.to_crs(epsg=CRS, inplace=True)

    def closest_point(self, point, points):
        """ Find closest point from a list of points. """
        return points[cdist([point], points).argmin()]

    def match_value(self, df, col1, x, col2):
        """ Match value x from col1 row to value in col2. """
        return df[df[col1] == x][col2].values[0]

    def utility_territory_mapping(self):
        map_dict = {}
        mo_coops = list(
            self.utilities_map[
                (self.utilities_map["STATE"] == "MO")
                & (self.utilities_map["NAME"].str.contains("COOP"))
            ]["NAME"]
        )
        map_dict["AECIZ"] = [mo_coops, "olive"]
        map_dict["ATC"] = [
            [
                "WISCONSIN ELECTRIC POWER CO",
                "WISCONSIN PUBLIC SERVICE CORP",
                "WISCONSIN POWER & LIGHT CO",
            ],
            "pink",
        ]
        map_dict["CBPC-NIPCO"] = [
            [
                "MIDLAND POWER COOP",
                "IOWA LAKES ELECTRIC COOP",
                "BUTLER COUNTY RURAL ELEC COOP - (IA)",
                "BOONE VALLEY ELECTRIC COOP",
                "FRANKLIN RURAL ELECTRIC COOP - (IA)",
                "RACCOON VALLEY ELECTRIC COOPERATIVE",
                "PRAIRIE ENERGY COOP",
                "CALHOUN COUNTY ELEC COOP ASSN",
                "GRUNDY COUNTY RURAL ELEC COOP",
                "NORTH WEST RURAL ELECTRIC COOP",
                "WOODBURY COUNTY RURAL E C A",
                "WESTERN IOWA POWER COOP",
                "HARRISON COUNTY RRL ELEC COOP",
                "NISHNABOTNA VALLEY R E C",
                "HEARTLAND POWER COOP",
            ],
            "darkseagreen",
        ]
        map_dict["CONS"] = [["CONSUMERS ENERGY CO"], "orange"]
        map_dict["DECO"] = [["DTE ELECTRIC COMPANY"], "g"]
        map_dict["EES-ARK"] = [["ENTERGY ARKANSAS INC"], "r"]
        map_dict["EES-TX"] = [["ENTERGY TEXAS INC."], "gold"]
        map_dict["IA-E"] = [
            [
                "EASTERN IOWA LIGHT & POWER COOP",
                "EAST-CENTRAL IOWA RURAL ELEC COOP",
                "FARMERS ELECTRIC COOP, INC - (IA)",
                "MAQUOKETA VALLEY RRL ELEC COOP",
                "LINN COUNTY REC",
                "ACCESS ENERGY COOP",
                "T I P RURAL ELECTRIC COOP",
                "CHARITON VALLEY ELEC COOP, INC",
                "SOUTHERN IOWA ELEC COOP, INC",
                "PELLA COOPERATIVE ELEC ASSN",
                "SOUTHWEST IOWA RURAL ELEC COOP",
                "CLARKE ELECTRIC COOP INC - (IA)",
                "ALLAMAKEE-CLAYTON EL COOP, INC",
                "GUTHRIE COUNTY RURAL E C A",
                "CONSUMERS ENERGY",
                "MIENERGY COOPERATIVE",
            ],
            "purple",
        ]
        map_dict["IL-C"] = [["AMEREN ILLINOIS COMPANY"], "darkred"]
        map_dict["IN-C"] = [
            ["INDIANAPOLIS POWER & LIGHT CO", "INDIANA MICHIGAN POWER CO"],
            "lime",
        ]
        map_dict["IN-S"] = [
            ["DUKE ENERGY INDIANA, LLC", "SOUTHERN INDIANA GAS & ELEC CO"],
            "indigo",
        ]
        map_dict["LA-GULF"] = [
            [
                "ENTERGY GULF STATES - LA LLC",
                "ENTERGY NEW ORLEANS, LLC",
                "SOUTH LOUISIANA ELEC COOP ASSN",
            ],
            "darkviolet",
        ]
        map_dict["LA-N"] = [
            [
                "CLECO POWER LLC",
                "SOUTHWEST LOUISIANA E M C",
                "NORTHEAST LOUISIANA POWER COOP INC.",
                "CLAIBORNE ELECTRIC COOP, INC",
            ],
            "aquamarine",
        ]
        map_dict["MEC"] = [["MIDAMERICAN ENERGY CO"], "darkgoldenrod"]
        map_dict["MISO-MO"] = [["UNION ELECTRIC CO - (MO)"], "lightgreen"]
        map_dict["MISO-MS"] = [
            ["ENTERGY MISSISSIPPI INC", "MISSISSIPPI POWER CO"],
            "royalblue",
        ]
        mn_coops = list(
            self.utilities_map[
                (self.utilities_map["STATE"] == "MN")
                & (self.utilities_map["NAME"].str.contains("COOP"))
            ]["NAME"]
        )
        mn_coops.remove("MIENERGY COOPERATIVE")
        map_dict["MN-NE"] = [
            [
                "LAKE COUNTRY POWER",
                "EAST CENTRAL ENERGY",
                "CONNEXUS ENERGY",
                "FEDERATED RURAL ELECTRIC ASSN",
                "SOUTH CENTRAL ELECTRIC ASSN",
                "RUNESTONE ELECTRIC ASSN",
                "ITASCA-MANTRAP CO-OP ELECTRICAL ASSN",
                "SIOUX VALLEY SW ELEC COOP",
            ]
            + mn_coops,
            "midnightblue",
        ]
        map_dict["MN-SE"] = [
            [
                "NORTHERN STATES POWER CO - MINNESOTA",
                "NORTHERN STATES POWER CO - WISCONSIN",
            ],
            "tan",
        ]
        map_dict["MN-C"] = [["OTTER TAIL POWER CO"], "salmon"]
        map_dict["NIPS"] = [["NORTHERN INDIANA PUB SERV CO"], "slategray"]
        map_dict["SIPC"] = [
            [
                "SOUTHWESTERN ELECTRIC COOP INC - (IL)",
                "SOUTHEASTERN IL ELEC COOP, INC",
                "PRAIRIE POWER, INC",
            ],
            "yellow",
        ]
        map_dict["UPPC"] = [
            ["UPPER PENINSULA POWER COMPANY", "CLOVERLAND ELECTRIC CO-OP"],
            "cyan",
        ]
        self.map_dict = map_dict

    def initialize_SEAMS_match(self, flipped=False):
        self.match_gdf = gpd.sjoin(self.utilities_map, self.miso_gdf, op="contains")
        self.match_gdf["SEAMS_ZONE"] = ["NA"] * len(self.match_gdf.index)
        self.match_gdf["SEAMS_COLOR"] = ["NA"] * len(self.match_gdf.index)

        # assign a SEAMS bus based on utility
        for k, v in self.map_dict.items():
            for utility in v[0]:
                for i in self.match_gdf[self.match_gdf.NAME == utility].index:
                    self.match_gdf.at[i, "SEAMS_ZONE"] = k
                    self.match_gdf.at[i, "SEAMS_COLOR"] = v[1]

        # check to make sure matching worked
        if (
            len(self.match_gdf[self.match_gdf.SEAMS_ZONE != "NA"].Name.unique()) < 1.0
            and flipped == False
        ):
            print("no matches, so flipping coordinates and trying again...")
            self.miso_map = self.reverse_gdf_coordinates()  # flip lat/lon order
            self.convert_CRS()  # convert CRS again
            self.initialize_SEAMS_match(flipped=True)  # recursion
        elif flipped == True:
            print(
                "...flipping complete, now there are "
                + str(
                    len(self.match_gdf[self.match_gdf.SEAMS_ZONE != "NA"].Name.unique())
                )
                + " matches"
            )

    def match_unmatched_buses(self):
        matched_names = list(
            self.match_gdf[self.match_gdf.SEAMS_ZONE != "NA"].Name.unique()
        )
        matched_points = self.miso_gdf[self.miso_gdf.Name.isin(matched_names)][
            ["Name", "Lat", "Lon"]
        ]  # these points have a matched SEAMS zone
        unmatched_points = self.miso_gdf[~self.miso_gdf.Name.isin(matched_names)][
            ["Name", "Lat", "Lon"]
        ]  # get the points that have yet to receive any matched SEAMS zone

        # create some new columns to help with matching info
        matched_points["point"] = [
            (x, y) for x, y in zip(matched_points["Lat"], matched_points["Lon"])
        ]
        matched_points["zone"] = [
            (x, y) for x, y in zip(matched_points["Lat"], matched_points["Lon"])
        ]
        unmatched_points["point"] = [
            (x, y) for x, y in zip(unmatched_points["Lat"], unmatched_points["Lon"])
        ]
        unmatched_points["closest"] = [
            self.closest_point(x, list(matched_points["point"]))
            for x in unmatched_points["point"]
        ]
        unmatched_points["zone"] = [
            self.match_value(matched_points, "point", x, "zone")
            for x in unmatched_points["closest"]
        ]
        unmatched_points["matchname"] = [
            self.match_value(matched_points, "point", x, "Name")
            for x in unmatched_points["closest"]
        ]

        # grab the best SEAMS zone associated with the new match
        for p in unmatched_points.index:

            mlist = list(
                self.match_gdf[
                    self.match_gdf.Name == unmatched_points.loc[p, "matchname"]
                ].SEAMS_ZONE.unique()
            )
            if "NA" in mlist:
                mlist.remove("NA")
            nameid = unmatched_points.loc[p, "Name"]
            for i in self.match_gdf[self.match_gdf.Name == nameid].SEAMS_ZONE.index:
                self.match_gdf.at[i, "SEAMS_ZONE"] = mlist[
                    0
                ]  # take first for now; this will be resolved by multi-matches
                self.match_gdf.at[i, "SEAMS_COLOR"] = self.map_dict[mlist[0]][1]
            if len(mlist) > 1 and hasattr(self, "multi_names"):
                self.multi_names.append(unmatched_points.loc[p, "matchname"])
            elif len(mlist) > 1:
                self.multi_names = [
                    unmatched_points.loc[p, "matchname"]
                ]  # create list if not exist

    def handle_multimatch_buses(self):

        if hasattr(self, "multi_names"):
            pass
        else:
            self.multi_names = []  # create if does not exist

        for matched_id in self.match_gdf[
            self.match_gdf.SEAMS_ZONE != "NA"
        ].Name.unique():
            zone_matches = list(
                self.match_gdf[self.match_gdf.Name == matched_id].SEAMS_ZONE.unique()
            )
            if "NA" in zone_matches:
                zone_matches.remove("NA")
            if len(zone_matches) > 1:
                self.multi_names.append(
                    self.match_gdf[self.match_gdf.Name == matched_id].Name.values[0]
                )

        self.multi_names = list(set(self.multi_names))  # filter to unique IDs

        multi_match_gdf = self.match_gdf[self.match_gdf.Name.isin(self.multi_names)]
        unique_match_gdf = self.match_gdf[~self.match_gdf.Name.isin(self.multi_names)]
        unique_match_gdf = unique_match_gdf[
            unique_match_gdf.SEAMS_ZONE != "NA"
        ]  # filter

        # use Nik's function to determine a best match for each multi-match among the unique matches
        pd.set_option("mode.chained_assignment", None)  # for now to suppress warnings
        unique_match_gdf["point"] = [
            (x, y) for x, y in zip(unique_match_gdf["Lat"], unique_match_gdf["Lon"])
        ]
        multi_match_gdf["point"] = [
            (x, y) for x, y in zip(multi_match_gdf["Lat"], multi_match_gdf["Lon"])
        ]
        multi_match_gdf["closest_unique"] = [
            self.closest_point(x, list(unique_match_gdf["point"]))
            for x in multi_match_gdf["point"]
        ]
        multi_match_gdf["matchname"] = [
            self.match_value(unique_match_gdf, "point", x, "Name")
            for x in multi_match_gdf["closest_unique"]
        ]
        multi_match_gdf["seamszonematch"] = [
            self.match_value(unique_match_gdf, "point", x, "SEAMS_ZONE")
            for x in multi_match_gdf["closest_unique"]
        ]

        # do the overwrite with matching in place
        new_zones = []
        new_colors = []
        for z, c, n in zip(
            self.match_gdf.SEAMS_ZONE, self.match_gdf.SEAMS_COLOR, self.match_gdf.Name
        ):
            if n in self.multi_names:
                new_zones.append(
                    multi_match_gdf[multi_match_gdf.Name == n].seamszonematch.unique()[
                        0
                    ]
                )
                new_colors.append(
                    self.map_dict[
                        multi_match_gdf[
                            multi_match_gdf.Name == n
                        ].seamszonematch.unique()[0]
                    ][1]
                )
                # new_zones.append()
            else:
                new_zones.append(z)
                new_colors.append(c)
        self.match_gdf["FINAL_SEAMS_ZONE"] = new_zones
        self.match_gdf["FINAL_SEAMS_COLOR"] = new_colors

    def plot_points(
        self, plotTitle="(Attempted) Bus Mapping to SEAMS zones", showplot=True
    ):

        zone_lab = "FINAL_SEAMS_ZONE"
        color_lab = "FINAL_SEAMS_COLOR"

        mm = self.match_gdf[self.match_gdf.FINAL_SEAMS_ZONE != "NA"][
            ["Lat", "Lon", zone_lab, color_lab]
        ]
        mm.reset_index(inplace=True)
        category_keys = list(mm.FINAL_SEAMS_ZONE.unique())
        category_dict = {}
        v = 0
        for k in category_keys:
            category_dict[k] = v
            v += 1
        mm["util_integer"] = [-1] * len(mm.index)
        for i in list(mm.index):
            mm.at[i, "util_integer"] = category_dict[mm.at[i, zone_lab]]
        colormap = mm.FINAL_SEAMS_COLOR.unique()
        categories = mm.util_integer

        fig, ax = plt.subplots(1, figsize=(8, 8))
        myaxes = plt.axes()
        myaxes.set_ylim([23, 50])
        myaxes.set_xlim([-108, -82])
        myaxes.set_title(plotTitle)
        self.iso_map[self.iso_map["NAME"] == self.iso_map.at[0, "NAME"]].plot(
            ax=myaxes, facecolor="b", edgecolor="r", alpha=0.1, linewidth=2
        )
        self.states_map.plot(ax=myaxes, edgecolor="k", facecolor="None")

        # hopefully this dataframe will have more matched zones, so then it'll plot more
        self.match_gdf[self.match_gdf.FINAL_SEAMS_ZONE != "NA"][
            ["Lat", "Lon", "FINAL_SEAMS_ZONE", "FINAL_SEAMS_ZONE"]
        ].plot.scatter(ax=myaxes, x="Lon", y="Lat", c=colormap[categories])

        leg_labels = []
        for k, v in self.map_dict.items():
            leg_labels.append(patches.Circle((0, 0), 1, color=v[1], label=k))
        leg_labels.append(patches.Circle((0, 0), 1, color="k", label="Unmatched"))
        leg_labels.append(
            patches.Rectangle(
                (0, 0), 1, 1, facecolor="b", edgecolor="r", alpha=0.1, label="MISO"
            )
        )
        myaxes.legend(
            handles=leg_labels,
            handler_map={
                patches.Rectangle: HandlerRect(),
                patches.Circle: HandlerCircle(),
            },
            fontsize=11,
            loc="lower left",
        )
        if showplot:
            plt.show()
        else:
            plt.savefig(
                os.path.join(self.folder_datapath, "miso_locs_to_SEAMS_zones.jpg"),
                dpi=300,
            )

    def returndata(self):
        return self.match_gdf


class LoadVREScenarios(object):
    def __init__(self, folder_datapath, sheet):
        self.vre_scenarios = pd.read_excel(
            os.path.join(folder_datapath, "Capacity to Luke.xlsx"), sheet_name=sheet
        )
        self.seams_load = pd.read_excel(
            os.path.join(folder_datapath, "NREL-Seams Model (MISO).xlsx"),
            sheet_name="Load",
        )

    def subset_VRE_df(self, df, VRE_type):
        colnames = df.iloc[0, 1:]
        if "Wind" in VRE_type:
            return_df = df.iloc[1:11, :]
            return_df.set_index(VRE_type, inplace=True)
            return_df.columns = list(colnames)

        else:
            slice_value = df[df["Utility Wind"] == VRE_type].index[0]
            return_df = df.iloc[slice_value + 2 : slice_value + 12, :]
            return_df.columns = return_df.columns[1:].insert(0, VRE_type)
            return_df.set_index(VRE_type, inplace=True)
            return_df.columns = list(colnames)

        return return_df.iloc[:, :10].reset_index()

    def create_seams_df(self, seams_dict):
        assert (type(seams_dict)) == dict

        self.SEAMS_LRZ_df = pd.DataFrame.from_dict(seams_dict, orient="index")
        self.SEAMS_LRZ_df.columns = ["LRZ"]
        self.SEAMS_LRZ_df = self.SEAMS_LRZ_df.join(self.seams_load.sum().rename("load"))

        load_sums = self.SEAMS_LRZ_df.groupby("LRZ")["load"].sum()
        self.SEAMS_LRZ_df["LRZLoad"] = [
            load_sums[self.SEAMS_LRZ_df.loc[i, "LRZ"]] for i in self.SEAMS_LRZ_df.index
        ]
        self.SEAMS_LRZ_df["LRZLoadFrac"] = (
            self.SEAMS_LRZ_df["load"] / self.SEAMS_LRZ_df["LRZLoad"]
        )

        for VRE_type in ["Utility Solar", "Utility Wind", "Distributed Solar"]:
            vre_df = self.subset_VRE_df(self.vre_scenarios, VRE_type)

            self.SEAMS_LRZ_df = (
                self.SEAMS_LRZ_df.reset_index()
                .merge(vre_df, how="left", left_on="LRZ", right_on=VRE_type)
                .set_index("index")
                .drop([VRE_type], axis=1)
            )
            self.SEAMS_LRZ_df.rename(
                lambda x: VRE_type + " " + str(x)
                if "." in str(x) and len(str(x)) < 5
                else x,
                axis=1,
                inplace=True,
            )
            for i in range(-10, 0):
                self.SEAMS_LRZ_df.iloc[:, i] = (
                    self.SEAMS_LRZ_df.loc[:, "LRZLoadFrac"]
                    * self.SEAMS_LRZ_df.iloc[:, i]
                )

        return self.SEAMS_LRZ_df


class LoadVREData(object):
    def __init__(
        self, folder_datapath, vre_types=["Rooftop", "Fixed", "Tracking", "Wind"]
    ):
        self.path = folder_datapath
        self.vre_types = vre_types

    def load_data(self, VRE_type, year_begin=2007, year_end=2012):
        for y in range(year_begin, year_end + 1):
            data_df = pd.read_csv(
                os.path.join(self.path, "MISO_data", VRE_type + "_" + str(y) + ".csv")
            )
            data_df.set_index("DateTime", inplace=True)
            if y != year_begin:
                self.VRE_type = pd.concat([self.VRE_type, data_df], axis=0, sort=False)
            else:
                self.VRE_type = data_df
        return self.VRE_type

    def load_all_dfs(self):
        print("loading all VRE profile dataframes...")
        df_dict = {}
        for v in self.vre_types:
            df_dict[v] = self.load_data(v)
        return df_dict
        print("... VRE dfs loaded")


class CreateHDF5(object):
    def __init__(
        self,
        folder_datapath,
        row_len,
        slice_in_index,
        metadata,
        vre_profile_df,
        miso_geography_df,
        vre_scenario_df,
    ):
        self.row_len = row_len
        self.slicer = slice_in_index
        self.metadata = metadata
        self.vre_profile_df = vre_profile_df
        self.miso_geography_df = miso_geography_df
        self.vre_scenario_df = vre_scenario_df
        self.folder_datapath = folder_datapath
        self.seams_transmission_df = pd.read_excel(
            os.path.join(folder_datapath, "NREL-Seams Model (MISO).xlsx"),
            sheet_name="Transmission",
        )
        self.seams_generation_df = pd.read_excel(
            os.path.join(folder_datapath, "NREL-Seams Model (MISO).xlsx"),
            sheet_name="Generation",
        )
        self.seams_load_df = pd.read_excel(
            os.path.join(folder_datapath, "NREL-Seams Model (MISO).xlsx"),
            sheet_name="Load",
        )
        self.seams_mapping_df = pd.read_excel(
            os.path.join(folder_datapath, "NREL-Seams Model (MISO).xlsx"),
            sheet_name="Mapping",
        )

        self.MISO_load_df = pd.read_excel(
            os.path.join(folder_datapath, "MISO_load_2009-2013_to_Luke.xlsx"),
            sheet_name="2009-2013",
        )

        # cleaning/formatting
        retain_cols = ["Line", "From", "To", "FW", "BW", "Area From", "Area To"]
        self.cleaned_seams_transmission_df = self.seams_transmission_df[
            retain_cols
        ].dropna(subset=retain_cols)

        # drop generators that aren't in MISO-mapped zones(for now)
        self.seams_generation_df = self.seams_generation_df[
            self.seams_generation_df["Bubble"].isin(
                list(self.seams_mapping_df["CEP Bus ID"])
            )
        ].reset_index()

    def create_gens_np(self):
        ### GENERATORS ###
        self.generators_dtype = np.dtype(
            [
                ("name", h5py.special_dtype(vlen=str)),
                ("category", h5py.special_dtype(vlen=str)),
                ("region", h5py.special_dtype(vlen=str)),
            ]
        )

        generators_data = np.zeros(
            (len(self.seams_generation_df.index),), dtype=self.generators_dtype
        )
        generators_data["name"] = tuple(self.seams_generation_df["Generator Name"])
        generators_data["category"] = tuple(self.seams_generation_df["category"])
        generators_data["region"] = tuple(self.seams_generation_df["Bubble"])
        self.generators_data = generators_data

    def create_zone_np(self):

        self.capacity_np = np.asarray(
            np.ones((self.row_len, 1))
            @ np.asmatrix(np.asarray(self.seams_generation_df["Max Capacity"])),
            dtype=np.int32,
        )
        self.failure_np = np.asarray(
            np.ones((self.row_len, 1))
            @ np.asmatrix(
                np.asarray(self.seams_generation_df["Forced Outage Rate"] * 0.0001)
            ),
            dtype=np.float,
        )
        self.repair_np = (
            np.asarray(
                np.ones((self.row_len, len(self.seams_generation_df.index))),
                dtype=np.float,
            )
            * 0.01
        )  # @np.ones(,dtype=np.int32)

        ### REGIONS ###
        regions_dtype = np.dtype([("name", h5py.special_dtype(vlen=str))])
        regions_data = np.zeros(
            (len(self.seams_mapping_df.index),), dtype=regions_dtype
        )
        regions_data["name"] = tuple(self.seams_mapping_df["CEP Bus ID"])
        self.regions_data = regions_data

        ### INTERFACES ###
        interfaces_dtype = np.dtype(
            [
                ("region_from", h5py.special_dtype(vlen=str)),
                ("region_to", h5py.special_dtype(vlen=str)),
            ]
        )

        interfaces_data = np.zeros(
            (len(self.cleaned_seams_transmission_df.index),), dtype=interfaces_dtype
        )
        interfaces_data["region_from"] = tuple(
            self.cleaned_seams_transmission_df["From"].astype(int).astype(str)
        )
        interfaces_data["region_to"] = tuple(
            self.cleaned_seams_transmission_df["To"].astype(int).astype(str)
        )
        self.interfaces_data = interfaces_data

        self.txfrom_np = np.asarray(
            np.ones((self.row_len, 1))
            @ np.asmatrix(np.asarray(self.cleaned_seams_transmission_df["FW"])),
            dtype=np.int32,
        )
        self.txto_np = np.asarray(
            np.ones((self.row_len, 1))
            @ np.asmatrix(np.asarray(self.cleaned_seams_transmission_df["BW"])),
            dtype=np.int32,
        )

    def create_tx_np(self):
        ### LINES ###
        lines_dtype = np.dtype(
            [
                ("name", h5py.special_dtype(vlen=str)),
                ("category", h5py.special_dtype(vlen=str)),
                ("region_from", h5py.special_dtype(vlen=str)),
                ("region_to", h5py.special_dtype(vlen=str)),
            ]
        )

        lines_data = np.zeros(
            (len(self.cleaned_seams_transmission_df.index),), dtype=lines_dtype
        )
        lines_data["name"] = tuple(self.cleaned_seams_transmission_df["Line"])
        lines_data["category"] = tuple(self.cleaned_seams_transmission_df["Area From"])
        lines_data["region_from"] = tuple(
            self.cleaned_seams_transmission_df["From"].astype(int).astype(str)
        )
        lines_data["region_to"] = tuple(
            self.cleaned_seams_transmission_df["To"].astype(int).astype(str)
        )
        self.lines_data = lines_data

        self.txfailure_np = (
            np.asarray(
                np.ones((self.row_len, len(self.cleaned_seams_transmission_df.index))),
                dtype=np.float,
            )
            * 0.0001
        )
        self.txrecovery_np = (
            np.asarray(
                np.ones((self.row_len, len(self.cleaned_seams_transmission_df.index))),
                dtype=np.float,
            )
            * 0.01
        )

    def modify_load_year(self, year, new_load_magnitude, new_load_profile):
        scalar = new_load_magnitude / (
            self.seams_load_df.iloc[:, 1:].sum(axis=1).mean()
        )
        self.seams_load_df.iloc[:, 1:] = self.seams_load_df.iloc[:, 1:] * scalar
        seams_profile = (
            self.seams_load_df.iloc[:, 1:].sum(axis=1)
            / self.seams_load_df.iloc[:, 1:].sum(axis=1).sum()
        )
        self.seams_load_df.iloc[:, 1:] = self.seams_load_df.iloc[:, 1:].mul(
            (new_load_profile.LoadMW.reset_index().LoadMW / seams_profile), axis=0
        )
        return None
        # zones 8, 9, 10 must be filled pro-rata w/ their share of miso load on similar days

    def get_vre_key(self, name):
        assert (type(name)) == str
        if "Wind" in name or "wind" in name:
            return "Wind"
        elif (
            "Distributed" in name
            or "Rooftop" in name
            or "distributed" in name
            or "rooftop" in name
        ):
            return "Rooftop"
        else:
            return "Fixed"

    def add_re_generator(
        self, name, zone, profile_ID, penetration, year, overwrite_MW=0
    ):

        zone_int = np.asarray(
            self.seams_mapping_df[self.seams_mapping_df["CEP Bus Name"] == zone][
                "CEP Bus ID"
            ]
        )[0]
        penetration_col = name + " " + penetration
        generators_data_list = list(self.generators_data)
        if overwrite_MW != 0:

            generators_data_list.append(
                (
                    zone.replace(" ", "") + name.replace(" ", "") + "2",
                    name.replace(" ", "_"),
                    zone_int,
                )
            )
        else:
            generators_data_list.append(
                (
                    zone.replace(" ", "") + name.replace(" ", ""),
                    name.replace(" ", "_"),
                    zone_int,
                )
            )
        self.generators_data = np.asarray(
            generators_data_list, dtype=self.generators_dtype
        )

        active_profile_df = self.vre_profile_df[self.get_vre_key(name)]

        # cut down to only the active year's profile
        # we can do something that mixes profiles later
        active_profile_df["datetime"] = pd.to_datetime(
            active_profile_df.index, infer_datetime_format=True
        )
        active_profile_df["year"] = [x.year for x in active_profile_df.datetime]
        active_profile_df["month_day"] = [
            str(x.month) + "-" + str(x.day) for x in active_profile_df.datetime
        ]
        # active_profile_df["day"] = [x.day for x in active_profile_df.datetime]
        year_profile_df = active_profile_df[
            (active_profile_df.year == year) & (active_profile_df.month_day != "2-29")
        ]

        # now grab the profile using the key, taking only the first row_len entries
        np_profile = np.asarray(
            year_profile_df.loc[:, str(profile_ID)][
                self.slicer : self.slicer + self.row_len
            ]
        )
        # and scale it by the capacity scenario
        zonal_capacity = self.vre_scenario_df.at[zone, penetration_col]
        final_capacity_array = np_profile * zonal_capacity  # scales profile
        if overwrite_MW != 0:
            print(overwrite_MW)
            final_capacity_array = np_profile * overwrite_MW

        # and append!
        self.capacity_np = np.hstack(
            (self.capacity_np, final_capacity_array.reshape(self.row_len, 1))
        )
        self.repair_np = np.hstack(
            (self.repair_np, np.ones(self.row_len).reshape(self.row_len, 1))
        )
        self.failure_np = np.hstack(
            (self.failure_np, np.zeros(self.row_len).reshape(self.row_len, 1))
        )

    def add_all_re_profs(
        self,
        penetration,
        year,
        profile_types=["Utility Wind", "Distributed Solar", "Utility Solar"],
        choice="random",
    ):
        assert (type(choice)) == str
        choice = choice.lower()  # removes casing issues
        for zone in self.vre_scenario_df.index:
            print("adding VRE in zone " + zone + " to generation profiles")
            for p in profile_types:
                if choice == "max":
                    id_list = [
                        str(i)
                        for i in list(
                            self.miso_geography_df[
                                self.miso_geography_df.FINAL_SEAMS_ZONE == zone
                            ].Name
                        )
                    ]
                    profile_ID = (
                        self.vre_profile_df[self.get_vre_key(p)][id_list].sum().idxmax()
                    )

                elif choice == "min":
                    id_list = [
                        str(i)
                        for i in list(
                            self.miso_geography_df[
                                self.miso_geography_df.FINAL_SEAMS_ZONE == zone
                            ].Name
                        )
                    ]
                    profile_ID = (
                        self.vre_profile_df[self.get_vre_key(p)][id_list].sum().idxmax()
                    )
                else:
                    profile_ID = np.random.choice(
                        self.miso_geography_df[
                            self.miso_geography_df.FINAL_SEAMS_ZONE == zone
                        ].Name.unique()
                    )
                self.add_re_generator(p, zone, profile_ID, penetration, year)
        print("...done adding VRE profiles")

    def hstack_helper(self, array, value):
        return np.hstack((array, np.ones((self.row_len, 1)) * value))

    def add_all_storage_resource(self, capacity, duration):
        for zone in self.vre_scenario_df.index:
            print(
                "adding "
                + str(capacity)
                + "MW, "
                + str(duration)
                + "-hour storage in zone "
                + zone
                + " to storage profiles"
            )
            self.add_storage_resource(zone, capacity, duration)

    def add_storage_resource(
        self,
        zone,
        capacity,
        duration,
        charge_efficiency=0.9,
        discharge_efficiency=1,
        p_repair=1.0,
        p_fail=0.0,
        carryover_efficiency=1.0,
    ):
        assert (type(zone)) == str
        zone_int = np.asarray(
            self.seams_mapping_df[self.seams_mapping_df["CEP Bus Name"] == zone][
                "CEP Bus ID"
            ]
        )[0]
        name = zone.replace(" ", "") + "_Storage_" + str(duration) + "hour"
        # add in name, zone of storage resource
        if hasattr(self, "storage_data_list"):
            self.storage_data_list.append((zone, name, zone_int))
            self.storage_charge_capacity_np = self.hstack_helper(
                self.storage_charge_capacity_np, capacity
            )
            self.storage_discharge_capacity_np = self.hstack_helper(
                self.storage_discharge_capacity_np, capacity
            )
            self.storage_energy_np = self.hstack_helper(
                self.storage_energy_np, capacity * duration
            )
            self.storage_discharge_efficiency_np = self.hstack_helper(
                self.storage_discharge_efficiency_np, discharge_efficiency
            )
            self.storage_charge_efficiency_np = self.hstack_helper(
                self.storage_charge_efficiency_np, charge_efficiency
            )
            self.storage_carryover_efficiency_np = self.hstack_helper(
                self.storage_carryover_efficiency_np, carryover_efficiency
            )
            self.storage_repair_np = self.hstack_helper(
                self.storage_repair_np, p_repair
            )
            self.storage_fail_np = self.hstack_helper(self.storage_fail_np, p_fail)
        else:
            self.storage_data_list = [(zone, name, zone_int)]
            self.storage_charge_capacity_np = np.ones((self.row_len, 1)) * capacity
            self.storage_discharge_capacity_np = np.ones((self.row_len, 1)) * capacity
            self.storage_energy_np = np.ones((self.row_len, 1)) * capacity * duration
            self.storage_discharge_efficiency_np = (
                np.ones((self.row_len, 1)) * discharge_efficiency
            )
            self.storage_charge_efficiency_np = (
                np.ones((self.row_len, 1)) * charge_efficiency
            )
            self.storage_carryover_efficiency_np = (
                np.ones((self.row_len, 1)) * carryover_efficiency
            )
            self.storage_repair_np = np.ones((self.row_len, 1)) * p_repair
            self.storage_fail_np = np.ones((self.row_len, 1)) * p_fail
        self.storage_data = np.asarray(
            self.storage_data_list, dtype=self.generators_dtype
        )
        return None

    def write_h5pyfile(self, filename, load_scalar=1):
        assert (type(filename)) == str
        pras_name = filename + ".pras"
        os.chdir(os.path.join(self.folder_datapath, "PRAS_files"))
        with h5py.File(pras_name, "w", track_order=True) as f:
            # attrs
            for k, v in self.metadata.items():
                f.attrs[k] = v
                if type(f.attrs[k]) == np.int32:
                    f.attrs[k] = np.int64(f.attrs[k])  # dtype(np.int64)
                    # print(type(f.attrs[k]))
            # regions
            regions_group = f.create_group("regions")
            regions_group.create_dataset("_core", data=self.regions_data)  # rcore =
            regions_group.create_dataset(
                "load",
                data=np.asarray(
                    load_scalar
                    * self.seams_load_df.iloc[
                        self.slicer : self.slicer + self.row_len, 1:
                    ],
                    dtype=np.int32,
                ),
                dtype=np.int32,
            )

            # generators
            generators_group = f.create_group("generators")
            generators_group.create_dataset(
                "_core", data=self.generators_data
            )  # gcore =
            generators_group.create_dataset(
                "capacity", data=self.capacity_np, dtype=np.int32
            )
            generators_group.create_dataset(
                "failureprobability", data=self.failure_np, dtype=np.float
            )
            generators_group.create_dataset(
                "repairprobability", data=self.repair_np, dtype=np.float
            )

            # storages, if they exist
            if hasattr(self, "storage_data"):
                storages_group = f.create_group("storages")
                storages_group.create_dataset("_core", data=self.storage_data)
                storages_group.create_dataset(
                    "chargecapacity",
                    data=self.storage_charge_capacity_np,
                    dtype=np.int32,
                )
                storages_group.create_dataset(
                    "dischargecapacity",
                    data=self.storage_discharge_capacity_np,
                    dtype=np.int32,
                )
                storages_group.create_dataset(
                    "energycapacity", data=self.storage_energy_np, dtype=np.int32
                )
                storages_group.create_dataset(
                    "chargeefficiency",
                    data=self.storage_charge_efficiency_np,
                    dtype=np.float,
                )
                storages_group.create_dataset(
                    "dischargeefficiency",
                    data=self.storage_discharge_efficiency_np,
                    dtype=np.float,
                )
                storages_group.create_dataset(
                    "carryoverefficiency",
                    data=self.storage_carryover_efficiency_np,
                    dtype=np.float,
                )
                storages_group.create_dataset(
                    "failureprobability", data=self.storage_fail_np, dtype=np.float
                )
                storages_group.create_dataset(
                    "repairprobability", data=self.storage_repair_np, dtype=np.float
                )
            # interfaces
            interfaces_group = f.create_group("interfaces")
            interfaces_group.create_dataset("_core", data=self.interfaces_data)
            interfaces_group.create_dataset(
                "forwardcapacity", data=self.txfrom_np, dtype=np.int32
            )
            interfaces_group.create_dataset(
                "backwardcapacity", data=self.txto_np, dtype=np.int32
            )

            # lines
            lines_group = f.create_group("lines")
            lines_group.create_dataset("_core", data=self.lines_data)
            lines_group.create_dataset(
                "forwardcapacity", data=self.txfrom_np, dtype=np.int32
            )
            lines_group.create_dataset(
                "backwardcapacity", data=self.txto_np, dtype=np.int32
            )
            lines_group.create_dataset(
                "failureprobability", data=self.txfailure_np, dtype=np.float
            )
            lines_group.create_dataset(
                "repairprobability", data=self.txrecovery_np, dtype=np.float
            )


class NRELEFSprofiles(object):
    def __init__(self, folder_datapath, scenario):
        self.folder_datapath = folder_datapath
        self.scenario = scenario

    def load_zip_archive(self):
        url = "https://data.nrel.gov/system/files/126/" + self.scenario + ".zip"
        r = requests.get(url, stream=True)
        self.zip_path = self.folder_datapath + "\\NRELdata.zip"
        with open(self.zip_path, "wb") as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
        return None

    def csv_to_df(self):
        csv_path = os.path.join(self.folder_datapath, self.scenario + ".csv")
        if os.path.exists(csv_path):
            print("csv already exists in folder, so reading...")
            self.df = pd.read_csv(csv_path)
        else:
            try:
                with ZipFile(self.zip_path, "r") as zipObj:
                    # Extract all the contents of zip file in different directory
                    zipObj.extractall(self.folder_datapath)
            except:
                print("An exception occurred extracting with Python ZipFile library.")
                print("Attempting to extract using 7zip")
                subprocess.Popen(
                    [
                        r"C:\Program Files\7-Zip\7z.exe",
                        "e",
                        f"{self.zip_path}",
                        f"-o{self.folder_datapath}",
                        "-y",
                    ]
                )
            self.df = pd.read_csv(csv_path)
        print("...NREL csv read into dataframe")
        return None

    def process_csv(
        self,
        year,
        miso_states=["MN", "ND", "MI", "WI", "IA", "IL", "IN", "AK", "LA", "MS"],
    ):
        # for now, just year, but may also subset states in future
        load = (
            self.df[
                (self.df.State.isin(miso_states)) & (self.df.Year == year)
            ].LoadMW.sum()
            / 8760.0
        )
        load8760 = (
            self.df[(self.df.State.isin(miso_states)) & (self.df.Year == year)]
            .groupby("LocalHourID")
            .sum()[["LoadMW"]]
        )
        norm8760load = load8760.div(load8760.sum(axis=0), axis=1)
        if load == 0.0:
            ylist = list(self.df.Year.unique())
            raise ValueError(
                "Invalid Year: only "
                + ", ".join(map(str, ylist))
                + " are valid years for this NREL scenario"
            )
        return (load, norm8760load)

    def run_all(self, year):
        self.load_zip_archive()
        self.csv_to_df()
        return self.process_csv(year)
