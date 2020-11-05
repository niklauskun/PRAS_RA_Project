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
import time
import h5py
import json
import datetime as dt

# script imports
from MISO_data_utility_functions import (
    LoadMISOData,
    LoadVREScenarios,
    LoadVREData,
    CreateHDF5,
    NRELEFSprofiles,
)


# general inputs
RE_sheet = "Wind-heavy by energy"
# RE_sheet = "More balanced by energy"
row_len = 8760  # for HDF5 file
slice_in_index = 0  # 0 if you want to start on 1/1
re_penetration = "0.2"
profile_year = 2012
NREL = False
NREL_year, NREL_profile = 2040, "EFSLoadProfile_Reference_Moderate"
pras_filename = "VRE0.2_wind_2012base100%_8760_18%IRM_nostorage"
load_scalar = 1  # how much to scale resulting load profile by... 1 should be default
target_IRM = 0.18  # as a fraction
use_target_IRM = True  #
storage_capacity = 0  # total storage capacity, in MW
# fliename convention is VREscenario_REscenario_year_hours_RMmodifier_storage

folder = "testPRAS11.3"  # whatever you name your folder when pulled from Github

if slice_in_index + row_len > 8760:
    raise ValueError("cannot index beyond 8760")

# datapaths
folder_datapath = join(os.environ["HOMEPATH"], "Desktop", folder)
miso_datapath = join(os.environ["HOMEPATH"], "Desktop", folder, "VREData")
hifld_datapath = join(os.environ["HOMEPATH"], "Desktop", folder, "HIFLD_shapefiles")
shp_path = (
    os.environ["CONDA_PREFIX"] + r"\Library\share\gdal"
)  # you need this to pull the retail shapefile, which doesn't come with everything else

## testing of new function in inline code ##
if NREL:
    nreltester = NRELEFSprofiles(folder_datapath, NREL_profile)
    load, normprofile = nreltester.run_all(NREL_year)

# VRE data load
vredata = LoadVREData(folder_datapath)
vre_data_df = vredata.load_all_dfs()  # all VRE profiles loaded into dictionary

# SEAMS bus assignment
SEAMS_LRZ_map = {}
SEAMS_LRZ_map["MN-C"] = ["LRZ 1"]
SEAMS_LRZ_map["MN-NE"] = ["LRZ 1"]
SEAMS_LRZ_map["MN-SE"] = ["LRZ 1"]
SEAMS_LRZ_map["ATC"] = ["LRZ 2"]
SEAMS_LRZ_map["UPPC"] = ["LRZ 2"]
SEAMS_LRZ_map["MEC"] = ["LRZ 3"]
SEAMS_LRZ_map["IA-E"] = ["LRZ 3"]
SEAMS_LRZ_map["CBPC-NIPCO"] = ["LRZ 3"]
SEAMS_LRZ_map["IL-C"] = ["LRZ 4"]
SEAMS_LRZ_map["SIPC"] = ["LRZ 4"]
SEAMS_LRZ_map["MISO-MO"] = ["LRZ 5"]
SEAMS_LRZ_map["AECIZ"] = ["LRZ 5"]
SEAMS_LRZ_map["IN-S"] = ["LRZ 6"]
SEAMS_LRZ_map["IN-C"] = ["LRZ 6"]
SEAMS_LRZ_map["NIPS"] = ["LRZ 6"]
SEAMS_LRZ_map["CONS"] = ["LRZ 7"]
SEAMS_LRZ_map["DECO"] = ["LRZ 7"]
SEAMS_LRZ_map["EES-ARK"] = ["LRZ 8"]
SEAMS_LRZ_map["LA-GULF"] = ["LRZ 9"]
SEAMS_LRZ_map["LA-N"] = ["LRZ 9"]
SEAMS_LRZ_map["EES-TX"] = ["LRZ 9"]
SEAMS_LRZ_map["MISO-MS"] = ["LRZ 10"]

# metadata for HDF5
orig_time = dt.datetime.strptime("2012-01-01", "%Y-%m-%d")
new_time = orig_time + dt.timedelta(
    hours=slice_in_index
)  # +24 accounts for leap for now

metadata = {
    "pras_dataversion": "v0.5.0",
    "start_timestamp": new_time.strftime("%Y-%m-%d") + "T00:00:00-05:00",
    "timestep_count": row_len,
    "timestep_length": 1,
    "timestep_unit": "h",
    "power_unit": "MW",
    "energy_unit": "MWh",
}

## MAIN CODE RUN ##
## DO NOT MODIFY ##
start_time = time.time()

# load bus-level wind and solar locs, match to MISO zones
miso_data = LoadMISOData(folder_datapath, miso_datapath, hifld_datapath, shp_path)
miso_data.convert_CRS()
miso_data.utility_territory_mapping()
miso_data.initialize_SEAMS_match()
miso_data.match_unmatched_buses()
miso_data.handle_multimatch_buses()
miso_data.plot_points(showplot=False)
final_miso_data = miso_data.returndata()

# assign capacity scenario to buses
vre_scenarios = LoadVREScenarios(folder_datapath, RE_sheet)
vre_capacity_scenarios = vre_scenarios.create_seams_df(SEAMS_LRZ_map)

# assign or modify load data?

# pras formatting
HDF5_data = CreateHDF5(
    folder_datapath,
    row_len,
    slice_in_index,
    metadata,
    vre_data_df,
    final_miso_data,
    vre_capacity_scenarios,
)
HDF5_data.create_gens_np()
HDF5_data.create_zone_np()
HDF5_data.create_tx_np()

if NREL:
    HDF5_data.modify_load_year(NREL_year, load, normprofile)

# add a single generator, if desired
# prof_id = np.random.choice(
#    final_miso_data[final_miso_data.FINAL_SEAMS_ZONE == "MEC"].Name.unique()
# )
# HDF5_data.add_re_generator("Utility Wind", "MEC", prof_id, "0.1", 2012)

# add a single storage resource, if desired
# HDF5_data.add_storage_resource("MEC", 100, 3)

# HDF5_data.calc_top_N_hours_load(8)

# add selected sceanario VRE capacity
HDF5_data.add_all_re_profs(re_penetration, profile_year, choice="random", N=15)

# add a generic sized storage resource at all buses, if desired
HDF5_data.add_all_storage_resource(
    storage_capacity, 6, alloc_method="prorataVRE"
)  # now is total capacity and duration

# irm adjustment to load, if desired
if use_target_IRM:
    load_scalar = HDF5_data.calc_IRM(target_IRM, storage_capacity)

# HDF5_data.add_re_generator(
#    "Utility Wind", "LA-GULF", 261010, "0.1", 2012, 8, overwrite_MW=100
# )


# finally, export PRAS case
HDF5_data.write_h5pyfile(pras_filename, load_scalar=load_scalar)


# how long did this take?
end_time = time.time() - start_time
print("time elapsed during run is " + str(round(end_time, 2)) + " seconds")
