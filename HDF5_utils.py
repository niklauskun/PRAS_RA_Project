import numpy as np
import pandas as pd
import h5py
import json
import os


def load_seams(
    sheet, wb="NREL-Seams Model (MISO).xlsx", path="C:/Users/llavin/Desktop/PRAS"
):
    return pd.read_excel(os.path.join(path, wb), sheet_name=sheet)


def clean_tx(df, cols):
    return df[cols].dropna(subset=cols)


def clean_gen(df, map_df):
    return df[df["Bubble"].isin(list(map_df["CEP Bus ID"]))].reset_index()


def add_gen(df, add_gens):
    # print(df.at[len(df) - 1, "index"])
    df.loc[len(df)] = [df.at[len(df) - 1, "index"]] + add_gens
    return df


def create_gen_failure_recovery_cols(df):
    df["RecoveryRate"] = [1.0 / df.at[i, "Mean Time to Repair"] for i in df.index]
    df["RecoveryRate"].fillna(1.0, inplace=True)
    df["RecoveryRate"].replace([np.inf, -np.inf], 1.0, inplace=True)

    df["FailureRate"] = [
        (0.01 * a * b) / (1.0 + 0.01 * a)
        for a, b in zip(df["Forced Outage Rate"], df["RecoveryRate"])
    ]
    df["FailureRate"].replace([np.inf, -np.inf], 0.0, inplace=True)
    return df


class HDF5Case(object):
    def __init__(
        self, tx, gen, load, mapping, tmps, solar_shape, wind_shape, include_vg=True
    ):
        self.tx = tx
        self.gen = gen
        self.load = load
        self.mapping = mapping
        self.tmps = tmps  # n of timepoints to use in case
        self.solar_shape = np.asarray(solar_shape[:tmps])
        self.wind_shape = np.asarray(wind_shape[:tmps])
        self.include_vg = include_vg

    def create_gens(self):
        ### GENERATORS ###
        generators_dtype = np.dtype(
            [
                ("name", h5py.special_dtype(vlen=str)),
                ("category", h5py.special_dtype(vlen=str)),
                ("region", h5py.special_dtype(vlen=str)),
            ]
        )

        self.gen_data = np.zeros((len(self.gen.index),), dtype=generators_dtype)
        self.gen_data["name"] = tuple(self.gen["Generator Name"])
        self.gen_data["category"] = tuple(self.gen["category"])
        self.gen_data["region"] = tuple(self.gen["Bubble"])
        self.capacity_np = np.asarray(
            np.ones((self.tmps, 1)) @ np.asmatrix(np.asarray(self.gen["Max Capacity"])),
            dtype=np.int32,
        )
        for index, label in enumerate(self.gen["category"]):
            if label == "Solar" and self.include_vg:
                self.capacity_np[:, index] = (
                    self.capacity_np[:, index] * self.solar_shape
                )
            elif label == "Wind" and self.include_vg:
                self.capacity_np[:, index] = (
                    self.capacity_np[:, index] * self.wind_shape
                )
            elif label == "Solar" or label == "Wind":
                self.capacity_np[:, index] = self.capacity_np[:, index] * 0

        self.failure_np = np.asarray(
            np.ones((self.tmps, 1)) @ np.asmatrix(np.asarray(self.gen["FailureRate"])),
            dtype=np.float,
        )
        self.repair_np = np.asarray(
            np.ones((self.tmps, 1)) @ np.asmatrix(np.asarray(self.gen["RecoveryRate"])),
            dtype=np.float,
        )

    def create_regions(self):
        regions_dtype = np.dtype([("name", h5py.special_dtype(vlen=str))])
        self.regions_data = np.zeros((len(self.mapping.index),), dtype=regions_dtype)
        self.regions_data["name"] = tuple(self.mapping["CEP Bus ID"])

    def create_interfaces(self):
        interfaces_dtype = np.dtype(
            [
                ("region_from", h5py.special_dtype(vlen=str)),
                ("region_to", h5py.special_dtype(vlen=str)),
            ]
        )

        self.interfaces_data = np.zeros((len(self.tx.index),), dtype=interfaces_dtype)
        self.interfaces_data["region_from"] = tuple(
            self.tx["From"].astype(int).astype(str)
        )
        self.interfaces_data["region_to"] = tuple(self.tx["To"].astype(int).astype(str))

        self.txfrom_np = np.asarray(
            np.ones((self.tmps, 1)) @ np.asmatrix(np.asarray(self.tx["FW"])),
            dtype=np.int32,
        )
        self.txto_np = np.asarray(
            np.ones((self.tmps, 1)) @ np.asmatrix(np.asarray(self.tx["BW"])),
            dtype=np.int32,
        )

    def create_lines(self):
        lines_dtype = np.dtype(
            [
                ("name", h5py.special_dtype(vlen=str)),
                ("category", h5py.special_dtype(vlen=str)),
                ("region_from", h5py.special_dtype(vlen=str)),
                ("region_to", h5py.special_dtype(vlen=str)),
            ]
        )

        self.lines_data = np.zeros((len(self.tx.index),), dtype=lines_dtype)
        self.lines_data["name"] = tuple(self.tx["Line"])
        self.lines_data["category"] = tuple(self.tx["Area From"])
        self.lines_data["region_from"] = tuple(self.tx["From"].astype(int).astype(str))
        self.lines_data["region_to"] = tuple(self.tx["To"].astype(int).astype(str))

        self.txfailure_np = (
            np.asarray(np.ones((self.tmps, len(self.tx.index))), dtype=np.float) * 0.0
        )
        self.txrecovery_np = np.asarray(
            np.ones((self.tmps, len(self.tx.index))), dtype=np.float
        )

    def create_storages(self):
        print("storage not yet written")
        return None

    def create_all(self):
        self.create_gens()
        self.create_regions()
        self.create_interfaces()
        self.create_lines()

    def write_HDF5(self, filename, **kwargs):
        with h5py.File(filename, "w", track_order=True) as f:
            # attrs
            for k, v in kwargs.items():
                f.attrs[k] = v
                if type(f.attrs[k]) == np.int32:
                    f.attrs[k] = np.int64(f.attrs[k])
            # regions
            regions_group = f.create_group("regions")
            regions_group.create_dataset("_core", data=self.regions_data)

            regions_group.create_dataset(
                "load",
                data=np.asarray(self.load.iloc[: self.tmps, 1:] * 2.0, dtype=np.int32),
                dtype=np.int32,
            )

            # generators
            generators_group = f.create_group("generators")
            generators_group.create_dataset("_core", data=self.gen_data)
            generators_group.create_dataset(
                "capacity", data=self.capacity_np, dtype=np.int32
            )
            generators_group.create_dataset(
                "failureprobability", data=self.failure_np, dtype=np.float
            )
            generators_group.create_dataset(
                "repairprobability", data=self.repair_np, dtype=np.float
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
