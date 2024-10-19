import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime


def drop_no_labels():
    for index in os.listdir("Data"):
        path = os.listdir("Data/" + index)
        if "labels.txt" not in path:
            shutil.rmtree("Data/" + index)


def data_integration_individual():
    for individual_id in os.listdir("Data"):
        path = "Data/" + individual_id + "/Trajectory/"
        data_individual = pd.DataFrame(columns=["Latitude", "Longitude", "Altitude", "Date"])
        for index in os.listdir(path):
            data_each = np.genfromtxt(path + index, dtype=[float, float, int, float], delimiter=",",
                                      skip_header=6, usecols=(0, 1, 3, 4),
                                      names=["Latitude", "Longitude", "Altitude", "Date"])
            data_each = pd.DataFrame(data=data_each)
            data_individual = pd.concat([data_individual, data_each], axis=0, ignore_index=True)
        data_individual.to_csv("Data/" + individual_id + "/" + individual_id + ".csv")
        data_individual.to_csv("Data/preprocessed_data/" + individual_id + ".csv")


def label_handle(label):
    label = pd.DataFrame(data=label)
    index_to_drop_label = label[label["Transportation Mode"] != "taxi"].index
    label.drop(index_to_drop_label, inplace=True)

    label["Start Time"] = label["Start Time"].apply(
        lambda x: int(datetime.fromisoformat(x.replace("/", "-")).timestamp()))
    label["End Time"] = label["End Time"].apply(lambda x: int(datetime.fromisoformat(x.replace("/", "-")).timestamp()))
    label.to_csv("b.csv")
    return label


def data_handle(data):
    a = 39298.1462037037 * 24 * 60 * 60
    b = datetime.fromisoformat("2007-08-04 03:30:32").timestamp()
    interval = a - b
    data["Date"] = data["Date"].apply(lambda x: int(x * 24 * 60 * 60 - int(interval + 1)))
    data.to_csv("a.csv")
    return data