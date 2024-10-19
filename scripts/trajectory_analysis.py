import pandas as pd
from datetime import datetime, date
import numpy as np
import os
from MapVisualization import map_matching_plot
from Utilities import timestamp_handle, timestamp2datestr, distance


def taxi_data_match(passenger_id):
    label = pd.read_csv("Data\\" + passenger_id + "\\labels.txt", delimiter='\t')
    label = label_handle(label)

    data = pd.read_csv("Data\\preprocessed_data\\" + passenger_id + ".csv")
    data = data_handle(data)

    index_taxi = []
    for index, value in data["Date"].items():
        for start, end in zip(label["Start Time"], label["End Time"]):
            if start <= value <= end:
                index_taxi.append(index)
                break

    data = data.loc[index_taxi]
    data.drop(columns=[data.columns[0]], inplace=True)
    data.to_csv("TrajectoryTaxi\\" + passenger_id + "TrajectoryTaxiBeijing.csv")


def map_matching_by_time_interval(passenger_id):
    taxi_data_match(passenger_id=passenger_id)

    demo_map = folium.Map(
        tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图',
        location=[39.93, 116.40],
        zoom_start=12)

    data_original = pd.read_csv(passenger_id + "test.csv")
    print(data_original)

    begin_a = date(2008, 7, 22)
    begin_b = date(2008, 7, 23)
    one_more_day = begin_b - begin_a

    begin = timestamp2datestr(data_original["Date"].min())
    end = timestamp2datestr(data_original["Date"].max())

    interval_days = []
    for d in range((end - begin).days + 1):
        day = begin
        for i in range(d):
            day += one_more_day
        interval_days.append(day)

    for day in interval_days:
        day = str(day).replace('-', '/')[:10]
        data = data_original.__deepcopy__()
        data_wanted = data[
            (timestamp_handle(day + " 00:00:00") < data["Date"]) & (
                    data["Date"] < timestamp_handle(day + " 23:59:59"))].index
        data = data.loc[data_wanted]

        locations = []
        for latitude, longitude in zip(data['Latitude'], data['Longitude']):
            locations.append([latitude, longitude])
        print(locations)

        if not locations:
            continue

        folium.PolyLine(
            locations=locations,
            weight=3,
            color='red',
            opacity=0.8
        ).add_to(demo_map)

    demo_map.save('demo_map.html')
    os.system(r"demo_map.html")


def denoise(data):
    data = data[(data["Latitude"] < (41 + 3 / 60)) & (data["Latitude"] > (39 + 26 / 60))]
    data = data[(data["Longitude"] < (117 + 30 / 60)) & (data["Longitude"] > (115 + 25 / 60))]
    data.drop(columns=[data.columns[0]], inplace=True)
    data.sort_values(by="Date", inplace=True)
    start = 0
    group_marks = []
    for index, value in data["Date"].items():
        if (index + 1) != data.shape[0]:
            try:
                if data["Date"][index + 1] - value > 30:
                    group_marks.append([start, index])
                    start = index + 1
            except:
                continue

    trajectories = []
    for each in group_marks:
        try:
            data_wanted_index = data[
                (data["Date"][each[0]] <= data["Date"]) & (data["Date"] <= data["Date"][each[1]])].index
            data_wanted = data.loc[data_wanted_index]
        except:
            continue

        trajectory = []
        for latitude, longitude in zip(data_wanted['Latitude'], data_wanted['Longitude']):
            trajectory.append([latitude, longitude])

        trajectories.append(trajectory)
    print(len(trajectories))
    return data


def POIVisual010(file):
    data = pd.read_csv(file)
    data.drop(columns=[data.columns[0]], inplace=True)
    data.sort_values(by="Date", inplace=True)
    start = 0
    group_marks = []
    for index, value in data["Date"].items():
        if (index + 1) != data.shape[0]:
            if data["Date"][index + 1] - value > 30:
                group_marks.append([start, index])
                start = index + 1

    data_POI = pd.DataFrame(columns=["Latitude", "Longitude", "Altitude", "Date"])
    for each in group_marks:
        new_row = [data["Latitude"][each[0]], data["Longitude"][each[0]], data["Altitude"][each[0]],
                   data["Date"][each[0]]]
        data_POI.loc[len(data_POI)] = new_row
    data_POI["Date"] = data_POI["Date"].apply(lambda x: timestamp2datestr(x))

    data_POI.to_csv(file.replace("TrajectoryTaxiBeijing", "POIStarts"))

    data_destination = pd.DataFrame(columns=["Latitude", "Longitude", "Altitude", "Date"])
    for each in group_marks:
        new_row = [data["Latitude"][each[1]], data["Longitude"][each[1]], data["Altitude"][each[1]],
                   data["Date"][each[1]]]
        data_destination.loc[len(data_destination)] = new_row
    data_destination.to_csv(file.replace("TrajectoryTaxiBeijing", "POIDestination"))


def trajectory_segment(file):
    demo_map = folium.Map(
        tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图',
        location=[39.93, 116.40],
        zoom_start=12)
    data_original = pd.read_csv(file)
    data = data_original.__deepcopy__()
    data_original.drop(columns=[data.columns[0]], inplace=True)
    data_original.sort_values(by="Date", inplace=True)
    start = 0
    group_marks = []
    for index, value in data_original["Date"].items():
        if (index + 1) != data_original.shape[0]:
            if data_original["Date"][index + 1] - value > 30:
                group_marks.append([start, index])
                start = index + 1

    for i, day in zip(range(len(group_marks)), group_marks):
        data = data_original.__deepcopy__()
        data.reset_index(inplace=True)
        data.drop(columns=data.columns[0], inplace=True)
        data = data.loc[group_marks[i][0]:group_marks[i][1]]

        locations = []
        for latitude, longitude in zip(data['Latitude'], data['Longitude']):
            locations.append([latitude, longitude])
        print(locations)

        if len(locations) < 100:
            continue
        if not locations:
            continue

        folium.PolyLine(
            locations=locations,
            weight=3,
            color='red',
            opacity=0.8
        ).add_to(demo_map)

    demo_map.save(file.replace(".csv", ".html"))