import random

import numpy as np
import pandas as pd
import os
import sys
import shutil
import folium
import webbrowser
import sklearn
from datetime import datetime, date
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

'''
font = {'family': 'SimSun',  # 宋体
        # 'weight': 'bold',  # 加粗
        'size': '10.5'  # 五号
        }
plt.rc('font', **font)
plt.rc('axes', unicode_minus=False)
plt.rcParams['figure.facecolor'] = "#FFFFF0"  # 设置窗体颜色
plt.rcParams['axes.facecolor'] = "#FFFFF0"  # 设置绘图区颜色
'''


class Kf_Params:
    B = 0  # 外部输入为0
    u = 0  # 外部输入为0
    K = float('nan')  # 卡尔曼增益无需初始化
    z = float('nan')  # 这里无需初始化，每次使用kf_update之前需要输入观察值z
    P = np.diag(np.ones(4))  # 初始P设为0 ??? zeros(4, 4)

    # 初始状态：函数外部提供初始化的状态，本例使用观察值进行初始化，vx，vy初始为0
    x = []
    G = []

    # 状态转移矩阵A
    # 和线性系统的预测机制有关，这里的线性系统是上一刻的位置加上速度等于当前时刻的位置，而速度本身保持不变
    A = np.eye(4) + np.diag(np.ones((1, 2))[0, :], 2)

    # 预测噪声协方差矩阵Q：假设预测过程上叠加一个高斯噪声，协方差矩阵为Q
    # 大小取决于对预测过程的信任程度。比如，假设认为运动目标在y轴上的速度可能不匀速，那么可以把这个对角矩阵
    # 的最后一个值调大。有时希望出来的轨迹更平滑，可以把这个调更小
    Q = np.diag(np.ones(4)) * 0.1

    # 观测矩阵H：z = H * x
    # 这里的状态是（坐标x， 坐标y， 速度x， 速度y），观察值是（坐标x， 坐标y），所以H = eye(2, 4)
    H = np.eye(2, 4)

    # 观测噪声协方差矩阵R：假设观测过程上存在一个高斯噪声，协方差矩阵为R
    # 大小取决于对观察过程的信任程度。比如，假设观测结果中的坐标x值常常很准确，那么矩阵R的第一个值应该比较小
    R = np.diag(np.ones(2)) * 0.1


def kf_init(px, py, vx, vy):
    # 本例中，状态x为（坐标x， 坐标y， 速度x， 速度y），观测值z为（坐标x， 坐标y）
    kf_params = Kf_Params()
    kf_params.B = 0
    kf_params.u = 0
    kf_params.K = float('nan')
    kf_params.z = float('nan')
    kf_params.P = np.diag(np.ones(4))
    kf_params.x = [px, py, vx, vy]
    kf_params.G = [px, py, vx, vy]
    kf_params.A = np.eye(4) + np.diag(np.ones((1, 2))[0, :], 2)
    kf_params.Q = np.diag(np.ones(4)) * 0.1
    kf_params.H = np.eye(2, 4)
    kf_params.R = np.diag(np.ones(2)) * 0.1
    return kf_params


def kf_update(kf_params):
    # 以下为卡尔曼滤波的五个方程（步骤）
    a1 = np.dot(kf_params.A, kf_params.x)
    a2 = kf_params.B * kf_params.u
    x_ = np.array(a1) + np.array(a2)

    b1 = np.dot(kf_params.A, kf_params.P)
    b2 = np.dot(b1, np.transpose(kf_params.A))
    p_ = np.array(b2) + np.array(kf_params.Q)

    c1 = np.dot(p_, np.transpose(kf_params.H))
    c2 = np.dot(kf_params.H, p_)
    c3 = np.dot(c2, np.transpose(kf_params.H))
    c4 = np.array(c3) + np.array(kf_params.R)
    c5 = np.linalg.matrix_power(c4, -1)
    kf_params.K = np.dot(c1, c5)

    d1 = np.dot(kf_params.H, x_)
    d2 = np.array(kf_params.z) - np.array(d1)
    d3 = np.dot(kf_params.K, d2)
    kf_params.x = np.array(x_) + np.array(d3)

    e1 = np.dot(kf_params.K, kf_params.H)
    e2 = np.dot(e1, p_)
    kf_params.P = np.array(p_) - np.array(e2)

    kf_params.G = x_
    return kf_params


def accuracy(predictions, labels):
    return np.array(predictions) - np.array(labels)


def kf_handle(data):
    # 真实路径
    path = './9.xlsx'
    data_A = pd.read_excel(path, header=None)
    data_A_x = list(data_A.iloc[::, 0])
    data_A_y = list(data_A.iloc[::, 1])
    A = np.array(list(zip(data_A_x, data_A_y)))

    # plt.subplot(131)
    plt.figure()
    plt.plot(data_A_x, data_A_y, 'b-+')
    # plt.title('实际的真实路径')

    # 检测到的路径
    path = './10.xlsx'
    data_B = pd.read_excel(path, header=None)
    data_B_x = list(data_B.iloc[::, 0])
    data_B_y = list(data_B.iloc[::, 1])
    B = np.array(list(zip(data_B_x, data_B_y)))

    # plt.subplot(132)
    plt.plot(data_B_x, data_B_y, 'r-+')
    # plt.title('检测到的路径')

    # 卡尔曼滤波
    kf_params_record = np.zeros((len(data_B), 4))
    kf_params_p = np.zeros((len(data_B), 4))
    t = len(data_B)
    kalman_filter_params = kf_init(data_B_x[0], data_B_y[0], 0, 0)
    for i in range(t):
        if i == 0:
            kalman_filter_params = kf_init(data_B_x[i], data_B_y[i], 0, 0)  # 初始化
        else:
            # print([data_B_x[i], data_B_y[i]])
            kalman_filter_params.z = np.transpose([data_B_x[i], data_B_y[i]])  # 设置当前时刻的观测位置
            kalman_filter_params = kf_update(kalman_filter_params)  # 卡尔曼滤波
        kf_params_record[i, ::] = np.transpose(kalman_filter_params.x)
        kf_params_p[i, ::] = np.transpose(kalman_filter_params.G)

    kf_trace = kf_params_record[::, :2]
    kf_trace_1 = kf_params_p[::, :2]

    # plt.subplot(133)
    plt.plot(kf_trace[::, 0], kf_trace[::, 1], 'g-+')
    plt.plot(kf_trace_1[1:26, 0], kf_trace_1[1:26, 1], 'm-+')
    legend = ['CMA最佳路径数据集', '检测路径', '卡尔曼滤波结果', '预测路径']
    plt.legend(legend, loc="best", frameon=False)
    plt.title('卡尔曼滤波后的效果')
    plt.savefig('result.svg', dpi=600)
    plt.show()
    # plt.close()

    p = accuracy(kf_trace, A)
    print(p)


# 删除所有没有labels.txt标明出行方式的文件数据
def drop_no_labels():
    for index in os.listdir(r"Data"):
        path = os.listdir("Data/" + index)
        if "labels.txt" not in path:
            shutil.rmtree("Data/" + index)


# 将属于同一个人的轨迹整合到一个csv文件
def data_integration_individual():
    for individual_id in os.listdir("Data"):
        path = "Data/" + individual_id + "/Trajectory/"
        data_individual = pd.DataFrame(columns=["Latitude", "Longitude", "Altitude", "Date"])
        for index in os.listdir(path):
            # 读取数据集，提取的数据是纬度，经度，海拔，时间
            # Date - number of days (with fractional part) that have passed since 12/30/1899
            data_each = np.genfromtxt(path + index, dtype=[float, float, int, float], delimiter=",",
                                      skip_header=6, usecols=(0, 1, 3, 4),
                                      names=["Latitude", "Longitude", "Altitude", "Date"])
            data_each = pd.DataFrame(data=data_each)
            data_individual = pd.concat([data_individual, data_each], axis=0, ignore_index=True)
        data_individual.to_csv("Data/" + individual_id + "/" + individual_id + ".csv")
        data_individual.to_csv("Data/preprocessed_data/" + individual_id + ".csv")


def map_matching(data):
    # sys.path.append('..')  # 更改文件调用路径

    """ 地图初始化
    Args:
        location: 经纬度，list 或者 tuple 格式，顺序为 latitude, longitude；一般选所有经纬度的中心
        zoom_start: 缩放值（初始地图大小），默认为 10，值越大，地图放大比例越大
        tiles: 地图样式
    """

    # tiles: OpenStreetMap, Stamen Terrain, Stamen Toner, Mapbox Bright, Mapbox Control Room
    demo_map = folium.Map(
        tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图',
        location=[39.93, 116.40],
        zoom_start=12)
    '''
    # 按行循环添加节点

    for i in range(data.shape[0]):
        tmp = data.iloc[i]  # 提取单行数据
        pop_content = 'Altitude: ' + str(tmp['Altitude']) + "<br>" + 'Date: ' + str(tmp['Date'])  # 弹窗内容设置

        # 向地图内添加圆形标记

        Args:
            location: 经纬度信息填充
            radius: 圆形直径
            popup: 点击圆后弹窗，可设置大小和内容
            color: 圆周颜色
            fill: 是否为实心圆
            fill_color: 圆内填充颜色
        

        folium.CircleMarker(
            location=(tmp['Latitude'], tmp['Longitude']),
            radius=5,
            popup=folium.Popup(pop_content, max_width=2000),
            color='red',
            fill=True,
            fill_color='red',
        ).add_to(demo_map)
    '''

    locations = []
    for latitude, longitude in zip(data['Latitude'], data['Longitude']):
        locations.append([latitude, longitude])
    print(locations)

    folium.PolyLine(  # polyline方法为将坐标用线段形式连接起来
        locations=locations,  # 将坐标点连接起来
        weight=3,  # 线的大小为3
        color='red',  # 线的颜色为红色
        opacity=0.8  # 线的透明度
    ).add_to(demo_map)  # 将这条线添加到刚才的区域m内

    # demo_map  # 在记事本中显示地图
    demo_map.save('demo_map.html')  # 如果需要保存地图为HTML文件，执行此命令
    os.system(r"demo_map.html")


def label_handle(label):
    label = pd.DataFrame(data=label)
    index_to_drop_label = label[label["Transportation Mode"] != "taxi"].index
    label.drop(index_to_drop_label, inplace=True)

    label["Start Time"] = label["Start Time"].apply(
        lambda x: int(datetime.fromisoformat(x.replace("/", "-")).timestamp()))
    label["End Time"] = label["End Time"].apply(lambda x: int(datetime.fromisoformat(x.replace("/", "-")).timestamp()))
    # print(label)
    label.to_csv("b.csv")
    return label


def data_handle(data):
    # 39298.1462037037,2007-08-04,03:30:32
    # Unix时间戳是从1970年1月1日
    # start_time = "1899:12:30:23:59:59"
    # end_time = "1970:01:01:00:00:00"
    #
    # start_time1 = datetime.strptime(start_time, "%Y:%m:%d:%H:%M:%S")
    # end_time1 = datetime.strptime(end_time, "%Y:%m:%d:%H:%M:%S")
    #
    # interval = end_time1 - start_time1
    # interval = interval.total_seconds()
    a = 39298.1462037037 * 24 * 60 * 60
    b = datetime.fromisoformat("2007-08-04 03:30:32").timestamp()
    interval = a - b
    data["Date"] = data["Date"].apply(lambda x: int(x * 24 * 60 * 60 - int(interval + 1)))
    # print("here:", a, b, interval)
    # print(label["Start Time"], label["End Time"], data["Date"])
    # print(data)
    data.to_csv("a.csv")
    return data


def timestamp_handle(date):
    date = datetime.fromisoformat(date.replace("/", "-")).timestamp()
    return int(date)


def timestamp2datestr(date):
    date = datetime.fromtimestamp(date)
    return date


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

    '''
    tiles = 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=8&ltype=11',
    attr = '高德-街道路网图',

    tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
    attr='高德-常规图',

    tiles='https://webrd02.is.autonavi.com/appmaptile?lang=en&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
    attr='高德-纯英文对照',
    '''

    demo_map = folium.Map(
        tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图',
        location=[39.93, 116.40],
        zoom_start=12)

    data_original = pd.read_csv(passenger_id + "test.csv")
    print(data_original)

    # 遍历日期
    begin_a = date(2008, 7, 22)
    begin_b = date(2008, 7, 23)
    one_more_day = begin_b - begin_a

    begin = timestamp2datestr(data_original["Date"].min())
    end = timestamp2datestr(data_original["Date"].max())

    # begin = date(2008, 7, 22)
    # end = date(2022, 9, 25)

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
        # map_matching(data)

        locations = []
        for latitude, longitude in zip(data['Latitude'], data['Longitude']):
            locations.append([latitude, longitude])
        print(locations)

        if not locations:
            continue

        folium.PolyLine(  # polyline方法为将坐标用线段形式连接起来
            locations=locations,  # 将坐标点连接起来
            weight=3,  # 线的大小为3
            color='red',  # 线的颜色为红色
            opacity=0.8  # 线的透明度
        ).add_to(demo_map)  # 将这条线添加到刚才的区域m内

    # demo_map  # 在记事本中显示地图
    demo_map.save('demo_map.html')  # 如果需要保存地图为HTML文件，执行此命令
    os.system(r"demo_map.html")


def map_matching_plot(data, file_name):
    demo_map = folium.Map(
        tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图',
        location=[39.93, 116.40],
        zoom_start=12)

    for i in range(data.shape[0]):
        tmp = data.iloc[i]  # 提取单行数据
        pop_content = 'Altitude: ' + str(tmp['Altitude']) + "<br>" + 'Date: ' + str(tmp['Date'])  # 弹窗内容设置

        folium.CircleMarker(
            location=(tmp['Latitude'], tmp['Longitude']),
            radius=2,
            popup=folium.Popup(pop_content, max_width=2000),
            color='red',
            fill=True,
            fill_color='red',
        ).add_to(demo_map)

    demo_map.save(file_name)
    os.system(file_name)


def dimension_expansion(data, destination_file):
    data["Date"] = data["Date"].apply(lambda x: timestamp_handle(x))
    data["DateStr"] = data["Date"].apply(lambda x: timestamp2datestr(x))  # 将数值时间戳转换成字符便于提取星期等操作
    data["Weekday"] = data["DateStr"].apply(lambda x: x.weekday())  # 新建维度：星期一~星期日（0-6）
    data["IsWeekend"] = data["Weekday"].apply(lambda x: 1 if x == 5 or x == 6 else 0)  # 新建维度：是否是周末
    # 北京市区坐标为：北纬39.9”，东经116. 3”
    dis2centers = []
    dis2destinations = []
    data_destination = pd.read_csv(destination_file)
    # print(data.shape)
    # print(data_destination.shape)
    for index in range(data.shape[0]):
        dis2center = distance(data["Longitude"][index], data["Latitude"][index], 116.3, 33.9)
        dis2centers.append(dis2center)

        dis2destination = distance(data["Longitude"][index], data["Latitude"][index],
                                   data_destination["Longitude"][index],
                                   data_destination["Latitude"][index])
        dis2destinations.append(dis2destination)

    data["Dis2Center"] = dis2centers
    data["Dis2Destination"] = dis2destinations

    def time_match(date_time):
        time = str(date_time)[11:13]
        return time

    data["Time"] = data["DateStr"].apply(time_match)
    time_group_dict = {"00": 0, "01": 0, "02": 0, "03": 0, "04": 0, "05": 0, "06": 1, "07": 1, "08": 2, "09": 2,
                       "10": 3, "11": 3, "12": 4, "13": 4, "14": 5, "15": 5, "16": 6, "17": 6, "18": 7, "19": 7,
                       "20": 8, "21": 8, "22": 8, "23": 8, "24": 8}
    data["Time"] = data["Time"].map(time_group_dict)
    return data


def denoise(data):
    # data = pd.read_csv(r"010TrajectoryTaxiBeijing.csv")
    # 北纬39”26’至41”03’，东经115”25’至 117”30’
    # 转化为小数（示例）：39+26/60
    data = data[(data["Latitude"] < (41 + 3 / 60)) & (data["Latitude"] > (39 + 26 / 60))]
    data = data[(data["Longitude"] < (117 + 30 / 60)) & (data["Longitude"] > (115 + 25 / 60))]
    data.drop(columns=[data.columns[0]], inplace=True)  # 重置index
    data.sort_values(by="Date", inplace=True)
    start = 0
    group_marks = []
    for index, value in data["Date"].items():
        # 300秒即5分钟，一般不可能在5分钟之内两次乘车/最后一个index+1会不存在报错，无需比较
        if (index + 1) != data.shape[0]:
            try:
                if data["Date"][index + 1] - value > 30:
                    group_marks.append([start, index])
                    start = index + 1
                    # print(index, timestamp2datestr(value))
            except:
                continue
    # print(group_marks)

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

        # if len(trajectory) <= 20:
        #     continue
        trajectories.append(trajectory)
    print(len(trajectories))
    return data


def POIVisual010(file):
    data = pd.read_csv(file)
    data.drop(columns=[data.columns[0]], inplace=True)  # 重置index
    data.sort_values(by="Date", inplace=True)
    start = 0
    group_marks = []
    for index, value in data["Date"].items():
        # 300秒即5分钟，一般不可能在5分钟之内两次乘车/最后一个index+1会不存在报错，无需比较
        if (index + 1) != data.shape[0]:
            if data["Date"][index + 1] - value > 30:
                group_marks.append([start, index])
                start = index + 1
                # print(index, timestamp2datestr(value))
    print(group_marks)

    data_POI = pd.DataFrame(columns=["Latitude", "Longitude", "Altitude", "Date"])
    for each in group_marks:
        new_row = [data["Latitude"][each[0]], data["Longitude"][each[0]], data["Altitude"][each[0]],
                   data["Date"][each[0]]]
        data_POI.loc[len(data_POI)] = new_row
    # print(data_POI)
    data_POI["Date"] = data_POI["Date"].apply(lambda x: timestamp2datestr(x))
    # data_POI.to_csv(r"data_POI.csv")
    # map_matching_plot(data_POI, file_name="010POI.html")

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
    data_original.drop(columns=[data.columns[0]], inplace=True)  # 重置index
    data_original.sort_values(by="Date", inplace=True)
    start = 0
    group_marks = []
    for index, value in data_original["Date"].items():
        # 300秒即5分钟，一般不可能在5分钟之内两次乘车/最后一个index+1会不存在报错，无需比较
        if (index + 1) != data_original.shape[0]:
            if data_original["Date"][index + 1] - value > 30:
                group_marks.append([start, index])
                start = index + 1
                # print(index, timestamp2datestr(value))

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

        folium.PolyLine(  # polyline方法为将坐标用线段形式连接起来
            locations=locations,  # 将坐标点连接起来
            weight=3,  # 线的大小为3
            color='red',  # 线的颜色为红色
            opacity=0.8  # 线的透明度
        ).add_to(demo_map)  # 将这条线添加到刚才的区域m内

    demo_map.save(file.replace(".csv", ".html"))
    # os.system(file.replace(".csv", ".html"))


def distance(latitude1, longitude1, latitude2, longitude2):
    from haversine import haversine
    # 输入的格式：经度，纬度
    location1 = (longitude1, latitude1)
    location2 = (longitude2, latitude2)
    dis = haversine(location1, location2)
    return dis


def one_hot_code_handle(data):
    print(data)
    # 部分数据维度需要创建数字或独热编码
    one_hot_encode = OneHotEncoder()
    data_sp = pd.concat([data["Weekday"], data["Time"]], axis=1)
    sp_encoded = one_hot_encode.fit_transform(data_sp)
    df_results = pd.DataFrame.sparse.from_spmatrix(sp_encoded)
    df_results.columns = one_hot_encode.get_feature_names_out(data_sp.columns)
    data.drop(["Weekday", "Time"], axis=1, inplace=True)
    data = pd.concat([data, df_results], axis=1)
    # data.to_csv(r"One_Hot_Coded.csv")
    # print(data)
    return data


def individual2dimension():
    ids = []
    data = pd.DataFrame()
    for index in os.listdir(r"Data"):
        if index != "preprocessed_data":
            # taxi_data_match(passenger_id=each)
            # trajectory_segment("TrajectoryTaxi\\" + each + "TrajectoryTaxiBeijing.csv")
            data_individual = pd.read_csv("TrajectoryTaxi\\" + index + "POIStarts.csv")
            # POIVisual010("TrajectoryTaxi\\" + index + "TrajectoryTaxiBeijing.csv")
            data_individual = dimension_expansion(data_individual, "TrajectoryTaxi\\" + index + "POIDestination.csv")
            data_individual = denoise(data_individual)
            if data_individual.shape[0] != 0:
                data_individual = one_hot_code_handle(data_individual)
            data_individual["PassengerID"] = data_individual["Latitude"].apply(lambda x: str(index))
            data_individual.dropna(inplace=True)
            data = pd.concat([data, data_individual], axis=0)
    data.to_csv("data_concat.csv")


def main():
    def knn_handle(k_range):
        print("----------1.KNN----------")
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import cross_val_score
        # from sklearn.metrics import accuracy_score
        cv_scores = []
        print("进行10折交叉验证:")
        for k in k_range:
            knn = KNeighborsRegressor(n_neighbors=k)
            scores = cross_val_score(knn, X_train, y_train, cv=10)  # 进行10折交叉验证
            cv_score = np.mean(scores)  # 取10次分数平均值
            print('k={}，验证集上的模型得分={:.3f}'.format(k, cv_score))
            cv_scores.append(cv_score)
        best_k = k_range[np.argmax(cv_scores)]
        best_knn = KNeighborsRegressor(n_neighbors=best_k)
        best_knn.fit(X_train, y_train)
        print('\n取得k=%d时，模型最优。模型得分：' % best_k, best_knn.score(X_test, y_test))
        y_predict = best_knn.predict(X_test)
        y_predict = pd.DataFrame(y_predict)

        y_test.reset_index(inplace=True)
        y_test.drop(columns=["index"], inplace=True)
        y_predict = y_predict.rename(columns={0: 'Latitude', 1: 'Longitude', 2: 'Altitude'})

        acc = []
        for index in range(y_test.shape[0]):
            dis = distance(y_test["Longitude"][index], y_test["Latitude"][index],
                           y_predict["Longitude"][index],
                           y_predict["Latitude"][index])
            acc.append(dis)
        print("预测位置与实际位置的球面距离的平均值：", np.average(acc), "km")
        print("预测位置与实际位置的球面距离的最小值（最佳预测）：", np.min(acc), "km")
        print("预测位置与实际位置的球面距离的最大值（最劣预测）：", np.max(acc), "km")
        print("预测位置与实际位置的球面距离小于1km的预测数：", len([x for x in acc if x <= 1]), "/", len(acc))
        print(51 / 66)
        '''
        # 可视化
        demo_map = folium.Map(
            tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
            attr='高德-常规图',
            location=[39.93, 116.40],
            zoom_start=12)

        data = y_test
        for i in range(data.shape[0]):
            tmp = data.iloc[i]  # 提取单行数据

            folium.CircleMarker(
                location=(tmp['Latitude'], tmp['Longitude']),
                radius=2,
                color='red',
                fill=True,
                fill_color='red',
            ).add_to(demo_map)

        data = y_predict.rename(columns={0: 'Latitude', 1: 'Longitude', 2: 'Altitude'})
        for i in range(data.shape[0]):
            tmp = data.iloc[i]  # 提取单行数据

            folium.CircleMarker(
                location=(tmp['Latitude'], tmp['Longitude']),
                radius=2,
                color='blue',
                fill=True,
                fill_color='red',
            ).add_to(demo_map)

        demo_map.save(r"knn.html")
        os.system(r"knn.html")
        '''

    def lr_handle(c_values):
        print("\n----------2.LLR----------")
        from sklearn.metrics import accuracy_score
        from sklearn.linear_model import LinearRegression
        mlr_model = LinearRegression()
        mlr_model.fit(X_train, y_train)
        y_predict = mlr_model.predict(X_test)

        y_predict = pd.DataFrame(y_predict)
        y_test.reset_index(inplace=True)
        y_test.drop(columns=["index"], inplace=True)
        y_predict = y_predict.rename(columns={0: 'Latitude', 1: 'Longitude', 2: 'Altitude'})

        acc = []
        for index in range(y_test.shape[0]):
            dis = distance(y_test["Longitude"][index], y_test["Latitude"][index],
                           y_predict["Longitude"][index],
                           y_predict["Latitude"][index])
            acc.append(dis)
        print("预测位置与实际位置的球面距离的平均值：", np.average(acc), "km")
        print("预测位置与实际位置的球面距离的最小值（最佳预测）：", np.min(acc), "km")
        print("预测位置与实际位置的球面距离的最大值（最劣预测）：", np.max(acc), "km")
        print("预测位置与实际位置的球面距离小于1km的预测数：", len([x for x in acc if x <= 1]), "/", len(acc))
        print(59 / 66)

    def dt_handle(max_depth_values):
        print("\n----------3.DecisionTree----------")
        from sklearn.tree import DecisionTreeClassifier
        for max_depth_val in max_depth_values:
            dt_model = DecisionTreeClassifier(max_depth=max_depth_val)
            dt_model.fit(X_train, y_train)
            print('max_depth=', max_depth_val)
            print('训练集上的准确率: {:.3f}'.format(dt_model.score(X_train, y_train)))
            print('测试集的准确率: {:.3f}'.format(dt_model.score(X_test, y_test)))

    def SVM_handle(c_values):
        print("\n----------4.SVM----------")
        from sklearn.metrics import accuracy_score
        from sklearn.svm import SVR
        for c_value in c_values:
            print("c=", c_value)
            predicts = []
            for dim in ["Latitude", "Longitude"]:
                svm_model = SVR(C=c_value, kernel='rbf')
                svm_model.fit(X_train, y_train[dim])
                y_predict = svm_model.predict(X_test)
                # acc = accuracy_score(y_test[dim], y_predict)
                # print('C={}，准确率：{:.3f}'.format(c_value, acc))
                y_predict = pd.DataFrame(y_predict)
                predicts.append(y_predict)

            y_predict_combine = pd.DataFrame()
            y_predict_combine["Latitude"] = predicts[0][0]
            y_predict_combine["Longitude"] = predicts[1][0]

            y_test.reset_index(inplace=True)
            y_test.drop(columns=y_test.columns[0], inplace=True)

            acc = []
            for index in range(y_test.shape[0]):
                dis = distance(y_test["Longitude"][index], y_test["Latitude"][index],
                               y_predict_combine["Longitude"][index],
                               y_predict_combine["Latitude"][index])
                acc.append(dis)
            print("预测位置与实际位置的球面距离的平均值：", np.average(acc), "km")
            print("预测位置与实际位置的球面距离的最小值（最佳预测）：", np.min(acc), "km")
            print("预测位置与实际位置的球面距离的最大值（最劣预测）：", np.max(acc), "km")
            print("预测位置与实际位置的球面距离小于1km的预测数：", len([x for x in acc if x <= 1]), "/", len(acc))

            demo_map = folium.Map(
                tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
                attr='高德-常规图',
                location=[39.93, 116.40],
                zoom_start=12)

            data = y_test
            for i in range(data.shape[0]):
                tmp = data.iloc[i]  # 提取单行数据

                folium.CircleMarker(
                    location=(tmp['Latitude'], tmp['Longitude']),
                    radius=2,
                    color='red',
                    fill=True,
                    fill_color='red',
                ).add_to(demo_map)

            data = y_predict_combine
            for i in range(data.shape[0]):
                tmp = data.iloc[i]  # 提取单行数据

                folium.CircleMarker(
                    location=(tmp['Latitude'], tmp['Longitude']),
                    radius=2,
                    color='blue',
                    fill=True,
                    fill_color='red',
                ).add_to(demo_map)

            demo_map.save(r"knn.html")
            os.system(r"knn.html")

    def random_forest_handle():
        from sklearn.ensemble import RandomForestRegressor
        # X 是特征变量，有五个训练数据，每个训练数据两个特征，所以特征值为2
        # X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        # y 是目标变量，是一个连续值
        # y = [1, 2, 3, 4, 5]
        # 弱学习机器设置为 10 random_state=123 保证采用相同的节点划分方式，即运行的结果相同
        model = RandomForestRegressor(n_estimators=15)
        # 训练模型
        model.fit(X_train, y_train)
        # predict 函数进行预测
        y_test.reset_index(inplace=True)
        y_predict = model.predict(X_test)
        acc = []
        y_predict = pd.DataFrame(y_predict)
        print(y_test, y_predict)
        for index in range(y_test.shape[0]):
            dis = distance(y_test["Longitude"][index], y_test["Latitude"][index],
                           y_predict[y_predict.columns[1]][index],
                           y_predict[y_predict.columns[0]][index])
            acc.append(dis)
        print("预测位置与实际位置的球面距离的平均值：", np.average(acc), "km")
        print("预测位置与实际位置的球面距离的最小值（最佳预测）：", np.min(acc), "km")
        print("预测位置与实际位置的球面距离的最大值（最劣预测）：", np.max(acc), "km")
        print("预测位置与实际位置的球面距离小于1km的预测数：", len([x for x in acc if x <= 1]), "/", len(acc))

    def lstm_handle():

        import numpy as np
        from keras.layers.core import Dense, Activation, Dropout
        from keras.layers import LSTM
        from keras.models import Sequential, load_model
        from keras.callbacks import Callback
        # import keras.backend.tensorflow_backend as KTF
        import tensorflow as tf
        import keras.callbacks
        import matplotlib.pyplot as plt

        # 设定为自增长
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # session = tf.Session(config=config)
        # KTF.set_session(session)

        def create_dataset(data, n_predictions, n_next):
            '''
            对数据进行处理
            '''
            dim = data.shape[1]
            train_X, train_Y = [], []
            for i in range(data.shape[0] - n_predictions - n_next - 1):
                a = data[i:(i + n_predictions), :]
                train_X.append(a)
                tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
                b = []
                for j in range(len(tempb)):
                    for k in range(dim):
                        b.append(tempb[j, k])
                train_Y.append(b)
            train_X = np.array(train_X, dtype='float64')
            train_Y = np.array(train_Y, dtype='float64')

            test_X, test_Y = [], []
            i = data.shape[0] - n_predictions - n_next - 1
            a = data[i:(i + n_predictions), :]
            test_X.append(a)
            tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
            b = []
            for j in range(len(tempb)):
                for k in range(dim):
                    b.append(tempb[j, k])
            test_Y.append(b)
            test_X = np.array(test_X, dtype='float64')
            test_Y = np.array(test_Y, dtype='float64')

            return train_X, train_Y, test_X, test_Y

        def NormalizeMult(data, set_range):
            '''
            返回归一化后的数据和最大最小值
            '''
            normalize = np.arange(2 * data.shape[1], dtype='float64')
            normalize = normalize.reshape(data.shape[1], 2)

            for i in range(0, data.shape[1]):
                if set_range == True:
                    list = data[:, i]
                    listlow, listhigh = np.percentile(list, [0, 100])
                else:
                    if i == 0:
                        listlow = -90
                        listhigh = 90
                    else:
                        listlow = -180
                        listhigh = 180

                normalize[i, 0] = listlow
                normalize[i, 1] = listhigh

                delta = listhigh - listlow
                if delta != 0:
                    for j in range(0, data.shape[0]):
                        data[j, i] = (data[j, i] - listlow) / delta

            return data, normalize

        def trainModel(train_X, train_Y):
            '''
            trainX，trainY: 训练LSTM模型所需要的数据
            '''
            model = Sequential()
            model.add(LSTM(
                120,
                input_shape=(train_X.shape[1], train_X.shape[2]),
                return_sequences=True))
            model.add(Dropout(0.3))

            model.add(LSTM(
                120,
                return_sequences=False))
            model.add(Dropout(0.3))

            model.add(Dense(
                train_Y.shape[1]))
            model.add(Activation("relu"))

            model.compile(loss='mse', optimizer='adam', metrics=['acc'])
            model.fit(train_X, train_Y, epochs=100, batch_size=64, verbose=1)
            model.summary()

            return model

        train_num = 6
        per_num = 1
        # set_range = False
        set_range = True

        # 读入时间序列的文件数据
        data = pd.read_csv(r"lstm.csv", sep=',').iloc[:, 0:2].values
        print("样本数：{0}，维度：{1}".format(data.shape[0], data.shape[1]))
        # print(data)

        # 画样本数据库
        plt.scatter(data[:, 1], data[:, 0], c='b', marker='o', label='traj_A')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()

        # 归一化
        data, normalize = NormalizeMult(data, set_range)
        # print(normalize)

        # 生成训练数据
        train_X, train_Y, test_X, test_Y = create_dataset(data, train_num, per_num)
        print("x\n", train_X.shape)
        print("y\n", train_Y.shape)

        # 训练模型
        model = trainModel(train_X, train_Y)
        loss, acc = model.evaluate(train_X, train_Y, verbose=2)
        print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

        # 保存模型
        np.save("./traj_model_trueNorm.npy", normalize)
        model.save("./traj_model_120.h5")

    def lstm_predict():
        import numpy as np
        from keras.layers.core import Dense, Activation, Dropout
        from keras.layers import LSTM
        from keras.models import Sequential, load_model
        from keras.callbacks import Callback
        # import keras.backend.tensorflow_backend as KTF
        import tensorflow as tf
        import pandas as pd
        import os
        import keras.callbacks
        import matplotlib.pyplot as plt
        import copy

        # # 设定为自增长
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # session = tf.Session(config=config)
        # KTF.set_session(session)

        def rmse(predictions, targets):
            return np.sqrt(((predictions - targets) ** 2).mean())

        def mse(predictions, targets):
            return ((predictions - targets) ** 2).mean()

        def reshape_y_hat(y_hat, dim):
            re_y = []
            i = 0
            while i < len(y_hat):
                tmp = []
                for j in range(dim):
                    tmp.append(y_hat[i + j])
                i = i + dim
                re_y.append(tmp)
            re_y = np.array(re_y, dtype='float64')
            return re_y

        # 多维反归一化
        def FNormalizeMult(data, normalize):

            data = np.array(data, dtype='float64')
            # 列
            for i in range(0, data.shape[1]):
                listlow = normalize[i, 0]
                listhigh = normalize[i, 1]
                delta = listhigh - listlow
                print("listlow, listhigh, delta", listlow, listhigh, delta)
                # 行
                if delta != 0:
                    for j in range(0, data.shape[0]):
                        data[j, i] = data[j, i] * delta + listlow

            return data

        # 使用训练数据的归一化
        def NormalizeMultUseData(data, normalize):

            for i in range(0, data.shape[1]):

                listlow = normalize[i, 0]
                listhigh = normalize[i, 1]
                delta = listhigh - listlow

                if delta != 0:
                    for j in range(0, data.shape[0]):
                        data[j, i] = (data[j, i] - listlow) / delta

            return data

        from math import sin, asin, cos, radians, fabs, sqrt

        EARTH_RADIUS = 6371  # 地球平均半径，6371km

        # 计算两个经纬度之间的直线距离
        def hav(theta):
            s = sin(theta / 2)
            return s * s

        def get_distance_hav(lat0, lng0, lat1, lng1):
            # "用haversine公式计算球面两点间的距离。"
            # 经纬度转换成弧度
            lat0 = radians(lat0)
            lat1 = radians(lat1)
            lng0 = radians(lng0)
            lng1 = radians(lng1)

            dlng = fabs(lng0 - lng1)
            dlat = fabs(lat0 - lat1)
            h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
            distance = 2 * EARTH_RADIUS * asin(sqrt(h))
            return distance

        test_num = 6
        per_num = 1
        data_all = pd.read_csv(r"lstm.csv", sep=',').iloc[
                   -2 * (test_num + per_num):-1 * (test_num + per_num), 0:2].values
        data_all.dtype = 'float64'

        data = copy.deepcopy(data_all[:-per_num, :])
        y = data_all[-per_num:, :]

        # #归一化
        normalize = np.load("./traj_model_trueNorm.npy")
        data = NormalizeMultUseData(data, normalize)

        model = load_model("./traj_model_120.h5")
        test_X = data.reshape(1, data.shape[0], data.shape[1])
        y_hat = model.predict(test_X)
        y_hat = y_hat.reshape(y_hat.shape[1])
        y_hat = reshape_y_hat(y_hat, 2)

        # 反归一化
        y_hat = FNormalizeMult(y_hat, normalize)
        print("predict: {0}\ntrue：{1}".format(y_hat, y))
        print('预测均方误差：', mse(y_hat, y))
        print('预测直线距离：{:.4f} KM'.format(get_distance_hav(y_hat[0, 0], y_hat[0, 1], y[0, 0], y[0, 1])))

        # 画测试样本数据库
        p1 = plt.scatter(data_all[:-per_num, 1], data_all[:-per_num, 0], c='b', marker='o', label='traj_A')
        p2 = plt.scatter(y_hat[:, 1], y_hat[:, 0], c='r', marker='o', label='pre')
        p3 = plt.scatter(y[:, 1], y[:, 0], c='g', marker='o', label='pre_true')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()

    # drop_no_labels()
    # data_integration_individual()
    # map_matching_by_time_interval("010")
    # kf_handle(data)
    # denoise()
    # POIVisual010()
    # trajectory_segment()
    # individual2dimension()

    # data = pd.read_csv("data_concat.csv")
    # data.drop(columns='Unnamed: 0', inplace=True)
    # data.fillna(0, inplace=True)
    # data.to_csv("data_concat.csv")

    data = pd.read_csv("lstm.csv")

    '''划分数据集'''
    data_dependents = data.columns.tolist()
    remove_list = ['Latitude', 'Longitude', 'Altitude', 'Date', 'DateStr']
    for each in remove_list[::]:
        data_dependents.remove(each)
    X = data[data_dependents]
    y = data[['Latitude', 'Longitude', 'Altitude']]
    # y = data['Latitude']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=0)

    # # 手动划分前3/4为训练集
    # X_length = X.shape[0]
    # split = int(X_length * 0.75)
    # X_train, X_test = X[:split], X[split:]
    # y_train, y_test = y[:split], y[split:]

    print('数据集样本数：{}，训练集样本数：{}，测试集样本数：{}'.format(len(X), len(X_train), len(X_test)), "\n")

    # 特征归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_name_param_dict = {'KNN': [1, 3, 50],
                             'LR': [0.01, 1, 100],
                             'DT': [5, 10, 15],
                             'SVM': [0.1, 1, 10, 0.0001]}

    # lstm_handle()
    # lstm_predict()
    random_forest_handle()

    '''
    demo_map = folium.Map(
        tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图',
        location=[39.93, 116.40],
        zoom_start=12)

    for i in range(data.shape[0]):
        tmp = data.iloc[i]  # 提取单行数据
        pop_content = 'Altitude: ' + str(tmp['Altitude']) + "<br>" + 'Date: ' + str(tmp['Date'])  # 弹窗内容设置

        folium.CircleMarker(
            location=(tmp['Latitude'], tmp['Longitude']),
            radius=2,
            popup=folium.Popup(pop_content, max_width=2000),
            color='red',
            fill=True,
            fill_color='red',
        ).add_to(demo_map)

    demo_map.save(r"test.html")
    os.system(r"test.html")
    '''
    # knn_handle(model_name_param_dict['KNN'])
    # lr_handle(model_name_param_dict['LR'])
    # dt_handle(model_name_param_dict['DT'])
    # SVM_handle(model_name_param_dict['SVM'])


if __name__ == "__main__":
    main()
