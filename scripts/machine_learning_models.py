import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Utilities import distance


def knn_handle(X_train, X_test, y_train, y_test, k_range):
    print("----------1.KNN----------")
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import cross_val_score
    cv_scores = []
    print("进行10折交叉验证:")
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10)
        cv_score = np.mean(scores)
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


def lr_handle(X_train, X_test, y_train, y_test):
    print("\n----------2.LLR----------")
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


def dt_handle(X_train, X_test, y_train, y_test, max_depth_values):
    print("\n----------3.DecisionTree----------")
    from sklearn.tree import DecisionTreeClassifier
    for max_depth_val in max_depth_values:
        dt_model = DecisionTreeClassifier(max_depth=max_depth_val)
        dt_model.fit(X_train, y_train)
        print('max_depth=', max_depth_val)
        print('训练集上的准确率: {:.3f}'.format(dt_model.score(X_train, y_train)))
        print('测试集的准确率: {:.3f}'.format(dt_model.score(X_test, y_test)))


def SVM_handle(X_train, X_test, y_train, y_test, c_values):
    print("\n----------4.SVM----------")
    from sklearn.svm import SVR
    for c_value in c_values:
        print("c=", c_value)
        predicts = []
        for dim in ["Latitude", "Longitude"]:
            svm_model = SVR(C=c_value, kernel='rbf')
            svm_model.fit(X_train, y_train[dim])
            y_predict = svm_model.predict(X_test)
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


def random_forest_handle(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=15)
    model.fit(X_train, y_train)
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

def main():
    data = pd.read_csv("lstm.csv")

    data_dependents = data.columns.tolist()
    remove_list = ['Latitude', 'Longitude', 'Altitude', 'Date', 'DateStr']
    for each in remove_list[::]:
        data_dependents.remove(each)
    X = data[data_dependents]
    y = data[['Latitude', 'Longitude', 'Altitude']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=0)

    print('数据集样本数：{}，训练集样本数：{}，测试集样本数：{}'.format(len(X), len(X_train), len(X_test)), "\n")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_name_param_dict = {'KNN': [1, 3, 50],
                             'LR': [0.01, 1, 100],
                             'DT': [5, 10, 15],
                             'SVM': [0.1, 1, 10, 0.0001]}

    random_forest_handle(X_train, X_test, y_train, y_test)