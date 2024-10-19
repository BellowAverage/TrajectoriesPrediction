import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Kf_Params:
    def __init__(self):
        self.B = 0
        self.u = 0
        self.K = float('nan')
        self.z = float('nan')
        self.P = np.diag(np.ones(4))
        self.x = []
        self.G = []
        self.A = np.eye(4) + np.diag(np.ones((1, 2))[0, :], 2)
        self.Q = np.diag(np.ones(4)) * 0.1
        self.H = np.eye(2, 4)
        self.R = np.diag(np.ones(2)) * 0.1


def kf_init(px, py, vx, vy):
    kf_params = Kf_Params()
    kf_params.x = [px, py, vx, vy]
    kf_params.G = [px, py, vx, vy]
    return kf_params


def kf_update(kf_params):
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


def kf_handle(data):
    path = './9.xlsx'
    data_A = pd.read_excel(path, header=None)
    data_A_x = list(data_A.iloc[::, 0])
    data_A_y = list(data_A.iloc[::, 1])
    A = np.array(list(zip(data_A_x, data_A_y)))

    plt.figure()
    plt.plot(data_A_x, data_A_y, 'b-+')

    path = './10.xlsx'
    data_B = pd.read_excel(path, header=None)
    data_B_x = list(data_B.iloc[::, 0])
    data_B_y = list(data_B.iloc[::, 1])
    B = np.array(list(zip(data_B_x, data_B_y)))

    plt.plot(data_B_x, data_B_y, 'r-+')

    kf_params_record = np.zeros((len(data_B), 4))
    kf_params_p = np.zeros((len(data_B), 4))
    t = len(data_B)
    kalman_filter_params = kf_init(data_B_x[0], data_B_y[0], 0, 0)
    for i in range(t):
        if i == 0:
            kalman_filter_params = kf_init(data_B_x[i], data_B_y[i], 0, 0)
        else:
            kalman_filter_params.z = np.transpose([data_B_x[i], data_B_y[i]])
            kalman_filter_params = kf_update(kalman_filter_params)
        kf_params_record[i, ::] = np.transpose(kalman_filter_params.x)
        kf_params_p[i, ::] = np.transpose(kalman_filter_params.G)

    kf_trace = kf_params_record[::, :2]
    kf_trace_1 = kf_params_p[::, :2]

    plt.plot(kf_trace[::, 0], kf_trace[::, 1], 'g-+')
    plt.plot(kf_trace_1[1:26, 0], kf_trace_1[1:26, 1], 'm-+')
    legend = ['CMA最佳路径数据集', '检测路径', '卡尔曼滤波结果', '预测路径']
    plt.legend(legend, loc="best", frameon=False)
    plt.title('卡尔曼滤波后的效果')
    plt.savefig('result.svg', dpi=600)
    plt.show()

    p = accuracy(kf_trace, A)
    print(p)


def accuracy(predictions, labels):
    return np.array(predictions) - np.array(labels)