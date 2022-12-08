import dis
import random

import numpy as np
import matplotlib.pyplot as plt
import time


def dist_o(x, y):
    return np.sqrt(np.sum((x - y) * (x - y)))


def generate_data(x_c, x_d, y_c, y_d, num):
    x = np.random.normal(x_c, x_d * np.random.rand(), num)
    y = np.random.normal(y_c, y_d * np.random.rand(), num)
    return np.stack((x, y), -1)  # 竖着放


def k_means(dataSet, k):
    global j
    center = dataSet[np.random.choice(dataSet.shape[0], k), :]
    m = dataSet.shape[0]

    belong = np.zeros(m)
    is_modify = True
    max_round = 500
    cnt_round = 0

    while is_modify and cnt_round < max_round:
        is_modify = False
        cnt_round += 1

        for i in range(m):
            minDist = 1e18
            idx = 0

            for j in range(k):
                dist = dist_o(center[j, :], dataSet[i, :])
                if dist < minDist:
                    minDist = dist
                    idx = j

            if belong[i] != idx:
                is_modify = True
                belong[i] = idx
        for i in range(k):
            points = dataSet[np.nonzero(belong == j)[0]]
            center[j, :] = np.mean(points, axis=0)

        plt.scatter(dataSet[:, 0], dataSet[:, 1], s=0.5, c=belong.astype(int))
        plt.scatter(center[:, 0], center[:, 1], marker='x', s=100, c='r')
        plt.savefig('./img/upd_{}.png'.format(cnt_round))
        plt.cla()
    return belong, center


if __name__ == "__main__":
    # print('input a random seed(int)')
    # seed = int(input())
    # seed = 2
    # np.random.seed(seed)
    k = 4
    p1 = generate_data(90, 50, 90, 50, 3000)
    p2 = generate_data(-90, 50, 90, 50, 3000)
    p3 = generate_data(90, 50, -90, 50, 3000)
    p4 = generate_data(-90, 50, -90, 50, 3000)
    ds = np.concatenate((p1, p2, p3, p4))
    plt.scatter(ds[:, 0], ds[:, 1], s=1)
    plt.savefig("./img/init.png")
    # plt.show()
    # print(ds)

    t_begin = time.time()
    result, center = k_means(ds, k)
    t_end = time.time()

    plt.scatter(ds[:, 0], ds[:, 1], s=0.5, c=result.astype(int))
    plt.scatter(center[:, 0], center[:, 1], marker='x', s=100, c='r')
    plt.savefig('./img/kmeans_result.png')

    print(u'kmeans，12000 points ,cost %.10f s' % (t_end - t_begin))
