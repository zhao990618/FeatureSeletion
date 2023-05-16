
import matplotlib.pyplot as plt

# 画图
import numpy as np
from numpy import ravel


def pltForWin(iteNum, pop,iteratorTime,name):
    plt.ion()
    length_LA = len(pop)
    # 存放 error
    x = np.ones(length_LA)
    # 存放 len
    y = np.ones(length_LA)
    for i in range(length_LA):
        x[i] = pop[i].proportionOfFeature
        y[i] = pop[i].mineFitness

    plt.xlabel("len")
    plt.ylabel("error")
    plt.title(name+"  "+str(iteNum))
    plt.plot(x, y, color='r', marker='o',linestyle='None')
    plt.pause(0.1)
    if iteNum != iteratorTime:
        plt.cla()