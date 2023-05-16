import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # MOEA/D_STAT
    y1=1 - np.asarray([0.908888889,0.898888889,0.897777778,0.888888889,0.877777778,0.858888889])
    x1=np.asarray([311,193,140,120,48,24])/729
    # MOEA/D_DYN
    y2=1 - np.asarray([0.927777778,0.917777778,0.897777778,0.877777778,0.867777778,0.834444444,0.825555556])
    x2=np.asarray([29,23,13,14,9,6,3])/729
    # MOPSO/D
    y3=1 - np.asarray([0.918888889,0.896666667,0.888888889,0.867777778,0.865555556,0.846666667,0.837777778,0.833333333,0.826666667,0.812222222,0.794444444,0.784444444,0.751111111])
    x3=np.asarray([83,69,62,42,29,25,22,18,16,14,11,6,2])/729
    # MOEA/D
    y4=1 - np.asarray([0.917777778,0.908888889,0.907777778,0.898888889,0.897777778,0.888888889,0.886666667,0.867777778,0.866666667,0.865555556])
    x4=np.asarray([360,358,345,343,294,290,264,259,258,255])/729

    plt.xlabel("len")
    plt.ylabel("error")
    plt.title("pareto front")

    plt.plot(x1, y1, color='r', marker='o', label='MOEA/D_STAT')
    plt.plot(x2, y2, color='b', marker='o', label='MOEA/D_DYN')
    plt.plot(x3, y3, color='y', marker='o', label='MOPSO/D')
    plt.plot(x4, y4, color='#ff7f0e', marker='o', label='MOEA/D')
    plt.legend(loc="upper right")