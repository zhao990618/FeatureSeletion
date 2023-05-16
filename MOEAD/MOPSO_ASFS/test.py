import random

import numpy as np
import matplotlib.pyplot as plt
import math
if __name__ == '__main__':
    # z = np.ones(2)
    # F_i = np.asarray([0.2, 0.3])
    # temp_v = np.asarray(z - F_i)
    # a = np.asarray([1,1])
    # c = np.dot(temp_v,a)
    # d = np.linalg.norm(a,ord=2)
    # m = math.exp(0)
    #
    # a = np.asarray([2,4,5,1])
    # print(a[np.argsort(a)])
    #
    # temp_1 = np.ones(3)
    # a = np.asarray([0.9,2,3])
    # b = np.asarray([3,4,5])
    # temp_2 = np.where(temp_1 < a)[0]
    # a[temp_2] = temp_1[temp_2]
    # c = b * 0.0002
    # c = np.where(a>1)

    # import numpy as np
    # import matplotlib.pyplot as plt
    # import matplotlib.animation as animation
    #
    # fig, ax = plt.subplots()
    #
    # x = np.arange(0, 2 * np.pi, 0.01)
    # line, = ax.plot(x, np.sin(x))
    #
    #
    # # 定义init_func,给定初始信息
    # def init():  # only required for blitting to give a clean slate.
    #     line.set_ydata([np.nan] * len(x))
    #     return line,
    #
    #
    # # 定义func,用来反复调用的函数
    # def animate(i):
    #     line.set_ydata(np.sin(x + i / 100))  # 跟随自变量的增加更新y值
    #     return line,
    #
    #
    # ani = animation.FuncAnimation(
    #     fig, animate, init_func=init, interval=2, blit=True, save_count=50)
    # plt.show()



    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots()
    # y1 = []
    # for i in range(50):
    #     y1.append(i)  # 每迭代一次，将i放入y1中画出来
    #     ax.cla()  # 清除键
    #     ax.bar(y1, label='test', height=y1, width=0.3)
    #     ax.legend()
    #     plt.pause(0.1)

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # plt.axis([0, 100, 0, 1])
    # plt.ion()
    #
    # xs = [0, 0]
    # ys = [1, 1]
    #
    # for i in range(100):
    #     y = np.random.random()
    #     xs[0] = xs[1]
    #     ys[0] = ys[1]
    #     xs[1] = i
    #     ys[1] = y
    #     plt.plot(xs, ys)
    #     plt.pause(0.1)
    a = np.ones((10,10))

    print(a)