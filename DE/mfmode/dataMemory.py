import numpy as np


class memory:
    # fre
    frequency = np.asarray([])
    # acc和len
    accLen = np.asarray([])
    # 任务数量
    taskNum = 0
    def __init__(self,taskNum):
        self.taskNum = taskNum
        self.setArraySize()
    # 设置存放大小
    def setArraySize(self):
        # 设置存放frequency 和 acclen的大小，几个任务就存放几个
        self.frequency = [[] for i in range(self.taskNum)]
        self.accLen = [[] for i in range(self.taskNum)]

    # 存放数据
    def saveData(self,freTask,acclentask,task):
        self.frequency[task] = freTask
        self.accLen[task] = acclentask
