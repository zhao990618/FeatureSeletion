import numpy as np
from dataProcessing.reliefF import reliefFScore

class FilterOfData:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 保存relifF的值
    scoreOfRelifF = np.asarray([])
    # 保存了信息增益
    inforGainArray = np.asarray([])
    # 保存了对称不确定SU
    featureClassSU = np.asarray([])

    # 初始化
    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY

    #计算relifF的值
    def computerRelifFScore(self):
        self.scoreOfRelifF = reliefFScore(self.dataX,self.dataY)
        return self.scoreOfRelifF

    # filter 过滤
    def filter_relifF(self,scoreOfRelifF,dataX,dataFeature):
        # 拐点分割特征，选择拐点前的数据
        if len(dataFeature) > 1000:
            # extratRatio = findKneePoint(data=self.scoreOfRelifF)
            extratRatio = 0.03
        else:
            extratRatio = 1
        dataLen = dataX.shape[1]
        exR = int(dataLen * extratRatio)
        # 降序排序 relifF评分
        indexOfRelifF = np.argsort(scoreOfRelifF)[::-1]
        tenpI = indexOfRelifF[:exR]
        scoreOfRelifF = scoreOfRelifF[tenpI]
        dataX = dataX[:, tenpI]
        dataFeature = dataFeature[tenpI]
        return scoreOfRelifF,dataX,dataFeature