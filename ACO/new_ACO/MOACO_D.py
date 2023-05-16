# 基于分解的ACO
###################
import copy
import math

import numpy as np
import random

import pandas as pd

from Filter.filter import FilterOfData
from dataProcessing.ReadDataCSV_new import ReadCSV
from dataProcessing.NonDominate import nonDominatedSort_PFAndPop, getPF
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from random import sample
from sklearn.preprocessing import MinMaxScaler

class MOACOD:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 特征的存放数组
    dataFeature = np.asarray([])
    # 特征的数量
    dataFeatureNum = 0
    # 每一个特征的relifF评分
    scoreOfReliefF = np.asarray([])
    # 每一个特征的评分
    scoreOfFeature = np.asarray([])
    # 种群
    population_ACO = np.asarray([])
    # 非支配前沿
    paretoFront = np.asarray([])
    # 每一个种群的数量
    populationNum = 0
    # 种群迭代次数
    iteratorTime = 0
    # 蒸发率 rho
    rho = 0
    # 信息素最小值
    tau_min = 0
    # 信息苏最大值
    tau_max = 0
    # 全局最优解
    globalSolution = np.asarray([])
    # 全局最优解的适应度值
    globalFitness = 0
    # 全局最优解的长度
    globalLen = 0
    # 保存数据名字
    dataName = ""
    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.dataFeature = np.arange(dataX.shape[1])
        # 对数据01 归一化，将每一列的数据缩放到 [0,1]之间
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        print("进行filter，提取一部分数据")
        # filter operator
        filter = FilterOfData(dataX=self.dataX, dataY=self.dataY)
        #  compute reliefF score
        self.scoreOfReliefF = filter.computerRelifFScore()
        self.MI = filter.computerInforGain()
        # 进行filter过滤数据 ----
        self.scoreOfReliefF, self.dataX, self.dataFeature = filter.filter_relifF(
            scoreOfRelifF=self.scoreOfReliefF, dataX=self.dataX, dataFeature=self.dataFeature)
        # 获取特征数量
        self.dataFeatureNum = len(self.dataFeature)
        # 初始化 信息素矩阵
        self.matrix_pheromone = np.ones((self.dataFeatureNum,self.dataFeatureNum))*self.MI.max()
        # 设置启发式信息
        self.setHeuristic()
    def setParameter(self, populationNum, iteratorTime):
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime

    def setHeuristic(self):
        # 计算特征评分
        self.comScoreOfFeature()
        # sigmoid 函数放缩
        exp_x = np.exp(self.scoreOfFeature)
        # 计算值
        exp_x = 1 + exp_x
        exp_x = 1 / exp_x
        #
        print()


    class Ant:
        # 每一个个体所选择出来的特征组合
        featureOfSelect = np.asarray([])
        # 特征组合和长度
        numberOfSolution = 0
        # 特征组合的索引
        indexOfSolution = np.asarray([])
        # 比例
        proportionOfFeature = 0
        # 当前的选择的fitness
        mineFitness = 0
        # 支配等级
        p_rank = 0
        # 已选择特征
        selectedL= []
        # 未选择特征
        unselectedL = []
        # 选择的状态
        stateOfSelect = np.asarray([])
        # 非支配数量
        numberOfNondominate = 0
        # 支配个体集合
        dominateSet = []

        def __init__(self, moacoD):
            self.initOfChromosome(moacoD=moacoD)

        def initOfChromosome(self, moacoD):
            # 得到解
            self.generateSolution(moacoD = moacoD)
            fX = moacoD.dataX[:,self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=fX,findData_y=moacoD.dataY,CV=10)
            # 获得 多少个特征。。
            self.getNumberOfSolution()

        # 产生解
        def generateSolution(self,moacoD):
            # 所有未选择的特征 -- 初始化
            self.unselectedL = np.arange(moacoD.dataFeatureNum)
            self.selectedL = []
            self.stateOfSelect = np.asarray([])

            # 计数
            count = 0
            self.featureOfSelect = np.zeros(moacoD.dataFeatureNum)
            # 蚂蚁所选择的初始特征
            firstFeature = np.random.randint(moacoD.dataFeatureNum)
            # 添加该特征
            self.selectedL.append(firstFeature)
            # 该特征在 solution 中为 1
            self.featureOfSelect[self.selectedL[count]] = 1
            # 将该特征从未选择特征中删除
            self.unselectedL = np.delete(self.unselectedL,self.selectedL[count])
            # 状态  该特征被选择就是1 否则是0 --- 第一个特征是 选择状态
            state = 1
            # 进行循环
            while(len(self.unselectedL) != 0):
                self.stateOfSelect = np.append(self.stateOfSelect,state)
                # 第i个被选择的特征
                nt = self.selectedL[count]
                ne = int(nt * 2)
                # 需要删除的列数索引
                deleteIndex = []
                # 获得每一个索引
                for i in range(count + 1):
                    deleteIndex.append(self.selectedL[i] * 2)
                    deleteIndex.append(self.selectedL[i] * 2 + 1)
                # 第 nf 个特征 与所有未选择特征的信息素
                if state == 0:
                    eta = np.delete(moacoD.matrix_pheromone[ne], deleteIndex)
                else:
                    eta = np.delete(moacoD.matrix_pheromone[ne + 1], deleteIndex)
                # 第 nf 个特征 与所有未选择特征的启发式信息
                tau = np.delete(moacoD.matrix_heuristic[nt], deleteIndex)

                #--------------------------获得下一个特征-------------------------#

                # 获得未选择特征中的第几个特征  并且状态为选择或者不选择
                nt, state = self.getNextFeature(tau=tau, eta=eta)
                # 计数加1 进入到下一个特征
                count += 1
                # 若state为1 则该特征被选中
                if state == 1:
                    self.featureOfSelect[nt] = 1
                # 从 选择 中添加该特征
                self.selectedL.append(self.unselectedL[nt])
                # 从 未选择 中删除该特征
                self.unselectedL = np.delete(self.unselectedL, nt)
            # 添加最后一个state
            self.stateOfSelect = np.append(self.stateOfSelect, state)

        # 得到下一个特征
        def getNextFeature(self,tau,eta):
            proF = np.zeros(len(eta))
            # 特征为0 的概率
            p_0 = 0
            # 特征为1 的概率
            p_1= 0
            for i in range(proF.shape[0]):
                # 偶数 为 0
                if i % 2 == 0:
                    p_0 += tau[i] * eta[i]
                elif i % 2 == 1:
                    p_1 += tau[i] * eta[i]
            tempP = (tau * eta) / (p_0 + p_1)
            # 轮盘赌
            rand1 = np.random.rand()
            count = 0
            sun_p = tempP[0]
            while sun_p < rand1:
                count += 1
                sun_p += tempP[count]

            # count 向下取整数 得到第几个特征
            fN = math.floor(count/2)
            # 是否选择了特征 0 未选择; 1 选择
            state = count % 2
            return fN,state

        # 得到选择特征数量是多少个
        def getNumberOfSolution(self):
            # 得到选择的组合里面 为1 的索引为多少个
            self.indexOfSolution = np.where(self.featureOfSelect == 1)[0]
            # 得到的索引的长度是多少
            self.numberOfSolution = self.indexOfSolution.shape[0]
            # 获得比例
            self.proportionOfFeature = self.numberOfSolution / self.featureOfSelect.shape[0]

    # 计算每一个特征的分数
    def comScoreOfFeature(self):
        # 用于存放每一个特征的err
        errArray = np.zeros(self.dataFeatureNum)
        for i in range(self.dataFeatureNum):
            featureX = self.dataX[:,i]
            errArray[i] = 1 - fitnessFunction_KNN_CV(findData_x=featureX,findData_y=self.dataY,CV=10)

        # 将err进行排序
        self.scoreOfFeature = np.ones(self.dataFeatureNum)
        tmp_1 = np.unique(errArray)

        for i in range(tmp_1.shape[0]):
            tmp_index = np.where(errArray == tmp_1[i])[0]
            self.scoreOfFeature[tmp_index] = i
        print()


    # 产生参考点
    def generateReferPoint(self):
        # 全为1的数组
        self.referPoint = np.ones(self.populationNum)
        for i in range(self.populationNum):
            self.referPoint[i] = (i + 1) / self.populationNum

    # 运行
    def run(self):
        # 初始化函数
        self.initPopulation()
        # 进行运行
        runTime = 0
        # 产生参考点
        self.generateReferPoint()
        # 初始化近邻
        self.getClosestReferPoint()
        # 计算每一个特征的acc
        self.calculateAccOfSingleFeature()
        while runTime < self.iteratorTime:

            print("第",runTime+1,"代")
            self.printOptimal(runTime=runTime)
            self.getNextPopulation()
            runTime += 1
            self.pltForWin(iteNum=runTime, referPoint=self.referPoint, pop=self.population_DE,
                           iteratorTime=self.iteratorTime, name="MOEA/D_STAT")
        print(" PF 解")
        for i in range(len(self.EP)):
            print(1 - self.EP[i].mineFitness, " ", self.EP[i].numberOfSolution)

        print("全局最优acc", self.globalBestFitness, " 全局最优len", self.globalBestLen)

if __name__ == '__main__':
    import time
    start = time.time()
    ducName = "BreastCancer1"
    path = "D:\MachineLearningBackUp\dataCSV\dataCSV_high\\" + ducName + ".csv"
    dataCsv = ReadCSV(path=path)
    print("获取文件数据")
    dataCsv.getData()
    moead = MOACOD(dataX=dataCsv.dataX, dataY=dataCsv.dataY)
    moead.setParameter(populationNum=100, iteratorTime=100)
    moead.run()
    print("================================================")
    print(f"all time = {time.time() - start} seconds")