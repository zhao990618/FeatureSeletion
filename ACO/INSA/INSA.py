import copy
import math

import numpy as np
import random

import pandas as pd

from Filter.filter import FilterOfData
from dataProcessing.ReadDataCSV_new import ReadCSV

from dataProcessing.NonDominate import getPF
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from random import sample
import skfeature.utility.entropy_estimators as ee
from skfeature.utility.mutual_information import su_calculation,information_gain
from sklearn.preprocessing import MinMaxScaler


class INSA:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 特征的存放数组
    dataFeature = np.asarray([])
    # 特征的数量
    dataFeatureNum = 0
    # 每一个特征的relifF评分
    scoreOfRelifF = np.asarray([])
    # 种群
    population_ACO = np.asarray([])
    # 互信息矩阵 -- feature --class
    featureMI = np.asarray([])
    # psi矩阵
    featurePSI = np.asarray([])
    # SU 矩阵
    matrix_SU = np.asarray([])
    # 互信息矩阵 -- feature --feature
    matrix_MI = np.asarray([])
    # 信息素矩阵
    matrix_pheromone = np.asarray([])
    # 启发式信息矩阵
    matrix_heuristic = np.asarray([])
    # 非支配前沿
    paretoFront = np.asarray([])
    # 每一个种群的数量
    populationNum = 0
    # 种群迭代次数
    iteratorTime = 0
    # 参数 alpha
    alpha = 0
    # 参数 beta
    beta = 0
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
        # self.dataX = MinMaxScaler(self.dataX)
        print("进行filter，提取一部分数据")
        # filter operator
        filter = FilterOfData(dataX=self.dataX, dataY=self.dataY)
        #  compute relifF score
        self.scoreOfRelifF = filter.computerRelifFScore()
        # self.MI = filter.computerInforGain()
        # 进行filter过滤数据 ----
        self.scoreOfRelifF, self.dataX, self.dataFeature = filter.filter_relifF(
            scoreOfRelifF=self.scoreOfRelifF, dataX=self.dataX, dataFeature=self.dataFeature)
        # self.MI, self.dataX, self.dataFeature = filter.filter_relifF(
        #     scoreOfRelifF=self.MI, dataX=self.dataX, dataFeature=self.dataFeature)
        # 获取特征数量
        self.dataFeatureNum = len(self.dataFeature)
        self.mutatePro = 1/self.dataFeatureNum
        # 操作 --- 获取SU 、 MI 和 psi
        self.calculateSUAndMI(dataX=self.dataX)
        # 初始化 信息素矩阵
        self.matrix_pheromone = np.ones((self.dataFeatureNum * 2,self.dataFeatureNum * 2))
        # 获得启发式信息矩阵
        self.calculateHeuristic()
        print()

    # 设置参数
    def setParameter(self,alpha,beta,rho,populationNum,iteratorTime,tau_min,tau_max):
        self.alpha = alpha
        # 参数 beta
        self.beta = beta
        # 蒸发率 rho
        self.rho = rho
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime
        self.tau_min = tau_min
        self.tau_max = tau_min

    def calculateHeuristic(self):
        # 初始化 启发式信息矩阵
        self.matrix_heuristic = np.ones((self.dataFeatureNum, self.dataFeatureNum * 2))
        # 偶数 为 0  奇数为 1
        for i in range(self.dataFeatureNum):
            for j in range(int(self.dataFeatureNum * 2)):
                # 向下取整
                t_j = math.floor(j/2)
                # 为 偶数数 --- 0
                if j % 2 == 0:
                    self.matrix_heuristic[i][j] = self.matrix_SU[i][t_j] * (1 - self.featurePSI[t_j])
                elif j % 2 == 1:
                    self.matrix_heuristic[i][j] = (1 - self.matrix_SU[i][t_j]) * self.featurePSI[t_j]

    # 计算获得SU 和 MI矩阵  ---
    def calculateSUAndMI(self,dataX):
        # feature -- class
        self.featureMI = np.zeros(self.dataFeatureNum)
        # feature -- feature
        self.matrix_SU = np.zeros((self.dataFeatureNum,self.dataFeatureNum))
        self.matrix_MI = np.zeros((self.dataFeatureNum,self.dataFeatureNum))
        # psi
        self.featurePSI = np.zeros(self.dataFeatureNum)
        for i in range(self.dataFeatureNum):
            f_i = dataX[:,i]
            self.featureMI[i] = information_gain(f1=f_i,f2=self.dataY)
            for j in range(i,self.dataFeatureNum):
                f_j = dataX[:,j]
                # 计算互信息
                t1 = information_gain(f_i, f_j)
                self.matrix_MI[i][j] = t1
                self.matrix_MI[j][i] = t1
                # 计算信息熵
                t2 = ee.entropyd(f_i)
                t3 = ee.entropyd(f_j)
                # 计算su
                su = 2.0 * t1 / (t2 + t3 + 0.1)
                self.matrix_SU[i][j] = su
                self.matrix_SU[j][i] = su
        # 计算矩阵psi
        fn = self.dataFeatureNum - 1
        for i in range(self.dataFeatureNum):
            tempSum = np.sum(self.matrix_MI[i])
            tempSum = (tempSum - self.matrix_MI[i][i])/fn
            self.featurePSI[i] = self.featureMI[i] - tempSum
        # 对矩阵psi执行最大最小归一化
        max_psi = np.max(self.featurePSI)
        min_psi = np.min(self.featurePSI)
        self.featurePSI = (self.featurePSI - min_psi)/(max_psi - min_psi)

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

        def __init__(self, INSA):
            self.initOfChromosome(INSA=INSA)

        def initOfChromosome(self, INSA):
            # 得到解
            self.generateSolution(INSA = INSA)
            fX = INSA.dataX[:,self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=fX,findData_y=INSA.dataY,CV=10)
            # 获得 多少个特征。。
            self.getNumberOfSolution()

        # 产生解
        def generateSolution(self,INSA):
            # 所有未选择的特征 -- 初始化
            self.unselectedL = np.arange(INSA.dataFeatureNum)
            self.selectedL = []
            self.stateOfSelect = np.asarray([])

            # 计数
            count = 0
            self.featureOfSelect = np.zeros(INSA.dataFeatureNum)
            # 蚂蚁所选择的初始特征
            firstFeature = np.random.randint(INSA.dataFeatureNum)
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
                    eta = np.delete(INSA.matrix_pheromone[ne], deleteIndex)
                else:
                    eta = np.delete(INSA.matrix_pheromone[ne + 1], deleteIndex)
                # 第 nf 个特征 与所有未选择特征的启发式信息
                tau = np.delete(INSA.matrix_heuristic[nt], deleteIndex)

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

    # 初始化种群
    def initChromosomeList(self):
        for i in range(0,self.populationNum):
            ant = self.Ant(INSA=self)
            self.population_ACO = np.append(self.population_ACO,ant)

    # 获得非支配前沿
    def getParetoFront(self):
        tempP = np.concatenate((self.paretoFront,self.population_ACO))
        # 去重复
        tempP = self.deleteDuolicate(conPop = tempP)
        self.paretoFront = getPF(pop = tempP)
        acc_array = np.zeros(len(self.paretoFront))
        # 非支配解集中进行循环
        for i in range(len(self.paretoFront)):
            # acc
            acc_array[i] = 1 - self.paretoFront[i].mineFitness
        # 排序
        acc_index = np.argsort(acc_array)[::-1]
        if acc_array[acc_index[0]] > self.globalFitness:
            self.globalFitness =  acc_array[acc_index[0]]
            self.globalSolution = self.paretoFront[acc_index[0]].featureOfSelect
            self.globalLen = self.paretoFront[acc_index[0]].numberOfSolution
        print("acc = ",self.globalFitness,"len = ",self.globalLen)

    # 更新信息素
    def updatePheromone(self):
        # 信息素蒸发
        self.matrix_pheromone = (1 - self.rho) * self.matrix_pheromone
        # 提取paretoFront中个体的信息素
        for i in range(len(self.paretoFront)):
            delta_extend = 1 / (self.paretoFront[i].mineFitness + self.paretoFront[i].proportionOfFeature)
            # 更新值
            state = self.paretoFront[i].stateOfSelect
            p = 0
            q = 1
            while q != self.dataFeatureNum - 1:
                if state[p] == 0:
                    temp_i = p * 2
                else:
                    temp_i = p * 2 + 1
                if state[q] == 0:
                    temp_j = q * 2
                else:
                    temp_j = q * 2 + 1
                self.matrix_pheromone[temp_i][temp_j] += delta_extend
                # 控制范围
                np.clip(self.matrix_pheromone, self.tau_min, self.tau_max)

                p += 1
                q += 1

    # 去重复
    def deleteDuolicate(self, conPop):
        # 首先合并solution
        temp = np.asarray(conPop[0].featureOfSelect)
        for i in range(1, len(conPop)):
            temp = np.vstack((temp, conPop[i].featureOfSelect))
        # 转化为dataFrame
        df = pd.DataFrame(temp)
        # 去重
        df_dup = df.drop_duplicates()
        # 获得索引
        index_dup = df_dup.index.values
        # 提取相对应的个体
        new_pop = conPop[index_dup]
        return new_pop

    # 下一代种群
    def getNextPop(self):
        for i in range(self.populationNum):
            self.population_ACO[i].generateSolution(INSA=self)
            fX = self.dataX[:, self.population_ACO[i].featureOfSelect == 1]
            self.population_ACO[i].mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=fX, findData_y=self.dataY, CV=10)
            # 获得 多少个特征。。
            self.population_ACO[i].getNumberOfSolution()


    # 运行
    def run(self):
        import time
        start = time.time()
        # 初始化种群
        print("初始化种群")
        # 初始化种群
        self.initChromosomeList()
        # 循环
        runTime = 0
        while runTime < self.iteratorTime:
            print(" 第", runTime + 1, "轮")
            self.getParetoFront()
            self.updatePheromone()
            # 产生新的解
            self.getNextPop()
            runTime += 1
        print("================================================")
        print(f"all time = {time.time() - start} seconds")
        print(1 - self.globalFitness, self.globalLen)
        print()
        for i in range(len(self.paretoFront)):
            print(self.paretoFront[i].mineFitness, " ", self.paretoFront[i].numberOfSolution)
        print("####################################")
        print("populationNum = ", self.populationNum)
        print("iteratorTime = ", self.iteratorTime)
        print(" ")



if __name__ == "__main__":
    ducName = "BreastCancer1"
    # ducName = "CLLSUB"

    path = "D:/MachineLearningBackUp/dataCSV/dataCSV_high/" + ducName + ".csv"
    from sklearn.datasets import load_iris
    data = load_iris()
    dataX = data.data
    dataY = data.target
    dataCsv = ReadCSV(path=path)
    # print("获取文件数据")
    # dataCsv.getData()
    # genetic = INSA(dataX=dataCsv.dataX, dataY=dataCsv.dataY)
    genetic = INSA(dataX=dataX, dataY=dataY)
    genetic.setParameter(alpha=1,beta=0.7,rho=0.3, populationNum=100, iteratorTime=100,tau_min=0.1,tau_max=6)
    genetic.run()
