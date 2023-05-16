import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from random import sample

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from dataProcessing.drawPlo import pltForWin
from mfmode.operators import mutate_rand1,crossover_num
from Filter.filter import FilterOfData
from dataProcessing.ReadDataCSV_new import ReadCSV
from MOEAD.ParetoFront.paretoEP import updatePF


class MOEA_D_STAT:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 特征的存放数组
    dataFeature = np.asarray([])
    # 特征的数量
    dataFeatureNum = 0
    # 粒子群
    population_DE = np.asarray([])
    # 每一个种群的数量
    populationNum = 0
    # 种群迭代次数
    iteratorTime = 0
    # 步长
    F = 0
    # 变异率
    CR = 0
    # alpha 用于适应度计算
    alpha  = 0
    # 用于判断Ne
    sigma = 0
    # 动态点的百分比
    movePb = 0

    # 用于存放临时种群，父类个体从中提取
    Ne = np.asarray([])
    # 区间数量
    intervalNum = 0
    # 区间
    interval = np.asarray([])
    # 动态点迭代boundary次进行移动
    boundary = 0
    # 动态参考点的数量
    M = 0
    # 存放每一个区间有那些参考点的索引
    inferRP = []
    # 近邻权重向量索引
    closestWeight = []
    # 理想点
    ideal_Z = np.asarray([])
    # 存放参考点
    referPoint = np.asarray([])
    # 存放每一个特征单独用于分类的acc
    accOfSingleFeature = np.asarray([])
    # 存放惩罚参数向量
    thetaArray = np.asarray([])
    # 惩罚值
    penalty = 0
    # H
    H = 0
    # 目标数量
    objNum = 2
    # 近邻数量
    T = 0
    # 保存pareto解
    EP = np.asarray([])
    # 每一个特征的relifF评分
    scoreOfReliefF = np.asarray([])
    # 全局最优选择
    globalBestSolution = np.asarray([])
    # 全局最优选择得到的fitness
    globalBestFitness = 0
    # 全局最优选择特征长度
    globalBestLen = 0

    globalAcc = np.asarray([])
    globalLen = np.asarray([])

    # 初始化
    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.dataFeature = np.arange(dataX.shape[1])

        # 对数据01 归一化，将每一列的数据缩放到 [0,1]之间
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        print("进行filter，提取一部分数据")
        # filter
        filter = FilterOfData(dataX=self.dataX,dataY=self.dataY)
        #  compute reliefF score
        self.scoreOfReliefF = filter.computerRelifFScore()
        # 进行filter过滤数据 ----  默认采用的是膝节点法
        self.scoreOfReliefF, self.dataX, self.dataFeature = filter.filter_relifF(
            scoreOfRelifF=self.scoreOfReliefF, dataX=self.dataX, dataFeature=self.dataFeature)
        # 更新 特征长度
        self.dataFeatureNum = len(self.dataFeature)
        # 初始化每一个特征的acc
        self.accOfSingleFeature = np.zeros(self.dataX.shape[1])



# initialization parameters
    def  setParameter(self,populationNum,iteratorTime,sigma,F,CR ,alpha,intervalNum,movePb):
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime
        self.F = F
        self.CR = CR
        self.alpha  = alpha
        self.H = populationNum - 1
        self.sigma = sigma
        self.intervalNum = intervalNum
        self.M = int(0.4 * self.populationNum)
        self.movePb = movePb
        self.boundary = int(self.iteratorTime / intervalNum)
        # 近邻的设定
        self.T = int(self.populationNum / 10)
        if self.T < 4:
            self.T = 4
        # 初始化 理想点 ，优于是最小化问题，所以初始化为[1,1]
        self.ideal_Z = np.ones(self.objNum)
        self.globalAcc = np.zeros(self.populationNum)
        self.globalLen = np.zeros(self.populationNum)

    class Genetic():
        # 每一个个体所选择出来的特征组合   --- 二进制
        featureOfSelect = np.asarray([])
        # 每一个个体所选择出来的特征组合   --- 实数制
        featureOfSelectNum = np.asarray([])
        # 特征组合的长度
        numberOfSolution = 0
        # 特征组合的索引
        indexOfSolution = np.asarray([])
        # 所选特征长度占总长度的比例
        proportionOfFeature = 0
        # 当前的选择的fitness
        mineFitness = 0
        # 惩罚系数
        theta = 0
        # 对比适应度
        funFitness = 0
        # 每个个体的最大特征数量
        n_ref = 0
        # 每个个体的最小特征数量
        n_ref_bottom = 0
        def __init__(self,moeaD_stat,mode):
            if mode == 'init':
                # 初始化一个位置
                self.defineInitPosition(numberOfFeature = moeaD_stat.dataFeature.shape[0],moeaD=moeaD_stat)

            elif mode == "iterator":
                self.iteratorOfChromosome(moeaD=moeaD_stat)
                pass

        def iteratorOfChromosome(self, moeaD):
            self.featureOfSelect = np.zeros(len(moeaD.dataFeature))
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()

        # 给每一个粒子初始化一个位置
        def defineInitPosition(self,numberOfFeature,moeaD):
            self.featureOfSelectNum = np.random.rand(numberOfFeature)
            self.featureOfSelect = np.where(self.featureOfSelectNum > 0.6 ,1,0)
            # 更新信息
            self.getNumberOfSolution()
            # 计算acc
            feature_x = moeaD.dataX[:, self.featureOfSelect == 1]
            self.mineFitness = 1 - moeaD.fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=moeaD.dataY, CV=10)


        # 得到选择特征数量是多少个
        def getNumberOfSolution(self):
            # 得到选择的组合里面 为1 的索引为多少个
            index = np.where(self.featureOfSelect == 1)[0]
            # 存放每一个被选择的索引索引
            self.indexOfSolution = np.copy(index)
            # 得到的索引的长度是多少
            self.numberOfSolution = index.shape[0]
            # 获得比例
            self.proportionOfFeature = self.numberOfSolution / self.featureOfSelect.shape[0]
    # 初始化任务种群
    def initPopulation(self):
        # 每一个小区间的特征数量
        numberOfInterval = int(self.dataX.shape[1] / self.populationNum)
        # 生成种群
        for i in range(self.populationNum):
            ind = self.Genetic(moeaD_stat=self,mode='init')
            ind.funFitness = self.getFitness(eRates=ind.mineFitness,fRatios=ind.proportionOfFeature,
                                             S=ind.numberOfSolution,n_ref=ind.n_ref)

            self.globalAcc[i] = 1 - ind.mineFitness
            self.globalLen[i] = ind.numberOfSolution

            # 新生成的解和全局最优对比
            if 1 - ind.mineFitness > self.globalBestFitness:
                self.globalBestFitness = 1 - ind.mineFitness
                self.globalBestSolution = ind.featureOfSelect
                self.globalBestLen = ind.numberOfSolution

            self.population_DE = np.append(self.population_DE,ind)
            self.EP = updatePF(EP=self.EP,individual=ind)

    # 产生初始化参考点  --- 用于还未找到边界点
    def generateReferPoint(self,numI,hBroundary):
        # 每一个小区间的特征数量
        numberOfInterval = int(self.dataX.shape[1] / self.intervalNum)
        # 全为1的数组
        self.referPoint = np.asarray([])
        # 确定每一个区间内部有那些参考点
        self.inferRP = [[] for i in range(self.intervalNum)]
        # 动态点的数量
        numberOfM = int(self.movePb * self.populationNum)
        # 静态点的数量
        numberOfF = self.populationNum - numberOfM
        # 临时静态存储数组
        tempArrayOfF = [[] for i in range(self.intervalNum)]
        # 索引
        indexOfInterval = 1
        # 分配静态参考点 -- F
        for i in range(numberOfF):
            tempF = int((i + 1) / numberOfF * self.dataX.shape[1])
            if tempF <= indexOfInterval * numberOfInterval:
                tempArrayOfF[indexOfInterval - 1].append(tempF)
            elif indexOfInterval == 4:
                tempArrayOfF[indexOfInterval - 1].append(tempF)
            else:
                indexOfInterval += 1
                tempArrayOfF[indexOfInterval - 1].append(tempF)

        # 临时动态参考点存储数组
        tempArrayOfM = []
        if hBroundary == True:
            # 更新 numI区域的参考点
            for i in range(numI+1):
                numberOfM += len(tempArrayOfF[i])
                # 分配动态参考点 -- M
                for i in range(numberOfM):
                    tempArrayOfM.append(int((i + 1) / numberOfM * numberOfInterval * (numI + 1)))
        else:
            # 更新 numI区域的参考点
            numberOfM += len(tempArrayOfF[numI])
            # 分配动态参考点 -- M
            for i in range(numberOfM):
                tempArrayOfM.append(self.interval[numI] + int((i + 1) / numberOfM * numberOfInterval))

        # 添加的参考点数量
        addNumRP = 0
        if hBroundary == True:
            self.referPoint = np.append(self.referPoint, np.asarray(tempArrayOfM))
            for i in range(numI+1):
                self.inferRP[i] = np.arange(addNumRP,addNumRP + int((i + 1)/(numI + 1) * len(tempArrayOfM)))
                addNumRP += int((i + 1)/(numI + 1) * len(tempArrayOfM))
            for j in range(numI + 1,self.intervalNum):
                self.referPoint = np.append(self.referPoint, np.asarray(tempArrayOfF[j]))
                self.inferRP[j] = np.arange(addNumRP, addNumRP + len(tempArrayOfF[j]))
                addNumRP += len(tempArrayOfF[j])
        else:
            # 输入参考点
            for j in range(self.intervalNum):
                if j == numI:
                    self.referPoint = np.append(self.referPoint,np.asarray(tempArrayOfM))
                    self.inferRP[j] = np.arange(addNumRP, addNumRP + len(tempArrayOfM))
                    addNumRP += len(tempArrayOfM)
                else:
                    self.referPoint = np.append(self.referPoint, np.asarray(tempArrayOfF[j]))
                    self.inferRP[j] = np.arange(addNumRP,addNumRP + len(tempArrayOfF[j]))
                    addNumRP += len(tempArrayOfF[j])

        # 更新每一个个体的参考点
        for i in range(self.populationNum):
            # 分配 每一个个体的 所能选择的最大特征数量
            self.population_DE[i].n_ref = self.referPoint[i]
            self.population_DE[i].funFitness = self.getFitness(eRates=self.population_DE[i].mineFitness,
                                                               fRatios=self.population_DE[i].proportionOfFeature,
                                                               S=self.population_DE[i].numberOfSolution,
                                                               n_ref=self.population_DE[i].n_ref)
            if i == 0:
                # 分配 每一个个体的 所能选择的最小特征数量
                self.population_DE[i].n_ref_bottom = 0
            else:
                # 分配 每一个个体的 所能选择的最小特征数量
                self.population_DE[i].n_ref_bottom = self.referPoint[i - 1]



    # 计算每一个特征单独分类的 acc
    def calculateAccOfSingleFeature(self):
        for i in range(self.dataX.shape[1]):
            feature_x = self.dataX[:,i]
            self.accOfSingleFeature[i] = self.fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=self.dataY, CV=10)


    # 获得本实验 评判适应度值
    def getFitness(self,eRates,fRatios,S,n_ref):
        # 惩罚项
        penalty = 100 * np.max([S - n_ref,0])
        temp_v = eRates + penalty + fRatios
        return temp_v


    # 得到T个近邻参考点的索引
    def getClosestReferPoint(self):
        # 生成一个全0矩阵用于存放 每一个参考点到其他参考点的距离
        EuclideanDistances = np.zeros((self.populationNum, self.populationNum))
        # 计算每一个参考点的 T个近邻参考点
        for i in range(self.populationNum):
            # 计算距离
            for j in range(i, self.populationNum):
                temp_1 = np.asarray(self.referPoint[i]) - np.asarray(self.referPoint[j])
                distance_ij = np.power(np.sum(np.power(temp_1, 2)), 0.5)
                EuclideanDistances[i][j] = distance_ij
                EuclideanDistances[j][i] = distance_ij
            # # 计算完毕距离后，用于添加每一个参考点的 前T个近邻参考点
            # 从小到大排序
            sortIndex = np.argsort(EuclideanDistances[i])
            # 提取前T个
            self.closestWeight.append(sortIndex[:self.T])

      # 得到Ne
    def getNe(self,index):
        rand1 = np.random.rand()
        # 如果小于 ， 则Ne 就是 该个体的近邻个体
        if rand1 < self.sigma:
            self.Ne = self.population_DE[np.asarray(self.closestWeight[index])]
        else:
            # 否则就是整个种群
            self.Ne = self.population_DE

    # 修复解 -- 添加个体  (n_ref_bottom,n_ref]
    def repairSolution_add(self,u_x,u_x_bin,u_S,n_ref_bottom):
        # 获取所有未选择的特征索引 -- 为0
        unSelectIndex = np.where(u_x_bin == 0)[0]
        # 根据每一个特征的分类acc来进行排序
        acc_feature = self.accOfSingleFeature[unSelectIndex]
        # 进行排序 -- 降序排序
        accIndex = np.argsort(acc_feature)[::-1]
        # 添加特征
        tempIndex = 0
        while u_S <= n_ref_bottom:
            # 未选择特征中 分类准确度最高的特征为1 --- 二进制
            u_x_bin[unSelectIndex[accIndex[tempIndex]]] = 1
            # 未选择特征中 分类准确度最高的特征为[0.6,1] --- 实数制
            u_x[unSelectIndex[accIndex[tempIndex]]] = np.random.uniform(0.6,1)
            # 加1
            tempIndex += 1
            u_S += 1
        return u_x,u_x_bin,u_S

    # 修复解 -- 减少个体
    def repairSolution_del(self,u_x,u_x_bin,u_S,n_ref):
        # 获取所有选择的特征索引 -- 为1
        selectIndex = np.where(u_x_bin == 1)[0]
        # 根据每一个特征的分类acc来进行排序
        acc_feature = self.accOfSingleFeature[selectIndex]
        # 进行排序 -- 升序排序
        accIndex = np.argsort(acc_feature)
        # 添加特征
        tempIndex = 0
        while u_S > n_ref:
            # 选择特征中 分类准确度最低的特征为0 --- 二进制
            u_x_bin[selectIndex[accIndex[tempIndex]]] = 0
            # 选择特征中 分类准确度最高的特征为[0,0.6] --- 实数制
            u_x[selectIndex[accIndex[tempIndex]]] = np.random.uniform(0,0.6)
            # 加1
            tempIndex += 1
            u_S -= 1
        return u_x, u_x_bin, u_S

    # 获得新解
    def getNewSolutionForDE(self,individual_i):
        # 随机提取两个个体作为父个体
        xArray = sample(self.Ne.tolist(),2)
        # 变异算子
        v_x = mutate_rand1(xMain=individual_i.featureOfSelectNum,
                           x1=xArray[0].featureOfSelectNum,
                           x2=xArray[1].featureOfSelectNum,F =self.F)
        # 交叉算子
        u_x = crossover_num(x=individual_i.featureOfSelectNum,
                            v_x = v_x, crossPro = self.CR)
        u_x = np.where(u_x < 0,0,u_x)
        u_x = np.where(u_x > 1,1,u_x)
        #  转化成二进制
        u_x_bin = np.where(u_x > 0.6 ,1,0)
        # 所选择特征的数量
        u_S = np.where(u_x_bin == 1)[0].shape[0]
        # 判单u_x是否需要修复
        if u_S > individual_i.n_ref:
            # 删除特征
            u_x, u_x_bin, u_S = self.repairSolution_del(u_x=u_x,u_x_bin=u_x_bin
                                                        ,u_S=u_S,
                                                        n_ref=individual_i.n_ref)
        elif u_S < individual_i.n_ref_bottom:
            # 添加特征
            u_x, u_x_bin, u_S = self.repairSolution_add(u_x=u_x, u_x_bin=u_x_bin
                                                        , u_S=u_S,
                                                        n_ref_bottom=individual_i.n_ref_bottom)
        # 所选特征数量占特征总数量的比例
        u_fRatios = u_S/self.dataX.shape[1]
        # 获得所选择的数据
        findData_x = self.dataX[:,u_x_bin == 1]
        # 计算error
        u_err = 1 - self.fitnessFunction_KNN_CV(findData_x=findData_x,findData_y=self.dataY,CV=10)
        # 惩罚项
        u_penalty = 100 * np.max([u_S - individual_i.n_ref,0])
        # 计算funFitness
        u_fit= u_err + u_penalty + self.alpha * u_fRatios

        if u_fit < individual_i.funFitness:
            individual_i.funFitness = u_fit
            individual_i.mineFitness = u_err
            individual_i.featureOfSelect = u_x_bin
            individual_i.featureOfSelectNum = u_x
            # 更新个体其他属性
            individual_i.getNumberOfSolution()
            self.EP = updatePF(EP=self.EP,individual=individual_i)
            if 1 - individual_i.mineFitness > self.globalBestFitness:
                self.globalBestFitness = 1 - individual_i.mineFitness
                self.globalBestSolution = individual_i.featureOfSelect
                self.globalBestLen = individual_i.numberOfSolution


    # 获得下一代种群
    def getNextPopulation(self):
        for i in range(self.populationNum):
            # 得到第一个个体的 Ne
            self.getNe(index=i)
            # 随机选择两个个体从Ne中，进行DE的操作，获得新解
            self.getNewSolutionForDE(individual_i=self.population_DE[i])


    # 输出每一代最优
    def printOptimal(self,runTime):
        accArray = np.zeros(self.populationNum)
        lenArray = np.zeros(self.populationNum)
        for i in range(self.populationNum):
            accArray[i] = 1 - self.population_DE[i].mineFitness
            lenArray[i] = self.population_DE[i].numberOfSolution

        # 排序
        indexAcc = np.argsort(accArray)[::-1]
        print("第",runTime+1,"代 ","acc: ",accArray[indexAcc[0]]," len:",lenArray[indexAcc[0]])
        print("acc: ",accArray[indexAcc[-1]]," len:",lenArray[indexAcc[-1]])

    def fitnessFunction_KNN_CV(self,findData_x, findData_y, CV):

        if findData_x.shape[0] == 0:
            return 0

        # 计算1
        # 先采用10折交叉验证的方式计算
        knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto", metric='manhattan')
        # 转化为二维数组
        if len(findData_x.shape) == 1:
            findData_x = findData_x.reshape((len(findData_x), 1))

        # kf = KFold(n_splits=CV, shuffle=True, random_state=1)
        # cv参数决定数据集划分比例，这里是按照9:1划分训练集和测试集
        scores = cross_val_score(knn, findData_x, findData_y, cv=CV, scoring='accuracy')
        # scores = cross_val_score(knn, findData_x, findData_y, cv=kf, scoring='accuracy',n_jobs=-1)
        accuracy = scores.mean()
        return accuracy

    # 区间
    def setInterval(self):
        numberOfInterval = int(self.dataX.shape[1] / self.intervalNum)
        # 存放的值是 [0,1 * numberOfInterval,2 * numberOfInterval,...]
        self.interval = np.arange(self.intervalNum + 1) * numberOfInterval

    # 边界区间
    def searchBoundaryI(self,numI):
        length1 = len(self.inferRP[numI])
        length2 = len(self.inferRP[numI + 1])
        # numI 所对应区间内的个体 的acc 和len
        err_previou = np.ones(length1)
        # len_previou = np.ones(length1)
        indexOferr_1 = np.argsort(err_previou)[::-1]
        err_previou = err_previou[indexOferr_1]
        for i in range(length1):
            err_previou[i] = self.population_DE[self.inferRP[numI][i]].mineFitness
            # len_previou[i] = self.population_DE[self.inferRP[numI][i]].proportionOfFeature
        # numI + 1 所对应区间内的个体 的acc 和len
        err_below = np.ones(length2)
        for i in range(length2):
            err_below[i] = self.population_DE[self.inferRP[numI + 1][i]].mineFitness
        # best below
        indexOferr_2 = np.argsort(err_below)
        errbestBelow = err_below[indexOferr_2[0]]
        # 对比
        for i in range(length1):
            # 若 previous中有解支配后一个区间的最优解，则后方为非冲突
            if err_previou[i] <= errbestBelow:
                # 重新分配参考点
                self.generateReferPoint(numI=numI,hBroundary=True)
                return False
            # 否则 previous中有解支配后一个区间的最优解，则后方为冲突
            else:
                if i == length1 - 1:
                    return True
    # 画图
    def pltForWin(self,iteNum, pop, iteratorTime, name,referPoint):
        plt.ion()
        length_LA = len(pop)
        # 存放 error
        x = np.ones(length_LA)
        # 存放 len
        y = np.ones(length_LA)
        for i in range(length_LA):
            x[i] = pop[i].proportionOfFeature
            y[i] = pop[i].mineFitness

        rp_y = np.zeros(len(referPoint))
        referPoint = referPoint / self.dataX.shape[1]
        plt.xlabel("len")
        plt.ylabel("error")
        plt.title(name+"  "+str(iteNum))
        plt.plot(x, y, color='r', marker='o', linestyle='None')
        plt.plot(referPoint, rp_y, color='b', marker='.', linestyle='None')
        plt.pause(0.1)
        if iteNum != iteratorTime:
            plt.cla()

    # 运行
    def run(self):
        # 迭代次数
        runTime = 0
        boundary_m = 0
        numI = 0
        # 设置区间
        self.setInterval()
        # 初始化函数
        self.initPopulation()
        # 产生参考点--- 初始化在0
        self.generateReferPoint(numI=numI,hBroundary=False)
        # 初始化近邻
        self.getClosestReferPoint()
        # 计算每一个特征的acc
        self.calculateAccOfSingleFeature()
        needMovePoint = True
        while runTime < self.iteratorTime:
            self.printOptimal(runTime = runTime)
            self.getNextPopulation()
            runTime += 1
            boundary_m += 1

            if needMovePoint == True and boundary_m == self.boundary:
                # 判断是否还需要移动参考点
                needMovePoint = self.searchBoundaryI(numI=numI)
                if needMovePoint == True:
                    numI += 1
                    self.generateReferPoint(numI=numI,hBroundary=False)
                boundary_m = 0
            # 画图
            self.pltForWin(iteNum=runTime, pop=self.population_DE,
                           iteratorTime=self.iteratorTime, name="MOEA/D_DYN",
                           referPoint=self.referPoint)

        print(" PF 解")
        for i in range(len(self.EP)):
            print(1 - self.EP[i].mineFitness, " ", self.EP[i].numberOfSolution)

        print("全局最优acc",self.globalBestFitness," 全局最优len",self.globalBestLen)


if __name__ == '__main__':
    import time
    start = time.time()
    ducName = "BreastCancer1"
    path = "D:\MachineLearningBackUp\dataCSV\dataCSV_high\\" + ducName + ".csv"
    dataCsv = ReadCSV(path=path)
    print("获取文件数据")
    dataCsv.getData()
    moead = MOEA_D_STAT(dataX=dataCsv.dataX, dataY=dataCsv.dataY)
    moead.setParameter(populationNum=100, iteratorTime=100, sigma = 0.85,
                       F = 0.7,CR = 0.6 ,alpha =0.01,intervalNum = 4,movePb = 0.4)
    moead.run()
    print("================================================")
    print(f"all time = {time.time() - start} seconds")
