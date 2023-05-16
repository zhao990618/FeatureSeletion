import copy
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sklearn.preprocessing import MinMaxScaler
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from Filter.filter import FilterOfData
from dataProcessing.ReadDataCSV_new import ReadCSV
from MOEAD.WeightVector.vector import Mean_vector

class MOEA_D:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 特征的存放数组
    dataFeature = np.asarray([])
    # 特征的数量
    dataFeatureNum = 0
    # 种群
    population_EA = []
    # 每一个种群的数量
    populationNum = 0
    # 种群迭代次数
    iteratorTime = 0




    # 近邻权重向量索引
    closestWeight = []
    # 匹配权重向量 的 个体的索引
    LA = np.asarray([])
    # 存放非支配解
    EP = np.asarray([])
    # 理想点
    ideal_Z = np.asarray([])
    # 存放权重向量
    weightArray = np.asarray([])
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

    # 每一个特征的relifF评分
    scoreOfRelifF = np.asarray([])
    # 全局最优选择
    globalBestSolution = np.asarray([])
    # 全局最优选择得到的fitness
    globalBestFitness = 0
    # 全局最优选择特征长度
    glboalBestLen = 0

    # 初始化
    def __init__(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.dataFeature = np.arange(dataX.shape[1])
        # 对数据01 归一化，将每一列的数据缩放到 [0,1]之间
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        print("进行filter，提取一部分数据")
        # filter
        filter = FilterOfData(dataX=self.dataX, dataY=self.dataY)
        #  compute relifF score
        self.scoreOfRelifF = filter.computerRelifFScore()
        # 进行filter过滤数据 ----  默认采用的是膝节点法
        self.scoreOfRelifF, self.dataX, self.dataFeature = filter.filter_relifF(
            scoreOfRelifF=self.scoreOfRelifF, dataX=self.dataX, dataFeature=self.dataFeature)
        # 更新 特征长度
        self.dataFeatureNum = len(self.dataFeature)

    # initialization parameters
    def setParameter(self, populationNum, iteratorTime):
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime
        self.H = populationNum - 1
        self.T = int(self.populationNum / 10)
        # 初始化 理想点 ，优于是最小化问题，所以初始化为[1,1]
        self.ideal_Z = np.ones(self.objNum)

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
        # 全局最优选择
        gBestSolution = np.asarray([])
        # 个体最优选择实数制
        gBestSolutionNum = np.asarray([])
        # 全局最优选择得到的fitness
        gBestFitness = 1

        def __init__(self,moDE,mode):
            if mode == "init":
                # 初始化一个位置
                self.defineInitPosition(numberOfFeature = moDE.dataFeature.shape[0],moDE=moDE)
            elif mode == "iterator":
                self.iteratorOfChromosome(MFEA=moDE)

        def iteratorOfChromosome(self, MFEA):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()

        # 给每一个粒子初始化一个位置
        def defineInitPosition(self, numberOfFeature, moDE):
            self.featureOfSelectNum = np.random.rand(numberOfFeature)
            self.featureOfSelect = np.where(self.featureOfSelectNum > 0.6, 1, 0)
            # 更新信息
            self.getNumberOfSolution()
            # 计算acc
            feature_x = moDE.dataX[:, self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=moDE.dataY, CV=10)


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
        # 生成种群
        for i in range(self.populationNum):
            particle = self.Genetic(moDE=self,mode="init")
            self.population_EA = np.append(self.population_EA,particle)
            self.LA = np.append(self.LA,particle)

    # 产生权重向量
    def generateWeightVector(self):
        # 设置了参数H决定生成权重数量 --- 设置权重的维度objNum
        weightV = Mean_vector(H=self.H, m=self.objNum)
        # 获得权重向量
        self.weightArray = weightV.generate()
        # 获得权重向量的 T 个近邻向量
        self.getClosestWeightVector()

    # 计算每一个权重向量的近邻向量
    def getClosestWeightVector(self):
        # 生成一个全0矩阵用于存放 每一个特征到其他特征的距离
        EuclideanDistances = np.zeros((self.populationNum,self.populationNum))
        # 计算每一个权重向量的 T个近邻向量
        for i in range(self.populationNum):
            # 计算距离
            for j in range(i,self.populationNum):
                temp_1 = np.asarray(self.weightArray[i]) - np.asarray(self.weightArray[j])
                distance_ij = np.power(np.sum(np.power(temp_1,2)),0.5)
                EuclideanDistances[i][j] = distance_ij
                EuclideanDistances[j][i] = distance_ij
        # 计算完毕距离后，用于添加每一个权重向量的 前T个近邻权重
        for i in range(self.populationNum):
            # 从小到大排序
            sortIndex = np.argsort(EuclideanDistances[i])
            # 提取前T个
            self.closestWeight.append(sortIndex[:self.T])

    # 设置理想点Z
    def setIdealPoint(self):
        # 从当代种群中抽取 每一个目标上的最优值作为理想点
        for i in range(self.populationNum):
            # if self.population_EA[i].mineFitness < self.ideal_Z[0]:
            #     self.ideal_Z[0] = self.population_EA[i].mineFitness
            # if self.population_EA[i].proportionOfFeature < self.ideal_Z[1]:
            #     self.ideal_Z[1] = self.population_EA[i].proportionOfFeature
            self.updataIdealPoint(indivudal=self.population_EA[i])

    # 更新理想点Z
    def updataIdealPoint(self,indivudal):
        if indivudal.mineFitness < self.ideal_Z[0]:
            self.ideal_Z[0] = indivudal.mineFitness
        if indivudal.proportionOfFeature < self.ideal_Z[1]:
            self.ideal_Z[1] = indivudal.proportionOfFeature

    # 计算PBI
    def calculatePBI(self, F_i, weight_i, theta_i):
        temp_v = np.dot(self.ideal_Z - F_i, weight_i)
        t_1 = np.linalg.norm(weight_i, ord=2)
        # 获得 d_1
        d_1 = np.asarray(np.abs(temp_v) / t_1)

        temp_v = self.ideal_Z + d_1 * weight_i / t_1
        d_2 = np.linalg.norm(F_i - temp_v, ord=2)
        PBI_i = d_1 + theta_i * d_2
        return PBI_i

    # 更新存储
    def updateArchive(self,pop):
        for i in range(self.populationNum):
            # 用于临时存放数据
            tempCache = np.asarray([])

            for j in range(len(pop)):
                # 获得第 j 个个体的两个目标
                F_i = np.asarray([pop[j].mineFitness,pop[j].proportionOfFeature])

                PBI_i = self.calculatePBI(F_i=F_i,weight_i=self.weightArray[i],theta_i=self.thetaArray[i])

                # 添加值
                tempCache = np.append(tempCache,PBI_i)

            # 获得最优个体的索引
            bestIndex = np.argsort(tempCache)[0]
            # 将pop中里权重i在PBI计算中最优的个体 提取出来
            bestIndiviaulofW_i = pop[bestIndex]

            # 用于存放 第i个权重选择的 最优solution
            self.LA[i] = copy.deepcopy(bestIndiviaulofW_i)
            # 将该个体删除
            pop = np.delete(pop, bestIndex).tolist()

        # 根据 LA 去将个体重新分配  ----  test用  看看加不加效果是由有提升
        self.population_EA = copy.deepcopy(self.LA)

    # 交叉  --
    def crossoverOperater(self,solution1,solution2):
        # 生成一个矩阵，若是存在rand值小于该矩阵对应位置的值，那就对应的位置就要进行交叉操作
        crossProArray = np.random.random(self.dataFeature.shape[0])
        # 交叉
        for i in range(self.dataFeature.shape[0]):
            # 若是该位置两个solution所存放的值是不一样的就可以进行交叉
            if solution1[i] != solution2[i]:
                # 生成的随机数，用于判断该位置是否需要进行交叉操作
                rand = np.random.rand()
                # 若是小于交叉矩阵里的概率值
                if rand > crossProArray[i]:
                    # 进行交叉
                    temp1 = solution1[i]
                    solution2[i] = solution1[i]
                    solution1[i] = temp1
        return solution1,solution2

    # 变异操作
    def mutateOperator(self, individual):
        # 每一个个体都有自己的变异率
        # 变异率
        p = 1 / len(self.dataFeature)
        for index in range(0, individual.featureOfSelect.shape[0]):
            if np.random.rand() < p:
                individual.featureOfSelect[index] = self.reserveAboutMutate(individual.featureOfSelect[index])
        # 跟新个体的solution索引
        individual.getNumberOfSolution()
        # 计算两个个体 fitness
        featureX = self.dataX[:, individual.featureOfSelect == 1]
        individual.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=featureX, findData_y=self.dataY, CV=10)

    # 交换 0 和 1，用于变异
    def reserveAboutMutate(self, value):
        y = -value + 1
        return y

    # 遗传算法操作
    def geneticOperate(self,parent_i,parent_j):
        start = time.time()

        # 交叉操作
        solution1,solution2 = self.crossoverOperater(solution1=parent_i.featureOfSelect,solution2=parent_j.featureOfSelect)

        # 生成新的个体1
        newIndividual_1 = self.Genetic(moDE=self,mode="iterator")
        newIndividual_1.featureOfSelect = solution1
        # 生成新的个体2
        newIndividual_2 = self.Genetic(moDE=self,mode="iterator")
        newIndividual_2.featureOfSelect = solution2

        # 变异操作--两个新个体变异
        self.mutateOperator(individual=newIndividual_1)
        self.mutateOperator(individual=newIndividual_2)

        # 选择操作
        if newIndividual_1.mineFitness < newIndividual_2.mineFitness:
            # 更新一下个体的状态
            newIndividual_1.getNumberOfSolution()
            return newIndividual_1
        else:
            # 更新一下个体的状态
            newIndividual_2.getNumberOfSolution()
            return newIndividual_2


    # 更新pareto解
    def updatePF(self,individual):
        # 先将这个新个体添加到 EP中
        self.EP = np.append(self.EP,individual)
        # 存放error的数组  --- [0,1]
        errorArray = np.zeros(self.EP.shape[0])
        # 存放len的数组  --- [0,1]
        lenArray = np.zeros(self.EP.shape[0])
        # 用于保存每一个个体的支配解集
        dominateFrontArray = []
        # 保存帕累托前沿,保存非支配数量为0的个体,
        F_i = []
        for i in range(self.EP.shape[0]):
            # 将每一个个体的 error 存储
            errorArray[i] = self.EP[i].mineFitness
            # 将每一个个体的 len 存储
            lenArray[i] = self.EP[i].proportionOfFeature

        ##---------------------------------- ENS  高效非支配排序 ----------------------------------------##
        # 首先对种群按照error进行一个排序，从小到大的一个排序

        # 先得到error进行排序后的索引
        sortErrorIndex = np.argsort(errorArray)
        # 将 error  population len 通过 该索引进行一个重新组合
        errorArray = errorArray[sortErrorIndex]
        parentPopulation = self.EP[sortErrorIndex]
        lenArray = lenArray[sortErrorIndex]
        # 对相同error的个体进行按照len的一个排序
        i = 0
        while i < len(errorArray) - 1:
            j = i + 1
            if errorArray[i] == errorArray[j]:
                while j < len(errorArray) and errorArray[i] == errorArray[j]:
                    j += 1
                if (j - i > 1):
                    newArray1 = copy.deepcopy(lenArray[i:j])

                    newArray3 = copy.deepcopy(parentPopulation[i:j])
                    indexNew = np.argsort(newArray1)
                    newArray1 = newArray1[indexNew]
                    newArray3 = newArray3[indexNew]
                    for m in range(len(newArray1)):
                        lenArray[i] = newArray1[m]
                        parentPopulation[i] = newArray3[m]
                        i += 1
                    i -= 1
            i += 1
            # ==========进行迭代 --- 用的是二分查找Front============#

        # 先让 Front集合添加第一个集合
        dominateFrontArray.append([])
        # 将第一个个体添加到第一个集合
        dominateFrontArray[0].append(parentPopulation[0])

        # 第0个已经被添加到dominateFrontArray中去了，所以要从第1个开始进行比较
        for i in range(1, len(parentPopulation)):
            # 设置head 和 tail
            # head 为上界
            head = 0
            # tail 为下界
            tail = len(dominateFrontArray) - 1

            # 第 i 个 个体的 error
            error_i = errorArray[i]
            # 第 i 个 个体的 len
            len_i = lenArray[i]
            # 获得第i个个体
            individual = parentPopulation[i]

            # k 为中间的前沿
            k = math.floor((head + tail) / 2 + 0.5)

            # 进行比较
            while True:
                # 第 k 个前沿点的最后一个个体
                leastIndividual = dominateFrontArray[k][-1]
                # 找到该个体在population的中位置，才能找到其转化后的error 和len
                leastIndex = np.where(parentPopulation == leastIndividual)[0]

                # 若生成的 leastIndex 并不是一个值，而是一堆个体，那么就挑第一个
                if len(leastIndex) > 1:
                    leastIndex = leastIndex[0]

                # 获得最后一个个体的error
                leastError = errorArray[leastIndex]
                # 获得最后一个个体的len
                leastLen = lenArray[leastIndex]
                # 若满足下条件 则  leastIndividual  支配 第i个个体
                if (leastError < error_i and leastLen < len_i) or (
                        leastError <= error_i and leastLen < len_i) or (
                        leastError < error_i and leastLen <= len_i):
                    # 需要往下找下一个前沿
                    head = k
                    # 如过 head 和 tail 中间没有前沿了，并且tail 并不是最后一个前沿，那就把个体添加在tail所在的前沿中
                    if (tail == head + 1 and tail < len(dominateFrontArray) - 1):
                        dominateFrontArray[tail].append(individual)
                        break
                    if (tail == head + 1 and tail == len(dominateFrontArray) - 1):
                        head = tail
                        k = tail
                    # 如果 tail 为最后一个前沿，则添加一个新的前沿到dominateFrontArray
                    elif (head == len(dominateFrontArray) - 1):
                        dominateFrontArray.append([])
                        dominateFrontArray[-1].append(individual)
                        break
                    # 换到下一个二分点
                    else:
                        k = math.floor((head + tail) / 2 + 0.5)
                else:
                    # 如果没有被支配
                    # 如果只剩下三个前沿，存放在中间区域
                    if k == head + 1:
                        tail = k
                        k = head
                    elif (k == head == tail):
                        dominateFrontArray[tail].append(individual)
                        break
                    elif k == head:
                        dominateFrontArray[head].append(individual)
                        break
                    else:
                        tail = k
                        k = math.floor((head + tail) / 2 + 0.5)
        # 将前沿保留下来
        self.EP = copy.deepcopy(dominateFrontArray[0])

    # 得到下一代种群
    def getNextPopulation(self):

        for i in range(self.populationNum):
            # 提取第一个权重向量的 前T个权重向量
            closestIndex = self.closestWeight[i]
            # 提取两个个体
            xArray = sample(closestIndex.tolist(), 2)
            # 个体k
            individual_k = self.population_EA[xArray[0]]
            # 个体l
            individual_l = self.population_EA[xArray[1]]
            # 遗传算法的 -- 交叉变异选择  --- 得到子代
            new_y = self.geneticOperate(parent_i=individual_k,parent_j=individual_l)
            # 更新Z
            self.updataIdealPoint(indivudal=new_y)
            # 更新当前解  --- 首先计算PBI
            # 原始解的PBI
            F_1 = np.asarray([self.population_EA[i].mineFitness,self.population_EA[i].proportionOfFeature])
            PBI_1 = self.calculatePBI(F_i=F_1,weight_i=self.weightArray[i],theta_i=self.thetaArray[i])
            # 新产生解的PBI
            F_2 = np.asarray([new_y.mineFitness,new_y.proportionOfFeature])
            PBI_2 = self.calculatePBI(F_i=F_2,weight_i=self.weightArray[i],theta_i=self.thetaArray[i])

            # 更新个体i
            if PBI_1 >= PBI_2:
                self.population_EA[i] = new_y
            # 更新pareto内存
            self.updatePF(individual=new_y)


    # 画图
    def pltForWin(self,iteNum,pop):
        plt.ion()
        length_LA = pop.shape[0]
        # 存放 error
        x = np.ones(length_LA)
        # 存放 len
        y = np.ones(length_LA)
        for i in range(length_LA):
            x[i] = pop[i].proportionOfFeature
            y[i] = pop[i].mineFitness

        plt.xlabel("len")
        plt.ylabel("error")
        plt.title("MOEA/D"+" "+str(iteNum))
        plt.plot(x,y,color='r',marker='o',linestyle='None')
        plt.pause(0.1)
        if iteNum != self.iteratorTime:
            plt.cla()

    # 运行
    def run(self):

        start = time.time()
        # 初始化函数
        self.initPopulation()
        # 进行运行
        runTime = 0
        # 产生权重向量
        self.generateWeightVector()
        # 初始化理想点
        self.setIdealPoint()
        # 初始化惩罚项  --- 全设置为1
        self.thetaArray = np.ones(self.populationNum)
        
        # 给每一个权重向量分配点
        self.updateArchive(pop=self.population_EA)

        while runTime < self.iteratorTime:
            print("第",runTime + 1,"代")
            self.getNextPopulation()
            # print("======= 更新PF =======")
            # print(f"all time = {time.time() - start} seconds")
            runTime += 1
            self.pltForWin(iteNum=runTime,pop=self.population_EA)

        print(" PF 解")
        for i in range(len(self.EP)):
            print(1 - self.EP[i].mineFitness," ",self.EP[i].numberOfSolution)

        # self.pltForWin(iteNum=runTime, pop=np.asarray(self.EP))

        print("================================================")
        print(f"all time = {time.time() - start} seconds")



if __name__ == '__main__':

    ducName = "BreastCancer1"
    path = "D:\MachineLearningBackUp\dataCSV\dataCSV_high\\" + ducName + ".csv"
    dataCsv = ReadCSV(path=path)
    print("获取文件数据")
    dataCsv.getData()
    moea_d = MOEA_D(dataX=dataCsv.dataX, dataY=dataCsv.dataY)
    moea_d.setParameter(populationNum=100,iteratorTime=100)
    moea_d.run()

