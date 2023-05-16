# MOBGA_AOS
import copy
import math
import random

import numpy as np
from random import sample
from dataProcessing.ReadDataCSV_new import ReadCSV
from dataProcessing.StringDataToNumerical import reserveNumerical
from dataProcessing.NonDominate import nonDominatedSort
from sklearn.preprocessing import MinMaxScaler
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from Filter.filter import FilterOfData


class Genetic:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 特征的存放数组
    dataFeature = np.asarray([])
    # 遗传算法 的种群
    population_GA = np.asarray([])
    # 种群数量
    populationNum = 0
    # 适应度评估次数
    maxFEs = 0
    # 奖励矩阵
    RD = np.asarray([])
    # 惩罚矩阵
    PN = np.asarray([])
    # 交叉率
    crossoverPro = 0
    # 变异率
    mutatePro = 0
    # 每一个特征的relifF评分
    scoreOfRelifF = np.asarray([])
    # 保存数据名字
    dataName = ""
    # 每过LP轮后进行重新分配每一个交叉算子的概率
    LP = 0
    # 代表了存在Q个交叉算子
    Q = 5
    # 每一个交叉算子被选择的概率数组
    cOperaterSelected = np.asarray([])
    # 全局最优解
    globalSolution = np.asarray([])
    # 全局最优解的实数制
    globalSolutionNum = np.asarray([])
    # 全局最优解的适应度值
    globalFitness = 1
    # 全局最优的score
    globalScore = 0

    # 初始化
    def __init__(self,dataX,dataY,dataName):
        self.dataX = dataX
        self.dataY = dataY
        self.dataName = dataName
        self.dataFeature = np.arange(self.dataX.shape[1])
        # 将class里的数据转为数值类型的
        # self.dataY = reserveNumerical(self.dataY)
        # 对数据01 归一化，将每一列的数据缩放到 [0,1]之间
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        print("进行filter，提取一部分数据")
        filter = FilterOfData(dataX=self.dataX,dataY=self.dataY)
        # 计算relifF值
        self.scoreOfRelifF = filter.computerRelifFScore()
        # 进行filter过滤数据 ----  默认采用的是膝节点法
        self.scoreOfRelifF,self.dataX,self.dataFeature = filter.filter_relifF(
            scoreOfRelifF=self.scoreOfRelifF,dataX=self.dataX,dataFeature=self.dataFeature)


    # 设置参数
    def setParameter(self, populationNum, maxFEs, crossoverPro,LP,Q ):
        self.populationNum = populationNum
        self.maxFEs = maxFEs
        self.crossoverPro = crossoverPro
        self.LP = LP
        self.Q = Q



    # 染色体
    class Chromosome:
        # 每一个个体所选择出来的特征组合   --- 二进制
        featureOfSelect = np.asarray([])
        # 特征组合的长度
        numberOfSolution = 0
        # 特征组合的索引
        indexOfSolution = np.asarray([])
        # 所选特征长度占总长度的比例
        proportionOfFeature = 0
        # 当前的选择的fitness
        mineFitness = 0
        # 交换率
        crossover = 0
        # 变异率
        mutation = 0
        # 支配等级
        p_rank = 0
        # 非支配数量
        numberOfNondominate = 0
        # 支配个体集合
        dominateSet = []

        def __init__(self, moea , mode):
            if mode == "init":
                self.initOfChromosome(MFEA=moea)
            elif mode == "iterator":
                self.iteratorOfChromosome(MFEA=moea)
        def iteratorOfChromosome(self,MFEA):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()

        def initOfChromosome(self, MFEA):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            # 首先随机初始化一个特征长度的随机数组
            rand = np.random.random(len(MFEA.dataFeature))
            for i in range(0, len(MFEA.dataFeature)):
                # 获取每一个特征上的随机数，通过这个随机数来判断初始化是否选择该特征
                if rand[i] >= 0.5:
                    self.featureOfSelect[i] = 1
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()
            # 计算acc
            feature_x = MFEA.dataX[:, self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=MFEA.dataY, CV=10)

        # 得到选择特征数量是多少个
        def getNumberOfSolution(self):
            # 得到选择的组合里面 为1 的索引为多少个
            index = np.where(self.featureOfSelect == 1)[0]
            # 存放每一个被选择的索引索引
            self.indexOfSolution = np.copy(index)
            # 得到的索引的长度是多少
            self.numberOfSolution = len(index)
            # 获得比例
            self.proportionOfFeature = self.numberOfSolution / len(self.featureOfSelect)

    # 初始化种群
    def initPopulation(self):
        # 种群
        for i in range(0, int(self.populationNum)):
            chromosome = self.Chromosome(moea=self,mode="init")
            self.population_GA = np.append(self.population_GA, chromosome)

    # 初始化 奖励矩阵 和 惩罚矩阵
    def initRPMatrice(self):
        # 初始化奖励矩阵
        self.RD = np.full((self.LP,self.Q),0)
        # 初始化惩罚矩阵
        self.PN = np.full((self.LP,self.Q),0)
        # 初始化矩阵  --- 初始每一个算子被选择的概率相同
        self.cOperaterSelected = np.full(self.Q,1/self.Q)

    # 交叉算子1 --- 单点交叉
    def crossoverOperater_1(self,solution1,solution2):
        # 随机选择一个点作为交叉的点
        singlePoint = np.random.randint(self.dataFeature.shape[0])
        # 进行右侧的交叉
        while singlePoint < self.dataFeature.shape[0]:
            # 交换两个位置的值
            temp = solution1[singlePoint]
            solution2[singlePoint] = solution1[singlePoint]
            solution1[singlePoint] = temp
            singlePoint += 1

    # 交叉算子2 --- 两点交叉
    def crossoverOperater_2(self,solution1,solution2):
        # 获得两个点
        point = np.asarray(sample(np.arange(self.dataFeature.shape[0]).tolist(),2))
        # 得到两个值中较小的值
        head = point.min()
        # 得到两个值中较大的值
        tail = point.max()
        while head <= tail:
            # 中间变量
            temp1 = solution1[head]
            solution2[head] = solution1[head]
            solution1[head] = temp1
            head += 1

    # 交叉算子3  --- 均匀交叉
    def crossoverOperater_3(self, solution1, solution2):
        # 生成一个矩阵，若是存在rand值小于该矩阵对应位置的值，那就对应的位置就要进行交叉操作
        crossProArray = np.random.random(self.dataFeature.shape[0])
        # 交叉
        for i in range(self.dataFeature.shape[0]):
            # 若是该位置两个solution所存放的值是不一样的就可以进行交叉
            if solution1[i] != solution2[i]:
                # 生成的随机数，用于判断该位置是否需要进行交叉操作
                rand = np.random.rand()
                # 若是小于交叉矩阵里的概率值
                if rand > crossProArray[i] :
                    # 进行交叉
                    temp1 = solution1[i]
                    solution2[i] = solution1[i]
                    solution1[i] = temp1


    # 交叉算子4 --- 洗牌交叉算子
    def crossoverOperater_4(self, solution1, solution2):
        # 先洗牌  -- 按照该随机索引进行洗牌
        shuffleIndex = np.random.permutation(solution1.shape[0])
        # 对 solution1 和 solution2 进行洗牌 -- 存在新的解中
        new1 = copy.deepcopy(solution1[shuffleIndex])
        new2 = copy.deepcopy(solution2[shuffleIndex])
        # 进行单点交叉
        self.crossoverOperater_1(solution1=new1,solution2=new2)
        #  2 4 1 3 --- 3 1 4 2
        # 1 2 3 4 ---
        # 逆洗牌
        for i in range(shuffleIndex.shape[0]):
            solution1[shuffleIndex[i]] = new1[i]
            solution2[shuffleIndex[i]] = new2[i]

    # 交叉算子5  --- 约简代理交叉算子
    def crossoverOperater_5(self, solution1, solution2):
        # 随机选择一个点作为交叉的点
        singlePoint = np.random.randint(self.dataFeature.shape[0])
        # 交叉
        for i in range(singlePoint,self.dataFeature.shape[0]):
            # 若是该位置两个solution所存放的值是不一样的就可以进行交叉
            if solution1[i] != solution2[i]:
                # 进行交叉
                temp1 = solution1[i]
                solution2[i] = solution1[i]
                solution1[i] = temp1


    # 选择是哪一个交叉算子
    def crossOperaterSelect(self,individual_i,individual_j):
        # 子代1
        spring_1 = copy.deepcopy(individual_i.featureOfSelect)
        # 子代2
        spring_2 = copy.deepcopy(individual_j.featureOfSelect)
        # 得到选择算子的索引
        operater_index = self.rouletteWheelSelection()
        # 交叉算子1
        if operater_index == 0:
            self.crossoverOperater_1(solution1=spring_1, solution2=spring_2)
        # 交叉算子2
        elif operater_index == 1:
            self.crossoverOperater_2(solution1=spring_1, solution2=spring_2)
        # 交叉算子3
        elif operater_index == 2:
            self.crossoverOperater_3(solution1=spring_1, solution2=spring_2)
        # 交叉算子4
        elif operater_index == 3:
            self.crossoverOperater_4(solution1=spring_1, solution2=spring_2)
        # 交叉算子5
        elif operater_index == 4:
            self.crossoverOperater_5(solution1=spring_1, solution2=spring_2)
        # 生成个体1
        s1 = self.Chromosome(moea=self,mode="iterator")
        s1.featureOfSelect = spring_1
        s1.getNumberOfSolution()
        # 生成个体2
        s2 = self.Chromosome(moea=self,mode="iterator")
        s2.featureOfSelect = spring_2
        s2.getNumberOfSolution()
        return s1,s2,operater_index

    # 轮盘赌
    def rouletteWheelSelection(self):
        # 首先得到一个随机数 [0,1]
        rand = np.random.rand()
        # 用于保存值的
        sum = 0
        # 其次进行轮盘赌
        for i in range(self.Q):
            sum += self.cOperaterSelected[i]
            if sum >= rand:
                return i

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
        tmp_u_len = len(np.where(individual.featureOfSelect == 1)[0])
        # 判断1的数量
        if tmp_u_len > 0:
            # 计算fit
            individual.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=featureX,
                                                                   findData_y=self.dataY, CV=10)
        else:
            individual.mineFitness = 1
        # individual.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=featureX,findData_y=self.dataY,CV=10)


    # 交换 0 和 1，用于变异
    def reserveAboutMutate(self, value):
        y = -value + 1
        return y

    # 遗传算法的主体 --- 变异交叉 然后添加值到相对应的惩罚和奖励矩阵
    def stage_BGA_1(self,population,nFE):
        # 子代种群
        springPopulation = np.asarray([])
        # 先随机打乱，然后每次抽取两个个体
        mateIndex = np.random.permutation(self.populationNum)
        # 本轮的奖励
        nReward = np.zeros(self.Q)
        # 本轮的惩罚
        nPenalty = np.zeros(self.Q)
        a = mateIndex[0]
        i = 0
        while i < self.populationNum:
            individual_1 = population[mateIndex[i]]
            i += 1
            individual_2 = population[mateIndex[i]]
            i += 1
            # 进行交叉算子
            spring1,spring2,operater_index = self.crossOperaterSelect(individual_i=individual_1,individual_j=individual_2)
            # 变异
            self.mutateOperator(individual=spring1)
            self.mutateOperator(individual=spring2)
            # 计算次数加2
            nFE += 2
            # 进行判断两个个体之间的优劣性
            paraent = np.asarray([individual_1,individual_2])
            spring = np.asarray([spring1,spring2])
            nReward,nPenalty = self.creditAssignment(parent=paraent,spring=spring,p=operater_index
                                                     ,nReward=nReward,nPenalty=nPenalty)
            # 添加到子代种群中去
            springPopulation = np.append(springPopulation,spring1)
            springPopulation = np.append(springPopulation,spring2)

        return springPopulation ,nFE ,nReward,nPenalty

    # reward penalty 矩阵分配
    """
        p  ：交叉算子
        parent : 父代种群
        spring : 子代种群
        nReward : 奖励数组
        nPenalty : 惩罚数组
    """
    def creditAssignment(self,parent,spring,p, nReward,nPenalty):
        # 获得两个父类个体之间的支配关系
        p_nd, p_d ,index = self.dominanceComparison(population=parent)
        # 判断情况 --- 支配不为空 相当于两个父类存在支配关系
        if p_d.shape[0] != 0:
            for i in range(2):
                tempPop = np.asarray([p_nd[0],spring[i]])
                ps_nd,ps_d,index_ps = self.dominanceComparison(population=tempPop)
                # 子代和父代存在支配关系
                if ps_d.shape[0] != 0:
                    # 如果等于0 ， 就相当于是第一个个体是父代个体，即父代个体支配了子代个体
                    if index_ps[0] == 0:
                        nPenalty[p] += 1
                    else:
                    # 如果等于1 ， 就相当于是第一个个体是子代个体，即子代个体支配了父代个体
                        nReward[p] += 1
        # 如果两个父代不存在支配关系
        else:
            for i in range(2):
                # 每一个子代不被父代支配，则相对应位置上为1
                linkParentAndSpring = np.zeros(2)
                for j in range(2):
                    tempPop = np.asarray([parent[j], spring[i]])
                    ps_nd, ps_d, index_ps = self.dominanceComparison(population=tempPop)
                    # 子代和父代存在支配关系
                    if ps_d.shape[0] != 0:
                        # 如果等于0 ， 就相当于是第一个个体是父代个体，即父代个体支配了子代个体
                        if index_ps[0] == 0:
                            nPenalty[p] += 1
                            break
                        else:
                        # 如果等于1 ， 就相当于是第一个个体是子代个体，即子代个体支配了父代个体
                            linkParentAndSpring[j] = 1
                    # 父代和子代不存在支配关系
                    else:
                        linkParentAndSpring[j] = 1
                # 即如果为2 ，说明该子代 与两个父代都存在非支配或者支配父代的情况
                if linkParentAndSpring.sum() == 2:
                    nReward[p] += 1

        return nReward , nPenalty



    # 比较两个个体之间的支配关系
    def dominanceComparison(self,population):
        # 存放非支配个体
        p_nd = np.asarray([])
        # 存放支配个体
        p_d = np.asarray([])
        # 用于存放两个父样本的fitness
        pError = np.asarray([population[0].mineFitness,population[1].mineFitness])
        # 用于存放两个父样本的len
        pLen = np.asarray([population[0].numberOfSolution,population[1].numberOfSolution])/self.dataFeature.shape[0]

        #首先先排序  ---  升序排序
        index_error = np.argsort(pError)
        index_len = np.argsort(pLen)
        if index_len[0] == index_error[0]:
            # 所以 两个样本之间 ， 第一个样本必定存在与 非支配数组内
            p_nd = np.append(p_nd, population[index_error[0]])
            # 第二个个体是 --- 支配数组
            p_d = np.append(p_d,population[index_error[1]])
        else:
            # 那就是两个都是非支配的,把整个population添加进去
            p_nd = np.append(p_nd, population)

        return p_nd,p_d,index_error


    # 跟新 奖励和惩罚矩阵
    def updataRPMatrice(self):
        # 存放当前 reward矩阵中 每一列就相当于每一个交叉算子中由多少成功的样本数量
        s_q1 = np.zeros(self.Q)
        # 存放当前 penal矩阵中 每一列就相当于每一个交叉算子中由多少失败的样本数量
        s_q2 = np.zeros(self.Q)
        for i in range(self.Q):
            s_q1[i] = self.RD[:,i].sum()
            s_q2[i] = self.PN[:,i].sum()

        if s_q1.sum() == 0 :
            s_q3 = np.ones(self.Q) * 0.0001
        else:
            s_q3 = s_q1

        # 获得 每一个算子中 reward占比多少
        s_q4 = s_q1/(s_q3 + s_q2 + 0.001)
        # 用于存放跟新后的每一个交叉算子概率
        tempCOSelect = s_q4 / s_q4.sum()
        self.cOperaterSelected = tempCOSelect

    # 总体运行
    def stage_all(self):
        # 评估次数
        nFE = 0
        # 迭代次数
        runTime = 0
        # 代表了 每5轮进行一次更新
        p = 0
        while nFE < self.maxFEs:
            print("stage:",runTime + 1,"    nFE:",nFE)
            # 得到子代种群，评估次数，奖励数组，惩罚数组
            springPopulation, nFE, nReward, nPenalty = self.stage_BGA_1(population=self.population_GA, nFE=nFE)
            # 跟新到全局的奖励、惩罚矩阵中
            self.RD[p] = nReward
            self.PN[p] = nPenalty
            runTime += 1
            p += 1
            # 判断
            if p % self.LP == 0:
                p = 0
                # 跟新矩阵
                self.updataRPMatrice()

            # 将子代和父代进行合并
            population = np.concatenate([self.population_GA, springPopulation])
            errorArray = np.asarray([])
            lenArray = np.asarray([])

            # 先获得排在前面的父代个体的 err 和 len
            for i in range(self.populationNum):
                errorArray = np.append(errorArray, population[i].mineFitness)
                lenArray = np.append(lenArray, population[i].numberOfSolution)
            # 通过err数组进行排序，升序排序
            errIndex = np.argsort(errorArray)
            # 将输出前三个
            for i in range(3):
                print("acc = ", 1 - self.population_GA[errIndex[i]].mineFitness,
                      "len = ", self.population_GA[errIndex[i]].numberOfSolution)
            print("======================================================")

            # 将spring个体添加进去
            for i in range(self.populationNum,population.shape[0]):
                errorArray = np.append(errorArray, population[i].mineFitness)
                lenArray = np.append(lenArray, population[i].numberOfSolution)
            # 将子代和父代进行非支配排序
            self.population_GA, score, bestIndividual = nonDominatedSort(parentPopulation=population,
                                                                         fitnessArray=errorArray,
                                                                         solutionlengthArray=lenArray
                                                                         , dataFeature=self.dataFeature,
                                                                         populationNum=self.populationNum)
            # 跟新全局最优个体
            if self.globalScore <= score:
                self.globalScore = score
                self.globalFitness = bestIndividual.mineFitness
                self.globalSolution = bestIndividual.featureOfSelect

    # 运行
    def run(self):
        # 初始化种群
        self.initPopulation()
        # 初始化奖励惩罚矩阵
        self.initRPMatrice()
        # 初始化矩阵
        # 运行次数
        import time
        start = time.time()
        self.stage_all()

        print(f"all time = {time.time() - start} seconds")
        print(1 - self.globalFitness)
        print(len(np.where(self.globalSolution == 1)[0]))

        print("####################################")
        print("populationNum = ", self.populationNum)
        print("maxFEs = ", self.maxFEs)
        print("crossoverPro = ", self.crossoverPro)
        print("LP = ", self.LP)
        print("Q = ", self.Q)
        print(" ")

if __name__ == '__main__':
    import time
    start = time.time()
    ducName = "BreastCancer1"
    path = "D:\MachineLearningBackUp\dataCSV\dataCSV_high\\" + ducName + ".csv"
    dataCsv = ReadCSV(path=path)
    print("获取文件数据")
    dataCsv.getData()
    genetic = Genetic(dataX=dataCsv.dataX, dataY=dataCsv.dataY, dataName=ducName)
    genetic.setParameter(populationNum=140, maxFEs=14000, crossoverPro=0.9,LP = 5,Q=5)
    genetic.run()