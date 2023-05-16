import copy
import math
import os

import numpy as np
import random

from dataProcessing.KneePointDivideData import findKneePoint
from dataProcessing.reliefF import reliefFScore
from dataProcessing.DataCSV import DataCSV
from dataProcessing.ReadDataCSV_new import ReadCSV
from dataProcessing.InforGain import InforGain
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from Classfier.SupportVectorMachines import fitnessFunction_SVM_CV, fitnessFunction_SVM_percent
from Classfier.DicisionTree import fitnessFunction_Tree_percent, fitnessFunction_Tree_CV
from write import readSimilar
from SoftmaxScale import softmax
from sklearn.preprocessing import  MinMaxScaler
from dataProcessing.GetParetoSolution import getParato

random.seed(1)

from Classfier.invokingClassfier import computeFitnessKNN,terminalComputeFitness
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
# 遗传算法  ---->
# Guiding and Repairing based Multi-Objective Evolutionary Algorithm
class GRmoea:

    # 保存特征表 备份
    oriDataFeature = np.asarray([])
    #   保存特征表
    dataFeature = np.asarray([])

    # 保存了数据中除了class意外的其他数据  多用于计算reliefF
    dataX = np.asarray([])
    # 保存类信息
    dataY = np.asarray([])

    dataX_t = np.asarray([])
    dataY_t = np.asarray([])

    # 特征和类的SU  su矩阵
    suArray = np.asarray([])
    # 用于备份
    initsuArray = np.asarray([])
    # 互信息
    mutualInforArray = np.asarray([])
    # 备份
    initmutalInforArray = np.asarray([])
    # 元素组的特征相似性
    originSimilarityOfFeature = np.asarray([])
    # 特征相似度
    similarityOfFeature = np.asarray([])
    # 每一个特征被选中为1 的概率
    probabilityOfFeatureToSelect = np.asarray([])
    # 每一个特征的权重
    weightOfFeatuer = np.asarray([])
    # 每个特征的relifF得分
    scoreOfRelifF = np.asarray([])

    # wrapper 上一轮的solution矩阵
    wrapperSolutionPrevious = []
    # wrapper 上一轮的solution的 Acc和Res矩阵
    accResMatrixPrevious = []
    # wrapper 当前轮的solution 矩阵
    wrapperSolutionRecent = []
    # wrapper 当前轮的solution的 Acc 和 Res 矩阵
    accResMatrixRencent = []
    # 特征的是否修复，每一个特征在本次实验中只能被修复一次
    repairFeature = []
    # filter 当前轮的solution 矩阵
    filterSolutionRecent = []
    # 进行Guiding and Repairing 时间
    grTime = 0

    # featureNumNormalize 特征数量总体的归一化    现在不用
    featureNumNormalize = np.asarray([])


    # 全局最优选择
    globalBestSolution = np.asarray([])
    # 全局最优fitness
    globalBestFitness = 0
    # 全局最优长度
    globalBestLength = 0


    # 染色体种群
    chromosomeList = []
    chromosomeFilterList = []
    chromosomeWrapperList = []
    # 个体数量
    numberOfChromosome = 0
    # 精英代
    elitistChromosome = []
    # 迭代次数
    iteratorTime = 0


    # 交换率
    probabilityOfCrossover = 0
    # 变异率
    probabilityOfMutation = 0

    # 启发式信息
    heuristicArray = []
    # 不平衡率
    imbalance = 0


    # 保存训练集
    saveTrainSet = []
    saveTestSet = []

    dataName = ""

    def __init__(self, dataCsv,  dataX, dataY,dataName):

        self.dataFeature = np.arange(dataX.shape[1])
        self.dataName = dataName
        self.dataX = dataX
        self.dataY = dataY
        # 更新  data_y
        self.reserveNumerical()
        # 判断是否平衡 -- 只可以用于二分类
        #self.imbalance = self.isBalance(self.dataY)

        self.oriDataFeature = np.copy(self.dataFeature)
        # test
        # 得到su矩阵

        # 对数据01 归一化，将每一列的数据缩放到 [0,1]之间
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        # 拐点分割特征，选择拐点前的数据

        # 通获取前0.03的数据
        self.filter()

        # 初始化每一个特征的权重  , 放在filter之后，因为filter会删除数据
        self.weightOfFeatuer = np.ones(len(self.dataFeature))
        self.repairFeature = np.ones(len(self.dataFeature))

        # GR-MOEA 不需要线性归一化这个，直接求未选择的比例为多少
        #self.normalizeOfFeature()

        print("")

    # 设置参数
    def setParameter(self, crossover, mutation, numberOfChromosome, iteratorTime,gtTime):
        self.probabilityOfCrossover = crossover
        self.probabilityOfMutation = mutation
        self.numberOfChromosome = numberOfChromosome
        self.iteratorTime = iteratorTime
        self.grTime = gtTime
        self.chromosomeWrapperList = []
        self.chromosomeFilterList = []

    # 初始化染色体种群
    def initChromosomeList(self):
        # filter 种群
        for i in range(0, int(self.numberOfChromosome)):
            chromosome = self.Chromosome(GRmoea=self,mode='filter')
            self.chromosomeFilterList.append(chromosome)
        # wrapper 种群
        for i in range(0, int(self.numberOfChromosome)):
            chromosome = self.Chromosome(GRmoea=self,mode='wrapper')
            self.chromosomeWrapperList.append(chromosome)
    # 判断 数据是否平衡
    def isBalance(self, data_y):
        x = np.where(data_y == 1)[0]
        y = np.where(data_y == 0)[0]
        numX = len(x)
        numY = len(y)
        imbalance = numX / numY
        return imbalance

    # 计算每一个特征被选择的概率
    def filterOfProFreture(self):
        print("通过SU和similar对每一个特征求其概率")
        Q = 20
        mutInfor = self.similarityOfFeature * Q + 0.00000001
        self.probabilityOfFeatureToSelect = self.mutualInforArray / mutInfor


    # 将class 的值转为数值类型
    def reserveNumerical(self):
        # 判断数据是否是str
        y = self.dataY
        if isinstance(y[0], str):
            valueOfClass = y
            tempy = np.asarray([])
            m = set(valueOfClass[i] for i in range(valueOfClass.shape[0]))
            m = list(m)
            m = np.asarray(m)
            y_value = np.arange(0, len(m))
            for i in range(0, len(valueOfClass)):
                index = np.where(m == valueOfClass[i])[0][0]
                tempy = np.append(tempy, y_value[index])
            y = tempy
        self.dataY = y

    # ======================================================================================
    # mutial information

    # 通过SU来进行筛选数据
    # fiter 方法  获取前 60% 的数据  ----------用于高维数据
    def filter(self):
        # 得到relifF
        print("得到所有特征的relifF")
        self.scoreOfRelifF = reliefFScore(self.dataX, self.dataY)

        # 得到su矩阵
        print("得到互信息矩阵")
        infor = InforGain(dataAttribute=self.dataFeature, dataX=self.dataX)
        # self.suArray = infor.getFeatureClassSU(dataY=self.dataY)  # 用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵
        self.mutualInforArray = infor.getMutualInformation_fc(dataX=self.dataX,dataY=self.dataY)  # 用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵

        # 得到相似度矩阵
        print("得到相似度矩阵")
        # 用于低维数据
        # similarityOfFeature = featureOfSimilar(dataX = self.dataX)
        # 用于高维数据
        self.featureOfSimilar_high()
        if len(self.dataFeature) > 1000:
            # extratRatio = findKneePoint(data=self.scoreOfRelifF)
            extratRatio = 0.03
        else:
            extratRatio = 1
        # 通过prob进行排序
        print("通过su进行筛数据")
        self.scoreOfRelifF, self.dataFeature, self.similarityOfFeature, self.mutualInforArray, self.dataX \
            = infor.getDataOfFilterIn_MFEA(
            scoreOfRelifF=self.scoreOfRelifF,
            extratRatio=extratRatio, dataAttribute=self.dataFeature,
            similarityOfFeature=self.similarityOfFeature, mutualInforArray=self.mutualInforArray,
            dataX=self.dataX)

        # 计算每个特征被选择的概率，通过 mutialInformation 和 similar来进行计算
        print("计算概率")
        self.filterOfProFreture()

        # 对 relifF进行归一化，放缩到[0.2,0.7]之间
        self.scoreOfRelifF = ((self.scoreOfRelifF - np.min(self.scoreOfRelifF)) / (
            np.max(self.scoreOfRelifF - np.min(self.scoreOfRelifF))) + 0.4) * 0.5
        # 对 pro进行归一化，放缩到[0.2,0.7]之间
        self.probabilityOfFeatureToSelect = ((self.probabilityOfFeatureToSelect - np.min(
            self.probabilityOfFeatureToSelect)) / (np.max(self.probabilityOfFeatureToSelect) - np.min(
            self.probabilityOfFeatureToSelect)) + 0.4) * 0.5

        print("")

    ###############################################################################################


    # 计算相似度
    def featureOfSimilar(self):
        # 用于保存每一个特征的相似度
        allFeatureSimilar = [[] for i in range(0, len(self.dataFeature))]
        # 用于保存每一个特征的power
        allFeaturePower = []
        import time
        start = time.time()
        # 获得每一个样本的平方开根号
        for i in range(0, len(self.dataFeature)):
            # print(i)
            p_i = np.asarray(self.dataX[:,i])
            # 先平方
            pow_i = np.power(p_i, 2)
            # 平方和
            sum_i = pow_i.sum()
            # 开根号
            sum_i = np.power(sum_i, 0.5)
            # 将该平方值保存
            allFeaturePower.append(sum_i)
        for i in range(0, len(self.dataFeature)):  # 是因为 datafeature包含了class标签
            # 用于保存特征i与其他特征的相似性
            cor_i = 0
            # 得到第i个特征所代表的数据 和 第j个特征所代表的数据
            c_i = np.asarray(self.dataX[:,i])
            # 获得第i个特征的值
            sum_pow_i = allFeaturePower[i]
            for j in range(i, len(self.dataFeature)):
                if i == j:
                    allFeatureSimilar[i].append(0)
                if i != j:
                    c_j = np.asarray(self.dataX[:,j])
                    # 分别获得第i个和第j个特征的数据
                    sum_ij = np.asarray(c_i * c_j)
                    # 计算得到分子,绝对值
                    sum_ij = np.abs(sum_ij.sum())
                    sum_pow_j = allFeaturePower[j]

                    # 最终得到这个特征i与其他所有特征之间的相关性
                    c_ij = sum_ij / (sum_pow_i * sum_pow_j + 0.00000001)

                    allFeatureSimilar[i].append(c_ij)
                    allFeatureSimilar[j].append(c_ij)

            # 计算平均相似度值   -2 的原因是： dataFeature最后一列是class 然后自己和自己比设置为0，所以要减去两个
            cor_i = np.sum(allFeatureSimilar[i], axis=0) / (len(self.dataFeature) - 1)
            # cor_i = 1 - cor_i
            # allFeatureSimilar = readSimilar(path='D:\MachineLearningBackUp\dataCSV\similarData\BreastSimilar.txt')
            self.originSimilarityOfFeature = np.append(self.originSimilarityOfFeature, cor_i)
            self.heuristicArray = np.append(self.heuristicArray, cor_i)
        print(f"similar time = {time.time() - start} seconds")
        # 不应该排序
        self.similarityOfFeature = np.asarray(self.originSimilarityOfFeature)

    #  # 用于高维数据  --- 提前处理好similar 保存在txt中
    def featureOfSimilar_high(self):
        allFeatureSimilar = readSimilar(path="D:\\MachineLearningBackUp\\dataCSV\\dataCSV_similar\\"+self.dataName+"Similar.txt")
        self.originSimilarityOfFeature = allFeatureSimilar
        self.heuristicArray = allFeatureSimilar
        # 不应该排序
        self.similarityOfFeature = np.asarray(self.originSimilarityOfFeature)
# ============================================================================================
    # RelifF

    # 通过RelifF方法进行筛选数据
    def getDataOfFilterInRelifF(self):
        # 判断数据是否是str
        y = dataCsv.dataAllColum[-1]
        if isinstance(y[0], str):
            valueOfClass = y
            tempy = np.asarray([])
            m = set(valueOfClass[i] for i in range(valueOfClass.shape[0]))
            m = list(m)
            m = np.asarray(m)
            y_value = np.arange(0, len(m))
            for i in range(0, len(valueOfClass)):
                index = np.where(m == valueOfClass[i])[0][0]
                tempy = np.append(tempy, y_value[index])
            y = np.copy(tempy)
        # 获得每个特征的relifF得分
        self.scoreOfRelifF = reliefFScore(self.dataX, y)
        # 获得需要被删除的特征索引，获得评分大于0的
        print("获得评分大于0的index")

        indexOfDelect = np.asarray([])

        # 从大到小排序
        sortIndexOfScore = np.asarray(sorted(self.scoreOfRelifF, reverse=True))
        tempSortIndex = np.asarray([])
        for i in range(0, int(len(sortIndexOfScore) * 0.03)):
            index = np.where(self.scoreOfRelifF == sortIndexOfScore[i])[0][0]
            tempSortIndex = np.append(tempSortIndex, index)
        indexOfDelect = tempSortIndex

        # 删除relifF评分小于0 的特征   ---- 删除了 dadaCloum 和 dadaFeature
        print("提取feature 和 cloum")
        self.removeData(indexOfDelect=indexOfDelect)

        # 删除得分小于0 的那些RelifF得分
        print("得到评分大于0的relifF")
        valueOfRelifF = list(self.scoreOfRelifF)
        self.remove(valueOfRelifF, indexOfDelect)
        # self.scoreOfRelifF = np.asarray(valueOfRelifF)
        print("")

    # data 是一个list类型的数据，index 是需要删除的索引
    def remove(self, data, index):
        score = np.asarray([])
        for i in range(0, len(index)):
            score = np.append(score, data[int(index[i])])
        self.scoreOfRelifF = score

    # 通过索引来删除数据  ，用来删除filter中不需要的数据，只对
    def removeData(self, indexOfDelect):
        # list 查找速度太快了
        colum = np.asarray([])
        feature = np.asarray([])
        for i in range(0, len(indexOfDelect)):
            # 删除数据，删除不需要的数据
            colum = np.append(colum, self.dataX[:,i])
            feature = np.append(feature, self.dataFeature)
        self.dataX = colum
        self.dataFeature = feature

    # ==========================================================================================

    # 数据线性归一化, 只能处理一行数据  1*n   array类型
    def linearNormalization(self, data):
        valueMax = data.max()
        valuemin = data.min()
        absoluteDistance = valueMax - valuemin
        data = data - valuemin
        data = data / (absoluteDistance + 0.000000001)
        return data

    # 染色体
    class Chromosome:
        # 每一个个体所选择出来的特征组合
        featureOfSelect = np.asarray([])
        # 特征组合和长度
        numberOfSolution = 0
        # 特征组合的索引
        indexOfSolution = np.asarray([])
        # 每一个个体选择出来的特征组合的fitness
        # featureOfFitness = 0
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

        def __init__(self, GRmoea , mode):
            self.crossover = GRmoea.probabilityOfCrossover
            self.mutation = GRmoea.probabilityOfMutation
            # self.initOfChromosome_su(GRmoea = GRmoea)
            # self.initOfChromosome_RelifF(GRmoea = GRmoea)
            if mode == 'filter':
                #self.initOfChromosome_Mutial(GRmoea=GRmoea)
                #self.initOfChromosome_su(GRmoea=GRmoea)
                self.initOfChromosome_Pro(GRmoea=GRmoea)
            elif mode == 'wrapper':
                self.initOfChromosome_wrapper(GRmoea=GRmoea)

        # 初始化染色体 通过wrapper方式
        def initOfChromosome_wrapper(self,GRmoea):
            #  高维数据，高维需要filter剔除数据
            self.featureOfSelect = np.zeros(len(GRmoea.dataFeature))
            #得到wrapper种群的总个体
            numChromosome = int(GRmoea.numberOfChromosome)
            # 向上取整
            numberOfGroup = int(GRmoea.dataX.shape[1]/numChromosome)
            # 将特征通过个体进行分组，每一个小组平均得到特定数量的特征
            group = [[] for i in range(0,numChromosome)]
            k = 0
            for i in range(0,len(group)):
                for j in range(0,numberOfGroup):
                    group[i].append(k)
                    k = k + 1
            # 如果无法均匀分配，那就把多余的特征添加到最后一个小组内
            while k < len(GRmoea.dataFeature) :
                group[-1].append(k)
                k = k + 1
            # 初始化
            for i in range(0,len(group)):
                head = group[i][0]
                tail = group[i][-1]
                rand1 = random.randint(head,tail)
                self.featureOfSelect[rand1] = 1
            self.getNumberOfSolution()

        # 初始化染色体 -- mutial
        def initOfChromosome_Mutial(self, GRmoea):
            self.featureOfSelect = np.zeros(len(GRmoea.mutualInforArray))
            for i in range(0, len(GRmoea.mutualInforArray)):
                # 获取每一个特征上的随机数，通过这个随机数来判断初始化是否选择该特征
                rand = np.random.random()
                if rand < GRmoea.mutualInforArray[i]:
                    self.featureOfSelect[i] = 1
                else:
                    self.featureOfSelect[i] = 0
            self.getNumberOfSolution()
        # 初始化染色体 -- su
        def initOfChromosome_su(self, GRmoea):
            self.featureOfSelect = np.zeros(len(GRmoea.suArray))
            for i in range(0, len(GRmoea.suArray)):
                # 获取每一个特征上的随机数，通过这个随机数来判断初始化是否选择该特征
                rand = np.random.random()
                if rand < GRmoea.suArray[i]:
                    self.featureOfSelect[i] = 1
                else:
                    self.featureOfSelect[i] = 0
            self.getNumberOfSolution()

        # 初始化染色体  -- Pro
        def initOfChromosome_Pro(self, GRmoea):
            self.featureOfSelect = np.zeros(len(GRmoea.probabilityOfFeatureToSelect))
            for i in range(0, len(GRmoea.probabilityOfFeatureToSelect)):
                # 获取每一个特征上的随机数，通过这个随机数来判断初始化是否选择该特征
                rand = np.random.random()
                if rand < GRmoea.probabilityOfFeatureToSelect[i]:
                    self.featureOfSelect[i] = 1
                else:
                    self.featureOfSelect[i] = 0
            self.getNumberOfSolution()

        # 初始化染色体 -- relifF
        def initOfChromosome_RelifF(self, GRmoea):
            for i in range(0, len(GRmoea.scoreOfRelifF)):
                # 获取每一个特征上的随机数，通过这个随机数来判断初始化是否选择该特征
                rand = np.random.random()
                if rand < GRmoea.scoreOfRelifF[i]:
                    self.featureOfSelect = np.append(self.featureOfSelect, 1)
                else:
                    self.featureOfSelect = np.append(self.featureOfSelect, 0)
            self.getNumberOfSolution()

        # 得到选择特征数量是多少个
        def getNumberOfSolution(self):
            # 得到选择的组合里面 为1 的索引为多少个
            index = np.where(self.featureOfSelect == 1)[0]
            # 存放每一个被选择的索引索引
            self.indexOfSolution = np.copy(index)
            # 得到的索引的长度是多少
            self.numberOfSolution = len(index)



    # 交叉算子  使用均匀交叉，使得子代拥有极高的多样性,需要传入种群
    def crossoverOperate(self, population,mode):
        # 将个体按照fitness 进行排序
        fitnessOfindividual = np.asarray([])
        sortFitness = np.asarray([])
        for i in range(0, len(population)):
            fitnessOfindividual = np.append(fitnessOfindividual, population[i].mineFitness)
        # 从高到低进行排序
        sortFitness = sorted(fitnessOfindividual, reverse=True)

        # 长度为 len(population) ， 间隔为2   , 80%的个体进行交叉
        for i in range(0, int(len(sortFitness)), 2):

            # 随机获得sortFitness 中的两个索引
            rand = random.sample(range(len(sortFitness)), 2)
            # 得到随机的两个个体
            index1 = np.where(fitnessOfindividual == sortFitness[rand[0]])[0][0]
            index2 = np.where(fitnessOfindividual == sortFitness[rand[1]])[0][0]

            # 获得第i个和第i+1 个父类
            individual_1 = population[index1]
            individual_2 = population[index2]
            # 获得每一个父类的选择特征
            individual_1_solution = individual_1.featureOfSelect
            individual_2_solution = individual_2.featureOfSelect
            # 进行均匀交叉
            for index in range(0, len(individual_1_solution)):
                # 设置一个随机数，在每一个位置上进行随机一个数字，
                # 如果随机数要高于self.probabilityOfCrossover，则进行交换，否则就不进行交换该位置
                temp1 = 0
                temp2 = 0
                rand_cross = np.random.random()
                temp1 = int(individual_1_solution[index])
                temp2 = int(individual_2_solution[index])
                # 如果父代相同位置一样则不进行交叉
                if temp1 != temp2 :
                    if rand_cross > self.probabilityOfCrossover:
                        # 进行交换
                        individual_1_solution[index] = temp2
                        individual_2_solution[index] = temp1

            # # 进行更新个体的特征数量
            individual_1.getNumberOfSolution()
            individual_2.getNumberOfSolution()

        # # fitness 排名最后20%的个体被舍弃，重新随机组成20%个新的个体添加进去
        # for i in range(int(len(sortFitness) * 0.8), len(sortFitness)):
        #     index3 = np.where(fitnessOfindividual == sortFitness[i])[0][0]
        #     individual = self.Chromosome(GRmoea=self,mode=mode)
        #     population[index3] = individual
        return population

    # 精英变异算子
    def mutateOperator_elist(self, parentPopulation):
        # 每一个个体都有自己的变异率
        for i in range(0, len(parentPopulation)):
            individual = copy.deepcopy(parentPopulation[i]) # 指针，对数组里面所指向的对象操作
            # # 个体solution中1 的数量
            # l_1 = len(np.where(individual.featureOfSelect == 1)[0])
            # # 个体solution中1 的数量
            # l_0 = len(np.where(individual.featureOfSelect == 0)[0])
            # # 1 的变异率
            # p_1 = 1/(len(self.dataFeature) + 0.00000001)
            # # 0 的变异率
            # p_0 = l_1/(l_0 + 0.00000001) * p_1
            # 变异率
            p = 1/len(self.dataFeature)
            for index in range(0, len(individual.featureOfSelect)):
                # if individual.featureOfSelect[i] == 0:
                #     rand_mutate_0 = np.random.random()
                #     if rand_mutate_0 < p_0:
                #         individual.featureOfSelect[index] = self.reserveAboutMutate(individual.featureOfSelect[index])
                # elif individual.featureOfSelect[i] == 1:
                #     rand_mutate_1 = np.random.random()
                #     if rand_mutate_1 < p_1:
                #         individual.featureOfSelect[index] = self.reserveAboutMutate(individual.featureOfSelect[index])
                if np.random.rand() < p:
                    individual.featureOfSelect[index] = self.reserveAboutMutate(individual.featureOfSelect[index])
            # 跟新个体的solution索引
            individual.getNumberOfSolution()
            parentPopulation[i] = individual

        return parentPopulation

    # 交换 0 和 1，用于变异
    def reserveAboutMutate(self, value):
        y = -value + 1
        return y

    # 计算该种群中每一个个体的适应度值和选择特征数量,并且得到生成下一代的父类代，即p_i+1
    def getNextParentPopulation(self, chromosomeList , iteratorNum,mode):
        # 临时存放种群
        tempPopulation = np.copy(chromosomeList)
        # tempPopulation = tempPopulation.tolist()
        # 存放每一个个体的适应度值--原生代
        fitnessOfEachIndividual_ori = np.asarray([])
        # 存放每一个个体的solution长度--原生代
        lengthOfEachIndividual_ori = np.asarray([])
        #import  time
        #start = time.time()
        # 计算 原生代 每一个个体的fitness
        for i in range(0, len(tempPopulation)):
            # 得到每一个个体
            chromosome_i = tempPopulation[i]
            # 进行计算该个体的fitness  ---- 本次算法使用的是错误率
            # acc2 = self.computeFitnessTree(chromosome_i=chromosome_i)
            # acc = acc2
            # error = 1 - acc
            # # 个体保存自己的fitness
            # chromosome_i.mineFitness = error

            error = chromosome_i.mineFitness
            acc = 1 - error
            # 保存该节点的fitness值
            fitnessOfEachIndividual_ori = np.append(fitnessOfEachIndividual_ori, error)
            # 保存该节点的选择特征的长度
            lengthOfEachIndividual_ori = np.append(lengthOfEachIndividual_ori, chromosome_i.numberOfSolution)
# 每过 self.grTime次迭代就要进行一次filter和wrapper之间的相互促进
            if mode == 'wrapper' and iteratorNum  == self.grTime - 1: # 上一代
                self.wrapperSolutionPrevious[i] = chromosome_i.featureOfSelect
                # 未选择率
                rec = 1 - chromosome_i.numberOfSolution/len(self.dataFeature)
                tempList = [acc,rec]
                # 保存了上一代的 acc 和 res
                self.accResMatrixPrevious[i] = tempList

            if mode == 'wrapper' and iteratorNum == self.grTime:# 当前代
                # 进行guiding    wrapper -----> filter
                self.wrapperSolutionRecent[i] = chromosome_i.featureOfSelect
                # 未选择率
                rec = 1 - chromosome_i.numberOfSolution/len(self.dataFeature)
                tempList = [acc,rec]
                # 保存了当前代的acc 和  res
                self.accResMatrixRencent[i] = tempList

            if mode == 'filter' and iteratorNum == self.grTime:
                #保存当前代的filter种群的特征选择
                self.filterSolutionRecent[i] = chromosome_i.featureOfSelect

        # 用于保存原生代fitness
        tempfitness = copy.deepcopy(fitnessOfEachIndividual_ori)
        temppop = chromosomeList

        #  进行 wrapper 对 filter的guiding  ， 然后  filter 对 wrapper 的repair
        if mode == 'wrapper' and iteratorNum == self.grTime:
            self.filterSolutionRecent = np.asarray(self.filterSolutionRecent)
            self.accResMatrixRencent = np.asarray(self.accResMatrixRencent)
            self.wrapperSolutionRecent = np.asarray(self.wrapperSolutionRecent)
            self.wrapperSolutionPrevious = np.asarray(self.wrapperSolutionPrevious)
            self.accResMatrixPrevious = np.asarray(self.accResMatrixPrevious)
            #print("执行wrapper -----> filter 的 guiding")
        # wrapper -----> filter 的guiding
            occFreqPrevious ,occNumberP= self.getOccFreq(wrapperSolution=self.wrapperSolutionPrevious)
            #  用occFreRenent 来接收每个特征计算归一化后的值，  occNumber来接收每个特征被选择的次数
            occFreqRecent , occNumberR = self.getOccFreq(wrapperSolution=self.wrapperSolutionRecent)
            for i in range(0, len(occFreqRecent)):
                # 当前代比上前一代
                occFreqRecent[i] = occFreqRecent[i] / occFreqPrevious[i]
            # 得到平均变化率，acc和rec在当前代和上一代的变化率为多少
            wpop = self.getAccRecMean(wrapperAccRecRecent=self.accResMatrixRencent,
                                      wrapperAccRecPrevious=self.accResMatrixPrevious)
            occFreq = occFreqRecent * wpop
            # 跟新weight
            self.weightOfFeatuer = self.weightOfFeatuer * occFreq

            # 进行guiding
            for i in range(0,len(self.weightOfFeatuer)):
                if self.weightOfFeatuer[i] >= 1:
                    # 如果大于1 ，则说明这个特征被选择的次数多，则互信息升高，相似度降低
                    self.mutualInforArray[i] = self.mutualInforArray[i] * self.weightOfFeatuer[i]
                    self.similarityOfFeature[i] = self.similarityOfFeature[i] / self.weightOfFeatuer[i]
                else:
                    # 如果该特征被选择的次数较少，说明效果并不是那么好，则互信息下降，相似度提高
                    self.mutualInforArray[i] = self.mutualInforArray[i] / self.weightOfFeatuer[i]
                    self.similarityOfFeature[i] = self.similarityOfFeature[i] * self.weightOfFeatuer[i]

            #print("执行 filter -----> wrapper 的 repair")
        # filter -----> wrapper 的repair
            # 得到特征被选择次数为0次的特征索引
            indexZero = np.where(occNumberR == 0)[0]
            # 得到特征被选择次数为1次的特征索引
            indexOne = np.where(occNumberR == len(self.chromosomeWrapperList))[0]
            # 如果都未被选中，并且选中次数不为全部种群
            if len(indexOne) == 0 and len(indexZero) == 0:
                #print(" 未全部被选中")
                pass
            else:
                # 将该索引链接到一起，并且正序排序
                allZeroOneIndex = sorted(np.append(indexZero,indexOne))
                # 得到每一个被筛选出来的特征的互信息
                mutialIforZeroOne = np.asarray([self.mutualInforArray[int(i)] for i in allZeroOneIndex])
                # 得到每一个被筛选出来的特征的相似度
                similarZeroOne = np.asarray([self.similarityOfFeature[int(i)] for i in allZeroOneIndex])
                # 将互信息进行一个排序，降序排序
                sortMutial = sorted(mutialIforZeroOne,reverse=True)
                # 将相似度进行一个排序，降序排序
                sortSimilar = sorted(similarZeroOne,reverse=True)
                # 获得每一个被筛选出来的特征通过各自的mutialInformation和similar值进行的各自排序
                rankM = self.getRank(sortedArray=sortMutial,originArray=mutialIforZeroOne)
                rankS = self.getRank(sortedArray=sortSimilar,originArray=similarZeroOne)
                # 总排名
                rank = rankM + rankS
                # 向上取整, repair K 个特征
                K = int(1/self.iteratorTime * len(allZeroOneIndex))
                if K < 1 :
                    K = 1
                # 按照rank的值进行排序  ,进行升序排序
                sortRank = sorted(rank)
                # 计数
                count = 0

                while count < K:
                    # 找到rank值中排名最靠前的一个值
                    indexRank_list = np.where(rank == sortRank[count])[0]
                    if len(indexRank_list) > 1:
                        indexRank = indexRank_list[0]
                        np.delete(indexRank_list,0)
                    else:
                        indexRank = indexRank_list[0]
                    indexRank = int(allZeroOneIndex[indexRank])
                    #print(indexRank)
                    # 每一个特征只能被修复一次
                    if self.repairFeature[indexRank] == 1:
                        # wrapper种群中 第indexRank列
                        indw = self.wrapperSolutionRecent[:,indexRank]
                        # filter种群中 第indexRank列
                        indf = self.filterSolutionRecent[:,indexRank]
                        # 进行交叉
                        for w in range(0,len(indw)):
                            r = random.random()
                            # 进行交换
                            if r < self.probabilityOfCrossover:
                                t = self.chromosomeWrapperList[w].featureOfSelect[indexRank]
                                self.chromosomeWrapperList[w].featureOfSelect[indexRank] = self.chromosomeFilterList[w].featureOfSelect[indexRank]
                                self.chromosomeFilterList[w].featureOfSelect[indexRank] = t
                            # 进行变异
                            r1 = random.random()
                            r2 = random.random()
                            if r1 < self.probabilityOfMutation:
                                self.chromosomeWrapperList[w].featureOfSelect[indexRank] = self.reserveAboutMutate(value=indw[w])
                            if r2 < self.probabilityOfMutation:
                                self.chromosomeFilterList[w].featureOfSelect[indexRank] = self.reserveAboutMutate(value=indf[w])
                        self.repairFeature[indexRank] == 0
                        count = count + 1
                    else:
                        # 如果该特征被修复过了，那么就选择下一个特征
                        count = count + 1
                        K = K + 1
            # # 进行归还该值 , 将这两个数组转化为list ，方便第二次的操作
            # self.wrapperSolutionRecent = [[] for c in range(0, len(self.chromosomeWrapperList))]
            # self.filterSolutionRecent = [[] for c in range(0, len(self.chromosomeFilterList))]
            # self.wrapperSolutionPrevious = [[] for c in range(0, len(self.chromosomeWrapperList))]
            # self.accResMatrixPrevious = [[] for c in range(0,len(self.chromosomeWrapperList))]
            # self.accResMatrixRencent = [[] for c in range(0,len(self.chromosomeWrapperList))]



        # 计算获得精英种群，通过二元竞争法  需要 交叉变异
        self.getElite(chromosomeList=tempPopulation,mode=mode)

        # 将原生代和精英代进行组合到一个数组中去
        parentPopulation = tempPopulation
        for i in range(0, len(self.elitistChromosome)):
            parentPopulation = np.append(parentPopulation, self.elitistChromosome[i])

        for i in range(0, len(self.elitistChromosome)):
            # 将精英种群中的 fitness 和 选择特征数量添加到相对应的数组中去
            # 所有的fitness
            fitnessOfEachIndividual_ori = np.append(fitnessOfEachIndividual_ori,
                                                   self.elitistChromosome[i].mineFitness)
            # 所有的选择特征的长度
            lengthOfEachIndividual_ori = np.append(lengthOfEachIndividual_ori,
                                                   self.elitistChromosome[i].numberOfSolution)

        # 输出fitness前3的三个个体
        tempsortfitness = sorted(fitnessOfEachIndividual_ori)
        # print(tempsortfitness)
        # if mode == 'filter':
        #     print(" filter ")
        # elif mode == 'wrapper':
        #     print(" wrapper ")
        # print("fitness前3个")
        for i in range(0, 3):
            # 得到索引
            tempindex = np.where(fitnessOfEachIndividual_ori == tempsortfitness[i])[0][0]
            tempindividual = parentPopulation[tempindex]
            # print("================================================")
            # #print("第", i, "个体的solution为：", tempindividual.indexOfSolution)
            # print("第", i, "个体的fitness为：",1 - tempindividual.mineFitness)
            # print("第", i, "个体的选择长度为：", tempindividual.numberOfSolution)
            # 每一个fitness里面保存的是error ， 所以要用 acc = 1 - error
            if self.globalBestFitness < (1 - tempindividual.mineFitness):
                self.globalBestFitness = 1 - tempindividual.mineFitness
                self.globalBestSolution = tempindividual.featureOfSelect
                self.globalBestLength = tempindividual.numberOfSolution
            elif self.globalBestLength == (1 - tempindividual.mineFitness) and self.globalBestLength > tempindividual.numberOfSolution:
                self.globalBestFitness =1 - tempindividual.mineFitness
                self.globalBestSolution = tempindividual.featureOfSelect
                self.globalBestLength = tempindividual.numberOfSolution



        #print("进行非支配排序")
        # 计算非支配排序  通过帕累托结集
        springsonPopulation = self.nonDominatedSort(parentPopulation=parentPopulation,
                                                    fitnessArray=fitnessOfEachIndividual_ori,
                                                    solutionlengthArray=lengthOfEachIndividual_ori,
                                                    mode= mode)

        #print("通过交叉变异产生子代")
        # 进行交叉算子，80%个体进行交叉，后20%删除，重新创建个体
        springsonPopulation = self.crossoverOperate(population=springsonPopulation,mode=mode)

        springsonPopulation = self.mutateOperator_elist(parentPopulation=springsonPopulation)

        # 计算每一个子代的适应度
        for i in range(len(springsonPopulation)):
            dataX = self.dataX[:, springsonPopulation[i].featureOfSelect == 1]
            acc = fitnessFunction_KNN_CV(findData_x=dataX, findData_y=self.dataY, CV=5)
            springsonPopulation[i].mineFitness =1 - acc
            tempfitness = np.append(tempfitness,1 - acc)
        tempPop = np.concatenate((temppop, springsonPopulation), axis=0)
        # 排序
        tempIndex = np.argsort(tempfitness)
        tempPop = tempPop[tempIndex]
        # 将前100当成下一代
        springsonPopulation = tempPop[0:self.numberOfChromosome]

        # 清理缓存
        self.clearIndividualCache(chromosomeList = springsonPopulation)

        return springsonPopulation

    # 将子代的数据清空
    def clearIndividualCache(self, chromosomeList):
        for i in range(0, len(chromosomeList)):
            # 支配集合为空
            chromosomeList[i].dominateSet = []
            # 支配等级
            chromosomeList[i].p_rank = 0
            # 非支配数量
            chromosomeList[i].numberOfNondominate = 0

    # 计算每一个节点的帕累托  非支配排序   传入的是归一化后的 error 和 solutionlength
    def nonDominatedSort(self, parentPopulation, fitnessArray, solutionlengthArray,mode):
        # 用于保存下一代解
        springsonPopulation = np.asarray([])

        # 用来保存每一个个体的函数值
        errorArray = np.asarray([])
        lenArray = np.asarray([])

        # 用于保存每一个个体的支配解集
        dominateFrontArray = []

        # 保存帕累托前沿,保存非支配数量为0的个体,
        F_i = []

        # 将每一个个体的error 和 len 的评分计算出来
        errorArray ,lenArray = self.sortFunction(error=fitnessArray,solutionLength_test=solutionlengthArray)

        #print("计算每一个个体的支配结集和非支配数量")

        ##---------------------------------- ENS  高效非支配排序 ----------------------------------------##
        # 首先对种群按照error进行一个排序，从小到大的一个排序

        # 先得到error进行排序后的索引
        sortErrorIndex = np.argsort(errorArray)
        # 将 error  population len 通过 该索引进行一个重新组合
        errorArray = errorArray[sortErrorIndex]
        parentPopulation = parentPopulation[sortErrorIndex]
        lenArray = lenArray[sortErrorIndex]
        # 对相同error的个体进行按照len的一个排序
        i = 0
        j = 0
        while i < len(errorArray) - 1:
            j = i + 1;
            if errorArray[i] == errorArray[j]:
                while j < len(errorArray) and errorArray[i] == errorArray[j]:
                    j += 1
                if (j - i > 1):
                    newArray1 = copy.deepcopy(lenArray[i:j])
                    # newArray2 = copy.deepcopy(errorArray[i,j])
                    newArray3 = copy.deepcopy(parentPopulation[i:j])
                    indexNew = np.argsort(newArray1)
                    newArray1 = newArray1[indexNew]
                    # newArray2 = newArray2[indexNew]
                    newArray3 = newArray3[indexNew]
                    for m in range(len(newArray1)):
                        lenArray[i] = newArray1[m]
                        # errorArray[k] = newArray2[m]
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

        # 进行添加个体到子代
        numPop = self.numberOfChromosome
        # 计数
        count = 0
        while len(springsonPopulation) < self.numberOfChromosome:
            front_i = dominateFrontArray[count]
            resideNum = numPop - len(springsonPopulation)
            # 如果长度够，则全部添加
            if resideNum - len(front_i) >= 0:
                # 将front_i中每一个个体添加到子代种群中去
                springsonPopulation = np.append(springsonPopulation, front_i)
            else:
                # 进行拥挤度排序
                # 保存error 和 len
                crossError = np.zeros(len(front_i))
                crossLen = np.zeros(len(front_i))
                # 将前沿中每一个个体的 error和len提取出来，然后准备计算
                for k in range(0, len(front_i)):
                    crossError[k] = front_i[k].mineFitness
                    crossLen[k] = front_i[k].numberOfSolution / len(self.dataFeature)
                crossError, crossLen = self.sortFunction(error=crossError, solutionLength_test=crossLen)
                # 计算拥挤度
                front_i, distance = self.crowdingDistance(F_i=front_i, error=crossError, lens=crossLen)
                # 从大到小排序
                sort_distance_index = np.argsort(distance)[::-1]
                # 还需要添加的个体数量
                for k in range(resideNum):
                    springsonPopulation = np.append(springsonPopulation, front_i[sort_distance_index[k]])
            count += 1

        # 返回子代种群
        return springsonPopulation

    def initIndividualFitness(self):
        for i in range(self.numberOfChromosome):

            acc1 = self.computeFitnessKNN(self.chromosomeWrapperList[i])
            self.chromosomeWrapperList[i].mineFitness =1 - acc1

            acc1 = self.computeFitnessKNN(self.chromosomeFilterList[i])
            self.chromosomeFilterList[i].mineFitness =1 - acc1

    #   用于帕累托的 ---- 拥挤度计算
    # 通过error 对F_i中的len进行一块排序，相当于坐标的排序,fitness里面存放的是error,
    def sortCoordinate(self, F_i, remaindFitness, remaindLength):
        # 先通过error 进行一个从小到大的排序
        sortError = sorted(remaindFitness)
        # 临时存放
        individal = np.asarray([])
        solutionLen = np.asarray([])
        # 然后再通过len进行排序
        for i in range(0, len(sortError)):
            index = np.where(remaindFitness == sortError[i])[0][0]
            individal = np.append(individal, F_i[index])
            solutionLen = np.append(solutionLen, remaindLength[index])
        F_i = individal
        remaindFitness = sortError
        remaindLength = solutionLen
        return F_i, remaindFitness, remaindLength

    # 拥挤度距离计算
    def crowdingDistance(self, F_i, error, lens):
        # 拥挤度距离
        crossedDistance = np.zeros(len(F_i))
        # 将两端的距离设置为 inf
        crossedDistance[0] = float(np.inf)
        crossedDistance[-1] = float(np.inf)

        # 计算每一个样本的拥挤度
        for i in range(1, len(F_i) - 1):
            crossedDistance[i] = crossedDistance[i] + (
                    error[i + 1] - error[i - 1]
            ) / (max(error) - min(error) + 0.00001)

        for i in range(1, len(F_i) - 1):
            crossedDistance[i] = crossedDistance[i] + np.abs(
                lens[i + 1] - lens[i - 1]
            ) / (max(lens) - min(lens) + 0.00001)
        return F_i, crossedDistance

    # 帕累托计算函数  ----> 1/exp(x)  1/exp(y)  [1/e,1]   -----> x == error : y == length#
    # 帕累托计算函数  ----> exp(x)  exp(y)  [1,e]   -----> x == error : y == length#
    def sortFunction(self, error, solutionLength_test):
        # 计算  e的 x次 和 e的y次
        error = np.exp(error)
        solutionLength = solutionLength_test / len(self.dataFeature)
        # 计算其倒数  将范围归一到[1/e,1]之间
        #error = 1 / error           # 错误率
        #solutionLength = 1 / np.exp(solutionLength) # 选择率
        solutionLength = np.exp(solutionLength)
        # 算出每个值个体的value值评分
        # valueOf_i = error + solutionLength
        return error, solutionLength




    # 生成Q_0 精英主义，用于和原生代组合生成下一代
    def getElite(self, chromosomeList,mode):
        # 用于存放临时的数组
        tempList = np.asarray([])
        # 二元竞赛
        for i in range(0, len(chromosomeList)):
            # 随机获得chromosomeList 中的两个索引
            rand = random.sample(range(0, len(chromosomeList)), 2)
            # 获得第i个个体
            individual_i = copy.deepcopy(chromosomeList[rand[0]])
            # 获得第j个个体
            individual_j = copy.deepcopy(chromosomeList[rand[1]])
            # 获得第i个个体的fitness
            fitness_i = individual_i.mineFitness
            # 获得第j个个体的fitness
            fitness_j = individual_j.mineFitness
            # 选择两者之间fitness低的添加到templist中去： 因为fitness是 error  error越低，acc越高
            if fitness_i < fitness_j:
                # 用深copy  ，形成一个新的对象
                individual = individual_i
            elif fitness_i > fitness_j:
                individual = individual_j
            else:
                # 如果 error 相等，那么选择特征长度越小的可以进入精英代
                l_i = individual_i.numberOfSolution
                l_j = individual_j.numberOfSolution
                if l_i < l_j:
                    individual = individual_i
                else:
                    individual = individual_j

            tempList = np.append(tempList, individual)

        l = tempList
        # 进行交叉变异

        tempList = self.crossoverOperate(population=tempList,mode=mode)
        tempList = self.mutateOperator_elist(parentPopulation=tempList)

        # 重新得到每个个体的选择的特征的fitness
        for i in range(0, len(tempList)):
            individual = tempList[i]
            # 计算新的个体fitness
            acc = self.computeFitnessKNN(individual)
            # 使用error来进行评判fitness
            error = 1 - acc
            individual.mineFitness = error
        # 添加完毕后
        self.elitistChromosome = copy.deepcopy(tempList)

    # 计算该个体的适应度值
    def computeFitnessKNN(self, chromosome_i):
        # 获得该个体的特征组合的 数据集合
        feature_x = self.getSolutionData(chromosome_i=chromosome_i)
        if len(feature_x) == 0:
            return 0
        # 获得类数据
        feature_y = self.dataY
        # 进行10折交叉验证
        acc = fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=feature_y, CV=10)
        return acc

    # 计算适应度值通过Tree
    def computeFitnessTree(self, chromosome_i):
        # 获得该个体的特征组合的 数据集合
        feature_x = self.getSolutionData(chromosome_i=chromosome_i)
        if len(feature_x) == 0:
            return 0
        # 获得类数据
        feature_y = self.dataY
        # 进行10折交叉验证
        acc = fitnessFunction_Tree_CV(findData_x=feature_x, CV=5, findData_y=feature_y)
        return acc

    # 获得该个体所选择的特征组合
    def getSolutionData(self, chromosome_i):
        tempFeatureSolutionData = self.dataX[:,chromosome_i.featureOfSelect == 1]
        return tempFeatureSolutionData

    # 将特征数量直接进行归一化：
    def normalizeOfFeature(self):
        # 得到的相对应的特征数量
        rangeOfSolution = np.arange(0, len(self.dataFeature))
        # 线性归一化
        self.featureNumNormalize = self.linearNormalization(rangeOfSolution)


    # 得到occurrence number ，每个特征在该代种群中被选中的比例 -----wrapper
    def getOccFreq(self,wrapperSolution):
        occurrenceNumber  = np.asarray([])
        # axis=0 获得一行数据，就是按照列相加； axis=1 获得一列数据，就是按照行相加
        occurrenceNumber = np.sum(wrapperSolution,axis=0)
        # for i in range(0,len(wrapperSolution[0])):
        #     # 得到第i列数据
        #     a = wrapperSolution[:,i]
        #     occurrenceNumber = np.append(occurrenceNumber,a.sum())
        # 求 softMax
        wij = softmax(numberOfAllFeatureSelect = occurrenceNumber)
        return wij , occurrenceNumber
    #  得到 OccFreq ,每个特征的正确率和未选择率的平均值 -----wrapper
    def getAccRecMean(self,wrapperAccRecPrevious,wrapperAccRecRecent):
        # 平均acc
        meanAccP = wrapperAccRecPrevious[:,0].mean()
        meanAccR = wrapperAccRecRecent[:,0].mean()
        # 平均rec
        meanRecP = wrapperAccRecPrevious[:,1].mean()
        meanRecR = wrapperAccRecRecent[:,1].mean()
        wPopProf = meanRecR/(meanRecP + 0.00000001) * meanAccR/(meanAccP + 0.00000001)
        return wPopProf

    # 传入两个数组， 一个是排序完毕的数组， 一个是未排序前的数组，返回一个数组，里面保存了每一个位置所对应的排名
    # 用于 filter 对 wrapper的一个repair操作
    def getRank(self,sortedArray,originArray):
        # 生成一个和原始数组一样大小的数组，用于存放每一个值所对应的排名
        rank = np.zeros(len(originArray))
        for i in range(0,len(sortedArray)):
            index = np.where(originArray == sortedArray[i])[0][0]
            rank[index] = i
        return rank

    # 运行
    def run(self):
        # 初始化种群
        # print("初始化种群")
        self.initChromosomeList()

        # 初始化 用于wrapper和filter的相互促进
        self.accResMatrixPrevious = [[] for i in range(0, len(self.chromosomeWrapperList))]
        self.wrapperSolutionPrevious = [[] for i in range(0,len(self.chromosomeWrapperList))]
        self.accResMatrixRencent = [[] for i in range(0, len(self.chromosomeWrapperList))]
        self.wrapperSolutionRecent = [[] for i in range(0,len(self.chromosomeWrapperList))]
        self.filterSolutionRecent = [[] for i in range(0,len(self.chromosomeFilterList))]

        # 运行时间
        runtime = 0
        # 用于计数
        count = 0
        import time
        start = time.time()
        # 计算初始个体的fitness
        self.initIndividualFitness()
        while runtime < self.iteratorTime:


            self.chromosomeFilterList = self.getNextParentPopulation(chromosomeList=self.chromosomeFilterList,mode='filter',iteratorNum=count + 1)

            self.chromosomeWrapperList = self.getNextParentPopulation(chromosomeList=self.chromosomeWrapperList,mode='wrapper',iteratorNum=count + 1)

            # 0到9，10个数，相当于循环了10轮，执行一次filter和wrapper的interact
            if count == self.grTime - 1:
                count = -1   #  底下有个加一，所以为0就会直接加一导致出现问题，所以变成-1，低下加1成为0，重新一轮迭代
                # filter 种群
                for i in range(0, int(self.numberOfChromosome / 2)):
                    chromosome = self.Chromosome(GRmoea=self, mode='filter')
                    self.chromosomeFilterList[i] = chromosome
                    chromosome.mineFitness = self.computeFitnessKNN(chromosome_i=chromosome)
            count  = count + 1
            runtime = runtime + 1

        tempPop = np.concatenate((self.chromosomeWrapperList,self.chromosomeWrapperList),axis=0)
        f_0 = getParato(pop=tempPop,popNum=self.numberOfChromosome*2)
        # 写入文件的路径
        path_txt = "../result/paretoSolution/GRMOEA/" + ducName + ".txt"

        # 写入文件的标题
        timeData = '   ' + str(time.ctime()) + '\n'
        # 将结果写入到文件中去
        with open(path_txt, 'a') as f:
            f.write(timeData)
            f.close()

        for a in range(len(f_0)):
            titleTxt = str(f_0[a].mineFitness)+'\t'+str(f_0[a].numberOfSolution)+'\n'
            # 将结果写入到文件中去
            with open(path_txt, 'a') as f:
                f.write(titleTxt)
                f.close()


if __name__ == "__main__":
    import time
    files = os.listdir("../dataCSV/dataCSV_high")
    for file in files:
        ducName = dataName = file.split('.')[0]
        path_csv = "../dataCSV/dataCSV_high/" + ducName + ".csv"

        # 写入文件的路径
        path_txt = "../result/result_txt/GRMOEA/" + ducName + ".txt"
        # 写入文件的标题
        titleTxt = '   ' + ducName + '.csv \n' + str(time.ctime()) + '\n'
        # 将结果写入到文件中去
        with open(path_txt, 'a') as f:
            f.write(titleTxt)
            f.close()

        dataCsv = ReadCSV(path=path_csv)
        print("获取文件数据",file)
        dataCsv.getData()
        grmoea = GRmoea(dataCsv=dataCsv,  dataX=dataCsv.dataX, dataY=dataCsv.dataY,dataName=file.split('.')[0])
        grmoea.setParameter(crossover=0.9, mutation=0.1, numberOfChromosome=70, iteratorTime=100,gtTime=10)
        # 循环多少次
        iterateT = 10

        acc = np.zeros(iterateT)
        length = np.zeros(iterateT)

        start = time.time()
        for i in range(iterateT):
            grmoea.run()
            acc[i] = grmoea.globalBestFitness
            length[i] = grmoea.globalBestLength
            # 重置
            grmoea.globalBestFitness = 0
            grmoea.globalBestSolution = np.asarray([])
            grmoea.globalBestFitness = 0
            grmoea.chromosomeFilterList = []
            grmoea.chromosomeWrapperList = []

            # 写入文件的值
            stringOfResult = str(acc[i]) + '\t' + str(length[i]) + '\n'
            # 将结果写入到文件中去
            with open(path_txt, 'a') as f:
                f.write(stringOfResult)
                f.close()
            print(acc[i], " ", length[i])

        # 向文件中写入均值
        # 写入文件的值
        stringOfResult = str(acc.mean()) + '\t' + str(acc.std()) + '\t' + str(length.mean()) +str(length.std()) + '\n'
        # 将结果写入到文件中去
        with open(path_txt, 'a') as f:
            f.write('   mean  \n')
            f.write(stringOfResult)
            f.close()

        print("acc:", acc.mean(), "  std:", acc.std(), "  len:", length.mean(),"  std:", length.std())
        print("种群1长度",len(grmoea.chromosomeWrapperList))
        print("种群2长度",len(grmoea.chromosomeFilterList))
        # 将结果写入到文件中去
        with open(path_txt, 'a') as f:
            f.write('\n')
            f.write(f"all time = {time.time() - start} seconds"+'\n')
            f.close()
        print(f"all time = {time.time() - start} seconds")
        print("===================================")
        print(" ")

