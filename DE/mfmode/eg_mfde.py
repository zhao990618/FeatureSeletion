import copy
import math
import os
import random


import numpy as np
from random import sample


from dataProcessing.ReadDataCSV_new import ReadCSV
from dataProcessing.GetParetoSolution import getParato

from dataProcessing.writeSimilarHigh import readSimilar
from dataProcessing.InforGain import InforGain
from dataProcessing.reliefF import reliefFScore
from dataProcessing.StringDataToNumerical import reserveNumerical
from Classfier.invokingClassfier import computeFitnessKNN
from sklearn.preprocessing import MinMaxScaler
from skfeature.utility.mutual_information import information_gain
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from scipy.stats import norm
from scipy.optimize import fminbound
from DE.mfmode.operators import mutate_best,mutate_rand1,mutate_rand2
from DE.mfmode.operators import crossover_num
from DE.mfmode.operators import getFreq
from DE.mfmode.dataMemory import memory


class MFMODE:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 特征的存放数组
    dataFeature = np.asarray([])
    # 特征的数量
    dataFeatureNum = 0
    # 每个特征与其他特征的相似程度
    similarityOfFeature = np.asarray([])
    # 每一个特征的relifF评分
    scoreOfRelifF = np.asarray([])
    # 互信息矩阵
    mutualInforArray = np.asarray([])
    # 每一个特征被选择的概率
    probabilityOfFeatureToSelect = np.asarray([])

    # 存放两两特征之间的冗余程度
    mrmrArray = np.asarray([])

    # 将种群中的solution提取出来
    solutionArray = []
    # 任务1 的种群
    population_task1 = np.asarray([])
    # 任务2 的种群
    population_task2 = np.asarray([])
    # 任务3 的种群
    #population_task3 = np.asarray([])

    # 每一个种群的数量
    populationNum = 0
    # 种群迭代次数
    iteratorTime = 0
    # 精英种群通过fitness排序得到的原始前 N% 个个体得到
    eliteProb = 0
    # 探索
    exStage = 0
    # 交叉率
    crossoverPro = 0
    # 变异率
    mutatePro = 0
    # 差分进化中变异操作的F 缩放因子
    F = 0
    # border0
    eliteFront = 0
    # 创建存储对象
    recentSave = memory(taskNum=2)
    currentSave = memory(taskNum=2)

    # 用于保存最优个体，在DE时用--2个任务
    p_DEarray = []
    # 特征权重--概率和relifF
    featureWeight= np.asarray([])

    # 全局最优解
    globalSolution = np.asarray([])
    # 全局最优解的实数制
    globalSolutionNum = np.asarray([])
    # 全局最优解的适应度值
    globalFitness = 1
    # 全局最优的score
    globalScore = 0

    dataName = ""

    oriX = np.asarray([])
    oriY = np.asarray([])
    saveTrainSet = []
    saveTestSet = []
    saveTrainSet_test = []
    dataX_t = np.asarray([])
    dataY_t = np.asarray([])

    def __init__(self, dataX, dataY,dataName):
        self.dataX = dataX
        # self.dataFeature = [i for i in range(self.dataX.shape[1])]
        self.dataFeature = np.arange(self.dataX.shape[1])
        self.dataFeatureNum = len(self.dataFeature)
        self.dataY = dataY
        self.dataName = dataName
        # 将class里的数据转为数值类型的
        self.dataY = reserveNumerical(self.dataY)

        # 得到相似度矩阵
        print("得到相似度矩阵")
        # 用于高维数据
        self.featureOfSimilar_high()
        # 用于低维数据
        # self.similarityOfFeature = featureOfSimilar(dataX=self.dataX)

        # 对数据01 归一化，将每一列的数据缩放到 [0,1]之间
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        print("进行filter，提取一部分数据")
        # filter
        self.filter()
        print(self.dataX.shape[1])

    # 设置参数
    def setParameter(self, populationNum, iteratorTime, eliteProb,crossoverPro, F,eliteFront,p,maxAge):
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime
        self.eliteProb = eliteProb
        #self.exStage = exStage
        self.crossoverPro = crossoverPro
        self.F = F
        self.eliteFront = eliteFront
        self.p = p
        self.maxAge = maxAge


    # 通过SU来进行筛选数据
    # fiter 方法  获取拐点位置的数据  ----------用于高维数据
    def filter(self):

        # 得到relifF
        print("得到所有特征的relifF")
        self.scoreOfRelifF = reliefFScore(self.dataX, self.dataY)

        # 得到su矩阵
        print("得到互信息矩阵")
        infor = InforGain(dataAttribute=self.dataFeature, dataX=self.dataX)

        self.mutualInforArray = infor.getMutualInformation_mode_fc(dataX=self.dataX)  # 用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵
        # self.mutualInforArray = infor.getMutualInformation_fc(dataX=self.dataX,dataY=self.dataY)  # 用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵

        # 拐点分割特征，选择拐点前的数据
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
        # 对 pro进行归一化，放缩到[0.2,0.7]之间
        self.probabilityOfFeatureToSelect = ((self.probabilityOfFeatureToSelect - np.min(
            self.probabilityOfFeatureToSelect)) / (np.max(self.probabilityOfFeatureToSelect) - np.min(
                                                 self.probabilityOfFeatureToSelect)) + 0.4) * 0.5
        # 对 relifF进行归一化，放缩到[0.2,0.7]之间
        self.scoreOfRelifF = ((self.scoreOfRelifF - np.min(self.scoreOfRelifF)) / (
            np.max(self.scoreOfRelifF - np.min(self.scoreOfRelifF))) + 0.4) * 0.5

        print("生成初始化的mrmr矩阵，用于后期保存两两特征的mrmr相关性")
        self.mrmrArray = np.full((len(self.dataFeature), len(self.dataFeature)), 0)
        # print("")

        # 初始化特征权重
        self.featureWeight = np.ones((2,len(self.dataFeature)))
    #  用于高维数据  --- 提前处理好similar 保存在txt中
    def featureOfSimilar_high(self):
        allFeatureSimilar = readSimilar(path="../dataCSV/dataCSV_similar/"+self.dataName+"Similar.txt")
        #allFeatureSimilar = readSimilar(path="D:\MachineLearningBackUp\dataCSV\dataCSV_similar\\arceneSimilar.txt")
        # 不应该排序
        self.similarityOfFeature = np.asarray(allFeatureSimilar)

    # 计算每一个特征被选择的概率
    def filterOfProFreture(self):
        # print("通过SU和similar对每一个特征求其概率")
        mutInfor = self.similarityOfFeature + 0.00000001
        self.probabilityOfFeatureToSelect = self.mutualInforArray / mutInfor
        # print(" ")

    # 初始化种群
    def initPopulation(self):
        # task1 种群
        for i in range(0, int(self.populationNum)):
            chromosome = self.Chromosome(mfea=self, mode='task1',task=0)
            chromosome.index = i
            self.population_task1 = np.append(self.population_task1, chromosome)

        # task2 种群
        for i in range(0, int(self.populationNum)):
            chromosome = self.Chromosome(mfea=self, mode='task2',task=1)
            chromosome.index = i
            self.population_task2 = np.append(self.population_task2, chromosome)



    # 染色体
    class Chromosome:
        # 每一个个体所选择出来的特征组合   --- 实数制
        featureOfSelectNum = np.asarray([])
        # 每一个个体所选择出来的特征组合   --- 二进制
        featureOfSelect = np.asarray([])
        # 特征组合的长度
        numberOfSolution = 0
        # 特征组合的索引
        indexOfSolution = np.asarray([])

        # 属于哪一个任务
        task = 0
        # 当前的选择的fitness
        mineFitness = 0
        # 交换率
        crossover = 0
        # 变异率
        mutation = 0

        # DMS 局部搜索  默认初始化为1
        alpha = 1

        # 支配等级
        p_rank = 0
        # 非支配数量
        numberOfNondominate = 0
        # 支配个体集合
        dominateSet = []

        # 存在时间
        age = 0
        # 索引
        index = 0
        # 解中不再冗余
        isNotRedundance = False

        # 终极
        terminus = False
        # 任务
        task = -1

        def __init__(self, mfea, mode,task):
            if mode == 'task1':
                self.initOfChromosome_Pro(MFEA=mfea,task=task)
                #self.initOfNomal(MFEA=mfea,task = task)
            elif mode == 'task2':
                self.initOfChromosome_RelifF(MFEA=mfea,task=task)
                #self.initOfNomal(MFEA=mfea, task=task)
            elif mode == 'elite':
                self.initOfChromosome_elite(MFEA=mfea,task=task)


        # 用于普通初始化
        def initOfNomal(self,MFEA,task):
            self.task = task

        # 用于精英种群的初始化
        def initOfChromosome_elite(self, MFEA,task):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            self.task = task

        # 初始化染色体  -- Pro  --- 用于正常生成个体，不适合用于反向学习
        def initOfChromosome_Pro(self, MFEA,task):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            self.featureOfSelectNum = np.zeros(len(MFEA.dataFeature))
            for i in range(0, len(MFEA.probabilityOfFeatureToSelect)):
                # 获取每一个特征上的随机数，通过这个随机数来判断初始化是否选择该特征
                rand = np.random.random()
                if rand <= MFEA.probabilityOfFeatureToSelect[i]:
                    self.featureOfSelect[i] = 1
                self.featureOfSelectNum[i] = rand
            # 计算其 acc
            acc = computeFitnessKNN(chromosome_i=self, data_x=MFEA.dataX, data_y=MFEA.dataY)
            self.mineFitness = 1 - acc
            self.getNumberOfSolution()
            self.task = task

        # 初始化染色体 通过relifF方式 --- 用于正常生成个体，不适合用于反向学习
        def initOfChromosome_RelifF(self, MFEA,task):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            self.featureOfSelectNum = np.zeros(len(MFEA.dataFeature))
            for i in range(0, len(MFEA.scoreOfRelifF)):
                rand = np.random.random()
                if rand <= MFEA.scoreOfRelifF[i]:
                    self.featureOfSelect[i] = 1
                self.featureOfSelectNum[i] = rand
            # 计算其 acc
            acc = computeFitnessKNN(chromosome_i=self, data_x=MFEA.dataX, data_y=MFEA.dataY)
            self.mineFitness = 1 - acc
            self.getNumberOfSolution()
            self.task = task

        # 得到选择特征数量是多少个
        def getNumberOfSolution(self):
            # 得到选择的组合里面 为1 的索引为多少个
            index = np.where(self.featureOfSelect == 1)[0]
            # 存放每一个被选择的索引索引
            self.indexOfSolution = np.copy(index)
            # 得到的索引的长度是多少
            self.numberOfSolution = len(index)
            # 更新冗余规则
            self.isNotRedundance = False

    # 生成下一代种群
    def getNextPopulation(self, population ,task):
        # 用于存储每一个个体的 error ，用于后面的排序操作
        saveFitness = np.asarray([])
        # 用于存储每一个个体的 len ，用于后面的排序操作
        saveLength = np.asarray([])
        acc1 = 0
        # 计算种群中每一个个体的fitness
        for i in range(len(population)):
            # 得到第一个个体
            individual_i = population[i]
            # 计算其 acc
            acc = 1 - individual_i.mineFitness
            acc1 += acc
            # 将每一个个体的fitness添加到存储数组中去
            saveFitness = np.append(saveFitness, 1 - acc)
            # 将每一个个体的length添加到存储数组中去
            saveLength = np.append(saveLength, individual_i.numberOfSolution)

        #print("mean--acc = ", acc1 / self.populationNum)


        # 将该数组内的数据进行一个升序排序，返回其索引
        errorSortIndex = np.argsort(saveFitness)
        # 将种群按照该索引进行一个排序
        population = population[errorSortIndex]
        # error 和 len 都要进行该排序
        saveFitness = saveFitness[errorSortIndex]
        saveLength = saveLength[errorSortIndex]

        # 输出前三个最优fitness对应的个体
        # for k in range(3):
        #     print(1 - population[k].mineFitness," ",population[k].numberOfSolution,)
        # print("获得精英代")

        # 获得精英种群,精英群fitness 和 精英群len
        elitePop, eliteFit, eliteLen = self.getElite(population=population,task=task)

        # 将精英代添加到种群中去 np.vstack 添加到列 ；np.hstack 添加到行
        population = np.concatenate((population, elitePop), axis=0)
        saveFitness = np.concatenate((saveFitness, eliteFit), axis=0)
        saveLength = np.concatenate((saveLength, eliteLen), axis=0)

        # 进行非支配排序
        population = self.nonDominatedSort(parentPopulation=population, fitnessArray=saveFitness,
                                           solutionlengthArray=saveLength)
        return population


    # 精英种群的计算
    def getElite(self, population,task):
        # 保存精英代个体的fitness
        fitnessE = np.asarray([])
        # 保存精英代个体的len
        lengthE = np.asarray([])
        # 提取fitness前10%个样本进行筛选
        rate = self.eliteProb
        # 精英种群
        elitePopulation = np.asarray([])
        # 临时存放种群
        tempPop = np.asarray([])
        # 抽取前rate的个体  --- 这些个体都得是 存在冗余的，如果不在冗余了则会isNotRedundance==Ture，则跳过该个体
        eliteNum = int(len(population) * rate)
        eIndex = 0
        while tempPop.shape[0] < eliteNum:

            if population[eIndex].isNotRedundance :
                eIndex += 1
            else:
                tempPop = np.append(tempPop, population[eIndex])
                eIndex += 1
            # 如果全部都是未冗余的又或者是到达最后了，并且没有选完，就随机选取
            if eIndex == len(population):
                tempArray = sample(population=population.tolist(), k=eliteNum - tempPop.shape[0])
                tempPop = np.append(tempPop, tempArray)
        # for i in range(int(len(population) * rate)):
        #     individual_i = population[i]
        #     tempPop = np.append(tempPop, individual_i)
        # 进行筛选特征，生成子代
        elitePopulation = self.removeIrralevantFeature(population=tempPop,task=task,ePro=self.eliteProb)
        # 计算每一个精英个体的fitness
        for i in range(len(elitePopulation)):
            # 获得第i个个体
            individual_e = elitePopulation[i]
            # 保存其fitness到数组中
            fitnessE = np.append(fitnessE, individual_e.mineFitness)
            # 保存其length到数组中
            lengthE = np.append(lengthE, individual_e.numberOfSolution)

        return elitePopulation, fitnessE, lengthE

    # 用于精英代,将个体的前10% 提取出来，然后删减其中特征
    def removeIrralevantFeature(self, population,task,ePro):
        # 用于保存精英代种群
        elitePopulation = np.asarray([])
        # 基因池，用于保存每一组solution中组合较好的特征
        genePool = np.zeros(len(self.dataFeature))
        # 先将F_i中的个体选择的特征提取出来，挡在一个二维的矩阵中去，方便计算
        solutionArray_Binary = [[] for i in range(len(population))]
        #  保存每一个个体所选择特征之间的相关性
        mrmrOfFeature = np.zeros((len(population), len(self.dataFeature)))
        for i in range(len(population)):
            solutionArray_Binary[i] = (population[i].featureOfSelect)
            # 进行筛减特征 --- 通过mrmr 计算两两特征之间的冗余程度
            # 获得选择中为1 的特征的索引 np.where返回了两个类型，一个是索引数组，一个是数据类型
            selectIndex = np.where(population[i].featureOfSelect == 1)[0]
            # 计算两个特征之间的信息增益
            for i1 in range(0, len(selectIndex)):
                id1 = selectIndex[i1]

                f1 = self.dataX[:, id1]
                # 将每一个ig累加起来，用于求平均数
                sum_f = 0
                for i2 in range(i1 + 1, len(selectIndex)):
                    id2 = selectIndex[i2]
                    if id1 != id2:
                        f2 = self.dataX[:, id2]
                        if self.mrmrArray[id1][id2] == 0:
                            # 计算不同特征之间的信息增益 ， 动态存储数组，两两特征之间的值进行保存，如果下次计算有这两个特征则直接调用，不需要计算了
                            ig = information_gain(f1, f2)
                            self.mrmrArray[id1][id2] = ig
                            self.mrmrArray[id2][id1] = ig
                        else:
                            ig = self.mrmrArray[id1][id2]

                        sum_f += ig
                # 计算得到 mrmr值   =  feature_i 和 class的互信息 - feature_i和其他所有的feature的互信息的平均值
                # 平均数
                mean = sum_f / (len(selectIndex) - 1 + 0.01)
                mrmrOfFeature[i][id1] = self.mutualInforArray[id1] - mean

        # 通过计算mrmr后的每一个feature的值，来决定每个特征的去还是留
        # 每一个个体需要产生的子代数量
        eachIndSubPopNum =int( 1 / ePro)

        # 提取mrmr计算后每一个大于0的特征索引
        for i in range(len(mrmrOfFeature)):

            # 将mrmr评分大于0的特征位置设置为1
            genePool = np.where(mrmrOfFeature[i] > 0, 1, 0)

            # 边界，即并未在genePool 中未1 的有一定概率也可以置为一
            #front = np.exp(-len(np.where(genePool == 1)[0]) / len(self.dataFeature)) * 0.2
            front = self.eliteFront

            # 生成精英代
            for j in range(int(eachIndSubPopNum)):
                # 初始化一个个体
                individual_elite = self.Chromosome(mfea=self, mode="elite",task=task)
                # 生成一个数组用于保存选择的特征
                solution1_B = np.zeros(len(self.dataFeature))
                solution1_N = np.zeros(len(self.dataFeature))
                solution2_B = np.zeros(len(self.dataFeature))
                solution2_N = np.zeros(len(self.dataFeature))
                solution3_B = np.zeros(len(self.dataFeature))
                solution3_N = np.zeros(len(self.dataFeature))

                for k in range(len(genePool)):
                    rand = random.random()
                    if (genePool[k] == 1):
                        # 0.2 > 0.3 > 0.4
                        if (rand > 0.1):
                            solution1_B[k] = 1
                            solution1_N[k] = np.random.uniform(0,0.8)
                        else:
                            solution1_N[k] = np.random.uniform(0.8,1)
                    else:
                        if rand < front:
                            solution1_B[k] = 1
                            solution1_N[k] = np.random.uniform(0,0.8)
                        else:
                            solution1_N[k] = np.random.uniform(0.8,1)

                # 取反操作,逆置
                for k in range(len(genePool)):
                    rand = random.random()
                    if (genePool[k] == 1):
                        if (rand > self.scoreOfRelifF[k]):
                            solution2_B[k] = 1
                            solution2_N[k] = np.random.uniform(0,self.scoreOfRelifF[k])
                        else:
                            solution2_N[k] = np.random.uniform(self.scoreOfRelifF[k],1)
                    else:
                        if (rand < front):
                            solution2_B[k] = 1
                            solution2_N[k] = np.random.uniform(0, self.scoreOfRelifF[k])
                        else:
                            solution2_N[k] = np.random.uniform(self.scoreOfRelifF[k], 1)

                for k in range(len(genePool)):
                    rand = random.random()
                    if (genePool[k] == 1):
                        if (rand > self.probabilityOfFeatureToSelect[k]):
                            solution3_B[k] = 1
                            solution3_N[k] = np.random.uniform(0,self.probabilityOfFeatureToSelect[k])
                        else:
                            solution3_N[k] = np.random.uniform(self.probabilityOfFeatureToSelect[k],1)
                    else:
                        if (rand < front):
                            solution3_B[k] = 1
                            solution3_N[k] = np.random.uniform(0, self.probabilityOfFeatureToSelect[k])
                        else:
                            solution3_N[k] = np.random.uniform(self.probabilityOfFeatureToSelect[k], 1)

                # 获得类数据
                feature_y = self.dataY
                # 获得该个体的特征组合的 数据集合
                if len(np.where(solution1_B == 1)[0]) == 0:
                    acc1 = 0
                else:
                    feature_x = self.dataX[:, solution1_B == 1]
                    acc1 = fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=feature_y, CV=5)
                if len(np.where(solution2_B == 1)[0]) == 0:
                    acc2 = 0
                else:
                    feature_x = self.dataX[:, solution2_B == 1]
                    acc2 = fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=feature_y, CV=5)
                if len(np.where(solution3_B == 1)[0]) == 0:
                    acc3 = 0
                else:
                    feature_x = self.dataX[:, solution3_B == 1]
                    acc3 = fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=feature_y, CV=5)

                len1 = len(np.where(solution1_B == 1)[0])
                len2 = len(np.where(solution2_B == 1)[0])
                len3 = len(np.where(solution3_B == 1)[0])

                # 生成矩阵来按照acc*0.9 + (1-(length/len(self.dataFeature))) * 0.1进行排序
                tempArray_B = np.asarray([solution1_B, solution2_B, solution3_B])
                tempArray_N = np.asarray([solution1_N, solution2_N, solution3_N])
                tempArray_acc = np.asarray([acc1, acc2, acc3])
                tempArray_len = 1 - np.asarray([len1,len2,len3])/len(self.dataFeature)
                tempArray_score = [[] for i in range(len(tempArray_acc))]
                # 进行计算排序规则， 0.9的acc 和 0.1的未选择长度
                for m in range(len(tempArray_acc)):
                    tempArray_score[m] = tempArray_acc[m] * 0.9 + tempArray_len[m] * 0.1
                # 通过上述的score进行排序--- 进行降序排序，评分越高越靠前
                tempIndex = np.argsort(tempArray_score)[::-1]
                # 将acc , 1-len, B, N 按照上述规则进行排序
                tempArray_B = tempArray_B[tempIndex]
                tempArray_N = tempArray_N[tempIndex]
                tempArray_acc = tempArray_acc[tempIndex]

                # 将按照score排序后的第一个B，N还有acc都保存起来，放到新个体中去
                solution_B = tempArray_B[0]
                solution_N = tempArray_N[0]
                error = 1 - tempArray_acc[0]

                # 将已选择的特征赋值传给初始化的个体
                individual_elite.featureOfSelect = solution_B
                individual_elite.featureOfSelectNum = solution_N
                # 将acc传给初始化的个体
                individual_elite.mineFitness = error
                # 得到本个体所选特征数量和索引
                individual_elite.getNumberOfSolution()
                # 将该个体添加到精英种群
                elitePopulation = np.append(elitePopulation, individual_elite)

        return elitePopulation


    # 计算每一个节点的帕累托  非支配排序   传入的是归一化后的 error 和 solutionlength
    def nonDominatedSort(self, parentPopulation, fitnessArray, solutionlengthArray):
        # 用于保存下一代解
        springsonPopulation = np.asarray([])

        # 用来保存每一个个体的函数值
        errorArray = np.asarray([])
        lenArray = np.asarray([])

        # 用于保存每一个个体的支配解集
        dominateFrontArray = []

        # 保存帕累托前沿,保存非支配数量为0的个体,
        F_i = []
        # 得到每一个个体转化后的 error  和 len
        errorArray, lenArray = self.sortFunction(error=fitnessArray, solutionLength_test=solutionlengthArray)

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
        numPop = self.populationNum
        # 计数
        count = 0
        while len(springsonPopulation) < self.populationNum:
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
                    crossLen[k] = front_i[k].numberOfSolution
                crossError, crossLen = self.sortFunction(error=crossError, solutionLength_test=crossLen)
                # 计算拥挤度
                front_i, distance = self.crowdingDistance(F_i=front_i, error=crossError, lens=crossLen)
                # 从大到小排序
                sort_distance_index = np.argsort(distance)[::-1]
                # 还需要添加的个体数量
                for k in range(resideNum):
                    springsonPopulation = np.append(springsonPopulation, front_i[sort_distance_index[k]])
            count += 1

        # 保存这一代种群中的最优解
        array_score = np.zeros(len(dominateFrontArray[0]))
        for m in range(len(array_score)):
            array_score[m] = (1 - dominateFrontArray[0][m].mineFitness) * 0.9
            + (1 - dominateFrontArray[0][m].numberOfSolution / len(self.dataFeature)) * 0.1
        # 降序排序
        score_index = np.argsort(array_score)[::-1]
        # 跟新
        array_score = array_score[score_index]
        array_individual = np.asarray(dominateFrontArray[0])
        array_individual = array_individual[score_index]

        if self.globalScore <= array_score[0]:
            self.globalScore = array_score[0]
            self.globalFitness = array_individual[0].mineFitness
            self.globalSolution = array_individual[0].featureOfSelect
            self.globalSolutionNum = array_individual[0].featureOfSelectNum


        return springsonPopulation

    # 拥挤度计算
    def crowdingDistance(self, F_i, error, lens):
        # 拥挤度距离
        crossedDistance = np.zeros(len(F_i))
        # 将两端的距离设置为 inf
        crossedDistance[0] = float(np.inf)
        crossedDistance[-1] = float(np.inf)

        # 计算每一个样本的拥挤度
        for i in range(1, len(F_i) - 1):
            crossedDistance[i] = crossedDistance[i] + (error[i + 1] - error[i - 1]) / (max(error) - min(error) + 0.01)

        for i in range(1, len(F_i) - 1):
            crossedDistance[i] = crossedDistance[i] + np.abs(lens[i + 1] - lens[i - 1]) / (max(lens) - min(lens) + 0.01)
        return F_i, crossedDistance

    # 帕累托计算函数  ----> 1/exp(x)  1/exp(y)  [1/e,1]   -----> x == error : y == length#
    # 帕累托计算函数  ----> exp(x)  exp(y)  [1,e]   -----> x == error : y == length#
    def sortFunction(self, error, solutionLength_test):

        # 计算  e的 x次 和 e的y次
        error = np.exp(error)
        # 将长度放缩到 [0,1]之间
        solutionLength = solutionLength_test / len(self.dataFeature)
        # 计算  e的 x次 和 e的x次
        solutionLength = np.exp(solutionLength)
        return error, solutionLength

    # 计算rmp矩阵
    def computeRmp(self):
        # 任务数量
        k = 2
        # 因为是两个任务，所以就设置一个二维的矩阵，该矩阵中主对角线都为1 其余都为0
        rmp_matrix = np.eye(k)
        # 保存每一个任务的mean和std
        ms_matrix = [[] for i in range(k)]
        # 长度
        l_matrix = np.ones(2)
        # 每一个任务数组的数据
        task1 = []
        task2 = []

        # 提取两个task种群中前50%个个体
        for i in range(int(self.populationNum * 0.5)):
            task1.append(self.population_task1[i].featureOfSelectNum)
            task2.append(self.population_task2[i].featureOfSelectNum)
        subpops = np.asarray([task1, task2])

        for i in range(k):
            subpop = subpops[i]
            num_sample = len(task1)
            num_random_sample = int(np.floor(0.1 * num_sample))
            rand_pop = np.random.rand(num_random_sample, len(self.dataFeature))
            mean = np.mean(np.concatenate([subpop, rand_pop]), axis=0)
            std = np.std(np.concatenate([subpop, rand_pop]), axis=0)
            ms_matrix[i].append(mean)
            ms_matrix[i].append(std)
            l_matrix[i] = num_sample

        for k_i in range(k - 1):
            for j in range(k_i + 1, k):
                probmatrix = [(np.ones([int(l_matrix[k_i]), 2])),
                              (np.ones([int(l_matrix[j]), 2]))]
                probmatrix[0][:, 0] = self.density(subpop=subpops[k_i], mean=ms_matrix[k_i][0], std=ms_matrix[k_i][1])
                probmatrix[0][:, 1] = self.density(subpop=subpops[j], mean=ms_matrix[k_i][0], std=ms_matrix[k_i][1])
                probmatrix[1][:, 0] = self.density(subpop=subpops[k_i], mean=ms_matrix[j][0], std=ms_matrix[j][1])
                probmatrix[1][:, 1] = self.density(subpop=subpops[j], mean=ms_matrix[j][0], std=ms_matrix[j][1])
                # fminbound找最小值用的是黄金分割法  0.382  0.618
                # scipy.optimize.fminbound(fun(),x,y) 在fun()中，给定范围内[x,y]里找到fun()函数值的最小值
                rmp = fminbound(lambda rmp: self.log_likelihood(rmp, probmatrix, k), 0, 1)
                rmp += np.random.randn() * 0.01
                # rmp = rmp * 0.5
                rmp = np.clip(rmp, 0, 1)  # 将rmp的数据限制在[0,1]之间，小于0则等于0，大于1则等于1
                rmp_matrix[k_i, j] = rmp
                rmp_matrix[j, k_i] = rmp


        return rmp_matrix

    # 计算密度
    def density(self, subpop, mean, std):
        N, D = subpop.shape
        prob = np.ones([N])
        for d in range(D):
            prob *= norm.pdf(subpop[:, d], loc=mean[d], scale=std[d])
        return prob

    # 计算
    def log_likelihood(self, rmp, prob_matrix, K):
        posterior_matrix = copy.deepcopy(prob_matrix)
        value = 0
        for k in range(2):
            for j in range(2):
                if k == j:
                    posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * (1 - 0.5 * (K - 1) * rmp) / float(K)
                else:
                    posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * 0.5 * (K - 1) * rmp / float(K)
            value = value + np.sum(-np.log(np.sum(posterior_matrix[k], axis=1)))
        return value





    # 交配产生下一代  ----- 使用差分进化来进行变异交叉
    def mate_new(self,taskNum):

        copyTask = [[] for i in range(2)]
        copyTask[0] = copy.deepcopy(self.population_task1)
        copyTask[1] = copy.deepcopy(self.population_task2)
        rmp = self.computeRmp()
        # 任务数量
        K = taskNum
        # 用于保存每一个任务的种群
        population = [[] for i in range(K)]
        # 随机选择的索引    长度为 self.populationNum 的一个数组，随机排序
        mateIndex = np.random.permutation(2 * self.populationNum)
        # 将两个任务种群添加到一起
        taskPopulation = np.concatenate((self.population_task1, self.population_task2), axis=0)

        for i in range(len(taskPopulation)):
            # 获得需要进行差分的个体
            ind1 = taskPopulation[mateIndex[i]]
            if ind1 not in self.p_DEarray:
                indMain = np.random.choice(self.p_DEarray)
            else:
                tempArray = np.delete(self.p_DEarray, np.where(self.p_DEarray == ind1)[0])
                # 通过随机找到--基底个体
                indMain = np.random.choice(tempArray)
            # 设置每一个个体的索引
            temp1 = np.arange(self.populationNum)
            xArray = [[] for i in range(2)]

            # 存放两个个体在整体大种群中的索引位置
            index1 = ind1.index
            index2 = indMain.index

            # 若两个个体是属于同一个任务的，则提取一个同任务较差的，然后随机选择一个-----寻优搜索
            # DE/current-to-best/1 ---- 加速收敛
            if (ind1.task == indMain.task):
                if ind1.task == 0:
                    # # 然后在该个体上删除 上面选择的 差分个体  和 基底个体
                    tempA1 = np.delete(self.population_task1, [index1, index2]).tolist()
                    xArray = sample(tempA1, 2)

                elif ind1.task == 1:
                    # # 然后在该个体上删除 上面选择的 差分个体  和 基底个体
                    # # 需要将同一个任务的两个个体都删除，这样接下来的筛选就不会重复选到这两个个体了
                    tempA1 = np.delete(self.population_task2, [index1, index2]).tolist()
                    xArray = sample(tempA1, 2)

                # 进行计算
                # 对第一个个体进行差分
                v_x = mutate_best(xDE=ind1.featureOfSelectNum, xMain=indMain.featureOfSelectNum,
                                  x1=xArray[0].featureOfSelectNum,
                                  x2=xArray[1].featureOfSelectNum, F=self.F)

            # 若是不同任务，并且小于其rmp值，在基底个体的任务中选择两个  or 差的抽一个，好的抽一个 or 差的抽一个 随机抽一个
            elif ind1.task != indMain.task and np.random.rand() < rmp[ind1.task][indMain.task]:

                # 最后一种情况就是在两个优秀任务里面都挑出来一个个体
                tempPop = [[] for i in range(K)]
                tempPop[0] = self.population_task1
                tempPop[1] = self.population_task2
                # 在优秀个体中选取
                if ind1 not in self.p_DEarray:
                    xArray[0] = np.random.choice(tempPop[ind1.task], 1)[0]
                else:
                    # 先找出来ind1任务种群中的随机一个个体
                    tempRange = np.delete(tempPop[ind1.task], ind1.index).tolist()
                    xArray[0] = sample(tempRange, 1)[0]

                # 先找出来ind1任务种群中的随机一个个体, indMain 肯定在self.p_DEarray中选取的
                tempRange = np.delete(tempPop[indMain.task], indMain.index).tolist()
                xArray[1] = sample(tempRange, 1)[0]
                # 进行计算
                # 对第一个个体进行差分 在另一个任务的优秀解周围进行搜索
                v_x = mutate_best(xDE=ind1.featureOfSelectNum, xMain=indMain.featureOfSelectNum,
                                  x1=xArray[0].featureOfSelectNum,
                                  x2=xArray[1].featureOfSelectNum, F=self.F)
            # --- 随机搜索，个体在自己本体种群上找到最优个体--- 扩大搜索范围
            # 随机  DE/rand/1
            else:
                # 随机  DE/rand/1
                tempPop = [[] for i in range(K)]
                tempPop[0] = self.population_task1
                tempPop[1] = self.population_task2

                # 在indMain 种群中选则个体
                tempRange = np.delete(tempPop[indMain.task], indMain.index).tolist()

                if np.random.rand() < 0.8:
                    xArrayN = sample(tempRange, 3)
                    # 进行计算
                    # 对第一个个体进行差分
                    v_x = mutate_rand1(xMain=xArrayN[2].featureOfSelectNum,
                                       x1=xArrayN[0].featureOfSelectNum,
                                       x2=xArrayN[1].featureOfSelectNum,
                                       F=self.F)
                else:
                    # 选择x1,x2

                    xArray = sample(tempRange, 2)
                    v_x = mutate_best(xDE=ind1.featureOfSelectNum, xMain=indMain.featureOfSelectNum,
                                      x1=xArray[0].featureOfSelectNum,
                                      x2=xArray[1].featureOfSelectNum, F=self.F)

            u_x = crossover_num(x=ind1.featureOfSelectNum, v_x=v_x, crossPro=self.crossoverPro)
            # 锁定范围， 超过1的部分就按照超过的来算，低于0的部分就是1 + 负值
            u_x = np.where(u_x >= 1, u_x - 1, u_x)
            u_x = np.where(u_x <= 0, 1 + u_x, u_x)
            # u_x = np.clip(u_x, 0, 1)

            # 计算不同种可能下的acc
            solution_u = np.zeros((3, len(self.dataFeature)))
            solution_acc = np.zeros(3)
            solution_len = np.zeros(3)
            solution_score = np.zeros(3)

            for k in range(len(u_x)):
                #  0.85 挺好用的有些最大值要比rilifF好，就算和relifF相等，但是长度更加的断
                #  0.8 和relifF比较接近，平均acc在0.85，就算和relifF相等，但是长度更加的断，更有效
                if u_x[k] > 0.8:
                    solution_u[0][k] = 1
                if u_x[k] < self.scoreOfRelifF[k]:
                    solution_u[1][k] = 1
                if u_x[k] < self.probabilityOfFeatureToSelect[k]:
                    solution_u[2][k] = 1

            featureY = self.dataY
            # 获的三个solution的acc 分别通过len 和 acc 来计算score
            for u in range(3):
                # 未选取的长度
                solution_len[u] = 1 - len(np.where(solution_u[u] == 1)[0]) / len(self.dataFeature)
                # 为选取的长度未1，则说明选取长度为0，那就直接返回acc = 0
                if solution_len[u] == 1:
                    solution_acc[u] == 0
                else:
                    featureX = self.dataX[:, solution_u[u] == 1]
                    solution_acc[u] = fitnessFunction_KNN_CV(findData_x=featureX, findData_y=featureY, CV=5)
                solution_score[u] = solution_acc[u] * 0.9 + solution_len[u] * 0.1
            # 从高到低分布排序
            solution_index = np.argsort(solution_score)[::-1]
            # 进行重新排序
            solution_score = solution_score[solution_index]
            solution_acc = solution_acc[solution_index]
            solution_len = solution_len[solution_index]
            solution_u = solution_u[solution_index]

            # 获得需要差分的个体的score  ---- ind1.mineFitness 是error所以要减一
            ind1_score = (1 - ind1.mineFitness) * 0.9 + (1 - ind1.numberOfSolution / len(self.dataFeature)) * 0.1

            # 进行比较  -- 让原有的score 和 最高的score进行比较
            if ind1_score <= solution_score[0]:
                # 跟新实数
                ind1.featureOfSelectNum = u_x
                # 跟新二进制
                ind1.featureOfSelect = solution_u[0]
                # 跟新error = 1 - acc
                ind1.mineFitness = 1 - solution_acc[0]
                # 更新数据
                ind1.getNumberOfSolution()
                # 重置时间
                ind1.age = 0
            else:
                ind1.age += 1
                # 不允许超过三代
                if ind1.age == self.maxAge:
                    # print("修减基因")
                    # 进行替换
                    # 效果好
                    ind1 = self.prunGene(individual=ind1)

                    # 效果负提升
                    # ind1 = self.prunGene_new(individual=ind1)
            # 将跟新后的ind1 传回去
            taskPopulation[mateIndex[i]] = ind1

    def prunGene(self, individual):
        # 如果是不冗余了的个体，则年龄直接重置
        if individual.isNotRedundance:
            individual.age = 0
            return individual

        # 获得该个体中为1的基因索引
        geneIndex = np.where(individual.featureOfSelect == 1)[0]
        if len(geneIndex) == 1:
            individual.age = 0
            return individual

        featureY = self.dataY
        # 进行计算
        rule = [[] for i in range(2)]
        # 因为任务1 是pro
        rule[0] = self.probabilityOfFeatureToSelect
        # 任务2 是relifF
        rule[1] = self.scoreOfRelifF

        # 得到信息
        getInfor = rule[individual.task][geneIndex]
        # 对其进行排序,从低到高排序
        tempIndex = np.argsort(getInfor)
        geneIndex = geneIndex[tempIndex]
        getInfor = getInfor[tempIndex]

        # 测试solution
        testSolution = copy.deepcopy(individual.featureOfSelect)
        # 进行遍历
        for i in range(geneIndex.shape[0]):
            isReserve = False
            testSolution[geneIndex[i]] = 0
            featureX = self.dataX[:, testSolution == 1]
            acc = fitnessFunction_KNN_CV(findData_x=featureX, findData_y=featureY, CV=5)
            if acc >= 1 - individual.mineFitness:
                individual.mineFitness = 1 - acc
                individual.featureOfSelect = copy.deepcopy(testSolution)
                individual.getNumberOfSolution()
                individual.featureOfSelectNum[geneIndex[i]] = np.random.uniform(
                    rule[individual.task][geneIndex[i]], 1)
                isReserve = True
                break
            else:
                testSolution[geneIndex[i]] = 1
        # 更新年龄
        individual.age = 0

        if isReserve:
            return individual
        else:
            individual.isNotRedundance = True
            return individual
    # 修减基因
    def prunGene_least(self,individual):
        # 获得该个体中为1的基因索引
        geneIndex = np.where(individual.featureOfSelect == 1)[0]
        if len(geneIndex) == 1:
            individual.age = 0
            return individual

        featureY = self.dataY
        # 进行计算
        rule = [[] for i in range(2)]
        # 因为任务1 是pro
        rule[0] = self.probabilityOfFeatureToSelect
        # 任务2 是relifF
        rule[1] = self.scoreOfRelifF

        # 得到信息
        getInfor = rule[individual.task][geneIndex]
        # 对其进行排序,从低到高排序
        tempIndex = np.argsort(getInfor)
        geneIndex = geneIndex[tempIndex]
        getInfor = getInfor[tempIndex]

        # 测试solution
        testSolution = copy.deepcopy(individual.featureOfSelect)
        # 标记是否结束
        noDone = True
        # 进行遍历
        while noDone:
            for i in range(geneIndex.shape[0]):
                testSolution[geneIndex[i]] = 0
                featureX = self.dataX[:, testSolution == 1]
                acc = fitnessFunction_KNN_CV(findData_x=featureX, findData_y=featureY, CV=5)
                if acc >= 1 - individual.mineFitness:
                    individual.mineFitness = 1 - acc
                    individual.featureOfSelect = copy.deepcopy(testSolution)
                    individual.getNumberOfSolution()
                    individual.featureOfSelectNum[geneIndex[i]] = np.random.uniform(rule[individual.task][geneIndex[i]],
                                                                                    1)
                    # 删除索引
                    geneIndex = np.delete(geneIndex,i)
                    #print("存在提升")
                    break
                else:
                    testSolution[geneIndex[i]] = 1
                    if i == geneIndex.shape[0] - 1:
                        noDone = False

        #print( 1 - individual.mineFitness)
        return individual



    # 得到当代最优的前self.DE_p个种群
    def getBestPop(self):
        #DE_p = np.random.uniform(0.1,0.3)
        DE_p = self.p
        # K 为多少个任务
        array1 = self.population_task1[:int(self.populationNum * DE_p)]
        array2 = self.population_task2[:int(self.populationNum * DE_p)]
        self.p_DEarray = np.concatenate((array1,array2),axis=0)

    # 最后的精炼
    def leastRefine(self,population1,population2):
        # 用于保存两个任务种群中 前6个acc最高的个体, 是存在冗余个体isNotRedundance == False
        tempPopulation = np.asarray([])
        # 第一个任务的索引
        index1 = 0
        # 第二个任务的索引
        index2 = 0
        # 添加两个任务中acc总排名中的前4个个体
        while tempPopulation.shape[0] < 4:
            # 判断是否是非冗余的，如果是则跳过；如果是冗余的，那么就要到下一步
            while index1 < population1.shape[0] and population1[index1].isNotRedundance:
                index1 += 1
            while index2 < population2.shape[0] and population2[index2].isNotRedundance:
                index2 += 1
            # 如果两个都到达了底端，那么就直接推出进行净化
            if index1 == population1.shape[0] and index2 == population2.shape[0]:
                break
            # 如果有一个到达数组底端了之后，另一个没有，那么直接添加另一个
            if index1 == population1.shape[0]:
                tempPopulation = np.append(tempPopulation, population2[index2])
                index2 += 1
            elif index2 == population2.shape[0]:
                tempPopulation = np.append(tempPopulation, population1[index1])
                index1 += 1
            else:
                # 如果两个都没到达底端
                if population1[index1].mineFitness < population2[index2].mineFitness:
                    tempPopulation = np.append(tempPopulation,population1[index1])
                    index1 += 1
                else:
                    tempPopulation = np.append(tempPopulation,population2[index2])
                    index2 += 1

        for i in range(tempPopulation.shape[0]):

            tempPopulation[i] = self.prunGene_least(individual=tempPopulation[i])
            # 进行比较新的个体
            # 如果分数高于全局最优的score，那么就更新
            score = tempPopulation[i].mineFitness * 0.9 + (
                    1 - tempPopulation[i].numberOfSolution / len(self.dataFeature)) * 0.1
            if score > self.globalScore:
                self.globalScore = score
                self.globalSolution = tempPopulation[i].featureOfSelect
                self.globalFitness = tempPopulation[i].mineFitness
                self.globalSolutionNum = tempPopulation[i].featureOfSelectNum

    # 运行
    def goRun(self):
        # 初始化种群
        #print("初始化种群")
        self.initPopulation()
        #print("进行迭代")
        # 运行次数
        runTime = 0
        import time
        start = time.time()
        while runTime < self.iteratorTime:
            #print(" 第",runTime + 1,"轮")
            # 任务1的进化
            #print("task1")
            self.population_task1 = self.getNextPopulation(population=self.population_task1,task=0)
            #print(f"task1 time = {time.time() - start} seconds")
            # 任务2的进化
            #print("task2")
            self.population_task2 = self.getNextPopulation(population=self.population_task2,task=1)
            #print(f"task2 time = {time.time() - start} seconds")
            # 得到当代最优
            self.getBestPop()
            # 变异交叉
            self.mate_new(taskNum=2)

            runTime += 1

        self.leastRefine(population1=self.population_task1,population2=self.population_task2)

        tempPop = np.concatenate((self.population_task1, self.population_task2), axis=0)
        f_0 = getParato(pop=tempPop, popNum=self.populationNum*2)
        # 写入文件的路径
        path_txt = "../result/paretoSolution/EG_MFDE/" + self.dataName + ".txt"
        # 写入时间
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

        print(f"all time = {time.time() - start} seconds")


if __name__ == '__main__':
    import time
    files = os.listdir("../dataCSV/dataCSV_high")
    for file in files:
        ducName = dataName = file.split('.')[0]
        path_csv = "../dataCSV/dataCSV_high/" + ducName + ".csv"

        # 写入文件的路径
        path_txt = "../result/result_txt/EG_MFDE/" + ducName + ".txt"
        # 写入文件的标题
        titleTxt = '   ' + ducName + '.csv '+ str(time.ctime()) +'\n'
        # 将结果写入到文件中去
        with open(path_txt, 'a') as f:
            f.write(titleTxt)
            f.close()

        dataCsv = ReadCSV(path=path_csv)
        print("获取文件数据", file)
        dataCsv.getData()
        genetic = MFMODE(dataX=dataCsv.dataX, dataY=dataCsv.dataY,dataName=ducName)

        genetic.setParameter(populationNum=70, iteratorTime=100, eliteProb=0.1,crossoverPro=0.9
                             ,F=0.5,eliteFront = 0.01,p = 0.3,maxAge=9)

        # 循环多少次
        iterateT = 10

        acc = np.zeros(iterateT)
        length = np.zeros(iterateT)

        start = time.time()
        for i in range(iterateT):
            #print("第",i,"次")
            genetic.goRun()
            acc[i] = 1 - genetic.globalFitness
            length[i] = len(np.where(genetic.globalSolution == 1)[0])
            # 重置
            genetic.population_task1 = np.asarray([])
            genetic.population_task2 = np.asarray([])
            genetic.globalScore = 0


            # 写入文件的值
            stringOfResult = str(acc[i]) + '\t' + str(length[i]) + '\n'
            # 将结果写入到文件中去
            with open(path_txt, 'a') as f:
                f.write(stringOfResult)
                f.close()
            print(acc[i], " ", length[i])

        # 向文件中写入均值
        # 写入文件的值
        stringOfResult = str(acc.mean()) + '\t'+str(acc.std()) + '\t' + str(length.mean()) + '\t'+str(length.std())+'\n'
        # 将结果写入到文件中去
        with open(path_txt, 'a') as f:
            f.write('   mean  \n')
            f.write(stringOfResult)
            f.close()

        print("acc:",acc.mean(),"  std:",acc.std(),"  len:",length.mean(),"  std:",length.std())
        # 将结果写入到文件中去
        with open(path_txt, 'a') as f:
            f.write('\n')
            f.write(f"all time = {time.time() - start} seconds" + '\n')
            f.close()
        print(f" time = {time.time() - start} seconds")
        print("===================================")
        print(" ")

