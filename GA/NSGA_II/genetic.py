import copy

import numpy as np
import random

from Filter.filter import FilterOfData
from dataProcessing.ReadDataCSV_new import ReadCSV
from dataProcessing.NonDominate import nonDominatedSort,getPF
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from random import sample
from sklearn.preprocessing import MinMaxScaler



# 遗传算法  ----> NSGA_II  非支配排序遗传算法--II
class Genetic:
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

    # 将种群中的solution提取出来
    solutionArray = []
    # 任务1 的种群
    population_EA = np.asarray([])

    # 每一个种群的数量
    populationNum = 0
    # 种群迭代次数
    iteratorTime = 0

    # 交叉率
    crossoverPro = 0
    # 变异率
    mutatePro = 0
    # 差分进化中变异操作的F 缩放因子
    F = 0
    # limit the number of selected feature
    eta = 0

    # 全局最优解
    globalSolution = np.asarray([])
    # 全局最优解的实数制
    globalSolutionNum = np.asarray([])
    # 全局最优解的适应度值
    globalFitness = 1
    # 全局最优的score
    globalScore = 0

    # 保存数据名字
    dataName = ""


    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.dataFeature = np.arange(dataX.shape[1])
        self.dataFeatureNum = len(self.dataFeature)
        # 对数据01 归一化，将每一列的数据缩放到 [0,1]之间
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        print("进行filter，提取一部分数据")
        # filter operator
        filter = FilterOfData(dataX=self.dataX, dataY=self.dataY)

        #  compute relifF score
        self.scoreOfRelifF = filter.computerRelifFScore()
        # 进行filter过滤数据 ----  默认采用的是膝节点法
        self.scoreOfRelifF, self.dataX, self.dataFeature = filter.filter_relifF(
            scoreOfRelifF=self.scoreOfRelifF, dataX=self.dataX, dataFeature=self.dataFeature)

        self.mutatePro = 1/self.dataX.shape[1]

    # 设置参数
    def setParameter(self,crossoverPro,populationNum,iteratorTime,eta):
        self.crossoverPro = crossoverPro
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime
        self.eta = eta

    # 初始化染色体种群
    def initChromosomeList(self):
        for i in range(0,self.populationNum):
            chromosome = self.Chromosome(moea=self,mode="init")
            self.population_EA = np.append(self.population_EA,chromosome)


# 染色体
    class Chromosome:
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



        def __init__(self,moea, mode):
            if mode == "init":
                self.initOfChromosome(MFEA=moea)
            elif mode == "iterator":
                self.iteratorOfChromosome(MFEA=moea)

        def iteratorOfChromosome(self, MFEA):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()
            #

        def initOfChromosome(self, MFEA):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            # 首先随机初始化一个特征长度的随机数组
            rand = np.random.random(len(MFEA.dataFeature))
            for i in range(0, len(MFEA.dataFeature)):
                # 获取每一个特征上的随机数，通过这个随机数来判断初始化是否选择该特征
                if rand[i] > MFEA.eta:
                    self.featureOfSelect[i] = 1
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()
            # 计算acc
            feature_x = MFEA.dataX[:, self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=MFEA.dataY, CV=10)

        # 得到选择特征数量是多少个
        def getNumberOfSolution(self):
            # 得到选择的组合里面 为1 的索引为多少个
            self.indexOfSolution = np.where(self.featureOfSelect == 1)[0]
            # 得到的索引的长度是多少
            self.numberOfSolution = self.indexOfSolution.shape[0]
            # 获得比例
            self.proportionOfFeature = self.numberOfSolution / self.featureOfSelect.shape[0]

    #   competition
    def selectOperater(self):
        elitePopulation = np.asarray([])
        # select number
        selectNum = int(0.1 * self.populationNum)

        for i in range(self.populationNum):
            selectCompetitor = sample(self.population_EA.tolist(),selectNum)
            scoreArray = np.zeros(selectNum)
            for i in range(selectNum):
                scoreArray[i] = (selectCompetitor[i].mineFitness) * 0.9 + (selectCompetitor[i].proportionOfFeature) * 0.1
            errIndex = np.argsort(scoreArray)
            elitePopulation = np.append(elitePopulation,copy.deepcopy(selectCompetitor[errIndex[0]]))

        return elitePopulation

    # 交叉算子  使用均匀交叉，使得子代拥有极高的多样性,需要传入种群
    def crossoverOperate(self, solution1, solution2):
        # 交叉算子3  --- 均匀交叉
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

    # 变异算子
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
    def reserveAboutMutate(self,value):
        y = -value + 1
        return  y


    # get spring populaiton
    def getSpringPopulation(self):

        errArray = np.zeros(int(2 * self.populationNum))
        lenArray = np.zeros(int(2 * self.populationNum))


        # select
        elitePopulation = self.selectOperater()

        # cross
        crossoverOrder = np.random.permutation(self.populationNum)

        number = 0
        while number < self.populationNum:
            self.crossoverOperate(solution1=elitePopulation[crossoverOrder[number]].featureOfSelect,
                                  solution2=elitePopulation[crossoverOrder[number + 1]].featureOfSelect)
            number += 2
        # mutate
        for i in range(self.populationNum):
            self.mutateOperator(individual=elitePopulation[i])

        # combine Population
        combinePop = np.concatenate([self.population_EA,elitePopulation])
        for i in range(combinePop.shape[0]):
            errArray[i] = combinePop[i].mineFitness
            lenArray[i] = combinePop[i].numberOfSolution

        # 输出fitness前3的三个个体
        tempA = errArray[:self.populationNum]
        tempB = lenArray[:self.populationNum]
        sortErrIndex = np.argsort(tempA)
        print("================================================")
        for i in range(0,3):
            print("acc= ",1 - tempA[sortErrIndex[i]], "  len =", tempB[sortErrIndex[i]])
        # nonDominate
        # parentPopulation, fitnessArray, solutionlengthArray,dataFeature,populationNum
        sprintPopulation,score,individual = nonDominatedSort(parentPopulation=combinePop,fitnessArray=errArray,solutionlengthArray=lenArray,
                         dataFeature=self.dataFeature,populationNum=self.populationNum)

        # 跟新全局最优个体
        if self.globalScore <= score:
            self.globalScore = score
            self.globalFitness = individual.mineFitness
            self.globalSolution = individual.featureOfSelect

        return sprintPopulation


    # 运行
    def run(self):
        import time
        start = time.time()
        # 初始化种群
        print("初始化种群")
        self.initChromosomeList()
        # 运行时间
        runtime = 0
        while runtime < self.iteratorTime:
            print("第", runtime + 1, "轮")
            self.population_EA = self.getSpringPopulation()
            runtime = runtime + 1
            # pltForWin(iteNum=runtime, pop=self.population_EA, iteratorTime=self.iteratorTime, name="NSGA-II")



        print("================================================")
        print(f"all time = {time.time() - start} seconds")
        print(1 - self.globalFitness,len(np.where(self.globalSolution == 1)[0]))
        print()
        # 得到最终前沿
        pf = getPF(pop=self.population_EA)
        for i in range(len(pf)):
            print(1 - pf[i].mineFitness, " ", pf[i].numberOfSolution)
        # # 画出前沿
        # pltForWin(iteNum=runtime,pop=pf,iteratorTime=self.iteratorTime,name = "NSGA-II")

        print("####################################")
        print("populationNum = ", self.populationNum)
        print("iteratorTime = ", self.iteratorTime)
        print("crossoverPro = ", self.crossoverPro)
        print("mutatePro = ", self.mutatePro)
        print("eta = ", self.eta)
        print(" ")





if __name__ == "__main__":
    ducName = "BreastCancer1"
    # ducName = "CLLSUB"
    path = "D:/MachineLearningBackUp/dataCSV/dataCSV_high/" + ducName + ".csv"
    dataCsv = ReadCSV(path=path)
    print("获取文件数据")
    dataCsv.getData()
    genetic = Genetic(dataX=dataCsv.dataX,dataY=dataCsv.dataY)
    genetic.setParameter(crossoverPro=0.9,populationNum=100,iteratorTime=100,eta=0.6)
    genetic.run()