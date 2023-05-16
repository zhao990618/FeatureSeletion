# 基于HV的多目标差分进化--用聚类思想来分配子种群
import copy
import math
import random


import numpy as np
from random import sample

from Filter.filter import FilterOfData
from dataProcessing.InforGain import InforGain
from dataProcessing.ReadDataCSV_new import ReadCSV
from dataProcessing.NonDominate import getPF, nonDominatedSort_PFAndPop, nonDominatedSort
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from Classfier.hv import HyperVolume
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV

class MOCDE:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])

    # 特征的存放数组
    dataFeature = np.asarray([])
    # 特征的数量
    dataFeatureNum = 0
    # 用于k-mean聚类
    k = 0
    # 迭代fGroup次后重新进行聚类
    fGroup = 0
    # 每一个特征的relifF评分
    scoreOfRelifF = np.asarray([])
    # DE种群
    population_DE = np.asarray([])
    # 聚类种群
    population_cluster = []
    # 聚类种群存档
    archive_cluster = []
    # 聚类种群停滞
    stagnant_cluster = np.asarray([])
    # 子种群的平均err
    meanErr_cluster = np.asarray([])
    # 全局存档
    archive_global = np.asarray([])
    # 每一个种群的数量
    populationNum = 0
    # 种群迭代次数
    iteratorTime = 0
    # 交叉率
    crossoverPro = 0
    # 差分进化中变异操作的F 缩放因子
    F = 0
    # 互信息矩阵
    mutualInforArray = np.asarray([])
    # 全局最优解
    globalSolution = np.asarray([])
    # 全局最优解的实数制
    globalSolutionNum = np.asarray([])
    # 全局最优解的适应度值
    globalFitness = 1
    # 全局最优的score
    globalScore = 0
    # 聚类索引
    clusterIndex = np.asarray([])
    # 保存数据名字
    dataName = ""

    # initialization function
    def __init__(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.dataFeature = np.arange(self.dataX.shape[1])
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
        # 获得互信息
        infor = InforGain(dataAttribute=self.dataFeature, dataX=self.dataX)
        self.mutualInforArray = infor.getMutualInformation_fc(dataX=self.dataX,dataY=self.dataY)
        # self.k = int(np.power(self.dataFeature.shape[0],0.5))
        self.k = 10
        # 分配每一个子种群
        self.population_cluster = [[] for i in range(self.k)]
        # 子档案
        self.archive_cluster = [[] for i in range(self.k)]
        # 子种群的平均err停滞变化次数
        self.stagnant_cluster = np.zeros(self.k)
        # 子种群的平均err
        self.meanErr_cluster = np.zeros(self.k)
        # 获取特征数量
        self.dataFeatureNum = len(self.dataFeature)
    # initialization parameters
    def  setParameter(self,populationNum,iteratorTime,crossoverPro,F,fGroup):
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime
        self.crossoverPro = crossoverPro
        self.F = F
        self.fGroup = fGroup

# 染色体
    class Chromosome:

        # 聚类标签
        clIndex = 0
        # HV 贡献
        HVDistribute = 0

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
        #
        def __init__(self, mocde, mode ,index=np.asarray([])):
            if mode == "init_1":
                self.initOfChromosome_1(MOCDE=mocde,index=index)
            elif mode == "init_2":
                self.initOfChromosome_2(MOCDE=mocde)
            elif mode == "iterator":
                self.iteratorOfChromosome(MOCDE=mocde)


        def iteratorOfChromosome(self,MOCDE):
            self.featureOfSelect = np.zeros(len(MOCDE.dataFeature))
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()

        # 基于MI 或者 MIC的初始化 -- 1
        def initOfChromosome_1(self, MOCDE ,index):
            # 产生个体
            self.featureOfSelect = np.zeros(len(MOCDE.dataFeature))
            self.featureOfSelectNum = np.zeros(len(MOCDE.dataFeature))
            self.featureOfSelect[index] = 1
            for i in range(0, len(MOCDE.dataFeature)):
                if self.featureOfSelect[i] == 1:
                    self.featureOfSelectNum[i] = np.random.uniform(MOCDE.mutualInforArray[i], 1)
                else:
                    self.featureOfSelectNum[i] = np.random.uniform(0, MOCDE.mutualInforArray[i])
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()
            # 计算acc
            feature_x = MOCDE.dataX[:, self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=MOCDE.dataY, CV=10)
        # 基于MI 或者 MIC的初始化 -- 2
        def initOfChromosome_2(self, MOCDE):
            self.featureOfSelect = np.zeros(len(MOCDE.dataFeature))
            # 首先随机初始化一个特征长度的随机数组
            rand = np.random.random(len(MOCDE.dataFeature))
            for i in range(0, len(MOCDE.dataFeature)):
                # 获取每一个特征上的随机数，通过这个随机数来判断初始化是否选择该特征
                if rand[i] >= MOCDE.mutualInforArray[i]:
                    self.featureOfSelect[i] = 1
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()
            self.featureOfSelectNum = rand
            # 计算acc
            feature_x = MOCDE.dataX[:, self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=MOCDE.dataY, CV=10)

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

    # 初始化种群
    def initPopulation(self):
        # 先对互信息排序
        tempMIIndex = np.argsort(self.mutualInforArray)
        indexC = 0
        # 种群
        for i in range(0, int(self.populationNum/2)):
            # 初始化1 -- 每一个个体只选择在MI中排名前i个特征
            chromosome = self.Chromosome(mocde=self, mode="init_1",index=tempMIIndex[:i+1])
            chromosome.clIndex = indexC
            indexC += 1
            self.population_DE = np.append(self.population_DE, chromosome)
            # 初始化2 -- 基于MI的随机初始化
            chromosome = self.Chromosome( mocde=self,mode="init_2")
            chromosome.clIndex = indexC
            indexC += 1
            self.population_DE = np.append(self.population_DE, chromosome)

    # 计算欧式距离
    def calculateED(self,solution_1,solution_2):
        dif_s = solution_1 - solution_2
        power_s = np.power(dif_s,2)
        power_s = np.power(power_s.sum(),0.5)
        return  power_s

    # 合并种群
    def conPop(self):
        temp = np.asarray(self.population_cluster[0])
        for i in range(1,self.k):
            temp = np.concatenate((temp,self.population_cluster[i]))
        self.population_DE = temp

    # 聚类产生k个子种群  --- k-means
    def dividePopulation(self,population,k):
        popArray = population[0].featureOfSelectNum
        for i in range(1,self.populationNum):
            popArray = np.vstack((popArray,population[i].featureOfSelectNum))
        kmeans = KMeans(n_clusters=k,random_state=0).fit(popArray)
        self.clusterIndex = kmeans.labels_
        # 根据kmeans计算出来的位置进行分 子种群
        for i in range(self.populationNum):
            # self.population_DE[i].clIndex = self.clusterIndex[i]
            self.population_cluster[self.clusterIndex[i]].append(self.population_DE[i])
        for i in range(self.k):
            self.population_cluster[i] = np.asarray(self.population_cluster[i])

    # 变异算子
    def mutateOperator(self,X_best,X_r1,X_r2):
        V_i = X_best - self.F * (X_r1 - X_r2)
        return V_i

    # 交叉算子
    def crossoverOperator(self,V_i,X_i):
        U_i = np.zeros(self.dataFeatureNum)
        rand_I = np.random.uniform(self.dataFeatureNum)
        for i in range(self.dataFeatureNum):
            rand = np.random.rand()
            if i == rand_I or rand < self.crossoverPro:
                U_i[i] = V_i[i]
            else:
                U_i[i] = X_i[i]
        return U_i

    # 判断传入数组中三个个体哪一个是最优 -- 将最优个体位置返回
    def findBestX(self,mArray):
        # 非支配
        PFArray = np.asarray(getPF(pop=mArray))
        # 判断个体 -- 若只有一个个体
        if PFArray.shape[0] == 1:
            tempIndex = np.zeros(len(mArray))
            for i in range(len(mArray)):
                tempIndex[i] = mArray[i].clIndex
            bestIndex = PFArray[0].clIndex
            indexB = np.where(tempIndex == bestIndex)[0]
            return indexB[0]
        else:
            errArray = np.zeros(PFArray.shape[0])
            for i in range(PFArray.shape[0]):
                errArray[i] = PFArray[i].mineFitness
            # 排序
            errIndex = np.argsort(errArray)
            return errIndex[0]

    # DE 操作
    def updataPopulation(self):
        springPop = np.asarray([])
        for i in range(self.k):
            cl = len(self.population_cluster[i])
            for j in range(cl):

                if self.stagnant_cluster[i] == 5:
                    newTPop = np.concatenate((np.asarray(self.archive_cluster[i]),self.archive_global),axis=0)
                    errTempA = np.zeros(newTPop.shape[0])
                    for m in range(errTempA.shape[0]):
                        errTempA[m] = newTPop[m].mineFitness
                    X_best = newTPop[np.argsort(newTPop)[0]]
                    # 随机抽取三个个体 用于变异
                    mA = sample(self.population_cluster[i].tolist(), 2)
                    X_r1 = mA[0].featureOfSelectNum
                    X_r2 = mA[1].featureOfSelectNum
                else:
                    # 随机抽取三个个体 用于变异
                    mA = sample(self.population_cluster[i].tolist(),3)
                    # 找到最优X的索引
                    index_best = self.findBestX(mArray=np.asarray(mA))
                    # 变异
                    tempA = np.arange(3)
                    tempA = np.delete(tempA,index_best)
                    X_best=mA[index_best].featureOfSelectNum
                    X_r1 = mA[tempA[0]].featureOfSelectNum
                    X_r2 = mA[tempA[1]].featureOfSelectNum
                V_i = self.mutateOperator(X_best=X_best,X_r1=X_r1,X_r2=X_r2)
                # 交叉
                U_i = self.crossoverOperator(V_i=V_i,X_i=self.population_DE[i].featureOfSelectNum)
                # 选择
                U_i_bin = np.where(U_i > 0.6,1,0)
                # 计算acc
                featureX = self.dataX[:,U_i_bin == 1]
                err_i = 1 - fitnessFunction_KNN_CV(findData_x=featureX, findData_y=self.dataY, CV=10)
                newIndividual = copy.deepcopy(self.population_cluster[i][j])
                newIndividual.mineFitness = err_i
                newIndividual.featureOfSelect = U_i_bin
                newIndividual.featureOfSelectNum = U_i
                newIndividual.getNumberOfSolution()
                # 放到子代中
                springPop = np.append(springPop,newIndividual)
            # 非支配，将非支配解集保存到子缓存中
            newPop = np.concatenate((np.asarray(self.population_cluster[i]),springPop),axis=0)
            # 获得非支配的前沿PF 和 非支配后的种群
            newPF,tempPop = nonDominatedSort_PFAndPop(parentPopulation=newPop,populationNum=cl)
            # 添加前沿PF到档案中
            # self.archive_cluster[i].append(np.asarray(newPF))
            for m in range(len(newPF)):
                self.archive_cluster[i] = np.append(self.archive_cluster[i],newPF[m])
            # 更新种群
            self.population_cluster[i] = tempPop
            # 保持种群多样性
            if len(self.archive_cluster[i]) > 2 * len(self.population_cluster[i]):
                # 保存多样性
                self.archive_cluster[i] = self.DiversityMaintenance(subArchive=np.asarray(self.archive_cluster[i]),ith=i)
            # 子缓存大小
            la = len(self.archive_cluster[i])
            # 判断每一个档案中平均err是否变化
            mArchive_err = np.zeros(la)
            for m in range(la):
                mArchive_err[m] = self.archive_cluster[i][m].mineFitness
            if np.mean(mArchive_err) == self.meanErr_cluster[i]:
                self.stagnant_cluster[i] += 1
            else:
                self.meanErr_cluster[i] = np.mean(mArchive_err)
                self.stagnant_cluster[i] = 0


    # 多样性
    def DiversityMaintenance(self,subArchive,ith):
        pN = len(self.population_cluster[ith])
        # 超出了多少个体
        newPF,newPop = nonDominatedSort_PFAndPop(parentPopulation=subArchive,populationNum=pN)
        return newPop

    # 保持收敛的全局缓存
    def maintainC(self):
        # 合并子档案
        tempPop = self.archive_cluster[0]
        for i in range(1,self.k):
            tempPop = np.concatenate((tempPop,np.asarray(self.archive_cluster[i])),axis=0)
        archivePF = getPF(pop=tempPop)
        PFNum = len(archivePF)
        # 判断是否存在重复

        # 获得前沿err和len
        front = np.zeros((PFNum, 2))
        for i in range(PFNum):
            front[i][0] = archivePF[i].mineFitness
            front[i][1] = archivePF[i].proportionOfFeature

        maxIndex = np.argmin(front[:, 0])
        print("acc = ", 1 - front[maxIndex][0], " ,len = ", int(front[maxIndex][1] * self.dataFeatureNum))

        if self.globalFitness > archivePF[maxIndex].mineFitness:
            self.globalFitness = archivePF[maxIndex].mineFitness
            self.globalSolution = archivePF[maxIndex].featureOfSelect
            self.globalSolutionNum = archivePF[maxIndex].featureOfSelectNum
        # 是否archivePF的长度超过了pop
        if PFNum > self.populationNum:
            # 计算非支配的 HV
            referencePoint = [1, 1]
            hv = HyperVolume(referencePoint)
            # PF的HV
            pf_volume = hv.compute(front)
            # 每一个前沿的HV
            eachHV = np.zeros(PFNum)
            #
            for i in range(PFNum):
                tempF = np.delete(front,i)
                eachHV[i] = hv.compute(tempF)
            eachHV = pf_volume - eachHV
            # 排序
            HV_index = np.argsort(eachHV)
            # 多余个体的数量
            exceedNum = PFNum - self.populationNum
            dI = 0
            while(exceedNum == dI):
                archivePF = np.delete(archivePF,HV_index[dI])
                dI += 1
            self.archive_global = archivePF



    def run(self):

        # 初始化种群
        self.initPopulation()
        runTime = 1
        import time
        start = time.time()
        self.dividePopulation(population=self.population_DE, k=self.k)
        while runTime <= self.iteratorTime:
            print("第",runTime,"代")
            # runTime 从1开始，所以要加1
            if runTime % (self.fGroup + 1) == 0:
                # 合并种群
                self.conPop()
                # 通过聚类方式将种群分配成k个子种群
                self.dividePopulation(population=self.population_DE,k=self.k)
            # 更新种群
            self.updataPopulation()
            # 保持全局收敛
            self.maintainC()
            print(f"all time = {time.time() - start} seconds")
            runTime += 1

        print(f"all time = {time.time() - start} seconds")
        print("global solution")
        print("acc = ",1 - self.globalFitness," len = ",len(np.where(self.globalSolution == 1)[0]))
        print("####################################")
        print("populationNum = ", self.populationNum)
        print("iteratorTime = ", self.iteratorTime)
        print("crossoverPro = ", self.crossoverPro)
        print("F = ", self.F)
        print(" ")

if __name__ == '__main__':

    ducName = "BreastCancer1"
    # ducName = "CLLSUB"
    path = "D:/MachineLearningBackUp/dataCSV/dataCSV_high/" + ducName + ".csv"
    dataCsv = ReadCSV(path=path)
    print("获取文件数据")
    dataCsv.getData()
    genetic = MOCDE(dataX=dataCsv.dataX, dataY=dataCsv.dataY)
    genetic.setParameter(populationNum=300, iteratorTime=100, crossoverPro=0.5, F=0.5,fGroup=10)
    genetic.run()


