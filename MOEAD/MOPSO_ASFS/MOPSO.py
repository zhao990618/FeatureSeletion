import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from Filter.filter import FilterOfData
from MOEAD.ParetoFront.paretoPop import updatePF
from dataProcessing.ReadDataCSV_new import ReadCSV
from MOEAD.WeightVector.vector import Mean_vector


class MOPSO_ASFS:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 特征的存放数组
    dataFeature = np.asarray([])
    # 特征的数量
    dataFeatureNum = 0
    # 粒子群
    population_PSO = []
    # 每一个种群的数量
    populationNum = 0
    # 种群迭代次数
    iteratorTime = 0
    # 粒子最大速度
    maxV = 0
    # 粒子最小速度
    minV = 0
    # 自我认知参数
    c_1 = 0
    # 全局认知参数
    c_2 = 0
    # omega
    omega = 0
    # 年龄阈值
    Ta = 0
    # 最大惩罚系数
    theta_max = 0
    # 最小惩罚系数
    theta_min = 0
    # 每一个权重对应的解
    LA = np.asarray([])
    # 领导缓存
    leaderArchive = np.asarray([])
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
    # 保存pareto
    EP = np.asarray([])

    # 每一个特征的relifF评分
    scoreOfRelifF = np.asarray([])
    # 全局最优选择
    globalBestSolution = np.asarray([])
    # 全局最优选择得到的fitness
    globalBestFitness = 0
    # 全局最优选择特征长度
    glboalBestLen = 0


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
        #  compute relifF score
        self.scoreOfRelifF = filter.computerRelifFScore()
        # 进行filter过滤数据 ----  默认采用的是膝节点法
        self.scoreOfRelifF, self.dataX, self.dataFeature = filter.filter_relifF(
            scoreOfRelifF=self.scoreOfRelifF, dataX=self.dataX, dataFeature=self.dataFeature)
        # 更新 特征长度
        self.dataFeatureNum = len(self.dataFeature)


    # initialization parameters
    def  setParameter(self,populationNum,iteratorTime,maxV,minV,c_1,c_2,theta_max,theta_min,Ta,omega):
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime
        self.maxV = maxV
        self.minV = minV
        self.c_1 = c_1
        self.c_2 = c_2
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.Ta = Ta
        self.omega = omega
        self.H = populationNum - 1
        self.T = int(self.populationNum / 10)
        # 初始化 理想点 ，优于是最小化问题，所以初始化为[1,1]
        self.ideal_Z = np.ones(self.objNum)



    class Particle():
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
        # 个体速度
        velocityOfAllFeature = np.asarray([])
        # 年龄 -- 用于判断个体是否停滞
        age = 0
        # 惩罚系数
        theta = 0
        # 个体最优选择
        pBestSolution = np.asarray([])
        # 个体最优选择实数制
        pBestSolutionNum = np.asarray([])
        # 个体最优选择得到的fitness
        pBestFitness = 1
        # 全局最优选择
        gBestSolution = np.asarray([])
        # 个体最优选择实数制
        gBestSolutionNum = np.asarray([])
        # 全局最优选择得到的fitness
        gBestFitness = 1

        def __init__(self,mopso,mode):


            if mode == 'init':
                # 初始化一个位置
                self.defineInitPosition(numberOfFeature = mopso.dataFeature.shape[0],mopso=mopso)
                # 初始化速度
                self.defineInitVelocity(numberOfFeature = mopso.dataFeature.shape[0])
            elif mode == "iterator":
                pass



        # 给每一个粒子初始化一个位置
        def defineInitPosition(self,numberOfFeature,mopso):
            self.featureOfSelectNum = np.random.rand(numberOfFeature)
            self.featureOfSelect = np.where(self.featureOfSelectNum > 0.6 ,1,0)
            # 更新信息
            self.getNumberOfSolution()
            # 计算acc
            feature_x = mopso.dataX[:, self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=mopso.dataY, CV=10)

            if self.mineFitness < self.pBestFitness:
                self.pBestFitness = self.mineFitness
                self.pBestSolution = self.featureOfSelect
                self.pBestSolutionNum = self.featureOfSelectNum

        # 给每一个粒子初始化一个速度
        def defineInitVelocity(self,numberOfFeature):
            self.velocityOfAllFeature = np.random.rand(numberOfFeature)


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
            particle = self.Particle(mopso=self,mode='init')
            self.population_PSO.append(particle)
            # self.EP = updatePF(EP=self.EP, individual=particle)
            # 用于存放每一个权重向量 的 最优solution
            self.LA = np.append(self.LA,particle)


    # 产生权重向量
    def generateWeightVector(self):
        # 设置了参数H决定生成权重数量 --- 设置权重的维度objNum
        weightV = Mean_vector(H=self.H,m=self.objNum)
        self.weightArray = weightV.generate()

    # 设置理想点Z
    def setIdealPoint(self):
        # 从当代种群中抽取 每一个目标上的最优值作为理想点
        for i in range(self.populationNum):
            if self.population_PSO[i].mineFitness < self.ideal_Z[0]:
                self.ideal_Z[0] = self.population_PSO[i].mineFitness
            if self.population_PSO[i].proportionOfFeature < self.ideal_Z[1]:
                self.ideal_Z[1] = self.population_PSO[i].proportionOfFeature


    # 计算PBI
    def calculatePBI(self,F_i,weight_i,theta_i):
        temp_v = np.dot(self.ideal_Z - F_i, weight_i)
        t_1 = np.linalg.norm(weight_i, ord=2)
        # 获得 d_1
        d_1 = np.asarray(np.abs(temp_v) / t_1)

        temp_v = self.ideal_Z + d_1 * weight_i / t_1
        d_2 = np.linalg.norm(F_i - temp_v, ord=2)
        PBI_i = d_1 + theta_i * d_2
        return d_2,PBI_i

    # 更新存储
    def updateArchive(self,pop,iteNum):
        for i in range(self.populationNum):
            # 用于临时存放数据
            tempCache = np.asarray([])
            # 用于临时存放d_2
            temp_d_2 = np.asarray([])
            for j in range(len(pop)):
                # 获得第 j 个个体的两个目标
                F_i = np.asarray([pop[j].mineFitness,pop[j].proportionOfFeature])

                d_2,PBI_i = self.calculatePBI(F_i=F_i,weight_i=self.weightArray[i],theta_i=self.thetaArray[i])

                # 保存d_2
                temp_d_2 = np.append(temp_d_2,d_2)
                # 添加值
                tempCache = np.append(tempCache,PBI_i)

            # 获得最优个体的索引
            bestIndex = np.argsort(tempCache)[0]
            # 将pop中里权重i在PBI计算中最优的个体 提取出来
            bestIndiviaulofW_i = pop[bestIndex]
            # 更新惩罚参数
            self.thetaArray[i] = self.adaptivePenaltyValue(iteNum=iteNum,d_2=temp_d_2[bestIndex])
            # 用于存放 第i个权重选择的 最优solution
            self.LA[i] = copy.deepcopy(bestIndiviaulofW_i)
            # 将该个体删除
            pop = np.delete(pop, bestIndex).tolist()


    # 自适应惩罚值
    def adaptivePenaltyValue(self,iteNum,d_2):
        genx = self.theta_min +(self.theta_max - self.theta_min) * iteNum / self.iteratorTime
        theta = 1 + (genx * math.exp(-1 * self.populationNum * d_2))
        return theta


    # 得到权重向量的T个近邻
    def getNeighborOfWeight_i(self,weight):
        # 存放距离差
        weightDistance = np.zeros(len(self.weightArray))
        # 计算该权重与其他所有权重之差
        for i in range(len(self.weightArray)):
            temp_1 = np.asarray(weight) - np.asarray(self.weightArray[i])
            # 欧几里得距离
            weightDistance[i] = np.linalg.norm(temp_1, ord=2)

        # 返回前T个权重索引
        weightSortIndex = np.argsort(weightDistance)
        return weightSortIndex[:int(self.T)]

    # 在T个近邻中选择最优个体
    def selectGBestOfT(self,weight_i,theta_i,weightSortIndex):
        # 存放weight_i的T个近邻所对应的solution 在 PBI上的值
        valueOfPBI = np.ones(int(self.T))
        # 计算这T个PBI值
        for i in range(self.T):
            F_i = np.asarray([self.LA[weightSortIndex[i]].mineFitness, self.LA[weightSortIndex[i]].proportionOfFeature])
            d_2,PBI_i = self.calculatePBI(F_i=F_i,weight_i=weight_i,theta_i=theta_i)
            valueOfPBI[i] = PBI_i
        # 进行排序
        tempSortIndex = np.argsort(valueOfPBI)
        # 得到这个个体
        gbest_individual = self.LA[weightSortIndex[tempSortIndex[0]]]
        gbest = gbest_individual.featureOfSelectNum
        return  gbest

    # 计算LA中的特征频率
    def calculateFeatureFrequenciesOfFA(self):
        # 用于保存特征被选择的数量
        FR = np.zeros(self.dataFeature.shape[0])
        # 用于保存 每一个维度上 实数制类型的最小值
        a_min = np.ones(self.dataFeature.shape[0])
        # 用于保存 每一个维度上 实数制类型的最大值
        a_max = np.zeros(self.dataFeature.shape[0])

        for i in range(self.populationNum):
            FR = FR + self.LA[i].featureOfSelect
            # 更新最小值
            minIndex = np.where(self.LA[i].featureOfSelectNum < a_min)[0]
            a_min[minIndex] = a_min[minIndex]
            # 更新最大值
            maxIndex = np.where(a_max < self.LA[i].featureOfSelectNum)[0]
            a_max[maxIndex] = a_max[maxIndex]
        return FR,a_min,a_max


    # 更新位置 -- 需要创建一个新的粒子
    def updatePosition(self,individual,gbest):
        # 更新速度
        new_velocity = self.omega * individual.velocityOfAllFeature + self.c_1 * np.random.rand()*(
            individual.pBestSolutionNum - individual.featureOfSelectNum
        ) + self.c_2 * np.random.rand() * (gbest - individual.featureOfSelectNum)
        # 限制速度在一定范围内
        new_velocity = np.where(new_velocity > self.maxV,self.maxV,new_velocity)
        new_velocity = np.where(new_velocity < self.minV,self.minV,new_velocity)
        # 保存新个体的速度
        individual.velocityOfAllFeature = new_velocity
        # 更新位置--实数制
        individual.featureOfSelectNum = individual.featureOfSelectNum + new_velocity
        # -- 二进制
        individual.featureOfSelect = np.where(individual.featureOfSelectNum > 0.6 ,1, 0)
        # 更新信息
        individual.getNumberOfSolution()
        return individual


    # 更新种群
    def updatePopulaiton(self):
        # 计算特征频率,每一个维度上的最大值和最小值
        FR, a_min, a_max = self.calculateFeatureFrequenciesOfFA()
        # 迭代每一个粒子
        for i in range(self.populationNum):
            # 如果年龄没到阈值
            if self.population_PSO[i].age < self.Ta:
                # weight_i 的前T个近邻的索引
                weightSortIndex_T = self.getNeighborOfWeight_i(weight=self.weightArray[i])
                # 选择第 i 个个体的最优个体--最优个体是从近邻T个个体中挑选
                gbest = self.selectGBestOfT(weight_i=self.weightArray[i],theta_i=self.thetaArray[i],
                                            weightSortIndex=weightSortIndex_T)
            else:
                # 计算新的gbest
                gbest = 0.5 * (np.random.rand()*(a_max + a_min) - FR + 1)

            # 更新个体
            newIndividual = self.updatePosition(individual=self.population_PSO[i],gbest=gbest)
            # 年龄加1
            newIndividual.age += 1
            # 判断 fitness
            # 计算acc
            feature_x = self.dataX[:, newIndividual.featureOfSelect == 1]
            newIndividual.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=feature_x,
                                                                   findData_y=self.dataY, CV=10)
            # 与 pbest进行比较
            if newIndividual.mineFitness < newIndividual.pBestFitness:
                self.pBestFitness = newIndividual.mineFitness
                self.pBestSolution = newIndividual.featureOfSelect
                self.pBestSolutionNum = newIndividual.featureOfSelectNum
                newIndividual.age = 0

            self.population_PSO[i] = newIndividual
            # self.EP = updatePF(EP=self.EP, individual=newIndividual)
        # 更新理想点
        self.setIdealPoint()


    # 画图
    def pltForWin(self,iteNum):
        plt.ion()
        length_LA = self.LA.shape[0]
        # 存放 error
        x = np.ones(length_LA)
        # 存放 len
        y = np.ones(length_LA)
        for i in range(length_LA):
            x[i] = self.LA[i].proportionOfFeature
            y[i] = self.LA[i].mineFitness

        plt.xlabel("len")
        plt.ylabel("error")
        plt.title("MOPSO/D"+ " " + str(iteNum))
        # plt.plot(x,y,color='r',marker='o',linestyle='dashed')
        plt.plot(x,y,color='r',marker='o',linestyle='None')
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))

        plt.pause(0.1)
        if iteNum != self.iteratorTime:
            plt.cla()

    # 评判当代最优
    def evaluateBestIndividual(self):
        length_LA = self.LA.shape[0]
        # 存放 error
        x = np.ones(length_LA)
        # 存放 len
        y = np.ones(length_LA)
        for i in range(length_LA):
            x[i] = self.LA[i].mineFitness
            y[i] = self.LA[i].numberOfSolution
        # 排序
        index_error = np.argsort(x)
        # 判断全局最优
        if self.globalBestFitness < 1 - x[index_error[0]]:
            self.globalBestFitness = 1 - x[index_error[0]]
            self.globalBestSolution = self.LA[index_error[0]].featureOfSelect
            self.glboalBestLen = y[index_error[0]]
        # 输出acc前三

        # 输出最优--最优按照error来判定的
        print("acc = ",1 - x[index_error[0]],"\t","len = ",y[index_error[0]])


    # 执行
    def run(self):

        plt.ion()

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
        # 初始化LA
        self.updateArchive(pop=self.population_PSO,iteNum=0)

        while runTime < self.iteratorTime:
            print("第",runTime + 1,"代")
            self.evaluateBestIndividual()
            # 更新种群
            self.updatePopulaiton()
            # 更新 LA
            tempPop = copy.deepcopy(np.concatenate([self.population_PSO,self.LA]))
            self.updateArchive(pop=tempPop,iteNum=runTime+1)
            runTime += 1
            self.pltForWin(iteNum=runTime)

        print(" ")
        print("global_acc = ",self.globalBestFitness," global_len = ",self.glboalBestLen)

        # self.EP = updatePF(pop=np.asarray(self.population_PSO))
        self.EP = updatePF(pop=np.asarray(self.LA))

        print(" PF 解")
        for i in range(len(self.EP)):
            print(1 - self.EP[i].mineFitness, " ", self.EP[i].numberOfSolution)



if __name__ == '__main__':
    import time
    start = time.time()
    ducName = "BreastCancer1"
    path = "D:\MachineLearningBackUp\dataCSV\dataCSV_high\\" + ducName + ".csv"
    dataCsv = ReadCSV(path=path)
    print("获取文件数据")
    dataCsv.getData()
    pso = MOPSO_ASFS(dataX=dataCsv.dataX, dataY=dataCsv.dataY)
    pso.setParameter(populationNum=100,iteratorTime=100,maxV=4,minV=-4,c_1=1.46,c_2=1.46,
                     theta_max=100,theta_min=1,omega=0.729,Ta=2)
    pso.run()
    print("================================================")
    print(f"all time = {time.time() - start} seconds")
