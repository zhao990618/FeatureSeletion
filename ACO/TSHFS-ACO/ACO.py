import copy
import random

import numpy
import numpy as np
import pandas as pd
from  InforGain import *
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import  random
from  dataProcessing.InforGain import InforGain

class ACO:
    # 样本值表
    allData = []
    # 保存特征表
    dataLabel = []
    # 保存的是每一纵列的总数据
    dataColum = []
    # 信息素表
    pheromone = []
    # 初始化信息素
    initPheromone = []
    # 启发式信息表
    heuristic = []

    # 每一个特征所在的位置索引
    featureIndex = []
    # 用于保存未进行filter方法前的特征表
    featureLable = []

    # 迭代次数
    iterationTime = 0
    # 蚂蚁的数量
    numberOfAnt = 0
    # 信息素蒸发率
    evaRate = 0
    # 蚂蚁群
    antList = []
    # 每一轮得到的最优组合
    GConduct = []
    # 每一轮的最优组合的fitness
    GFitness = float("-inf") # 无限大   float("-inf") 无限小
    # 历史最优组合
    GlConduct = []
    # 历史最优组合的fitness
    GlFitness = float("-inf") #无限大
    # 贪婪指标
    greedyRate = 0
    # 信息素指数
    alpha = 0
    # 启发式信息指数
    beta = 0
    # 信息素更新 Delta 的系数
    e = 0
    #信息素更新Delta
    delta = 0
    # 适应度函数参数
    omega = 0
    # 需要查找到的特征数量
    needFindFN = 0
    # 将相同类的数据存放在一起
    computeDistanceData = []
    # 用于得到每一个类所对应的样本数有多少个
    differntClassInstacneNumber = []
    # 有多少个类
    classOfSpecies = []
    # 保存数据
    dataX = np.asarray([])
    # 保存每一种蚂蚁遍历特征的结果
    eachSelectFeatureValue = []
    eachSelectFeatureFitness = np.asarray([])


#    testACO = ACO(dataCsv=dataCsv,allData=dataCsv.allData,dataColum=dataCsv.dataAllColum,featureLabel=dataCsv.dataAttribute,iterationTime=50,
#                  numberOfAnt=12,evarate=0.15,greedyRate=0.4,alpha=5,beta=1,e=0.125,needFindFN=4,omega=0.125)


    def __init__(self,dataCsv,allData,dataColum,featureLabel):
        self.allData = allData
        self.dataColum = dataColum
        self.dataLabel = featureLabel
        self.featureLable = featureLabel
        #初始化信息素矩阵
        print("计算信息素矩阵")
        self.initPheromone(dataCsv)
        print("初始化信息素完成")
        #print(self.pheromone)
        print("用SU进行filter，获取效果好的一些特征")
        # 保留了 3%的数据
        self.fiterOfSU(0.03)

        # 低维数据可以用
        # self.initHeuristic()


# 测试的时候先注释掉，然后添加回来
        # 将矩阵按照类来进行排序
        print("距离矩阵")
        self.initdataClassify()

        print("当前特征数量 = ",len(self.pheromone))




    def getParameter(self,iterationTime,numberOfAnt,evarate,greedyRate,alpha,beta,e,omega):
        self.iterationTime = iterationTime
        self.numberOfAnt = numberOfAnt
        self.evaRate = evarate
        self.greedyRate = greedyRate
        self.alpha = alpha
        self.beta = beta
        self.e = e
        self.omega = omega
        print("蚁群迭代",iterationTime,"代")

    # 信息素矩阵的初始化   ---- TSHFS-ACO
    def initPheromone(self,dataCsv):
        infor = InforGain(dataAttribute=dataCsv.dataAttribute, dataAllColum=dataCsv.dataAllColum)
        self.pheromone = infor.getFeatureClassSU()  #  用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵
        self.initPheromone = copy.deepcopy(self.pheromone)

    # fiter 方法  获取前 60% 的数据  ----------用于高维数据
    def fiterOfSU(self,extratRatio):
        # 得到su矩阵
        infor = InforGain(dataAttribute=self.dataLabel, dataAllColum=self.dataColum, dataX=self.dataX)
        print("以得到SU矩阵")
        self.suArray = infor.getFeatureClassSU()  # 用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵
        #self.suArray = infor.getMutualInformation()  # 用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵
        self.initsuArray = copy.deepcopy(self.suArray)
        print("得到相似度矩阵")
        # 用于低维数据
        # self.featureOfSimilar()
        # 用于高维数据
        self.featureOfSimilar_high()
        # 计算概率
        print("计算概率")
        self.filterOfProFreture()
        print("通过su进行筛数据")

        # 通过prob进行排序
        self.pheromone, self.dataLabel, self.dataColum ,self.dataX= infor.getDataOfFilterInPro(
            probabilityOfFeatureToSelect=self.probabilityOfFeatureToSelect, extratRatio=extratRatio)
        # print("  ")





    # 启发式信息矩阵, 方便计算
    # 适用于低维数据，因为是形成一个n*n的矩阵存放启发式信息
    def initHeuristic(self):
        heuristic_ij = 0
        feature_i = []
        feature_j = []
        side = len(self.dataLabel)

        tempList = np.random.uniform(0, 0, (side, side))  # 二维全0数组


        # 得到每一个特征到其他每一个特征的sim矩阵
        for i in range(0, side):
            for j in range(0, side):
                feature_i = self.dataColum[i]
                feature_j = self.dataColum[j]
                if (i != j):
                    heuristic_ij = self.computeHeuristicMatrix(feature_i, feature_j)
                    tempList[i][j] = heuristic_ij
                else:
                    tempList[i][j] = 0
        self.heuristic = tempList  # 将计算出来的矩阵保存

    # 计算启发式信息矩阵函数
    def computeHeuristicMatrix(self, feature_i, feature_j):
        value_i = pow(feature_i, 2)  # 计算特征i 的所有样本值的平方
        value_j = pow(feature_j, 2)  # 计算特征j 的所有样本值的平方
        sum_i = np.sum(value_i)  # i 的平方和
        sum_j = np.sum(value_j)  # j 的平方和
        sum_ij = np.sum(feature_i * feature_j)  # 特征i和特征j的所有值相乘再相加
        heuristic_ij = abs(sum_ij / (pow(sum_i, 0.5) * pow(sum_j, 0.5) + 0.00000001))
        return heuristic_ij


    class Ant:
        # 特征节点表
        dataFeature = []
        # 已选择的特征   保存的都是索引
        selectedFeature = []
        # 未选择的特征的索引  保存的都是索引
        unSelectFeature = []
        # 未访问特征节点表
        tour = []
        # 特征总数量
        numberFeature = 0
        # 需要访问的节点数量
        findNumber = 0
        # 已经查找到的特征节点数量
        nowFeatureNumber = 0


        def __init__(self,ACO):
            # 初始化启发式矩阵
            #self.initHeuristic()
            # 获得查找节点数量
            self.findNumber = ACO.needFindFN
            # 获得特征矩阵
            self.dataFeature = ACO.dataLabel

            # 获得特征长度
            #print(self.dataFeature)  因为 ACO.dataLabel里面含有class标签，所以要减一
            self.numberFeature = len(self.dataFeature) - 1
            #print("self.dataFeature",len(self.dataFeature))

            # 保存未访问的特征节点索引
            tempListUnSelect = range(0,self.numberFeature)
            self.unSelectFeature = np.asarray(tempListUnSelect)
            #print("self.unSelectFeature",self.unSelectFeature)
            # 随机选择初始节点  区间范围[0,self.numberFeature - 1]
            initFeature = random.randint(0,self.numberFeature - 1)

            # 将随机选择的初始特征节点保存到selectedFeature中;
            #self.selectedFeature = np.asarray(range(0,int(self.findNumber)))
            #self.selectedFeature = np.asarray(np.zeros(int(self.findNumber)))
            self.selectedFeature = np.asarray([])
            self.selectedFeature = np.append(self.selectedFeature,initFeature)

            # 在unSelectFeature中删除节点
            self.removeNode(initFeature)

            # 已经存入的节点数量
            self.nowFeatureNumber = 1

        # tour 是个list next Node是下一个特征节点
        def removeNode(self,nextNode):

            # 找不到array删除数据的函数，所以转成list来删除
            self.unSelectFeature = list(self.unSelectFeature)
            # 得到该节点的索引位置
            removeIndex = self.unSelectFeature.index(nextNode)
            #removeIndex = np.where(unSelectFeature == nextNode)[0][0]
            # del p[index]  删除list 中索引值为index的数据
            del self.unSelectFeature[removeIndex]
            # 转化回array
            self.unSelectFeature = np.asarray(self.unSelectFeature)





        def chooseNextFeature(self,ACO):

            # 用于保存下一个节点
            nextFeature = 0
            # 当前特征的信息素
            tau = 0
            # 当前特征的启发式信息
            eta = 0

            # 当前节点和信息素

            # 贪婪值 ，用于确定是用 贪婪法还是概率法
            q = random.random()
            q = np.round(q,4) # 限定小数点后4位

            # 轮盘赌的随机数
            rs = 0

            listFkPoint = list(np.zeros(self.unSelectFeature.shape[0]))

            sumFkPoint = 0
            maxFkPoint = -1
            if(q < ACO.greedyRate):  #进入贪婪算法
                print("==================================================")
                print("进入贪婪算法",self.unSelectFeature.shape[0],"个特征")
                for i in range(0,self.unSelectFeature.shape[0]):
                    tempNextFeature = self.unSelectFeature[i]
                    #print("计算tau",tau)
                    tau = ACO.pheromone[tempNextFeature]
                    #print("计算eta",eta)
                    eta = self.heuristicHighIJ(tempNextFeature,ACO=ACO)
                    fkNextPoint = np.power(tau,ACO.alpha)*np.power(eta,ACO.beta)
                    if(maxFkPoint < fkNextPoint):
                        # 存放了最大的Fk值
                        maxFkPoint = fkNextPoint
                        # 存放了当Fk最大时的特征节点
                        nextFeature = tempNextFeature
            else:
                # 进入概率法
                print("==================================================")
                print("进入概率法,计算",self.unSelectFeature.shape[0],"个特征")
                for i in range(0, self.unSelectFeature.shape[0]):
                    tempNextFeature = self.unSelectFeature[i]

                    #(ACO.pheromone)
                    tau = ACO.pheromone[tempNextFeature]

                    eta = self.heuristicHighIJ(tempNextFeature , ACO=ACO)
                    fkNextPoint = (np.power(tau, ACO.alpha)) * (np.power(eta, ACO.beta))

                    listFkPoint[i] = fkNextPoint
                    sumFkPoint += fkNextPoint
                #listFkPoint = np.asarray(listFkPoint)
                # 得到每一个未选择特征的概率
                listFkPoint = np.asarray(listFkPoint)
                listFkPoint = listFkPoint/sumFkPoint

                # 轮盘赌
                randomValue = np.random.random()

                mselect = 0
                for j in range(0,len(listFkPoint)):
                    mselect += listFkPoint[j]
                    if(randomValue <= mselect):
                        nextFeature = self.unSelectFeature[j]
                        break

            #print("================================================")

            return nextFeature

        # 形成特征组合
        def constructSolution(self,ACO):
            # 当前选择的节点数量 小于 需要选择的节点数量
            findNumber = int(self.findNumber)
            #print("当前需要找到",findNumber,"个节点")
            for i in range(1,findNumber):
                print("计算第",i,"个特征的效果")
                # 当前选择的节点
                nextPoint = self.chooseNextFeature(ACO=ACO)
                #print("找到了第",i,"个特征")
                # 移除未选择特征列表中的当前被选择的节点
                self.removeNode(nextPoint)

                self.selectedFeature= np.append(self.selectedFeature,nextPoint)
                self.nowFeatureNumber += 1

        # 计算第i个特征的启发式信息     ----- 用于低维数据
        def heuristicIJ(self,nextNode ,ACO):
            eta_i = 0
            dataSimSum = 0
            # 计算未选择的特征到已选择的特征的
            for selectF in self.selectedFeature:
                dataSimSum += ACO.heuristic[nextNode][selectF]
            eta_i = 1 - (dataSimSum/len(self.selectedFeature))
            eta_i = np.asarray(eta_i)
            return eta_i

        # 计算第i个特征的启发式信息   ---------用于高维数据
        def heuristicHighIJ(self,nextNode,ACO):
            eta_i = 0
            dataSimSUm = 0
            # 获得已选择的特征
            selectFeature = self.selectedFeature
            # 计算未选择的特征与已选择特征的启发式值

            for selectF in selectFeature:
                feature_i = ACO.dataColum[nextNode]
                feature_j = ACO.dataColum[int(selectF)]
                dataSimSUm += ACO.computeHeuristicMatrix(feature_i=feature_i,feature_j=feature_j)
            eta_i = 1 - (dataSimSUm/len(selectFeature))
            eta_i = np.asarray(eta_i)
            return eta_i
    # 创建 numberOfAnt 只蚂蚁
    def initAnts(self,numberOfAnt):

        for i in range(0,self.numberOfAnt):
            ant = self.Ant(ACO=self)
            self.antList.append(ant) #[ant1,ant2]


    # 所有蚂蚁完成查找后，进行选择最优蚂蚁路径
    def selectBestPath(self,ACO):
        tempAnts = self.antList
        # 存放每一只蚂蚁计算得到的适应度值
        antFitness = np.zeros(self.numberOfAnt)
        # 保存最优蚂蚁的精度
        bestAntAccuracy = 0
        bestAnt = 0
        # 用于临时存放每一个节点的精确度
        tempAccuracy = []
        findedData_x = []
        # 临时存放数据 ，要最后赋值给self.eachSelectFeatureValue
        tempD = self.eachSelectFeatureValue
        for i in range(0,self.numberOfAnt):
            # 获得每一只蚂蚁对象
            ant = tempAnts[i]
            selectFeature = np.sort(ant.selectedFeature)
            #selectFeature = list(selectFeature)
            # 将找到的第一个数据赋值给findedData
            findedData_x = self.dataX[:,selectFeature]
            #findedData_x = self.dataColum[int(selectFeature[0])]
            # for j in range(1,int(len(selectFeature))):
            #     tempData = self.dataColum[int(selectFeature[j])]
            #     # 将本只蚂蚁选择的特征包含的数据保存在findeedData_x中
            #     findedData_x = np.c_[findedData_x,tempData]
            # 获取得到类标签所对应的数据
            findedData_y = self.dataColum[-1]


            # 计算适应度函数
            fitness = self.fitnessFunOfACO(findedData_x,findedData_y,selectFeature)

            # test 计算不使用距离函数
            fitness = round(fitness,4)

            # 得到每一只蚂蚁的10折交叉的fitness
            antFitness[i] = fitness


        # 得到fitness最高的那一个值
        bestAntAccuracy =  antFitness.max()

        # 得到这个值所对应蚂蚁的index
        bestAntIndex =  np.where(antFitness == bestAntAccuracy)[0][0]

        # 找到这个蚂蚁
        bestAnt = self.antList[bestAntIndex]
        self.GFitness = bestAntAccuracy
        self.GConduct = bestAnt.selectedFeature
        self.delta = self.GFitness
        #print("bestAntAccurancy = ",)

    # fitness 适应度函数   TSHFS-ACO
    def fitnessFunOfACO(self,findedData_x,findedData_y,selectFeature):
        findedData_x = np.asarray(findedData_x)
        findedData_y = np.asarray(findedData_y)
        if len(findedData_x.shape) == 1:
            findedData_x = findedData_x.reshape(len(findedData_x),1)

        # 选择该特征的索引
        selectfeatureIndex = 0
        knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto", metric='manhattan')

        # cv参数决定数据集划分比例，这里是按照9:1划分训练集和测试集
        scores = cross_val_score(knn, findedData_x, findedData_y, cv=5, scoring='accuracy')
        accuracy = scores.mean()

        if self.omega == 0:
            fitness = accuracy
        else:
            tempD = self.eachSelectFeatureValue
            boolean_i = False
            if (len(tempD) == 0):
                # 计算距离
                # 将选择的特征传入进来
                distance = self.distancs_ACO(selectFeature)
                fitness = accuracy + self.omega * distance
                # 记录每一种可能
                tempD.append(selectFeature)
                self.eachSelectFeatureFitness = np.append(self.eachSelectFeatureFitness, fitness)
            else:
                for featureConduct_i in range(0, len(tempD)):
                    a = list(tempD[featureConduct_i])
                    b = list(selectFeature)
                    if (a == b):
                        boolean_i = True
                        selectfeatureIndex = tempD.index(b)

                if boolean_i:
                    distance = self.eachSelectFeatureFitness[selectfeatureIndex]
                    fitness = accuracy + self.omega * distance
                else:
                    # 计算距离
                    # 将选择的特征传入进来
                    distance = self.distancs_ACO(selectFeature)
                    fitness = accuracy + self.omega * distance

                    # 记录每一种可能
                    tempD.append(selectFeature)
                    self.eachSelectFeatureFitness = np.append(self.eachSelectFeatureFitness, fitness)
            # 一轮跑完， 将tempD的值还给self.eachSelectFeatureValue
            self.eachSelectFeatureValue = tempD
        return fitness


    # 将数据按照类来排放   ----TSHFS-ACO
    def initdataClassify(self):
        dataClass = self.dataColum[-1]
        # 获得总共有多少个类
        y_value_class = set(dataClass[i] for i in range(dataClass.shape[0]))
        y_value_class = list(y_value_class)
        y_value_class = np.asarray(y_value_class)

        temp1 = self.allData

        # 用于存放不同分类的数据值   differentClass[[],[],[]]
        computeDistanceData =[]#np.asarray([])# np.zeros(self.allData.shape[0])
        # 用于得到每一个类所对应的样本数有多少个
        differntClassInstacneNumber = np.asarray(np.zeros(len(y_value_class)))
        # 将相同类别的样本放在同一块 ， 并且获取得到每一个类的样本数量
        for j in range(0, y_value_class.shape[0]):
            number = 0
            for i in range(0, temp1.shape[0]):  # 遍历每一行数据
                if (temp1[i][-1] == y_value_class[j]):

                    #tempList = list(temp1[i])
                    # 排序矩阵
                    #computeDistanceData = np.append(computeDistanceData,[temp1[i]])
                    computeDistanceData.append(temp1[i][0:-1])
                    number += 1
            differntClassInstacneNumber[j] = number

        self.computeDistanceData = np.asarray(computeDistanceData)
        self.differntClassInstacneNumber = differntClassInstacneNumber
        self.classOfSpecies = y_value_class
        #To eliminate the influence of the selected number of features and the range of attribute values,
        # all data need to be scaled to the range of [0,1]
        # 将数据缩小到 [0,1] 之间
        self.computeDistanceData = self.computeDistanceData / self.computeDistanceData.sum()

    # 计算平均距离   ----TSHFS-ACO
    def distancs_ACO(self,selectFeature):
        # 用于存放不同分类的数据值   differentClass[[],[],[]]
        differentClass = np.asarray(self.computeDistanceData)
        #print(type(self.computeDistanceData))
        # 获得总共有多少个样本
        allInstanceNumber = self.allData.shape[0]
        # 用于得到每一个类所对应的样本数有多少个
        differntClassInstacneNumber = self.differntClassInstacneNumber
        # 得到类的种类
        y_value_class = self.classOfSpecies
        # 存放平均距离
        balanced_distance = 0
        # 所有样本的总距离
        endSumData = 0
        # 存放每一个instance计算出来的distance
        tempInstanceDistance = np.asarray(np.zeros(allInstanceNumber))
        eachInstanceDistance = np.asarray(np.zeros(allInstanceNumber))
        # 将找到的第一个数据赋值给findedData
        findedData_x = differentClass[:,selectFeature]
        #findedData_x = differentClass[:,selectFeature[0]]

        # # 获得需要那几个特征
        # for j in range(1, int(self.needFindFN)):
        #     tempData = differentClass[:,int(selectFeature[j])]
        #     # 将本只蚂蚁选择的特征包含的数据保存在findeedData_x中
        #     findedData_x = np.c_[findedData_x, tempData]

        #存放每一个类的平均距离
        eachClassDistance = np.asarray(np.zeros(len(y_value_class)))
        # 设置头索引
        headIndex = 0
        for class_i in range(len(y_value_class)):
            # 设置尾索引
            tailIndex = int(headIndex + differntClassInstacneNumber[class_i])

        # 得到每一样本的distance
            for instance_i in range(headIndex, tailIndex):
                for instance_j in range(0,allInstanceNumber):
                    if(instance_j != instance_i ):
                        distance_ij = self.distacne_ij(findedData_x[instance_i],findedData_x[instance_j])
                        tempInstanceDistance[instance_j] = distance_ij
                        #print("tempInstanceDistance[j] = ",tempInstanceDistance[j])
                    else:
                        tempInstanceDistance[j] = 0

                # 得到了   相同类的距离
                # tempInstanceDistance[headIndex:tailIndex:1] 是一维数组，
                # 所以 切片headIndex:tailIndex 步长为1
                sameClassData = tempInstanceDistance[headIndex:tailIndex:1]
                # 得到了   不同类的距离
                #diffClassData = tempInstanceDistance[differntClassInstacneNumber[0],-1]
                diffClassData = np.delete(tempInstanceDistance , [headIndex,tailIndex],axis=0)
                # 相同类要计算平均距离
                tempSum = -(sameClassData.sum()/(differntClassInstacneNumber[class_i]-1))
                # 和不同类计算最小距离
                minDiffDis = diffClassData.min()
                # 得到了该样本的distance值
                eachInstanceDistance[instance_i] = tempSum + minDiffDis
        # 计算balanced _ distance
            tempDistance = eachInstanceDistance[headIndex:tailIndex:1]
            eachClassDistance[class_i] = tempDistance.sum()/differntClassInstacneNumber[class_i]
        balanced_distance = eachClassDistance.sum()/len(y_value_class)
        return balanced_distance

    # 计算每一个类的距离函数
    def distacne_ij(self,instance_i,instance_j):
        distance = sum(map(lambda i, j: abs(j - i), instance_i, instance_j))
        distance = round(distance,4)
        return distance



    # 信息素的更新
    def updataPheromone(self):
        # 信息素的蒸发
        self.pheromone = self.pheromone*(1-self.evaRate)

        # 获得当前轮的最优组合
        tempConduct = self.GConduct
        # 数组的特征
        featureList = self.dataLabel

        featureList = range(0,len(self.dataLabel))
        #del featureList[-1] # 删除掉类标签
        #print("featureList = ",featureList)

        # 存放当前最优特征组合的每一个特征
        tempNode = 0
        # 该特征在特征表中的索引
        tempindex = 0
        for i in range(0,len(tempConduct)):
            tempNode = tempConduct[i]
            # 得到当前节点的索引
            #tempindex = featureList.index(tempNode)

            tempindex = np.where(featureList == tempNode)[0][0]
            self.pheromone[tempindex] = self.pheromone[tempindex] + self.e * self.delta

        for i in range(0, len(self.pheromone)):
            if self.pheromone[i] > 0.15:
                self.pheromone[i] = 0.1
            elif self.pheromone[i] < 0.05:
                self.pheromone[i] = 0.055


    # 运行
    def runACOFunction(self,ACO,needFindFN):
        self.needFindFN = needFindFN
        runTime = 0
        while(runTime <= self.iterationTime):
            # 初始化蚁群数组
            self.antList = []
            # 生成蚁群
            self.initAnts(self.numberOfAnt)
            print("生成蚁群")
            # 进行一次迭代（让所有蚂蚁找到自己的路径）
            for ant_i in range(0,self.numberOfAnt):
                print("第",ant_i,"只蚂蚁计算路径")
                self.antList[ant_i].constructSolution(ACO=ACO)
            print("所有蚂蚁寻找完毕各自路径")
            # 找到最优路径
            self.selectBestPath(ACO=ACO)
            print("寻找最优路径完成。。")
            # 跟新信息素
            self.updataPheromone()
            print("第",runTime+1,"次信息素",self.pheromone)
            mGConduct = self.getBestCondution(self.GConduct)
            print("第",runTime+1,"次迭代，最优路径为",mGConduct)
            print("适应度值为：",self.GFitness)
            print("更新信息素完成")
            runTime+=1

            # 保存所有次迭代里的最优组合

            if(self.GlFitness < self.GFitness):
                # 将当前最优适应度值保存为历史最优
                self.GlFitness = self.GFitness
                self.GlConduct = self.GConduct
                # 获取特征的原始index
                oriGlFitness = self.getBestCondution(tempGlConduct=self.GlConduct)
            else:
                oriGlFitness =self.getBestCondution(tempGlConduct=self.GConduct)
        self.pheromone = self.initPheromone
        print("最优路径为",self.GlConduct)
        print("适应度值为",self.GlFitness)
        return self.GlFitness , self.GlConduct ,oriGlFitness

    def getBestCondution(self,tempGlConduct):
        mGlConduct = []
        for i in tempGlConduct:
            mGlConduct.append(self.dataLabel[int(i)])
        return mGlConduct

# __init__(self,dataColum,featureLabel,iterationTime,numberOfAnt,
#                  evarate,greedyRate,alpha,beta,e,needFindFN):

if __name__ == "__main__":
    #dataCsv  = DataCSV(path="D:\MachineLearningBackUp\\data\\diabetes.csv")
    dataCsv = DataCSV(path="D:\MachineLearningBackUp\data\ionosphere.csv")
    dataCsv.getData()
    testACO = ACO(dataCsv=dataCsv,allData=dataCsv.allData,dataColum=dataCsv.dataAllColum,featureLabel=dataCsv.dataAttribute)
    testACO.getParameter(iterationTime=20,numberOfAnt=100,evarate=0.15,greedyRate=0.4,alpha=5,beta=1,e=0.125,omega=0)
    testACO.runACOFunction(ACO=testACO,needFindFN=70)


