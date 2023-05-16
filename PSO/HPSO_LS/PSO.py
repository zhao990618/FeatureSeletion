import copy

import  numpy as np
import pandas as pd
import random
from dataProcessing.DataCSV import DataCSV
from dataProcessing.ReadDataCSV_new import ReadCSV
from dataProcessing.KneePointDivideData import findKneePoint
from dataProcessing.reliefF import reliefFScore
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from dataProcessing.writeSimilarHigh import readSimilar
from dataProcessing.InforGain import InforGain
from sklearn.preprocessing import MinMaxScaler

#  hybirdPSO ----------->HPSO_LS  通过控制特征数量来进行局部搜索
class PSO:
    # 样本值表
    allData = []
    # 保存特征表
    dataFeature = []
    # 数据
    dataX = np.asarray([])
    # class
    dataY = np.asarray([])


    # 特征相似性
    similarityOfFeature = np.asarray([])

    # 粒子需要查找的特征数量
    sf = 0
    # 特征和类的SU
    suArray = np.asarray([])
    initsuArray = np.asarray([])
    # 保存不相似的特征
    featureOfDissimilar =np.asarray([])
    # 保存相似的特征
    featureOfSimilar = np.asarray([])
    # 迭代次数
    numberOfIterator = 0

    # 粒子群
    particleList = []
    # 粒子数量
    numberOfParticle = 0
    # 粒子最大速度
    maxV = 0
    # 粒子最小速度
    minV = 0
    # 自我认知参数
    c_1 = 0
    # 全局认知参数
    c_2 = 0

    #用于分割相似特征数量 和不相似特征数量的
    alpha = 0
    # rilifF 数组
    scoreOfRelifF = []
    # 全局最优选择
    globalBestSolution = np.asarray([])
    # 全局最优选择得到的fitness
    globalBestFitness = 0

    def __init__(self,dataCsv,dataX,dataY):

        self.dataFeature = np.arange(dataX.shape[1])
        self.dataX = dataX
        self.dataY = dataY

        print("计算每一个特征的相似度")
        # 用于高维数据
        self.featureOfSimilar_high(dataName="")
        # 对数据01 归一化，将每一列的数据缩放到 [0,1]之间
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        # 进行filter
        self.filter()

        print("现在所剩余的特征数量为",len(self.dataFeature))

        print("将每一个特征按照相似度进行分类")
        # 进行分类
        self.classifyFeature()

        print("len.featureOfSimilar = ",len(self.featureOfSimilar))
        print("len.featureOfDissimilar = ",len(self.featureOfDissimilar))



    # 设置参数
    def setParameter(self,maxV,minV,c_1,c_2,alpha,numberOfParticle,numberOfIterator):
        # 定义最大最小速度
        self.maxV = maxV
        self.minV = minV
        # 学习力
        self.c_1 = c_1
        self.c_2 = c_2
        # 相似特征占所选特征数量比例
        self.alpha = alpha
        # 粒子数量
        self.numberOfParticle = numberOfParticle
        # 迭代次数
        self.numberOfIterator = numberOfIterator

    #  用于高维数据  --- 提前处理好similar 保存在txt中
    def featureOfSimilar_high(self, dataName):
        # allFeatureSimilar = readSimilar(path="D:\\MachineLearningBackUp\\dataCSV\\dataCSV_similar\\"+dataName+"Similar.txt")
        allFeatureSimilar = readSimilar(path="D:\\MachineLearningBackUp\\dataCSV\\dataCSV_similar\\BreastCancer1Similar.txt")
        # 不应该排序
        self.similarityOfFeature = np.asarray(allFeatureSimilar)


    # fiter 方法  获取拐点位置的数据  ----------用于高维数据
    def filter(self):

        # 得到relifF
        print("得到所有特征的relifF")
        self.scoreOfRelifF = reliefFScore(self.dataX, self.dataY)

        # 得到su矩阵
        print("得到互信息矩阵")
        infor = InforGain(dataAttribute=self.dataFeature, dataX=self.dataX)
        self.suArray = infor.getFeatureClassSU(dataX=self.dataX,dataY=self.dataY)  # 用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵
        #self.mutualInforArray = infor.getMutualInformation_fc(dataX=self.dataX)  # 用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵

        # 拐点分割特征，选择拐点前的数据
        if len(self.dataFeature) > 1000:
            # extratRatio = findKneePoint(data=self.scoreOfRelifF)
            extratRatio = 0.03
        else:
            extratRatio = 1
        # 通过prob进行排序
        print("通过relifF进行筛数据")
        self.scoreOfRelifF, self.dataFeature, self.similarityOfFeature, self.suArray, self.dataX \
            = infor.getDataOfFilterIn_MFEA(
            scoreOfRelifF=self.scoreOfRelifF,
            extratRatio=extratRatio, dataAttribute=self.dataFeature,
            similarityOfFeature=self.similarityOfFeature, mutualInforArray=self.suArray,
            dataX=self.dataX)

        # 计算每个特征被选择的概率，通过 mutialInformation 和 similar来进行计算
        print("计算概率")
        # 对 relifF进行归一化，放缩到[0.2,0.7]之间
        self.scoreOfRelifF = ((self.scoreOfRelifF - np.min(self.scoreOfRelifF)) / (
            np.max(self.scoreOfRelifF - np.min(self.scoreOfRelifF))) + 0.4) * 0.5



    # 确定粒子需要查找多少个特征
    def defineNumberOfFeature(self):
        originNumberOfFeature = len(self.dataFeature)
        # 高维数组
        # 控制了边界的最大值
        epsilon = np.random.random() * 0.55 + 0.25
        # 边界的最小值
        borderMin = 3

        # # 低维数组  用于测试
        # epsilon = np.random.random() * 0.45 + 0.4
        # borderMin = 1

        # 边界的最大值
        borderMax = int(epsilon * originNumberOfFeature)
        #print("borderMax = ",borderMax)
        probabilityOfsf = np.asarray([])
        # [borderMin , borderMax]
        #sf = int(random.random() * (borderMax - borderMin) + borderMin)
        #sf = np.random.randint(borderMin ,borderMax)
        for sf in range(borderMin , borderMax+1):
            l = originNumberOfFeature - sf
            sum = 0
            for i in range(1,l+1):
                sum += originNumberOfFeature - i
            l_sf = l/sum
            probabilityOfsf = np.append(probabilityOfsf,l_sf)
        # 轮盘赌
        rand = np.random.random()# 随机一个数
        mselect = 0
        indexOfsf = 0
        for j in probabilityOfsf:
            mselect += j
            if rand <= mselect:
                # 获得所对应的那个sf的概率的索引
                indexOfsf = np.where(probabilityOfsf == j)[0][0]
                break
        # 最小区间是从3开始 ，而索引从0开始，即索引加上3 ，就可以得到所需要的sf的数量
        sf = indexOfsf + borderMin
        self.sf = sf


    # 初始化粒子群，创建 n个粒子
    def createParticleList(self,numberOfParticle):
        # 设置 numberOfParticle 个粒子
        for i in range(0,numberOfParticle):
            particle = self.Particle(PSO=self)
            self.particleList = np.append(self.particleList,particle)
            # print(particle,"  ",i,particle.featureOfParticleChoose)
            # print(particle,"  ",i,particle.velocityOfAllFeature)
        print("给每一个粒子初始化位置和速度完成")
        # 将其设置为array类型
        self.particleList = np.asarray(self.particleList)



    # 将特征进行分类，
    def classifyFeature(self):
        half = int(len(self.similarityOfFeature)/2)
        # 将特征进行分类  ， 前一半特征放在 dissimilarFeature  后一半特征放在 similarFeature
        tempIndex = np.argsort(self.similarityOfFeature)
        tempS = self.similarityOfFeature[tempIndex]
        self.featureOfDissimilar = tempS[0:half]
        self.featureOfSimilar = tempS[half:-1]
        # for i in range(0,len(self.similarityOfFeature)):
        #     if i < half:
        #         # 将前半特征存放在不相似组
        #         self.featureOfDissimilar = np.append(self.featureOfDissimilar,self.similarityOfFeature[i])
        #     else:
        #         # 将后半特征存放在相似组
        #         self.featureOfSimilar = np.append(self.featureOfSimilar,self.similarityOfFeature[i])




    # 粒子
    class Particle:
        # 每一个粒子所选择的特征, 以 0 ，1 来表示,长度为所有特征数量
        featureOfParticleChoose = np.asarray([])
        # 每一个粒子的每一个维度上的初始速度，范围[0,1]
        velocityOfAllFeature = np.asarray([])
        # 特征总数
        numberOfFeature  = 0
        # 最大速度
        maxVelocity = 0
        # 最小速度
        minVelocity = 0

        # 个体最优选择
        individualBestSolution = np.asarray([])
        # 个体最优选择得到的fitness
        individualBestFitness = 0
        # 全局最优选择
        globalBestSolution = np.asarray([])
        # 全局最优选择得到的fitness
        globalBestFitness = 0

        # c_1,c_2  ， 自我学习力 和 社会学习力
        c_1 = 2
        c_2 = 2


        def __init__(self,PSO):
            self.maxVelocity = PSO.maxV
            self.minVelocity = PSO.minV
            self.numberOfFeature = len(PSO.dataFeature)
            self.c_1 = PSO.c_1
            self.c_2 = PSO.c_2


            # 初始化一个位置
            self.defineInitPosition()
            # 初始化速度
            self.defineInitVelocity()


        # 给每一个粒子初始化一个位置
        def defineInitPosition(self):
            # temArray = []
            # for i in range(0,self.numberOfFeature):
            #     temArray.append(np.random.randint(0, 2))
            # self.featureOfParticleChoose = np.random.randint(0,2,(self.numberOfFeature))
            rand = np.random.rand(self.numberOfFeature)
            self.featureOfParticleChoose = np.where(rand > 0.6 ,1,0)

        # 给每一个粒子初始化一个速度
        def defineInitVelocity(self):
            #self.velocityOfAllFeature = np.random.uniform(0,1,(0,self.numberOfFeature))# 起码是个二维数组
            self.velocityOfAllFeature = np.random.rand(self.numberOfFeature)
            #print(self.velocityOfAllFeature)

        # 跟新速度
        def updateVelocity(self):
            r1 = random.uniform(0,1)
            r2 = random.uniform(0,1)
            a = np.asarray(self.individualBestSolution - self.featureOfParticleChoose)
            b = np.asarray(self.globalBestSolution - self.featureOfParticleChoose)
            a = a * self.velocityOfAllFeature
            temp1 = self.c_1 * r1 * a
            temp2 = self.c_2 * r2 * b
            self.velocityOfAllFeature = self.velocityOfAllFeature + temp1 + temp2
            #print(self.velocityOfAllFeature)
            # print(self.velocityOfAllFeature[0])
        # 更新粒子位置
        def updataParticleSolution(self):
            tempArray = np.asarray([])
            # 将速度限定在一个范围内   [minVelocity,maxVelocity]

            for i in range(0,len(self.velocityOfAllFeature)):
                # 小于最小速度，则将该维度的速度设置为 minVelocity
                if self.velocityOfAllFeature[i] < self.minVelocity:
                    self.velocityOfAllFeature[i] = self.minVelocity
                    # 大于最大速度，则将该维度的速度设置为 maxVelocity
                elif self.velocityOfAllFeature[i] > self.maxVelocity:
                    self.velocityOfAllFeature[i] = self.maxVelocity
            # 计算公式
            tempExp = np.exp((self.velocityOfAllFeature * -1))
            tempExp = 1 + tempExp
            tempExp = 1 / (tempExp + 0.00000001)
            tempArray = np.asarray(tempExp)
            # 获得一个随机值 [tempArrat.min() , tempArray.max()]
            rand = random.uniform(tempArray.min(),tempArray.max())
            # 更新粒子的下一轮位置
            for i in range(0,len(tempArray)):
                if rand < tempArray[i]:
                    self.featureOfParticleChoose[i] = 1
                else:
                    self.featureOfParticleChoose[i] = 0


    # 更新粒子群的速度和位置
    def updateParticle(self):
        for particle_i in self.particleList:

            # 更新速度
            particle_i.updateVelocity()

            # 更新位置
            particle_i.updataParticleSolution()


    # 获得最优粒子
    def selectBestSolution(self):
        # 获得临时存放的粒子群
        tempParticleList = self.particleList
        # 存放每一个粒子的适应度函数
        eachParticleFitness = np.asarray([])
        # 判断是否出现了全局最优粒子
        boolean_is_global = False
        # 保存本轮的最优粒子
        bestParticle = 0
        bestFitness = 0
        # 最优粒子的索引
        bestFitnessIndex = 0
        # 计算每一个粒子的适应度函数
        for i in range(0,len(tempParticleList)):
            #print("i = ",i+1)
            #粒子精度
            acc = 0
            # 得到第i个粒子
            particle_i = tempParticleList[i]
            # 获得每一个粒子所选择的特征
            selectFeature_i = np.copy(particle_i.featureOfParticleChoose)

            # 让 feature_x 存放训练的数据
            feature_x = self.dataX[:,selectFeature_i == 1]
            # 让 feature_y 存放相对因的类的值
            feature_y = self.dataY

            # 计算每一个粒子的acc  通过分类的方式
            # acc_1 =  self.fitnessFunction_percent(findData_x=feature_x,findData_y=feature_y)
            acc_1 =  fitnessFunction_KNN_CV(findData_x=feature_x,findData_y=feature_y,CV=10)
            # 进行每一个粒子的局部搜索,改变了该粒子的所选择的特征组合，然后进行下一轮比较
            acc_2,selectFeature_2  = self.localSearch(particle_i)
            # print("acc_1 = ",acc_1,selectFeature_i)
            # print("acc_2 = ",acc_2,selectFeature_2)

            # 准确度哪一个大就选哪一个
            if acc_1 > acc_2:
                acc = acc_1
            elif acc_1 < acc_2:
                acc = acc_2
                selectFeature_i = selectFeature_2
            else:# 如果准确度一样大   那么选择的特征数那个小，选择哪一个
                len_1 = np.where(selectFeature_i == 1)[0]
                len_2 = np.where(selectFeature_2 == 1)[0]
                if len(len_1) <= len(len_2):
                    acc = acc_1
                elif len(len_1) > len(len_2):
                    acc = acc_2
                    selectFeature_i = selectFeature_2


            eachParticleFitness = np.append(eachParticleFitness , acc)
            # 进行跟新，当前粒子所选择的特征
            particle_i.featureOfParticleChoose = np.copy(selectFeature_i)

            # 跟新粒子的历史最优
            # 判断当前粒子在该轮计算的fitness 是否大于该粒子的历史最优fitness，是则更新firness和solution，不是则跳过
            if acc > particle_i.individualBestFitness:
                particle_i.individualBestFitness = acc
                #print("更新路径")
                particle_i.individualBestSolution =copy.deepcopy(selectFeature_i)
            #print("第",i,"个粒子的solution为",selectFeature_i)
            # print("第",i,"个粒子的fitness为",acc)
            # print("第",i,"个粒子的个体最优solution为",particle_i.individualBestSolution)
            # print("第",i,"个粒子的个体最优fitness为",particle_i.individualBestFitness)
            # print("全局最优路径",particle_i.globalBestSolution)
            # print("============================================================")

        # 获得最优粒子的fitness
        bestFitness = eachParticleFitness.max()
        print(bestFitness)
        # 获得最优粒子的索引
        beatFitnessIndex = np.where(eachParticleFitness == bestFitness)[0][0]
        # 得到了本轮的最优粒子
        bestParticle = tempParticleList[bestFitnessIndex]
        # 得到全局最优，判断当前粒子的fitness 是否是全局最优 的，是则更新firness和solution，不是则跳过

        if bestFitness > self.globalBestFitness:
            self.globalBestFitness = bestFitness
            self.globalBestSolution = selectFeature_i
            boolean_is_global = True

        #更新每一个粒子的全局最优粒子参数
        if boolean_is_global :
            for p_i in tempParticleList:
                p_i.globalBestSolution = self.globalBestSolution
                p_i.globalBestFitness = self.globalBestFitness



    # 数据缩放 ，将数据缩放在[-1,1]之间，这种有利于减少噪音样本对模型的影响，主要适用于距离计算的分类器，KNN SVM等
    # 而用概率计算的根本不考虑这个距离，只关心该值在所有样本中的一个概率分布，距离对于这个来说无关紧要
    def lineNormalization(self,featureData):
        # 先转化成array类型，也不知道上面转没转，反转这里转一下
        eachFeatureData = np.asarray(featureData)
        # 获取这个特征所包含数据的最大值和最小值
        featureMaxData = eachFeatureData.max()
        featureMinData = eachFeatureData.min()
    # 进行归一化
        denominator = featureMaxData - featureMinData + 0.00000001 # 避免除数为0
        # 将数据缩放到[0,1]之间
        eachFeatureData = (eachFeatureData - featureMinData)/denominator
        # 然后继续缩放  获得边界最大 边界最小 和 边界范围
        borderMax = 1  #eachFeatureData.max()
        borderMin = -1 #eachFeatureData.min()
        border = borderMax - borderMin
        # 将数据缩放到[-1,1]之间
        eachFeatureData = borderMin + border * eachFeatureData
        return eachFeatureData

    # 适应度函数 10折交叉
    def fitnessFunction_CV(self, findData_x, findData_y):
        # 计算1
        # 先采用10折交叉验证的方式计算
        knn = KNeighborsClassifier(n_neighbors=1, algorithm="auto", metric='manhattan')
        # cv参数决定数据集划分比例，这里是按照9:1划分训练集和测试集
        scores = cross_val_score(knn, findData_x, findData_y, cv=10, scoring='accuracy')
        accuracy = scores.mean()
        return accuracy

    # 适应度函数  70% 训练  30%实验   -----  实验结果偏高 因为是测试集。可能分出来的测试集容易分类
    def fitnessFunction_percent(self, findData_x, findData_y):
        # 计算2
        # 创建了一个knn分类器的实例，并拟合数据
        x_train, x_test, y_train, y_test = train_test_split(findData_x, findData_y, test_size=0.3, random_state=1)
        clf = neighbors.KNeighborsClassifier(5, weights="distance")
        # 转化为二维数组
        if len(x_train.shape) == 1:
            x_train = x_train.reshape((len(x_train), 1))

        clf.fit(x_train, y_train)
        if len(x_test.shape) == 1:
            x_test = x_test.reshape((len(x_test), 1))
        predictOfTest = clf.predict(x_test)
        numberOfTrue = 0
        for i in range(0, len(predictOfTest)):
            if predictOfTest[i] == y_test[i]:
                numberOfTrue += 1
        accuracy = numberOfTrue / len(predictOfTest)
        return accuracy


    # 加强局部搜索   用的适应度函数是  10-CV  10折交叉
    def localSearch(self,particle_i):
        # 用于保存已选择的相似特征x_s 和不相似特征x_d
        x_s = np.asarray([])
        x_d = np.asarray([])

        # 用于保存相似矩阵 和不相似矩阵  ,用于删除
        tempSimilarArray = self.featureOfSimilar
        tempDissimilarArray = self.featureOfDissimilar

        # 相似性array 不相似array
        similarArrayIndex = np.asarray([])
        dissimilarArrayIndex = np.asarray([])

        # 相似特征的数量
        n_s = int(self.alpha * self.sf)
        # 不相似特征的数量
        n_d = self.sf - n_s

        # 得到了该粒子所选择的特征
        featureOfSelect = np.copy(particle_i.featureOfParticleChoose)
        # 首先得到每个粒子所选择的索引的list
        featureOfSelectIndex = np.where(featureOfSelect == 1)[0]
        # 得到每一个索引代表的特征的相似度是多少
        similarOfSelectFeature = np.asarray([])
        for i in featureOfSelectIndex:
            # 将每一个所选择的特征的索引带入到相对应的array中，用于获得所对应的相似度值
            similarOfSelectFeature = np.append(similarOfSelectFeature,self.similarityOfFeature[i])

        # 将所选择的特征进行分类，通过相似度值的一个排序
        half = int(len(self.similarityOfFeature)/2)
        # 获取正序排序完成后的中间值 ： halfOfSimilar
        halfOfSimilar = self.similarityOfFeature[half]

        for index  in range(0,len(similarOfSelectFeature)):
            similarValue = similarOfSelectFeature[index]

            # 大于中间值的 存入 相似
            if similarValue >= halfOfSimilar:
                x_s = np.append(x_s,similarValue)
                # 每当有一个特征被选中后，在这个数组中删除其备份，即这个数组中只保留了不被选择的特征
                #print("similar",np.where(tempSimilarArray == similarValue), " ", similarValue)
                tempIndex = np.where(tempSimilarArray == similarValue)[0]
                tempSimilarArray = np.delete(tempSimilarArray,tempIndex)
                # 将该特征在原数据上的索引添加到x_s
                similarArrayIndex = np.append(similarArrayIndex,featureOfSelectIndex[index])
            else:
                # 否则就是 不相似
                x_d = np.append(x_d,similarValue)
                # 每当有一个特征被选中后，在这个数组中删除其备份，即这个数组中只保留了不被选择的特征
                tempIndex = np.where(tempDissimilarArray == similarValue)[0]
                #print("dissimilar",np.where(tempDissimilarArray == similarValue), " ", similarValue)
                tempDissimilarArray = np.delete(tempDissimilarArray,tempIndex)
                # 将该特征在原数据上的索引添加到x_d
                dissimilarArrayIndex = np.append(dissimilarArrayIndex,featureOfSelectIndex[index])

        # 将所获取的索引，通过其相似度值进行排序
        similarArrayIndex = self.sortOFIndexAboutSimilar(similarArrayIndex)
        dissimilarArrayIndex = self.sortOFIndexAboutSimilar(dissimilarArrayIndex)


        # 获取 x_d中有多少个数据
        x_d_number = x_d.shape[0]
        # 获取 x_s 中有多少个数据
        x_s_number = x_s.shape[0]

        if n_d >= x_d_number:
            # 需要添加的特征数量
            gap = n_d - x_d_number
            for i in range(0,gap):
                addFeatureSimilar = tempDissimilarArray[i]
                addFeatureIndex = np.where(self.similarityOfFeature == addFeatureSimilar)[0][0]
                featureOfSelect[addFeatureIndex] = 1
        else:
            # 需要删除的特征数量
            gap = x_d_number - n_d
            for i in range(1,gap+1):
                tempIndex = int(dissimilarArrayIndex[-i])
                featureOfSelect[tempIndex] = 0
        if n_s >=x_s_number:
            # 需要添加的特征数量
            gap = n_s - x_s_number
            for i in range(0,gap):
                addFeatureSimilar = tempSimilarArray[i]
                addFeatureIndex = np.where(self.similarityOfFeature == addFeatureSimilar)[0][0]
                featureOfSelect[addFeatureIndex] = 1
        else:
            # 需要删除的特征数量
            gap = x_s_number - n_s
            for i in range(1,gap+1):
                tempIndex = int(similarArrayIndex[-i])
                featureOfSelect[tempIndex] = 0

        # 让 feature_x 存放训练的数据
        feature_x = self.dataX[:,featureOfSelect == 1]
        # 让 feature_y 存放相对因的类的值
        feature_y = self.dataY

        acc = fitnessFunction_KNN_CV(findData_x=feature_x,findData_y=feature_y,CV=10)
        # acc = self.fitnessFunction_percent(findData_x=feature_x,findData_y=feature_y)
        return  acc , featureOfSelect

    # 通过索引，来对传入的索引数组进行一个相似度排序
    def sortOFIndexAboutSimilar(self,indexArray):
        # 用来保存排序好的similar的index
        sortIndex = np.asarray([])
        # 用来保存排序好的similar
        arraySort = []
        #print("indexArray = ",indexArray)
        for i in indexArray:
            similarValue = self.similarityOfFeature[int(i)]
            #print("similarValue = ",similarValue)
            arraySort.append(similarValue)
        #print("arraySort = ", arraySort)
        arraySort = sorted(arraySort)
        #print("arraySort = ",arraySort)
        for j in range(0,len(arraySort)):
            index_1 = np.where(self.similarityOfFeature == arraySort[j])[0][0]
            sortIndex = np.append(sortIndex,index_1)
        return sortIndex


    # 返回原数据的索引
    def getOriIndex(self,solution):
        # 得到了被选择特征的索引，在filter后的索引
        solutionIndex = np.where(solution == 1)[0]
        # 通过索引找到其su值
        su_feature = np.asarray([])
        for i in solutionIndex:
            value_su = self.suArray[i]
            # 将su添加到su_feature
            su_feature = np.append(su_feature,value_su)
        # 通过这些su值找到了原始的，未筛选的su数组
        index_feature_ori = np.asarray([])
        for j in su_feature:
            index_feature = np.where(self.initsuArray == j)[0][0]
            index_feature_ori = np.append(index_feature_ori,index_feature)
        return index_feature_ori

    # 运行函数
    def runOfPSO(self):
        # 先定义要多少个特征
        self.defineNumberOfFeature()
        print("确定完成所需要的特征数 = ",self.sf)
        runTime = 0
        # 初始化粒子群数组
        self.particleList = np.asarray([])
        # 生成粒子群  ---> 每个粒子初始化时，会初始化自己的位置和速度
        print("初始化粒子群")
        self.createParticleList(self.numberOfParticle)

        while (runTime <= self.numberOfIterator):
            print("第",runTime+1,"轮")
            print("选择最优粒子")
            #选择最优粒子
            self.selectBestSolution()
            print("跟新最优粒子")
            # 跟新粒子群
            self.updateParticle()
            runTime +=1
        #print("全局最优粒子的solution" ,self.globalBestSolution)
        print("全局最优粒子的fitness" ,self.globalBestFitness)
        print(len(np.where(self.globalBestSolution == 1)[0]))

        # index_feature = self.getOriIndex(self.globalBestSolution)
        # print(index_feature)
        # print("最优粒子的原始数据上的特征选择索引为 ：",index_feature)


#maxV,minV,c_1,c_2,alpha,numberOfParticle,numberOfIterator
if  __name__ == "__main__":
    dataCsv = ReadCSV(path="D:\MachineLearningBackUp\dataCSV\dataCSV_high\\BreastCancer1.csv")
    print("获取文件数据")
    dataCsv.getData()
    pso = PSO(dataCsv=dataCsv, dataX = dataCsv.dataX ,dataY=dataCsv.dataY)
    pso.setParameter(maxV=4,minV=-4,c_1=2,c_2=2,alpha=0.65,numberOfParticle=140,numberOfIterator=100)
    print("参数设置完成")
    print("进行运行")
    pso.runOfPSO()