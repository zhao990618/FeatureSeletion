import numpy as np
#import pandas as pd

#import DataCSV as dc
from dataProcessing.DataCSV import *
from skfeature.utility.mutual_information import su_calculation,information_gain

class InforGain:
    # 特征矩阵
    pheromoneMatrix = []
    # 特征之间SU表
    featureSU = []
    # 特征与类之间的SU表
    featureClassSU = []
    # 特征信息熵表
    featureEntropy = []
    # 类标签的信息熵
    classEntropy = []
    #   返回所有特征数组
    dataAttribute = []
    #   返回所有样本值，除类标签值
    dataInstance = []
    #   类标签的所有值
    dataClass = []
    #   总样本数量
    numOfInstance = 0
    #   保存了每一列数据
    dataAllColum = []
    # 保存数据
    dataX = np.asarray([])
    # 保存互信息
    mutualInformation = []
    # 特征和类的SU
    suArray = np.asarray([])
    # 用于备份
    initsuArray = np.asarray([])






    def __init__(self,dataAttribute,dataX):
        #self.dataClass = np.asarray(dataClass)
        self.dataAttribute = np.asarray(dataAttribute)
        #self.dataInstance = np.asarray(dataInstance)
        #self.dataAllColum = np.asarray(dataAllColum)
        self.dataX =dataX
        # 运行计算信息熵矩阵  和 条件熵矩阵
        #self.getFeatureInforEnttopy()


    def getInforEntropy(self,feature_i):
        feature_i_entropy = 0
        #featureCloum_i = self.dataAllColum[feature_i]
        featureCloum_i = self.dataX[:,feature_i]
        featureCloum_i = np.asarray(featureCloum_i)
        # 得到一个没有重复的数组，将testDataClass中的没有重复的数组提取出来
        y_value_class = set(featureCloum_i[i] for i in range(featureCloum_i.shape[0]))
        for y_value in y_value_class:
            # 计算每一个不重复样本的 概率 p
            p = float(len(featureCloum_i[featureCloum_i == y_value]) / featureCloum_i.shape[0])
            logP = np.log2(p)
            feature_i_entropy -= p * logP
        return feature_i_entropy # 特征i的信息熵


    # 计算i j 的条件熵   并且返回了 特征i 个特征j 的信息熵 还有ij的条件熵
    def getConditionEntropy(self,feature_i,feature_j):
        feature_i_entropy = 0
        feature_j_entropy = 0
        #信息熵
        ent = 0
        #信息熵类
        entClass = []
        testDataInstance = self.dataAllColum
        length = testDataInstance.shape[0]  # 得到多少个数组   length-1 则是不包括最后一列 类标签
        # testDataClass = self.dataClass  # 计算类的
        testDataClass = testDataInstance[feature_j]
        testDataClass = np.asarray(testDataClass)
        #print(testDataClass)
        #print(testDataClass.shape)
        # 得到一个没有重复的数组，将testDataClass中的没有重复的数组提取出来
        y_value_class = set(testDataClass[i] for i in range(testDataClass.shape[0]))
        for y_value in y_value_class:
            # 计算每一个不重复样本的 概率 p
            p = float(len(testDataClass[testDataClass == y_value]) / testDataClass.shape[0])
            logP = np.log2(p)
            ent -= p * logP
            entClass.append(ent)  # 类标签的信息熵
        #获得特征j的信息熵
        feature_j_entropy = ent
        y_value_class = list(y_value_class)
        y_value_class = np.asarray(y_value_class)
        #print(y_value_class)

        # 计算不同特征和类之间的信息熵

        x_value_class = []
        a = testDataInstance[feature_i]  # 一个特征所具有的数据
        # b = testDataInstance[length-1] # 类标签所具有的数据
        b = testDataInstance[feature_j]  # 类标签所具有的数据
        c = np.vstack((a, b))  # 结合一个特征和类标签  或者 结合两个特征
        temp1 = set(a[i] for i in range(a.shape[0]))  # 将该特征标签所带有的数据提取出来，不重复提取,并且升序排序
        x_value_class = list(temp1)


        x_value_class = np.asarray(x_value_class)

        # 用于保存每一个特征所指向的类，分别有多少   行代表了多少个样本  列代表了多少个类
        featureClass = np.random.uniform(0, 0, (x_value_class.shape[0], y_value_class.shape[0]))

        # 执行 得到矩阵
        tempData = []
        featureIndex = 0
        classIndex = 0
        for i in range(0, len(c[0])):
            tempData = c[:, i]

            #featureIndex = x_value_class.index(tempData[0])
            featureIndex = np.where(x_value_class == tempData[0])[0][0]
            #classIndex = y_value_class.index(tempData[1])
            classIndex = np.where(y_value_class == tempData[1])[0][0]
            featureClass[featureIndex][classIndex] = featureClass[featureIndex][classIndex] + 1


        # 特征i中每一个值占总数量的比例
        featureClassProbability = np.asarray(np.zeros(x_value_class.shape[0]))
        instanceNumberofFeature = 0
        for i in range(0, x_value_class.shape[0]):
            tempData = featureClass[i]
            featureClassProbability[i] = sum(tempData)
        instanceNumberofFeature = sum(featureClassProbability)
        featureClassProbability = featureClassProbability / instanceNumberofFeature

        # 获得特征i的信息熵
        feature_i_entropy = sum(-(featureClassProbability * np.log2(featureClassProbability)))


        # 计算该特征上的每一个未重复样本的信息熵
        length_f = x_value_class.shape[0]  # 得到该特征中有多少个未重复的样本
        sumFeature_class = 0
        #print(x_value_class.shape[0])
        eachValue_entropy = np.random.uniform(0,0,(1,x_value_class.shape[0]))[0]
        #print(eachValue_entropy)
        # print("======================")
        for i in range(0, length_f):
            boolean_is_1 = True
            tempData = featureClass[i]
            sumFeature_class = sum(tempData)
            tempData = tempData / sumFeature_class
            # print(tempData)
            for j in range(0, len(tempData)):
                if (tempData[j] == 1):
                    boolean_is_1 = False
                # elif(tempData[j] == 0):
                #     boolean_is_1 = False
            if (boolean_is_1):
                #eachValue_entropy[i] = sum(-(tempData * np.log2(tempData + 1e-5)))  #
                eachValue_entropy[i] = sum(-(tempData * np.log2(tempData)))  #
            else:
                eachValue_entropy[i] = 0

        f_ij_ConditionEntropy = sum(eachValue_entropy * featureClassProbability)
        #f_ij_ConditionEntropy = round(f_ij_ConditionEntropy, 5)

        return  feature_i_entropy,feature_j_entropy,f_ij_ConditionEntropy



    #  得到SU矩阵  返回了su矩阵  feature_i  和 feature_j的su矩阵
    def getSUMatrix(self):
        feature_i_entropy = 0 #self.feature_i_entropy
        feature_j_entropy = 0 #self.feature_j_entropy
        f_ij_ConditionEntropy = 0
        # 信息熵
        testDataInstance = self.dataAllColum
        length = testDataInstance.shape[0]# 得到多少个数组   length-1 则是不包括最后一列 类标签
        #print(length)
        #生成一个n*n的矩阵
        tempPheromoneMatrix = np.random.uniform(0,0,(length-1,length-1))

        for i in range(0,length-1):
            for j in range(0,length-1):
                if (j == i):
                    tempPheromoneMatrix[i][j] = 0
                else:
                    feature_i_entropy,feature_j_entropy,f_ij_ConditionEntropy  = self.getConditionEntropy(i,j)
                    SU_value = 2*(feature_j_entropy - f_ij_ConditionEntropy)/(feature_i_entropy + feature_j_entropy)
                    SU_value = round(SU_value,5)
                    tempPheromoneMatrix[i][j] = SU_value
        sumProbability = sum(sum(index) for index in tempPheromoneMatrix )

        self.featureSU = tempPheromoneMatrix/sumProbability

        for i in range(0,length-1):
            for j in range(0,length-1):
                self.featureSU[i][j] = round(self.featureSU[i][j],5)

        return  self.featureSU


    # 得到su矩阵  feature ---class
    def getFeatureClassSU(self,dataX,dataY):
        instanceOfClass = -1
        length = dataX.shape[1]
        tempSU = 0

        #  test
        acc = 0
        count = 0
        import time
        start = time.time()
        self.featureClassSU = np.zeros(length)
        f1 = dataY
        for feature_i in range(0,length-1):

            f2 = dataX[:,feature_i]

            tempSU_test = su_calculation(f1,f2)

            # 引用别人的方法
            self.featureClassSU[feature_i] = tempSU_test

        print(f"SU time = {time.time() - start} seconds")
        return self.featureClassSU

    # 特征i和特征j的互信息
    def getMutualInformation_ff(self,dataX):
        instanceOfClass = -1
        length = len(dataX[0])
        tempSU = 0
        # self.mutualInformation = np.random.uniform(0, 0, (1, length - 1))[0]
        mutualInformation1 = [[] for i in range(0,length)]
        import time
        start = time.time()
        for feature_i in range(0, length):
            f1_sum = 0
            f1 = dataX[:,feature_i]
            for feature_j in range(feature_i,length):
                if feature_i != feature_j:
                    f2 = dataX[:,feature_j]
                    tempSU = information_gain(f1, f2)
                    # f1_sum = f1_sum + tempSU
                    mutualInformation1[feature_i].append(tempSU)
                    mutualInformation1[feature_j].append(tempSU)
                else:
                    mutualInformation1[feature_i].append(1)
        self.mutualInformation = np.mean(mutualInformation1,axis=1)
        # print(self.mutualInformation.shape)
        print(f"mutualInforamation time = {time.time() - start} seconds")
        return mutualInformation1

    # 获取特征和类之间的互信息  feature ---class
    def getMutualInformation_mode_fc(self, dataX):
        instanceOfClass = -1
        length = len(dataX[0])
        tempSU = 0
        # self.mutualInformation = np.random.uniform(0, 0, (1, length - 1))[0]
        self.mutualInformation = np.zeros(length)
        f1 = self.dataX[:, instanceOfClass]
        import time
        start = time.time()
        for feature_i in range(0, length):
            # featureOfEntropy, classOfEntropy, FCConditionRNtropy = self.getConditionEntropy(feature_i, instanceOfClass)
            # print(featureOfEntropy," ",classOfEntropy," ",FCConditionRNtropy)
            # tempSU = classOfEntropy - FCConditionEntropy
            f2 = dataX[:, feature_i]

            # class - H(class|feature)
            tempSU = information_gain(f1, f2)

            temp1 = information_gain(f2, f2)

            self.mutualInformation[feature_i] = tempSU
        print(f"mutualInforamation time = {time.time() - start} seconds")
        # self.mutualInformation = self.mutualInformation
        return self.mutualInformation



    # 获取特征和类之间的互信息  feature ---class
    def getMutualInformation_fc(self,dataX,dataY):
        length = len(dataX[0])
        tempSU = 0
        self.mutualInformation = np.zeros(length)
        f1 = dataY
        import time
        start = time.time()
        for feature_i in range(0, length):

            f2 = dataX[:,feature_i]
            tempSU = information_gain(f1,f2)

            self.mutualInformation[feature_i] = tempSU
        print(f"mutualInforamation time = {time.time() - start} seconds")

        return self.mutualInformation

    # 通过SU来进行筛选数据
    # fiter 方法  获取前 0.03% 的数据  ----------用于高维数据
    def getDataOfFilterInSU(self,suArray ,extratRatio,dataAttribute,dataX,dataAllColum):
        tempArray = []
        tempArray.append(np.arange(0, len(dataAttribute)).tolist())
        tempArray.append(list(suArray))
        tempArray = np.asarray(tempArray)
        # print("tempArray = ",tempArray)
        # 得到了按照最后一行进行排序后的索引排序  ， 正序排序
        allIndex = np.lexsort(tempArray)
        #   将该索引翻转，得到降序排序
        # b = a[::-1]
        # 得到了该索引排序所对应的数据排序，是一个按照最后一行降序排序的数组
        # tempArray = tempArray.T[b].T
        # 得到了该索引排序所对应的数据排序，是一个按照最后一行升序排序的数组
        tempArray = tempArray.T[allIndex].T

        # print(tempArray)
        delectRatio = 1 - extratRatio  # delectRatio = 0.97

        delectIndex = allIndex[0:int(len(suArray) * delectRatio)]
        featureIndex = allIndex[int(len(suArray) * delectRatio):]
        # 删除该数组中前 delectRatio*len(self.pheromone)个特征
        #newArray = np.delete(tempArray, range(0, int(len(suArray) * delectRatio)), axis=1)
        # 将 tempArray转化成list 加速删除速度
        newArray = np.zeros((2,len(featureIndex)))
        for i in range(0,len(featureIndex)):
           newArray[0][i] =  tempArray[0][int(len(suArray) * delectRatio) + i]
           newArray[1][i] =  tempArray[1][int(len(suArray) * delectRatio) + i]

        # 将类标签赋值给中间变量，用于最后添加上去
        # 将类标签添加回上去
        dataAttribute = np.asarray(newArray[0])
        # 将信息素矩阵传回上去
        suArray = np.asarray(newArray[1])
        splie = np.asarray(dataAttribute, dtype='int')
        dataX = dataX[:, splie]
        # newArray = np.delete(self.dataAllColum, delectIndex, axis=0)
        dataAllColum = dataAllColum[splie]
        return suArray , dataAttribute ,dataAllColum,dataX

    # 通过Pro来进行筛选数据
    # fiter 方法  获取前 0.03% 的数据  ----------用于高维数据
    def getDataOfFilterInPro(self,probabilityOfFeatureToSelect ,mutualInforArray,similarityOfFeature,extratRatio,dataAttribute,dataX,dataAllColum):
        tempArray = []
        tempArray.append(np.arange(0, len(dataAttribute)).tolist())
        tempArray.append(list(mutualInforArray))
        tempArray.append(list(similarityOfFeature))
        tempArray.append(list(probabilityOfFeatureToSelect))
        tempArray = np.asarray(tempArray)

        # 得到了按照最后一行进行排序后的索引排序  ， 正序排序
        allIndex = np.lexsort(tempArray)
        # 将tempArray中每一行数据按照allIndex进行一个排序
        tempArray = tempArray.T[allIndex].T

        delectRatio = 1 - extratRatio  # delectRatio = 0.97

        # featureIndex = allIndex[int(len(scoreOfRelifF) * delectRatio):]

        newArray = tempArray[:, int(len(probabilityOfFeatureToSelect) * delectRatio):]

        # 将类标签赋值给中间变量，用于最后添加上去
        # 将类标签添加回上去
        dataAttribute = np.asarray(newArray[0])
        mutualInforArray = np.asarray(newArray[1])
        similarityOfFeature = np.asarray(newArray[2])
        # 将信息素矩阵传回上去
        probabilityOfFeatureToSelect = np.asarray(newArray[3])

        splie = np.asarray(dataAttribute,dtype='int')
        dataX = dataX[:,splie]

        dataAllColum = dataAllColum[splie]

        return probabilityOfFeatureToSelect , dataAttribute ,similarityOfFeature,mutualInforArray,dataAllColum,dataX

    #通过relifF进行筛选数据
    # 用于 MFEA
    def getDataOfFilterIn_MFEA(self,scoreOfRelifF ,mutualInforArray,similarityOfFeature,extratRatio,dataAttribute,dataX):
        # 特征长度
        dataL = int(scoreOfRelifF.shape[0] * extratRatio)
        # 从大到小排序
        allIndex = np.argsort(scoreOfRelifF)[::-1]
        allIndex = allIndex[:dataL]
        # 进行重组
        scoreOfRelifF = scoreOfRelifF[allIndex]
        mutualInforArray = mutualInforArray[allIndex]
        similarityOfFeature = similarityOfFeature[allIndex]
        dataAttribute = dataAttribute[allIndex]
        dataX = dataX[:,allIndex]

        return scoreOfRelifF , dataAttribute ,similarityOfFeature,mutualInforArray,dataX

if __name__ == "__main__":
    dataCsv  = DataCSV("D:\MachineLearningBackUp\dataCSV\Breast.csv")
    dataCsv.getData()
    infor = InforGain(dataCsv.dataAttribute,dataCsv.dataAllColum,dataCsv.dataX)
    #infor.getSUMatrix()
    #infor.getFeatureClassSU()
    #infor.getMutualInformation()
    #infor.getDataOfFilterInPro()
    #infor.getMutualInformation_fc(dataCsv.dataX)

    print()
