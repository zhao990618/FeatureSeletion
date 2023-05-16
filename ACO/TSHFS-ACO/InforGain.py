import numpy as np
import pandas as pd

#import DataCSV as dc
from dataProcessing.DataCSV import *

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
    # 保存互信息
    mutualInformation = []




    def __init__(self,dataAttribute,dataAllColum):
        #self.dataClass = np.asarray(dataClass)
        self.dataAttribute = np.asarray(dataAttribute)
        #self.dataInstance = np.asarray(dataInstance)
        self.dataAllColum = np.asarray(dataAllColum)
        # 运行计算信息熵矩阵  和 条件熵矩阵
        #self.getFeatureInforEnttopy()


    def getInforEntropy(self,feature_i):
        feature_i_entropy = 0
        featureCloum_i = self.dataAllColum[feature_i]
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

        # print(x_value_class)
        x_value_class = np.asarray(x_value_class)
        # print(x_value_class)
        #print(x_value_class.shape)
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
                eachValue_entropy[i] = sum(-(tempData * np.log2(tempData + 1e-5)))  #
            else:
                eachValue_entropy[i] = 0

        f_ij_ConditionEntropy = sum(eachValue_entropy * featureClassProbability)
        f_ij_ConditionEntropy = round(f_ij_ConditionEntropy, 5)

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
        # print(self.featureSU)
        # print(type(self.featureSU))
        return  self.featureSU


    # 得到su矩阵  feature ---class
    def getFeatureClassSU(self):
        instanceOfClass = -1
        length = len(self.dataAllColum)
        tempSU = 0
        self.featureClassSU = np.random.uniform(0,0,(1,length-1))[0]
        for feature_i in range(0,length-1):

            featureOfEntropy,classOfEntropy,FCConditionRNtropy = self.getConditionEntropy(feature_i,instanceOfClass)
            #print(featureOfEntropy," ",classOfEntropy," ",FCConditionRNtropy)
            tempSU = 2*(classOfEntropy - FCConditionRNtropy)/(featureOfEntropy + classOfEntropy)
            #print("第",feature_i,"个：",tempSU)
            self.featureClassSU[feature_i] = tempSU
        tempSU = self.featureClassSU
        # 控制到小数点后4位
        # for i in range(0,length-1):
        #     tempSU[i] = round(tempSU[i],4)
        tempV = sum(tempSU)
        tempSU = tempSU / tempV
        self.featureClassSU = tempSU
        #print(self.featureClassSU)
        #print(type(self.featureClassSU))
        return self.featureClassSU

    # 获取特征和类之间的互信息  feature ---class
    def getMutualInformation(self):
        instanceOfClass = -1
        length = len(self.dataAllColum)
        tempSU = 0
        #self.mutualInformation = np.random.uniform(0, 0, (1, length - 1))[0]
        self.mutualInformation = np.zeros(length - 1)
        for feature_i in range(0, length - 1):
            featureOfEntropy, classOfEntropy, FCConditionRNtropy = self.getConditionEntropy(feature_i, instanceOfClass)
            # print(featureOfEntropy," ",classOfEntropy," ",FCConditionRNtropy)
            tempSU = classOfEntropy - FCConditionRNtropy
            # print("第",feature_i,"个：",tempSU)
            self.mutualInformation[feature_i] = tempSU
        return self.featureClassSU



if __name__ == "__main__":
    dataCsv  = DataCSV("D:\MachineLearningBackUp\dataCSV\Breast.csv")
    dataCsv.getData()
    infor = InforGain(dataCsv.dataAttribute,dataCsv.dataAllColum)
    #infor.getSUMatrix()
    infor.getFeatureClassSU()
