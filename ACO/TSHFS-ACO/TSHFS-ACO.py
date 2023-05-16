import numpy as np
import pandas as pd
from  InforGain import *
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import  random

from MachineLearn.ACO import ACO


# 二阶段蚁群    -----第一阶段粗略的找到一个特征数量，二阶段通过区间法和梯度下降法来确定最优特征数量

class TSHFSACO:
    # 所有数据
    allData = []
    #每一列的数据
    dataAllColum = []
    # 特征标签 包含类标签
    dataAttribute = []
    # feature 和class 的SU矩阵
    suOfFeatureAndClass = []
    #需要查找的特征数量
    needFindFN = 0
    #特征数量
    numberOfFeature= 0
    #一阶段要进行多少个回合
    boutOfOneStage = 0
    # 在每一个回合中要进行多少次切片
    eachBoutSplitNumber = []
    # k 的更新步长
    k_0 = 0
    # 梯度下降法用的下降比例
    rp = 0
    # 近似最优特征数量
    similarNumberOfFeature = 0
    # 历史最优fitness
    historyOptimalFitness = 0
    # 历史最优特征组合
    historyOptimalConduct = np.asarray([])
    # 定义蚂蚁


    # 传进来dataCsv才可以用ACO
    def __init__(self,dataCsv,boutOfOneStage,eachBoutSplitNumber,k_0,rp):
        dataCsv.getData()

        self.boutOfOneStage = boutOfOneStage
        self.eachBoutSplitNumber = eachBoutSplitNumber
        self.k_0 = k_0
        self.rp = rp
        print("初始化数据")
        self.aco = ACO(dataCsv=dataCsv, allData=dataCsv.allData,
                    dataColum=dataCsv.dataAllColum, featureLabel=dataCsv.dataAttribute)
        self.allData = self.aco.allData
        self.dataAllColum = self.aco.dataColum
        self.dataAttribute = self.aco.dataLabel
        #将跟新好的SU计算
        self.suOfFeatureAndClass = np.copy(self.aco.initPheromone)




    # 一阶段函数    ：区间法
    def stage_one(self):

        print("执行第一阶段，运用切片法得到一个粗略的特征数量")

        #头标志
        head = 0
        #尾标志
        tail = len(self.dataAttribute)-1
        # 找到当前最优切片后 向后再找k个切片
        k = 1
        # 初始化数据
        aco = self.aco

        # 定义第一阶段参数
        self.aco.getParameter(iterationTime=1, numberOfAnt=1, evarate=0.15,
                    greedyRate=0.8, alpha=5, beta=1, e=0.5,
                    omega=0)

        print("start")
        for bout_i in range(0,self.boutOfOneStage):
            m = k
            # 获得总共多少个数据
            self.numberOfFeature = tail - head
            tempNumberOfFeature = head
            # 保存最优fitness
            maxFitness = float("-inf")  # 负无穷
            # 保存每一个fitness
            fitnessArray = np.asarray([])
            # 保存每一个对应的fitness的切片右端点  ， 即保存了 该fitness的蚁群搜索的数量
            numberOfFeatureArray = np.asarray([])
            # 得到了每一轮切片时，每一个切片中的特征数量
            eachSplit_FeatureNumber = self.numberOfFeature / self.eachBoutSplitNumber[bout_i]
            # 进行self.eachBoutSplitNumber[bout_i]次的切片
            for i in range(0,self.eachBoutSplitNumber[bout_i]):

                if m == 0 :
                    break
                else:
                    #每一个切片的右端点的特征数量为 tempNumberOfFeature
                    tempNumberOfFeature += eachSplit_FeatureNumber
                    # 返回了第一个切片的Glfitness,该fitness对应的GConduct
                    print("进行第",bout_i,"轮的第",i,"阶段蚁群及搜索最优解 :",)
                    GFitness,GConduct = self.aco.runACOFunction(ACO=self.aco , needFindFN= int(tempNumberOfFeature))
                    print("第",bout_i,"阶段的，第",i,"轮 :",GConduct,"   ",GFitness)

                    if (maxFitness < GFitness) : # 如果找到了当前最优切片，则继续往下找k个位置
                        maxFitness = GFitness
                        fitnessArray = np.append(fitnessArray,GFitness)
                        numberOfFeatureArray = np.append(numberOfFeatureArray,tempNumberOfFeature)
                        m = k
                    else:# 没有找到当前最优切片，则k--
                        fitnessArray = np.append(fitnessArray,GFitness)
                        numberOfFeatureArray = np.append(numberOfFeatureArray, tempNumberOfFeature)
                        m -= 1

                    # # 如果特征数迭代到当前最后一个节点后，那么就要
                    # if tempNumberOfFeature == tail:
                    #     break

            # 当前数据保存了多少个节点
            fitArrayLenght =  fitnessArray.shape[0]
            # 减1 是因为索引从0开始 ，  -k是因为要多计算了k个切片 m是有可能没有计算完全，如m = 1
            bestFitIndex = fitArrayLenght - 1 - k + m
            # 得到head 和tail指向的位置
            head = numberOfFeatureArray[bestFitIndex - 1]
            tail = numberOfFeatureArray[bestFitIndex + 1]
            k += self.k_0
            print("第一阶段的第", bout_i, "轮，最优fitness为",fitnessArray[bestFitIndex])
            print("第一阶段的第", bout_i, "轮，最优特征数量为",numberOfFeatureArray[bestFitIndex])

        # 跑完所有的轮次
        # 近似的最优特征数 不选择第三轮最优 而是选择第三轮最优的后一个切片位置，毕竟范围大
        self.similarNumberOfFeature = tail
        self.historyOptimalFitness = maxFitness
        print(" 近似特征数量为 = ",self.similarNumberOfFeature)


    # 二阶段   ：梯度下降法
    def stage_two(self):

        print("执行第二阶段，运用梯度下降法计算")

        findedData_x = []
        findedData_y = []
        # 二阶段的蚁群计算
        aco = self.aco
        self.aco.getParameter(iterationTime=1, numberOfAnt=1, evarate=0.15,
                    greedyRate=0.4, alpha=5, beta=1, e=0.125,
                    omega=1)
        # 得到了在similarNumberOfFeature条件下的历史最优fitness  和 特征组合
        GFitness, GConduct ,oriGConduct= self.aco.runACOFunction(ACO=self.aco, needFindFN=self.similarNumberOfFeature)

        print("在当前特征数量下选择的特征组合为 = ",GConduct)
        print("适应度值为 = ",GFitness)
        print("被选择的特征在原始数据中的索引为" , oriGConduct)

        l = int(self.similarNumberOfFeature)

        # 用于保存每一个已选择特征的SU
        arrayOfSU = np.asarray(np.zeros(l))
        # 初始化p ,表示不能连续执行p次梯度下降
        p = 0

        print("进入梯度下降法，通过轮盘赌选择被删除的特征")

        while( p < self.similarNumberOfFeature * self.rp):
            for i in range(0,len(GConduct)):
                #self.suOfFeatureAndClass保存了每一个特征和类的相关性，相关性越高，则被选择当作删除特征的概率就越小
                # 即 self.suOfFeatureAndClass越高   ；arrayOfSU的值就越低
                index = int(GConduct[i])
                arrayOfSU[i] =1 - self.suOfFeatureAndClass[index]
            sumSU = arrayOfSU.sum()
            # 计算出每一个feature被选择的概率
            arrayOfSU = arrayOfSU / sumSU

            delectFeature = 0
            # 轮盘赌
            randomValue = np.random.random()
            fSelect = 0
            for i in range(0,self.similarNumberOfFeature):
                fSelect += arrayOfSU[i]
                if randomValue <= fSelect:
                    delectFeature = GConduct[i]
                    break

            print("选择了特征",delectFeature," 计算其fitness")

            # 将最优组合赋值给newSelectFeature
            newSelectFeature = GConduct
            # 需要删除的特征的索引
            delectIndex = np.where(newSelectFeature == delectFeature)
            newSelectFeature = np.delete(newSelectFeature,delectIndex)

            # 获得选择特征的所有数据放在findedData_x ， 类的数据放在 findedData_y
            selectFeature = sorted(newSelectFeature)
            # 将找到的第一个数据赋值给findedData
            findedData_x = self.dataAllColum[int(selectFeature[0])]
            for j in range(1, self.needFindFN):
                tempData = self.dataAllColum[int(selectFeature[j])]
                # 将本只蚂蚁选择的特征包含的数据保存在findeedData_x中
                findedData_x = np.c_[findedData_x, tempData]
            # 获取得到类标签所对应的数据
            findedData_y = self.dataAllColum[-1]

            # 获得新的fitness
            newFitness = aco.fitnessFunOfACO(findedData_x,findedData_y,selectFeature)
            print("newFitness = ",newFitness)
            if(newFitness > GFitness):
                GFitness = newFitness
                GConduct = newSelectFeature
                p = 0
            else:
                p += 1

        self.historyOptimalFitness = GFitness
        self.historyOptimalConduct = GConduct
        oriGConduct = aco.getBestCondution(GConduct)
        # 最优特征数量
        print("最终选择的特征组合为", self.historyOptimalConduct)
        print("最终选择的最优值为",self.historyOptimalFitness)
        print("最终选择的特征在原始数据上的索引为",oriGConduct)


    # 得CLass与feature的SU矩阵
    def getFeatureAndClassSU(self):
        infor = InforGain(dataAttribute=self.dataAttribute, dataAllColum=self.dataAllColum)
        self.suOfFeatureAndClass = infor.getFeatureClassSU()  # 用信息熵和条件熵组成SU矩阵，来当作初始化信息素矩阵

    def run(self):
        self.stage_one()
        self.stage_two()

if __name__ == "__main__":
        dataCsv  = DataCSV(path="D:\MachineLearningBackUp\dataCSV\Breast.csv")
        #test = TSHFS-ACO(dataCsv = dataCsv ,boutOfOneStage = 3 , eachBoutSplitNumber=[100,8,8],k_0=2,rp= 0.04)
        test = TSHFSACO(dataCsv = dataCsv ,boutOfOneStage = 3 , eachBoutSplitNumber=[10,8,8],k_0=2,rp= 0.04)
        test.run()
        # print("test.allData",test.allData)
        # print("test.dataAttribute",test.dataAttribute)
        # print("test.dataAllColum",test.dataAllColum)