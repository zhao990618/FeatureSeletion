# from dataProcessing.DataCSV import DataCSV
from dataProcessing.ReadDataCSV_new import ReadCSV
import numpy as np
import os

# 得到每一个数据的similar
def writeSimilarFun(path,dataName):
    #file = open('D:\MachineLearningBackUp\dataCSV\similarData\BreastSimilar.txt', mode='w')
    #file = open('D:\MachineLearningBackUp\dataCSV\similarData\\arceneSimilar.txt', mode='w')
    reservePath = "D:\\MachineLearningBackUp\\dataCSV\\dataCSV_similar\\"+dataName+"Similar.txt"
    file = open(reservePath, mode='w')
    dataCsv = ReadCSV(path=path)
    # 读取数据
    dataCsv.getData()
    # dataCsv.dataX = dataCsv.dataX[1:,1:]  # 用于马欢的数据集，第一行是特征标签的
    print("读取"+dataName+"数据完毕，进行计算similar")
    # 计算similar
    originSimilarFeature = featureOfSimilar(dataX=dataCsv.dataX)
    print("向"+dataName+"文件写入数据")
    for i in range(0,len(originSimilarFeature)):
        m = originSimilarFeature[i]
        m = str(m) + "\n"
        file.writelines(m)
    file.close()
    print("===========done===========")


def readSimilar(path):
    print("读取文件的数据")
    similarArray = np.asarray([])
    with open(path,'r') as file:
        data = file.read().splitlines()
    similarArray = np.asarray(data,dtype='float')

    return similarArray

 # 计算相似度
def featureOfSimilar(dataX):
    originSimilarityOfFeature = []
    #dataFeature = np.arange(dataX.shape[1])
    # 用于保存每一个特征的相似度
    allFeatureSimilar = [[] for i in range(0, dataX.shape[1])]
    # 用于保存每一个特征的power
    allFeaturePower = []
    import time
    start = time.time()
    # 获得每一个样本的平方开根号
    for i in range(0, dataX.shape[1]):
        # print(i)
        p_i = np.asarray(dataX[:,i])
        # 先平方
        pow_i = np.power(p_i, 2)
        # 平方和
        sum_i = pow_i.sum()
        # 开根号
        sum_i = np.power(sum_i, 0.5)
        # 将该平方值保存
        allFeaturePower.append(sum_i)
    for i in range(0, dataX.shape[1]):
        # 用于保存特征i与其他特征的相似性
        cor_i = 0
        # 得到第i个特征所代表的数据 和 第j个特征所代表的数据
        c_i = np.asarray(dataX[:,i])
        # 获得第i个特征的值
        sum_pow_i = allFeaturePower[i]
        for j in range(i, dataX.shape[1]):
            if i == j:
                allFeatureSimilar[i].append(0)
            if i != j:
                c_j = np.asarray(dataX[:,j])
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
        cor_i = np.sum(allFeatureSimilar[i], axis=0) / (dataX.shape[1] - 1)
        # cor_i = 1 - cor_i
        originSimilarityOfFeature = np.append(originSimilarityOfFeature, cor_i)
    print(f"similar time = {time.time() - start} seconds")
    return originSimilarityOfFeature

# 得到文件夹下所有文件的名字
def getDocumentAll(path):
    files = os.listdir(path)
    i = 0
    for file in files:
        try:
            # 将路径组合
            dataPath = path +"\\"+ file
            data_name = file.split('.')[0]
            # 写similar
            writeSimilarFun(path=dataPath,dataName=data_name)

        except:
            ## 跳过一些系统隐藏文档
            pass
        i += 1

if __name__ == "__main__":
    #writeSimilarFun()
    #readSimilar(path='D:\MachineLearningBackUp\dataCSV\similarData\BreastSimilar.txt')
    getDocumentAll(path="D:\\MachineLearningBackUp\\dataCSV\\dataCSV_temp")
