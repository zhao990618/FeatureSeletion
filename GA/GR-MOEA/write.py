from dataProcessing.DataCSV import DataCSV
import numpy as np

# 得到每一个数据的similar
def writeSimilarFun():
    file = open('D:\MachineLearningBackUp\dataCSV\similarData\BreastSimilar.txt', mode='w')
    dataCsv = DataCSV("D:\MachineLearningBackUp\dataCSV\Breast.csv")
    #dataCsv = DataCSV("D:\MachineLearningBackUp\data\ionosphere.csv")
    dataCsv.getData()
    print("读取数据完毕，进行计算similar")
    originSimilarFeature = featureOfSimilar(dataFeature=dataCsv.dataAttribute,dataColum=dataCsv.dataAllColum)
    print("向文件写入数据")
    for i in range(0,len(originSimilarFeature)):
        m = originSimilarFeature[i]
        m = str(m) + "\n"
        file.writelines(m)
    file.close()


def readSimilar(path):
    with open(path,'r') as file:
        data = file.read().splitlines()
    similarArray = np.asarray(data,dtype='float')
    return similarArray

 # 计算相似度
def featureOfSimilar(dataFeature,dataColum):
    originSimilarityOfFeature = []
    # 用于保存每一个特征的相似度
    allFeatureSimilar = [[] for i in range(0, len(dataFeature) - 1)]
    # 用于保存每一个特征的power
    allFeaturePower = []
    import time
    start = time.time()
    # 获得每一个样本的平方开根号
    for i in range(0, len(dataFeature) - 1):
        # print(i)
        p_i = np.asarray(dataColum[i])
        # 先平方
        pow_i = np.power(p_i, 2)
        # 平方和
        sum_i = pow_i.sum()
        # 开根号
        sum_i = np.power(sum_i, 0.5)
        # 将该平方值保存
        allFeaturePower.append(sum_i)
    for i in range(0, len(dataFeature) - 1):  # 是因为 datafeature包含了class标签
        # 用于保存特征i与其他特征的相似性
        cor_i = 0
        # 得到第i个特征所代表的数据 和 第j个特征所代表的数据
        c_i = np.asarray(dataColum[i])
        # 获得第i个特征的值
        sum_pow_i = allFeaturePower[i]
        for j in range(i, len(dataFeature) - 1):
            if i == j:
                allFeatureSimilar[i].append(0)
            if i != j:
                c_j = np.asarray(dataColum[j])
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
        cor_i = np.sum(allFeatureSimilar[i], axis=0) / (len(dataFeature) - 2)
        # cor_i = 1 - cor_i
        originSimilarityOfFeature = np.append(originSimilarityOfFeature, cor_i)
    print(f"similar time = {time.time() - start} seconds")
    return originSimilarityOfFeature


if __name__ == "__main__":
    #writeSimilarFun()
    readSimilar(path='D:\MachineLearningBackUp\dataCSV\similarData\BreastSimilar.txt')
