import  numpy as np


# 传入一个数组，是所有个体所选择所有特征的个数
def softmax(numberOfAllFeatureSelect):
    # 转化为array类型
    numSelect = np.asarray(numberOfAllFeatureSelect)
    # 得到平均数
    meanNum = numSelect.mean()
    # 计算方差
    numSelect = meanNum - numSelect
    #   均差 由 sigma表示
    sigma = np.power(numSelect,2).sum()
    sigma = sigma/len(numSelect)
    # 计算 softmax后的值
    # 指数
    exponent = -(numSelect - meanNum)/sigma
    Wij = np.exp(exponent)
    Wij = 1/(1 + Wij)
    return Wij
