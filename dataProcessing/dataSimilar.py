import  numpy as np
# 计算相似度
def featureOfSimilar(dataX):
    dataFeatureLen = dataX.shape[1]
    # 用于保存每一个特征的相似度
    allFeatureSimilar = [[] for i in range(0, dataFeatureLen)]
    # 用于保存每一个特征的power
    allFeaturePower = []
    # 用于保存每一个特征最后的相似度
    originS = []
    import time
    start = time.time()
    # 获得每一个样本的平方开根号
    for i in range(0, dataFeatureLen):
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
    for i in range(0, dataFeatureLen):  # 是因为 datafeature包含了class标签
        # 用于保存特征i与其他特征的相似性
        cor_i = 0
        # 得到第i个特征所代表的数据 和 第j个特征所代表的数据
        c_i = np.asarray(dataX[:,i])
        # 获得第i个特征的值
        sum_pow_i = allFeaturePower[i]
        for j in range(i, dataFeatureLen):
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
        # sum(axis = 0) 是最后得到行   axis = 1 是最后的到列；如果是一维数据，是 N*1的，所以axis = 0，是得到一行
        cor_i = np.sum(allFeatureSimilar[i], axis=0) / (dataFeatureLen - 1)

        originS = np.append(originS, cor_i)

    print(f"similar time = {time.time() - start} seconds")
    return originS