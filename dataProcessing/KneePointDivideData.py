import numpy as np

# 通过计算传进来的数据的拐点，来返回拐点所在位置的比例
def findKneePoint(data):
    # 获得排序后的索引 --- 从大到小
    sort_index = np.argsort(data)[::-1]
    # 将数据进行排序
    data = data[sort_index]
    # 进行连接最大值和最小值组成的线段函数
    minValue = data[-1]  # 最小值
    maxValue = data[0]  # 最大值
    # 该最小值和最大值组成的线段的梯度  min的索引为 len(data) - 1  max的索引为 0
    k_maxmin = (minValue - maxValue)/(len(data) - 1)
    b_maxmin = compute_b(knee=k_maxmin,x=0,y=data[0])
    # 与该线段相互垂直，并且过每一个数据点的线段梯度为 - 1/k
    k_knee = -1/k_maxmin
    # 保存在maxmin相连接线段上的点的坐标
    croodinate = np.zeros(2)

    # max距离
    maxDistance = -1
    # max索引
    maxDistanceIndex = 0

    # count = 5
    for i in range(0,len(data)):
        # 计算该点的b
        b_knee = compute_b(knee=k_knee,x=i,y=data[i])
        # 计算x
        croodinate[0] = (b_knee - b_maxmin)/(k_maxmin - k_knee)
        # 计算y = x * k + b
        croodinate[1] = croodinate[0] * k_maxmin + b_maxmin
        # 计算两个点的距离
        x_2 = np.power(croodinate[0] - i,2)
        y_2 = np.power(croodinate[1] - data[i],2)
        distance = np.power(x_2 + y_2,0.5)

        # count -= 1
        # 如果当前计算的距离，要大于最大距离，则更新
        if distance > maxDistance:
            # 更新最大距离
            maxDistance = distance
            # 保存最大距离的索引
            maxDistanceIndex = i
        #     # 重新计数
        #     count = 5
        # if count == 0:
        #     break
    # 获得该最大距离点在全局的比例值为多少
    extratRatio = maxDistanceIndex / len(data)
    return extratRatio

def compute_b(knee,x,y):
    b = y - knee * x
    return b