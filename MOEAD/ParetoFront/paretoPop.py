import copy
import math

import numpy as np
# 更新pareto解
def updatePF(pop):

    # 存放error的数组  --- [0,1]
    errorArray = np.zeros(pop.shape[0])
    # 存放len的数组  --- [0,1]
    lenArray = np.zeros(pop.shape[0])
    # 用于保存每一个个体的支配解集
    dominateFrontArray = []
    # 保存帕累托前沿,保存非支配数量为0的个体,
    F_i = []
    for i in range(pop.shape[0]):
        # 将每一个个体的 error 存储
        errorArray[i] = pop[i].mineFitness
        # 将每一个个体的 len 存储
        lenArray[i] = pop[i].proportionOfFeature

    ##---------------------------------- ENS  高效非支配排序 ----------------------------------------##
    # 首先对种群按照error进行一个排序，从小到大的一个排序

    # 先得到error进行排序后的索引
    sortErrorIndex = np.argsort(errorArray)
    # 将 error  population len 通过 该索引进行一个重新组合
    errorArray = errorArray[sortErrorIndex]
    parentPopulation = pop[sortErrorIndex]
    lenArray = lenArray[sortErrorIndex]
    # 对相同error的个体进行按照len的一个排序
    i = 0
    while i < len(errorArray) - 1:
        j = i + 1
        if errorArray[i] == errorArray[j]:
            while j < len(errorArray) and errorArray[i] == errorArray[j]:
                j += 1
            if (j - i > 1):
                newArray1 = copy.deepcopy(lenArray[i:j])

                newArray3 = copy.deepcopy(parentPopulation[i:j])
                indexNew = np.argsort(newArray1)
                newArray1 = newArray1[indexNew]
                newArray3 = newArray3[indexNew]
                for m in range(len(newArray1)):
                    lenArray[i] = newArray1[m]
                    parentPopulation[i] = newArray3[m]
                    i += 1
                i -= 1
        i += 1
        # ==========进行迭代 --- 用的是二分查找Front============#

    # 先让 Front集合添加第一个集合
    dominateFrontArray.append([])
    # 将第一个个体添加到第一个集合
    dominateFrontArray[0].append(parentPopulation[0])

    # 第0个已经被添加到dominateFrontArray中去了，所以要从第1个开始进行比较
    for i in range(1, len(parentPopulation)):
        # 设置head 和 tail
        # head 为上界
        head = 0
        # tail 为下界
        tail = len(dominateFrontArray) - 1

        # 第 i 个 个体的 error
        error_i = errorArray[i]
        # 第 i 个 个体的 len
        len_i = lenArray[i]
        # 获得第i个个体
        individual = parentPopulation[i]

        # k 为中间的前沿
        k = math.floor((head + tail) / 2 + 0.5)

        # 进行比较
        while True:
            # 第 k 个前沿点的最后一个个体
            leastIndividual = dominateFrontArray[k][-1]
            # 找到该个体在population的中位置，才能找到其转化后的error 和len
            leastIndex = np.where(parentPopulation == leastIndividual)[0]

            # 若生成的 leastIndex 并不是一个值，而是一堆个体，那么就挑第一个
            if len(leastIndex) > 1:
                leastIndex = leastIndex[0]

            # 获得最后一个个体的error
            leastError = errorArray[leastIndex]
            # 获得最后一个个体的len
            leastLen = lenArray[leastIndex]
            # 若满足下条件 则  leastIndividual  支配 第i个个体
            if (leastError < error_i and leastLen < len_i) or (
                    leastError <= error_i and leastLen < len_i) or (
                    leastError < error_i and leastLen <= len_i):
                # 需要往下找下一个前沿
                head = k
                # 如过 head 和 tail 中间没有前沿了，并且tail 并不是最后一个前沿，那就把个体添加在tail所在的前沿中
                if (tail == head + 1 and tail < len(dominateFrontArray) - 1):
                    dominateFrontArray[tail].append(individual)
                    break
                if (tail == head + 1 and tail == len(dominateFrontArray) - 1):
                    head = tail
                    k = tail
                # 如果 tail 为最后一个前沿，则添加一个新的前沿到dominateFrontArray
                elif (head == len(dominateFrontArray) - 1):
                    dominateFrontArray.append([])
                    dominateFrontArray[-1].append(individual)
                    break
                # 换到下一个二分点
                else:
                    k = math.floor((head + tail) / 2 + 0.5)
            else:
                # 如果没有被支配
                # 如果只剩下三个前沿，存放在中间区域
                if k == head + 1:
                    tail = k
                    k = head
                elif (k == head == tail):
                    dominateFrontArray[tail].append(individual)
                    break
                elif k == head:
                    dominateFrontArray[head].append(individual)
                    break
                else:
                    tail = k
                    k = math.floor((head + tail) / 2 + 0.5)
    # 将前沿保留下来
    EP = copy.deepcopy(dominateFrontArray[0])
    return EP