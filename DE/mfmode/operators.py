
import numpy as np


# 交叉 crossover
# from mfmode.SoftmaxScale import softmax


def crossover_num(x, v_x, crossPro):
    D = len(x)
    u_x = np.zeros(D)
    jRand = np.random.randint(0,D)
    for i in range(D):
        if i == jRand or np.random.rand() < crossPro:
            u_x[i] = v_x[i]
        else:
            u_x[i] = x[i]
    return u_x


# 变异
def mutate_best(xDE,xMain,x1,x2,F):
    temp1 = xMain - xDE
    temp1 = F * temp1
    temp2 = x1 - x2
    temp2 = F * temp2
    v_x =xDE + temp1 + temp2
    return v_x

# 变异
def mutate_rand1(xMain,x1,x2,F):
    temp2 = x1 - x2
    temp2 = F * temp2
    v_x =xMain + temp2
    return v_x

def mutate_rand2(x1,x2,x3,x4,x5,F):
    temp1 = x2 - x3
    temp1 = F * temp1
    temp2 = x4 - x5
    temp2 = F * temp2
    v_x =x1 + temp1 + temp2
    return v_x


def variable_swap(p1, p2, probswap):
  D = p1.shape[0]
  swap_indicator = np.random.rand(D) <= probswap
  c1, c2 = p1.copy(), p2.copy()
  c1[np.where(swap_indicator)] = p2[np.where(swap_indicator)]   # 为True 就把值进行交换，为False就不交换
  c2[np.where(swap_indicator)] = p1[np.where(swap_indicator)]
  return c1, c2


# 得到occurrence number ，每个特征在该代种群中被选中的比例
def getFreq(solution):
    # axis=0 获得一行数据，就是按照列相加； axis=1 获得一列数据，就是按照行相加
    occurrenceNumber = np.sum(solution,axis=0)
    # 求 softMax
    wij = softmax(numberOfAllFeatureSelect = occurrenceNumber)
    return wij

# 传入一个数组，是所有个体所选择所有特征的个数
def softmax(numberOfAllFeatureSelect):
    # 转化为array类型
    numSelect = np.asarray(numberOfAllFeatureSelect)
    # 得到平均数
    meanNum = numSelect.mean()
    # 方差
    sigma = numSelect.std()
    # 计算 softmax后的值
    # 指数
    exponent = -(numSelect - meanNum)/sigma
    Wij = np.exp(exponent)
    Wij = 1/(1 + Wij)
    return Wij

#  得到 OccFreq ,每个特征的正确率和未选择率的平均值
def getAccRecMean(self,accRecPrevious,accRecRecent):
    # 平均acc
    meanAccP = accRecPrevious[:,0].mean()
    meanAccR = accRecRecent[:,0].mean()
    # 平均rec
    meanRecP = accRecPrevious[:,1].mean()
    meanRecR = accRecRecent[:,1].mean()
    popProf = meanRecR/(meanRecP + 0.00000001) * meanAccR/(meanAccP + 0.00000001)
    return popProf