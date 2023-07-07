import copy
import os
import time
import numpy as np
import random
import xlwt
from filter.Filter import load_csv, reliefFScore, top_select
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier


# 初始化特征阈值集合thresSet
def init_thresSet(D, initThres):
    """
    :param D: 特征数目
    :param initThres: 算法的初始阈值
    :return: 特征阈值集合
    """
    thresSet = np.ones(D) * initThres
    return thresSet


def init_pop(X, Y, N, D, model, thresSet, kf, gp, bp, I):
    """
    :param X: dataSet
    :param Y: label
    :param model: 分类器模型
    :param thresSet: 特征阈值集合
    :param D: 特征数目
    :param kf: K折交叉检验
    :param gp: 优秀特征的选择范围,例如:gp = 30,表示取前30%特征,[0, 0.30)
    :param bp: 较差特征的选择范围,例如: bp = 30, 表示取后30%特征,[0.7, 1]
    :param I: 特征阈值每次更新的变化幅度, 默认 I = 0.01
    :return:
    """
    M = np.identity(D).tolist()  # 生成一个单位阵
    score = fit(X, Y, M, model, thresSet, D, kf)  # 得到所有特征的评分
    ######
    # 获取优秀特征和较差特征
    first = int(D * gp)
    last = int(D * bp)
    lo_goodFeature = np.argsort(score)[0:first]  # 得到优秀特征的位置
    lo_badFeature = np.argsort(score)[last:D]  # 得到较差特征的位置
    ########
    # 更新优秀特征和较差特征的阈值
    thresSet[lo_goodFeature] = thresSet[lo_goodFeature] + I
    thresSet[lo_badFeature] = thresSet[lo_badFeature] - I
    ##########
    # 生成种群
    pop = getPop(thresSet, N, D)
    return pop, thresSet


def getPop(thresSet, N, D):
    """
    :param thresSet: 特征阈值集合
    :param N: 个体数目
    :param D: 特征数目
    :return: 种群pop
    """
    pop = np.zeros([N, D])
    for i in range(N):
        for j in range(D):
            if random.random() < thresSet[j]:
                pop[i, j] = 1.0
    return pop


def fit(X, Y, pop, model, thresSet, D, kf):
    """
    :param X: dataSet
    :param Y: label
    :param pop: 输入的种群
    :param model: 所选择的分类器模型
    :param thresSet: 特征阈值集合
    :param D: 特征数目
    :param kf: K折交叉检验
    :return: 返回适应度集合fit(所有个体准确度的集合)
    """
    fit = []
    N = np.shape(pop)[0]  # 得到种群长度
    for i in range(N):
        if np.count_nonzero(pop[i]) == 0:  # 如果某个个体没有选择特征,会报错,所以重新生成一个个体
            pop[i] = getPop(thresSet, 1, D)[0]
        row = np.array(pop[i], dtype=bool)  # 从[0, 1, 1, 0, 1]转化为[False, True, True, False, True]
        select_X = X[:, row]  # 提取相应位置的特征
        acc = np.mean(cross_val_score(model, select_X, Y, cv=kf))  # 返回准确度acc
        err = 1 - acc
        fit.append(err)
    return np.array(fit)


def getScore(pop, fit, D):
    """
    :param pop: 种群
    :param fit: 包含所有个体适应度的集合
    :param D: 特征数目
    :return: 包含所有特征的特征评分的集合score
    """
    score = np.zeros([D, 2])
    for i in range(D):
        # 计算每个特征的平均评分和最低评分
        col = np.array(pop[:, i], dtype=bool)
        if len(fit[col]) != 0:
            # 当N个个体中存在所选特征
            score[i, 0] = np.sum(fit[col]) / len(fit[col])
            score[i, 1] = np.min(fit[col])
        else:
            # 当N个个体中都不存在所选特征
            score[i, 0] = 0
            score[i, 1] = 0
    return score


def update_thresSet(thresSet, score, gp, bp, D, minThres, maxThres, I):
    """
    :param thresSet: 特征阈值集合
    :param score: 特征评分
    :param gp: 优秀特征的选择范围,例如:gp = 30,表示取前30%特征,[0, 0.30)
    :param bp: 较差特征的选择范围,例如: bp = 30, 表示取后30%特征,[0.7, 1]
    :param D: 特征数目
    :return: 更新后的特征阈值集合thresSet
    """
    pareto = fast_non_dominated_sort(score[:, 0], score[:, 1])
    # 对每个前沿进行重新排序
    pareto_sort = []
    for front in pareto:
        if len(front) == 1:
            pareto_sort.append(front[0])
        else:
            lo_front_sort = np.argsort(score[front, 0])
            for i in lo_front_sort:
                pareto_sort.append(front[i])

    # 获取优秀特征和较差特征
    first = int(D * gp)
    last = int(D * bp)
    lo_goodFeature = pareto_sort[0:first]  # 得到优秀特征的位置
    lo_badFeature = pareto_sort[last:D]  # 得到较差特征的位置
    ########
    # 更新优秀特征和较差特征的阈值
    thresSet[lo_goodFeature] = thresSet[lo_goodFeature] + I
    thresSet[lo_badFeature] = thresSet[lo_badFeature] - I
    ##########
    # 防止特征阈值超出范围
    lo_minThres = np.where(thresSet < minThres)[0]  # 得到低于最小阈值的特征位置
    lo_maxThres = np.where(thresSet > maxThres)[0]  # 得到高与最大阈值的特征位置
    thresSet[lo_minThres] = minThres
    thresSet[lo_maxThres] = maxThres
    return thresSet


def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (
                    values1[p] <= values1[q] and values2[p] < values2[q]) or (
                    values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)  # 把p支配的点放入S[p]
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (
                    values1[q] <= values1[p] and values2[q] < values2[p]) or (
                    values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1  # 如果p被哪个点支配，则n[p]+1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


def championships(pop, fit, N, num=3):
    """
    :param pop: 种群
    :param fit: 适应度值
    :param N: 种群数目
    :param num: 从几个个体中选择
    :return: 一个个体
    """
    lo = random.sample(range(N), num)
    bestFit = None
    bestP = None
    for ele in lo:
        if bestFit is None or fit[ele] < bestFit:
            bestFit = fit[ele]
            bestP = pop[ele]
    return bestP


def cross(p1, p2, D, t, T, thresSet):
    """
    :param p1: 个体1
    :param p2: 个体2
    :param D: 特征数目
    :param t: 当前的迭代次数
    :param T: 最大迭代次数
    :param thresSet: 特征阈值集合
    :return:
    """
    s1 = copy.deepcopy(p1)
    s2 = copy.deepcopy(p2)
    if t <= int(T / 2):
        for i in range(D):
            if random.random() < 0.5:
                if s1[i] != s2[i]:
                    tmp = s1[i]
                    s1[i] = s2[i]
                    s2[i] = tmp
        return s1, s2
    else:
        for i in range(D):
            if random.random() < thresSet[i]:
                if s1[i] != s2[i]:
                    tmp = s1[i]
                    s1[i] = s2[i]
                    s2[i] = tmp
        return s1, s2


def mutation(p, thresSet, D, t, T):
    c = copy.deepcopy(p)
    rand = random.randint(1, int(D / 5))  # 从1~D/5,中随机选择一个一个数字
    H = []  # 存放变异位点的数组
    for i in range(rand):
        lo = random.sample(range(D), 3)
        best_thres = None
        best_lo = None
        for ele in lo:
            if best_thres is None or thresSet[ele] < best_thres:
                best_thres = thresSet[ele]
                best_lo = ele  # 所选变异位点
        H.append(best_lo)
    H = list(set(H))  # 去除重复位点
    if t <= int(T / 2):
        for ele in H:
            if random.random() < 0.5:
                if c[ele] == 0:
                    c[ele] = 1
                else:
                    c[ele] = 0
        return c
    else:
        for ele in H:
            if random.random() < thresSet[ele]:
                if c[ele] == 0:
                    c[ele] = 1
                else:
                    c[ele] = 0
        return c


def getOffspring(pop1, fit1, pop2, fit2, N, D, t, T, thresSet):
    offspring = []
    for i in range(int(N / 2)):
        # 锦标赛分别从pop和newPop中选择一个个体
        s1 = championships(pop1, fit1, N)
        s2 = championships(pop2, fit2, N)
        c1, c2 = cross(s1, s2, D, t, T, thresSet)
        m1 = mutation(c1, thresSet, D, t, T)
        m2 = mutation(c2, thresSet, D, t, T)
        offspring.append(m1)
        offspring.append(m2)
    return np.array(offspring)


def FTGGA(opts):
    # parameter

    # dataSet
    if 'X' in opts:
        X = opts['X']
    else:
        print("please input parameter X...")
        return -1
    # label
    if 'Y' in opts:
        Y = opts['Y']
    else:
        print("please input parameter Y...")
        return -1
    # 个体数目
    if 'N' in opts:
        N = opts['N']
    else:
        print("please input parameter N...")
        return -1
    # 迭代次数
    if 'T' in opts:
        T = opts['T']
    else:
        print("please input parameter T...")
        return -1
    # 特征数目
    if 'D' in opts:
        D = opts['D']
    else:
        print("please input parameter D...")
        return -1
    # 分类器模型
    if 'model' in opts:
        model = opts['model']
    else:
        print("please input parameter model...")
        return -1
    # K折交叉检验
    if 'kf' in opts:
        kf = opts['kf']
    else:
        print("please input parameter kf...")
        return -1
    # 所有特征的初始阈值
    if 'initThres' in opts:
        initThres = opts['initThres']
    else:
        print("please input parameter initThres...")
        return -1
    # 优秀特征的选取范围,取前gp%
    if 'gp' in opts:
        gp = opts['gp']
    else:
        print("please input parameter gp...")
        return -1
    # 较差特征的选取范围,取后bp%
    if 'bp' in opts:
        bp = opts['bp']
    else:
        print("please input parameter bp...")
        return -1
    # 特征阈值每次更新的变化幅度, 默认 I = 0.01
    if 'I' in opts:
        I = opts['I']
    else:
        print("please input parameter I...")
        return -1
    # 特征的最小阈值
    if 'minThres' in opts:
        minThres = opts['minThres']
    else:
        print("please input parameter minThres...")
        return -1
    # 特征的最大阈值
    if 'maxThres' in opts:
        maxThres = opts['maxThres']
    else:
        print("please input parameter maxThres...")
        return -1
    # # 结果输出路径
    # if 'outputFile' in opts:
    #     outputFile = opts['outputFile']
    # else:
    #     outputFile = False

    # if outputFile:
    #     file = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
    #     sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表

    # code
    t = 0
    curve = []  # 用于存放每一代最优解
    thresSet = init_thresSet(D, initThres)  # 初始化特征阈值集合
    pop, thresSet = init_pop(X, Y, N, D, model, thresSet, kf, gp, bp, I)  # 初始化种群
    while t < T:
        fit_pop = fit(X, Y, pop, model, thresSet, D, kf)
        score = getScore(pop, fit_pop, D)
        thresSet = update_thresSet(thresSet, score, gp, bp, D, minThres, maxThres, I)
        newPop = getPop(thresSet, N, D)
        fit_newPop = fit(X, Y, newPop, model, thresSet, D, kf)
        offspring = getOffspring(pop, fit_pop, newPop, fit_newPop, N, D, t, T, thresSet)
        fit_offspring = fit(X, Y, offspring, model, thresSet, D, kf)
        # 保留前N个个体
        mergePop = np.concatenate((pop, newPop))
        mergePop = np.concatenate((mergePop, offspring))
        mergeFit = np.append(fit_pop, fit_newPop)
        mergeFit = np.append(mergeFit, fit_offspring)
        lo = np.argsort(mergeFit)[0:N]  # 获取前100位置
        pop = mergePop[lo, :]
        fit_pop = mergeFit[lo]
        fitG = np.min(fit_pop)
        curve.append(1 - fitG)
        print("第{}代,准确率为:{}".format(t + 1, 1 - fitG))
        t = t + 1

    Acc = curve[-1]
    parameter = "相关参数: N:{} T:{} initThres:{} gp:{} bp:{} " \
        .format(N, T, initThres, gp, bp)
    Acc = "Accuracy:" + str(100 * Acc)
    featureNumber = "Feature number:" + str(np.count_nonzero(pop[np.argmin(fit_pop)]))

    FTGGA_data = {'parameter': parameter, 'Acc': Acc, 'featureNumber': featureNumber, 'curve': curve}
    return FTGGA_data


if __name__ == '__main__':
    # load data
    filename = ".\\dataCSV_high\\end\\arcene(200,10000).csv"
    X, Y = load_csv(filename)
    minmax = preprocessing.MinMaxScaler()  # 标准化
    X = minmax.fit_transform(X)
    Score = reliefFScore(X, Y)
    top_index = top_select(Score)  # 取前5%
    X = X[:, top_index]
    NUMBER_OF_FEATURES = X.shape[1]

    # parameter
    X = X  # 特征数据
    Y = Y  # 标签数据
    N = 100
    T = 100
    D = np.shape(X)[1]
    K = 5  # K折交叉检验
    model = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
    kf = KFold(n_splits=K, shuffle=True, random_state=1)
    initThres = 0.1  # 定义初始个体产生概率
    gp = 0.5  # 优秀特征的选择范围,例如:gp = 30,表示取前30%特征,[0, 0.30)
    bp = 0.5  # 较差特征的选择范围,例如: bp = 30, 表示取后30%特征,[0.7, 1]
    I = 0.01  # 特征阈值每次更新的变化幅度, 默认 I = 0.01
    minThres = 0.05  # 特征的最小阈值
    maxThres = 0.75  # 特征的最大阈值

    opts = {'X': X, 'Y': Y, 'N': N, 'D': D, 'T': T, 'model': model,
            'kf': kf, 'initThres': initThres, 'gp': gp, 'bp': bp,
            'I': I, 'minThres': minThres, 'maxThres': maxThres}
    FTGGA(opts)
