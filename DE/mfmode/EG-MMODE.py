import copy
import math
import random
import numpy as np
from random import sample
import pandas as pd
from sklearn.metrics import pairwise_distances
from Classfier.invokingClassfier import computeFitnessKNN
from sklearn.preprocessing import MinMaxScaler
from skfeature.utility.mutual_information import information_gain
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
from scipy.stats import norm
from scipy.optimize import fminbound

random.seed(1)
class Genetic:
    dataX = np.asarray([])
    dataY = np.asarray([])
    dataFeature = np.asarray([])
    dataFeatureNum = 0
    similarityOfFeature = np.asarray([])
    scoreOfRelifF = np.asarray([])
    mutualInforArray = np.asarray([])
    probabilityOfFeatureToSelect = np.asarray([])
    mrmrArray = np.asarray([])
    solutionArray = []
    population_task1 = np.asarray([])
    population_task2 = np.asarray([])
    populationNum = 0
    iteratorTime = 0
    eliteProb = 0
    levyRate = 0
    crossoverPro = 0
    F = 0
    p = 0
    p_DEarray = []
    eliteFront = 0
    globalSolution = np.asarray([])
    globalSolutionNum = np.asarray([])
    globalFitness = 1
    globalScore = 0
    dataName = ""

    def __init__(self, dataX, dataY ,dataName):
        self.dataX = dataX
        self.dataFeature = np.arange(self.dataX.shape[1])
        self.dataFeatureNum = len(self.dataFeature)
        self.dataY = dataY
        print("得到相似度矩阵")
        if self.dataX.shape[1] > 1000:
            self.featureOfSimilar_high(dataName=dataName)
        else:
            self.similarityOfFeature = self.featureOfSimilar(dataX=self.dataX)
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        self.filter()

    def setParameter(self, populationNum, iteratorTime, eliteProb,crossoverPro, F,p,eliteFront,maxAge):
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime
        self.eliteProb = eliteProb
        self.p = p
        self.crossoverPro = crossoverPro
        self.F = F
        self.eliteFront = eliteFront
        self.maxAge = maxAge

    def filter(self):
        self.scoreOfRelifF = self.reliefFScore(self.dataX, self.dataY)
        self.mutualInforArray = self.getMutualInformation_mode_fc(dataX=self.dataX)
        if len(self.dataFeature) > 1000:
            extratRatio = 0.03
            self.scoreOfRelifF, self.dataFeature, self.similarityOfFeature, self.mutualInforArray, self.dataX \
                = self.getDataOfFilterIn_MFEA(
                scoreOfRelifF=self.scoreOfRelifF,
                extratRatio=extratRatio, dataAttribute=self.dataFeature,
                similarityOfFeature=self.similarityOfFeature, mutualInforArray=self.mutualInforArray,
                dataX=self.dataX)
        else:
            extratRatio = 1
        print("计算概率")
        self.filterOfProFreture()
        self.probabilityOfFeatureToSelect = ((self.probabilityOfFeatureToSelect - np.min(
            self.probabilityOfFeatureToSelect)) / (np.max(self.probabilityOfFeatureToSelect) - np.min(
                                                 self.probabilityOfFeatureToSelect)) + 0.4) * 0.5
        self.scoreOfRelifF = ((self.scoreOfRelifF - np.min(self.scoreOfRelifF)) / (
            np.max(self.scoreOfRelifF - np.min(self.scoreOfRelifF))) + 0.4) * 0.5
        self.mrmrArray = np.full((len(self.dataFeature), len(self.dataFeature)), 0)

    def reliefFScore(self,X, y, **kwargs):
        if "k" not in kwargs.keys():
            k = 5
        else:
            k = kwargs["k"]
        n_samples, n_features = X.shape
        distance = pairwise_distances(X, metric='manhattan')
        score = np.zeros(n_features)
        for idx in range(n_samples):
            near_hit = []
            near_miss = dict()
            self_fea = X[idx, :]
            c = np.unique(y).tolist()
            stop_dict = dict()
            for label in c:
                stop_dict[label] = 0
            del c[c.index(y[idx])]
            p_dict = dict()
            p_label_idx = float(len(y[y == y[idx]])) / float(n_samples)
            for label in c:
                p_label_c = float(len(y[y == label])) / float(n_samples)
                p_dict[label] = p_label_c / (1 - p_label_idx)
                near_miss[label] = []
            distance_sort = []
            distance[idx, idx] = np.max(distance[idx, :])
            for i in range(n_samples):
                distance_sort.append([distance[idx, i], int(i), y[i]])
            distance_sort.sort(key=lambda x: x[0])
            for i in range(n_samples):
                if distance_sort[i][2] == y[idx]:
                    if len(near_hit) < k:
                        near_hit.append(distance_sort[i][1])
                    elif len(near_hit) == k:
                        stop_dict[y[idx]] = 1
                else:
                    if len(near_miss[distance_sort[i][2]]) < k:
                        near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                    else:
                        if len(near_miss[distance_sort[i][2]]) == k:
                            stop_dict[distance_sort[i][2]] = 1
                stop = True
                for (key, value) in stop_dict.items():
                    if value != 1:
                        stop = False
                if stop:
                    break
            near_hit_term = np.zeros(n_features)
            for ele in near_hit:
                near_hit_term = np.array(abs(self_fea - X[ele, :])) + np.array(near_hit_term)
            near_miss_term = dict()
            for (label, miss_list) in near_miss.items():
                near_miss_term[label] = np.zeros(n_features)
                for ele in miss_list:
                    near_miss_term[label] = np.array(abs(self_fea - X[ele, :])) + np.array(near_miss_term[label])
                score = score + near_miss_term[label] / (k * p_dict[label])
            score -= near_hit_term / k
        return score

    def getMutualInformation_mode_fc(self, dataX):
        instanceOfClass = -1
        length = len(dataX[0])
        self.mutualInformation = np.zeros(length)
        f1 = self.dataX[:, instanceOfClass]
        for feature_i in range(0, length):
            f2 = dataX[:, feature_i]
            tempSU = information_gain(f1, f2)
            self.mutualInformation[feature_i] = tempSU
        return self.mutualInformation

    def getDataOfFilterIn_MFEA(self,scoreOfRelifF ,mutualInforArray,similarityOfFeature,extratRatio,dataAttribute,dataX):
        dataL = int(scoreOfRelifF.shape[0] * extratRatio)
        allIndex = np.argsort(scoreOfRelifF)[::-1]
        allIndex = allIndex[:dataL]
        scoreOfRelifF = scoreOfRelifF[allIndex]
        mutualInforArray = mutualInforArray[allIndex]
        similarityOfFeature = similarityOfFeature[allIndex]
        dataAttribute = dataAttribute[allIndex]
        dataX = dataX[:,allIndex]
        return scoreOfRelifF , dataAttribute ,similarityOfFeature,mutualInforArray,dataX

    def mutate_best(self,xDE, xMain, x1, x2, F):
        temp1 = xMain - xDE
        temp1 = F * temp1
        temp2 = x1 - x2
        temp2 = F * temp2
        v_x = xDE + temp1 + temp2
        return v_x

    def mutate_rand1(self,xMain, x1, x2, F):
        temp2 = x1 - x2
        temp2 = F * temp2
        v_x = xMain + temp2
        return v_x

    def crossover_num(self,x, v_x, crossPro):
        D = len(x)
        u_x = np.zeros(D)
        jRand = np.random.randint(0, D)
        for i in range(D):
            if i == jRand or np.random.rand() < crossPro:
                u_x[i] = v_x[i]
            else:
                u_x[i] = x[i]
        return u_x

    def featureOfSimilar(self,dataX):
        dataFeatureLen = dataX.shape[1]
        allFeatureSimilar = [[] for i in range(0, dataFeatureLen)]
        allFeaturePower = []
        originS = []
        for i in range(0, dataFeatureLen):
            p_i = np.asarray(dataX[:, i])
            pow_i = np.power(p_i, 2)
            sum_i = pow_i.sum()
            sum_i = np.power(sum_i, 0.5)
            allFeaturePower.append(sum_i)
        for i in range(0, dataFeatureLen):
            cor_i = 0
            c_i = np.asarray(dataX[:, i])
            sum_pow_i = allFeaturePower[i]
            for j in range(i, dataFeatureLen):
                if i == j:
                    allFeatureSimilar[i].append(0)
                if i != j:
                    c_j = np.asarray(dataX[:, j])
                    sum_ij = np.asarray(c_i * c_j)
                    sum_ij = np.abs(sum_ij.sum())
                    sum_pow_j = allFeaturePower[j]
                    c_ij = sum_ij / (sum_pow_i * sum_pow_j + 0.00000001)
                    allFeatureSimilar[i].append(c_ij)
                    allFeatureSimilar[j].append(c_ij)
            cor_i = np.sum(allFeatureSimilar[i], axis=0) / (dataFeatureLen - 1)
            originS = np.append(originS, cor_i)
        return originS

    def readSimilar(self,path):
        print("读取文件的数据")
        with open(path, 'r') as file:
            data = file.read().splitlines()
        similarArray = np.asarray(data, dtype='float')
        return similarArray

    def featureOfSimilar_high(self,dataName):
        allFeatureSimilar = self.readSimilar(path="D:\\MachineLearningBackUp\\dataCSV\\dataCSV_similar\\"+dataName+"Similar.txt")
        self.similarityOfFeature = np.asarray(allFeatureSimilar)

    def filterOfProFreture(self):
        mutInfor = self.similarityOfFeature + 0.00000001
        self.probabilityOfFeatureToSelect = self.mutualInforArray / mutInfor

    def initPopulation(self):
        for i in range(0, int(self.populationNum)):
            chromosome = self.Chromosome(mfea=self, mode='task1',task=0)
            chromosome.index = i
            self.population_task1 = np.append(self.population_task1, chromosome)
        for i in range(0, int(self.populationNum)):
            chromosome = self.Chromosome(mfea=self, mode='task2',task=1)
            chromosome.index = i
            self.population_task2 = np.append(self.population_task2, chromosome)

    # 染色体
    class Chromosome:
        featureOfSelectNum = np.asarray([])
        featureOfSelect = np.asarray([])
        numberOfSolution = 0
        indexOfSolution = np.asarray([])
        mineFitness = 0
        index = 0
        p_rank = 0
        numberOfNondominate = 0
        dominateSet = []
        age = 0
        isNotRedundance = False
        task = -1

        def __init__(self, mfea, mode,task):
            if mode == 'task1':
                self.initOfChromosome_Pro(MFEA=mfea,task=task)
            elif mode == 'task2':
                self.initOfChromosome_RelifF(MFEA=mfea,task=task)
            elif mode == 'elite':
                self.initOfChromosome_elite(MFEA=mfea,task=task)

        def initOfNomal(self,MFEA,task):
            self.task = task

        def initOfChromosome_elite(self, MFEA,task):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            self.task = task

        def initOfChromosome_Pro(self, MFEA,task):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            self.featureOfSelectNum = np.zeros(len(MFEA.dataFeature))
            for i in range(0, len(MFEA.probabilityOfFeatureToSelect)):
                rand = np.random.random()
                if rand <= MFEA.probabilityOfFeatureToSelect[i]:
                    self.featureOfSelect[i] = 1
                self.featureOfSelectNum[i] = rand
            acc = computeFitnessKNN(chromosome_i=self, data_x=MFEA.dataX, data_y=MFEA.dataY)
            self.mineFitness = 1 - acc
            self.getNumberOfSolution()
            self.task = task

        def initOfChromosome_RelifF(self, MFEA,task):
            self.featureOfSelect = np.zeros(len(MFEA.dataFeature))
            self.featureOfSelectNum = np.zeros(len(MFEA.dataFeature))
            for i in range(0, len(MFEA.scoreOfRelifF)):
                rand = np.random.random()
                if rand <= MFEA.scoreOfRelifF[i]:
                    self.featureOfSelect[i] = 1
                self.featureOfSelectNum[i] = rand
            acc = computeFitnessKNN(chromosome_i=self, data_x=MFEA.dataX, data_y=MFEA.dataY)
            self.mineFitness = 1 - acc
            self.getNumberOfSolution()
            self.task = task

        def getNumberOfSolution(self):
            index = np.where(self.featureOfSelect == 1)[0]
            self.indexOfSolution = np.copy(index)
            self.numberOfSolution = len(index)
            self.isNotRedundance = False

    def getNextPopulation(self, population ,task):
        saveFitness = np.asarray([])
        saveLength = np.asarray([])
        acc1 = 0
        for i in range(len(population)):
            individual_i = population[i]
            acc = 1 - individual_i.mineFitness
            acc1 += acc
            saveFitness = np.append(saveFitness, 1 - acc)
            saveLength = np.append(saveLength, individual_i.numberOfSolution)
        errorSortIndex = np.argsort(saveFitness)
        population = population[errorSortIndex]
        saveFitness = saveFitness[errorSortIndex]
        saveLength = saveLength[errorSortIndex]
        # 输出前三个最优fitness对应的个体
        for k in range(3):
            print(1 - population[k].mineFitness," ",population[k].numberOfSolution,)
        elitePop, eliteFit, eliteLen = self.getElite(population=population,task=task)
        population = np.concatenate((population, elitePop), axis=0)
        saveFitness = np.concatenate((saveFitness, eliteFit), axis=0)
        saveLength = np.concatenate((saveLength, eliteLen), axis=0)
        population = self.nonDominatedSort(parentPopulation=population, fitnessArray=saveFitness,
                                           solutionlengthArray=saveLength)
        return population

    def getElite(self, population,task):
        fitnessE = np.asarray([])
        lengthE = np.asarray([])
        rate = self.eliteProb
        elitePopulation = np.asarray([])
        tempPop = np.asarray([])
        eliteNum = int(len(population) * rate)
        eIndex = 0
        while tempPop.shape[0] < eliteNum:
            if population[eIndex].isNotRedundance :
                eIndex += 1
            else:
                tempPop = np.append(tempPop, population[eIndex])
                eIndex += 1
            if eIndex == len(population):
                tempArray = sample(population=population.tolist(), k=eliteNum - tempPop.shape[0])
                tempPop = np.append(tempPop, tempArray)
        elitePopulation = self.removeIrralevantFeature(population=tempPop,task=task,ePro=self.eliteProb)
        for i in range(len(elitePopulation)):
            individual_e = elitePopulation[i]
            fitnessE = np.append(fitnessE, individual_e.mineFitness)
            lengthE = np.append(lengthE, individual_e.numberOfSolution)
        return elitePopulation, fitnessE, lengthE

    def removeIrralevantFeature(self, population,task,ePro):
        elitePopulation = np.asarray([])
        genePool = np.zeros(len(self.dataFeature))
        mrmrOfFeature = np.zeros((len(population), len(self.dataFeature)))
        for i in range(len(population)):
            selectIndex = np.where(population[i].featureOfSelect == 1)[0]
            for i1 in range(0, len(selectIndex)):
                id1 = selectIndex[i1]
                f1 = self.dataX[:, id1]
                sum_f = 0
                for i2 in range(1, len(selectIndex)):
                    id2 = selectIndex[i2]
                    if id1 != id2:
                        f2 = self.dataX[:, id2]
                        if self.mrmrArray[id1][id2] == 0:
                            ig = information_gain(f1, f2)
                            self.mrmrArray[id1][id2] = ig
                            self.mrmrArray[id2][id1] = ig
                        else:
                            ig = self.mrmrArray[id1][id2]
                        sum_f += ig
                mean = sum_f / (len(selectIndex) + 0.01)
                mrmrOfFeature[i][id1] = self.mutualInforArray[id1] - mean
        eachIndSubPopNum =int( 1 / ePro)
        for i in range(len(mrmrOfFeature)):
            temoLen1 = len(np.where(mrmrOfFeature[i] > 0, 1, 0))
            genePool = np.where(mrmrOfFeature[i] > 0, 1, 0)
            front = self.eliteFront
            for j in range(int(eachIndSubPopNum)):
                individual_elite = self.Chromosome(mfea=self, mode="elite",task=task)
                solution1_B = np.zeros(len(self.dataFeature))
                solution1_N = np.zeros(len(self.dataFeature))
                if task == 0:
                    border = self.scoreOfRelifF
                else:
                    border = self.probabilityOfFeatureToSelect
                for k in range(temoLen1):
                    rand = random.random()
                    if (genePool[k] == 1):
                        if (rand > 0.2):
                            solution1_B[k] = 1
                            solution1_N[k] = np.random.uniform(0,border[k])
                        else:
                            solution1_N[k] = np.random.uniform(border[k],1)
                    else:
                        if rand < front:
                            solution1_B[k] = 1
                            solution1_N[k] = np.random.uniform(0,border[k])
                        else:
                            solution1_N[k] = np.random.uniform(border[k],1)
                feature_y = self.dataY
                if len(np.where(solution1_B == 1)[0]) == 0:
                    acc1 = 0
                else:
                    feature_x = self.dataX[:, solution1_B == 1]
                    acc1 = fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=feature_y, CV=5)
                individual_elite.featureOfSelect = solution1_B
                individual_elite.featureOfSelectNum = solution1_N
                individual_elite.mineFitness = 1 - acc1
                individual_elite.getNumberOfSolution()
                elitePopulation = np.append(elitePopulation, individual_elite)
        return elitePopulation

    def nonDominatedSort(self, parentPopulation, fitnessArray, solutionlengthArray):
        springsonPopulation = np.asarray([])
        errorArray = np.asarray([])
        lenArray = np.asarray([])
        dominateFrontArray = []
        F_i = []
        errorArray, lenArray = self.sortFunction(error=fitnessArray, solutionLength_test=solutionlengthArray)
        sortErrorIndex = np.argsort(errorArray)
        errorArray = errorArray[sortErrorIndex]
        parentPopulation = parentPopulation[sortErrorIndex]
        lenArray = lenArray[sortErrorIndex]
        i = 0
        j = 0
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
        dominateFrontArray.append([])
        dominateFrontArray[0].append(parentPopulation[0])
        for i in range(1, len(parentPopulation)):
            head = 0
            tail = len(dominateFrontArray) - 1
            error_i = errorArray[i]
            len_i = lenArray[i]
            individual = parentPopulation[i]
            k = math.floor((head + tail) / 2 + 0.5)
            while True:
                leastIndividual = dominateFrontArray[k][-1]
                leastIndex = np.where(parentPopulation == leastIndividual)[0]
                leastError = errorArray[leastIndex]
                leastLen = lenArray[leastIndex]
                if (leastError < error_i and leastLen < len_i) or (
                        leastError <= error_i and leastLen < len_i) or (
                        leastError < error_i and leastLen <= len_i):
                    head = k
                    if (tail == head + 1 and tail < len(dominateFrontArray) - 1):
                        dominateFrontArray[tail].append(individual)
                        break
                    if (tail == head + 1 and tail == len(dominateFrontArray) - 1):
                        head = tail
                        k = tail
                    elif (head == len(dominateFrontArray) - 1):
                        dominateFrontArray.append([])
                        dominateFrontArray[-1].append(individual)
                        break
                    else:
                        k = math.floor((head + tail) / 2 + 0.5)
                else:
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
        numPop = self.populationNum
        count = 0
        while len(springsonPopulation) < self.populationNum:
            front_i = dominateFrontArray[count]
            resideNum = numPop - len(springsonPopulation)
            if resideNum - len(front_i) >= 0:
                springsonPopulation = np.append(springsonPopulation, front_i)
            else:
                crossError = np.zeros(len(front_i))
                crossLen = np.zeros(len(front_i))
                for k in range(0, len(front_i)):
                    crossError[k] = front_i[k].mineFitness
                    crossLen[k] = front_i[k].numberOfSolution
                crossError, crossLen = self.sortFunction(error=crossError, solutionLength_test=crossLen)
                front_i, distance = self.crowdingDistance(F_i=front_i, error=crossError, lens=crossLen)
                sort_distance_index = np.argsort(distance)[::-1]
                for k in range(resideNum):
                    springsonPopulation = np.append(springsonPopulation, front_i[sort_distance_index[k]])
            count += 1
        array_score = np.zeros(len(dominateFrontArray[0]))
        for m in range(len(array_score)):
            array_score[m] = (1 - dominateFrontArray[0][m].mineFitness) * 0.9+ (
            1 - dominateFrontArray[0][m].numberOfSolution / len(self.dataFeature)) * 0.1
        score_index = np.argsort(array_score)[::-1]
        array_score = array_score[score_index]
        array_individual = np.asarray(dominateFrontArray[0])
        array_individual = array_individual[score_index]
        if self.globalScore <= array_score[0]:
            self.globalScore = array_score[0]
            self.globalFitness = array_individual[0].mineFitness
            self.globalSolution = array_individual[0].featureOfSelect
            self.globalSolutionNum = array_individual[0].featureOfSelectNum
        return springsonPopulation

    def crowdingDistance(self, F_i, error, lens):
        crossedDistance = np.zeros(len(F_i))
        crossedDistance[0] = float(np.inf)
        crossedDistance[-1] = float(np.inf)
        for i in range(1, len(F_i) - 1):
            crossedDistance[i] = crossedDistance[i] + (
                    error[i + 1] - error[i - 1]
            ) / (max(error) - min(error))
        for i in range(1, len(F_i) - 1):
            crossedDistance[i] = crossedDistance[i] + np.abs(
                lens[i + 1] - lens[i - 1]
            ) / (max(lens) - min(lens))
        return F_i, crossedDistance

    def sortFunction(self, error, solutionLength_test):
        error = np.exp(error)
        solutionLength = solutionLength_test / len(self.dataFeature)
        solutionLength = np.exp(solutionLength)
        return error, solutionLength

    def computeRmp(self):
        k = 2
        rmp_matrix = np.eye(k)
        ms_matrix = [[] for i in range(k)]
        l_matrix = np.ones(2)
        task1 = []
        task2 = []
        for i in range(int(self.populationNum * 0.5)):
            task1.append(self.population_task1[i].featureOfSelectNum)
            task2.append(self.population_task2[i].featureOfSelectNum)
        subpops = np.asarray([task1, task2])
        for i in range(k):
            subpop = subpops[i]
            num_sample = len(task1)
            num_random_sample = int(np.floor(0.1 * num_sample))
            rand_pop = np.random.rand(num_random_sample, len(self.dataFeature))
            mean = np.mean(np.concatenate([subpop, rand_pop]), axis=0)
            std = np.std(np.concatenate([subpop, rand_pop]), axis=0)
            ms_matrix[i].append(mean)
            ms_matrix[i].append(std)
            l_matrix[i] = num_sample
        for k_i in range(k - 1):
            for j in range(k_i + 1, k):
                probmatrix = [(np.ones([int(l_matrix[k_i]), 2])),
                              (np.ones([int(l_matrix[j]), 2]))]
                probmatrix[0][:, 0] = self.density(subpop=subpops[k_i], mean=ms_matrix[k_i][0], std=ms_matrix[k_i][1])
                probmatrix[0][:, 1] = self.density(subpop=subpops[j], mean=ms_matrix[k_i][0], std=ms_matrix[k_i][1])
                probmatrix[1][:, 0] = self.density(subpop=subpops[k_i], mean=ms_matrix[j][0], std=ms_matrix[j][1])
                probmatrix[1][:, 1] = self.density(subpop=subpops[j], mean=ms_matrix[j][0], std=ms_matrix[j][1])
                rmp = fminbound(lambda rmp: self.log_likelihood(rmp, probmatrix, k), 0, 1)
                rmp = rmp * 2
                rmp = np.clip(rmp, 0, 1)
                rmp_matrix[k_i, j] = rmp
                rmp_matrix[j, k_i] = rmp
        return rmp_matrix

    def density(self, subpop, mean, std):
        N, D = subpop.shape
        prob = np.ones([N])
        for d in range(D):
            prob *= norm.pdf(subpop[:, d], loc=mean[d], scale=std[d])
        return prob

    def log_likelihood(self, rmp, prob_matrix, K):
        posterior_matrix = copy.deepcopy(prob_matrix)
        value = 0
        for k in range(2):
            for j in range(2):
                if k == j:
                    posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * (1 - 0.5 * (K - 1) * rmp) / float(K)
                else:
                    posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * 0.5 * (K - 1) * rmp / float(K)
            value = value + np.sum(-np.log(np.sum(posterior_matrix[k], axis=1)))
        return value

    def mate_new(self, taskNum):
        copyTask = [[] for i in range(2)]
        copyTask[0] = copy.deepcopy(self.population_task1)
        copyTask[1] = copy.deepcopy(self.population_task2)
        rmp = self.computeRmp()
        K = taskNum
        mateIndex = np.random.permutation(2 * self.populationNum)
        taskPopulation = np.concatenate((self.population_task1, self.population_task2), axis=0)
        for i in range(len(taskPopulation)):
            ind1 = taskPopulation[mateIndex[i]]
            if ind1 not in self.p_DEarray:
                indMain = np.random.choice(self.p_DEarray)
            else:
                tempArray = np.delete(self.p_DEarray,np.where(self.p_DEarray == ind1)[0])
                indMain = np.random.choice(tempArray)
            xArray = [[] for i in range(2)]
            index1 = ind1.index
            index2 = indMain.index
            if (ind1.task == indMain.task):
                if ind1.task == 0:
                    tempA1 = np.delete(self.population_task1, [index1, index2]).tolist()
                    xArray = sample(tempA1, 2)
                elif ind1.task == 1:
                    tempA1 = np.delete(self.population_task2, [index1, index2]).tolist()
                    xArray = sample(tempA1, 2)
                v_x = self.mutate_best(xDE=ind1.featureOfSelectNum, xMain=indMain.featureOfSelectNum,
                                  x1=xArray[0].featureOfSelectNum,
                                  x2=xArray[1].featureOfSelectNum, F=self.F)
            elif ind1.task != indMain.task and np.random.rand() < rmp[ind1.task][indMain.task]:
                tempPop = [[] for i in range(K)]
                tempPop[0] = self.population_task1
                tempPop[1] = self.population_task2
                if ind1 not in self.p_DEarray:
                    xArray[0] = np.random.choice(tempPop[ind1.task], 1)[0]
                else:
                    tempRange = np.delete(tempPop[ind1.task], ind1.index).tolist()
                    xArray[0] = sample(tempRange, 1)[0]
                tempRange = np.delete(tempPop[indMain.task], indMain.index).tolist()
                xArray[1] = sample(tempRange, 1)[0]
                v_x = self.mutate_best(xDE=ind1.featureOfSelectNum, xMain=indMain.featureOfSelectNum,
                                  x1=xArray[0].featureOfSelectNum,
                                  x2=xArray[1].featureOfSelectNum, F=self.F)
            else:
                tempPop = [[] for i in range(K)]
                tempPop[0] = self.population_task1
                tempPop[1] = self.population_task2
                tempRange = np.delete(tempPop[indMain.task],indMain.index).tolist()
                if np.random.rand() < 0.8:#0.8
                    xArrayN = sample(tempRange,3)
                    v_x = self.mutate_rand1(xMain=xArrayN[2].featureOfSelectNum,x1=xArrayN[0].featureOfSelectNum,
                                       x2=xArrayN[1].featureOfSelectNum,F=self.F)
                else:
                    xArray = sample(tempRange,2)
                    v_x = self.mutate_best(xDE=ind1.featureOfSelectNum, xMain=indMain.featureOfSelectNum,
                                      x1=xArray[0].featureOfSelectNum,x2=xArray[1].featureOfSelectNum, F=self.F)
            u_x = self.crossover_num(x=ind1.featureOfSelectNum, v_x=v_x, crossPro=self.crossoverPro)
            u_x = np.where(u_x >= 1 , u_x - 1, u_x)
            u_x = np.where(u_x <= 0 , 1 + u_x, u_x)
            if ind1.task == 0:
                border = self.scoreOfRelifF
            else:
                border = self.probabilityOfFeatureToSelect
            solutionU = np.where(u_x < border,1,0)
            featureX = self.dataX[:, solutionU == 1]
            solutionA = fitnessFunction_KNN_CV(findData_x=featureX, findData_y=self.dataY, CV=10)
            solutionL = 1 - len(np.where(solutionU == 1)[0]) / len(self.dataFeature)
            solution_score = solutionA * 0.9 + solutionL * 0.1
            ind1_score = (1 - ind1.mineFitness) * 0.9 + (1 - ind1.numberOfSolution / len(self.dataFeature))*0.1

            if ind1_score <= solution_score:
                ind1.featureOfSelectNum = u_x
                ind1.featureOfSelect = solutionU
                ind1.mineFitness = 1 - solutionA
                ind1.getNumberOfSolution()
                ind1.age = 0
            else:
                ind1.age += 1
                if ind1.age == self.maxAge:
                    ind1 = self.prunGene(individual=ind1)
            taskPopulation[mateIndex[i]] = ind1
    def prunGene(self, individual):
        if individual.isNotRedundance:
            individual.age = 0
            return individual
        geneIndex = np.where(individual.featureOfSelect == 1)[0]
        if len(geneIndex) == 1:
            individual.age = 0
            return individual
        featureY = self.dataY
        rule = [[] for i in range(2)]
        rule[0] = self.probabilityOfFeatureToSelect
        rule[1] = self.scoreOfRelifF
        getInfor = rule[individual.task][geneIndex]
        tempIndex = np.argsort(getInfor)
        geneIndex = geneIndex[tempIndex]
        getInfor = getInfor[tempIndex]
        testSolution = copy.deepcopy(individual.featureOfSelect)
        for i in range(geneIndex.shape[0]):
            isReserve = False
            testSolution[geneIndex[i]] = 0
            featureX = self.dataX[:, testSolution == 1]
            acc = fitnessFunction_KNN_CV(findData_x=featureX, findData_y=featureY, CV=5)
            if acc >= 1 - individual.mineFitness:
                individual.mineFitness = 1 - acc
                individual.featureOfSelect = copy.deepcopy(testSolution)
                individual.getNumberOfSolution()
                individual.featureOfSelectNum[geneIndex[i]] = np.random.uniform(
                    rule[individual.task][geneIndex[i]],1)
                isReserve = True
                break
            else:
                testSolution[geneIndex[i]] = 1
        individual.age = 0
        if isReserve:
            return individual
        else:
            individual.isNotRedundance = True
            return individual

    def prunGene_least(self,individual):
        geneIndex = np.where(individual.featureOfSelect == 1)[0]
        if len(geneIndex) == 1:
            individual.age = 0
            return individual
        featureY = self.dataY
        rule = [[] for i in range(2)]
        rule[0] = self.probabilityOfFeatureToSelect
        rule[1] = self.scoreOfRelifF
        getInfor = rule[individual.task][geneIndex]
        tempIndex = np.argsort(getInfor)
        geneIndex = geneIndex[tempIndex]
        getInfor = getInfor[tempIndex]
        testSolution = copy.deepcopy(individual.featureOfSelect)
        noDone = True
        while noDone:
            for i in range(geneIndex.shape[0]):
                testSolution[geneIndex[i]] = 0
                featureX = self.dataX[:, testSolution == 1]
                acc = fitnessFunction_KNN_CV(findData_x=featureX, findData_y=featureY, CV=5)
                if acc >= 1 - individual.mineFitness:
                    individual.mineFitness = 1 - acc
                    individual.featureOfSelect = copy.deepcopy(testSolution)
                    individual.getNumberOfSolution()
                    individual.featureOfSelectNum[geneIndex[i]] = np.random.uniform(rule[individual.task][geneIndex[i]],1)
                    geneIndex = np.delete(geneIndex,i)
                    break
                else:
                    testSolution[geneIndex[i]] = 1
                    if i == geneIndex.shape[0] - 1:
                        noDone = False
        return individual

    def getBestPop(self):
        DE_p = self.p
        array1 = self.population_task1[:int(self.populationNum * DE_p)]
        array2 = self.population_task2[:int(self.populationNum * DE_p)]
        self.p_DEarray = np.concatenate((array1,array2),axis=0)

    def leastRefine(self,population1,population2):
        tempPopulation = np.asarray([])
        index1 = 0
        index2 = 0
        while tempPopulation.shape[0] < 4:
            while index1 < population1.shape[0] and population1[index1].isNotRedundance:
                index1 += 1
            while index2 < population2.shape[0] and population2[index2].isNotRedundance:
                index2 += 1
            if index1 == population1.shape[0] and index2 == population2.shape[0]:
                break
            if population1[index1].mineFitness < population2[index2].mineFitness:
                tempPopulation = np.append(tempPopulation,population1[index1])
                index1 += 1
            else:
                tempPopulation = np.append(tempPopulation,population2[index2])
                index2 += 1
        for i in range(tempPopulation.shape[0]):
            tempPopulation[i] = self.prunGene_least(individual=tempPopulation[i])
            score = (1 - tempPopulation[i].mineFitness) * 0.9 + (
                    1 - tempPopulation[i].numberOfSolution / len(self.dataFeature)) * 0.1
            if score > self.globalScore:
                self.globalScore = score
                self.globalSolution = tempPopulation[i].featureOfSelect
                self.globalFitness = tempPopulation[i].mineFitness
                self.globalSolutionNum = tempPopulation[i].featureOfSelectNum

    def goRun(self):
        self.initPopulation()
        runTime = 0
        import time
        start = time.time()
        while runTime < self.iteratorTime:
            print(" 第",runTime + 1,"轮")
            print("task1")
            self.population_task1 = self.getNextPopulation(population=self.population_task1,task=0)
            print(f"task1 time = {time.time() - start} seconds")
            print("task2")
            self.population_task2 = self.getNextPopulation(population=self.population_task2,task=1)
            print(f"task2 time = {time.time() - start} seconds")
            self.getBestPop()
            self.mate_new(taskNum=2)
            runTime += 1
        self.globalScore = (1 - self.globalFitness) * 0.9 + (
                1 - len(np.where(self.globalSolution == 1)[0])/len(self.dataFeature)) * 0.1
        self.leastRefine(population1=self.population_task1,population2=self.population_task2)
        # print(f"all time = {time.time() - start} seconds")
        print("============================================")
        print("dataset : BreastCancer1")
        print("acc = " , 1 - self.globalFitness)
        print("len = " , len(np.where(self.globalSolution == 1)[0]))
        print(" ")

if __name__ == '__main__':
    import time
    start = time.time()
    ducName = "BreastCancer1"
    # ducName = "CLLSUB"
    path = "D:\MachineLearningBackUp\dataCSV\dataCSV_high\\" + ducName + ".csv"# 实验
    data = pd.read_csv(path, header=None)  # 获得数组
    dataX = data.values[:, 0:-1]
    dataY = data.values[:, -1]
    genetic = Genetic(dataX=dataX, dataY=dataY ,dataName=ducName)
    genetic.setParameter(populationNum=70, iteratorTime=100, eliteProb=0.2,crossoverPro=0.9
                        ,F=0.5,p=0.3,eliteFront=0.01,maxAge=3)#p = 0.3
    genetic.goRun()
