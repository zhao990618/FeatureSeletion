import numpy as np
import random
from skfeature.utility.mutual_information import su_calculation
from skfeature.function.information_theoretical_based.MRMR import mrmr


if __name__ == "__main__":

    # a = [1,2,3]
    # b = [4,5,6]
    # e = np.copy(a)
    # print(e)
    # print(a.index(1))
    # print(np.power(2,3))
    # p = np.concatenate((a,b))
    #
    # q = a + b
    # print(p,p.shape,type(p))
    # print(q,type(q))
    #
    # c = np.exp(0.1)
    # print(1/c)
    #
    # a = np.asarray([23,3,5,6,75])
    # b = [1,2,3,4,5,6,7]
    # print("a.sum() = ",np.sum(b,axis=0))
    # c = sorted(a)
    # print(b/c)
    # print(sorted(a,reverse=True))
    # print(type(c))
    # print(a)
    #
    # d = np.power(2,2)
    # print(d,type(d))
    # f = [-1,-2,-3]
    # print(np.abs(f))
    # print(np.arange(0,10))
    #
    # c = [0 for i in range(0,10)]
    # print(c)
    # print(np.zeros((10,10)))
    # print([[] for i in range(0,10)])
    # a = np.exp(1)
    # b = np.exp(2)
    # c = np.exp(3)
    # print(a/(a*3+b*2+c*3))
    # d = np.asarray([1,2,3])
    # print(np.power(d,2))
    # d = np.ones(10)
    # print(d)
    a = 123
    result = 'acc = ' + str(a)
    print(result)
    titleTxt = "one" + '.csv \n'
    # 将结果写入到文件中去
    with open("D:/MachineLearningBackUp/实验/test.txt", 'a') as f:
        f.write(titleTxt)
        f.close()
    with open("D:/MachineLearningBackUp/实验/test.txt",'a') as f:
        f.write(str(123)+"\t"+str(2.1) + "\n")
        f.close()






