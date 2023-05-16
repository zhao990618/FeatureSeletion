import random
import time

import numpy as np


def test1():
    a1 = 10
    a2 = 23
    return a1 , a2

if __name__ == "__main__":

    # a = np.asarray([[1,2],[1,2]])
    # a = np.unique(a,axis=0)
    # print(a)
    # a = np.unique(a, axis=0)
    # print(a)
    # print(np.delete(a,1))
    # print(len([]))
    a = []
    a.append([1,2,3,4])
    print(a)
    a.append([4,5,6,7])
    print(a)
    print(np.delete(a,1,axis=0))
    print(time.ctime())