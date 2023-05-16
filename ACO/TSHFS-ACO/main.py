import numpy as np

# class A:
#     p = []
#     number = 0
#     def __init__(self , number):
#         self.number = number
#         self.initTempList()
#     class B:
#         name = "ant "
#         high = 1
#         def __init__(self):
#             pass
#         def myName(self):
#             print("asd")
#     def initTempList(self):
#         for i in range(0,self.number):
#             b = self.B()
#             self.p.append(b)
#
#     def getB(self):
#         for c in self.p:
#             print(c.name)
#             print(c.high)

if __name__ == "__main__":
    # p = range(1,6)
    # p = list(p)
    # print(p)
    # print(len(p))
    # print(p.index(3))
    # m = p.index(3)
    # del p[3]
    # print(p)
    # print(p[3])
    # a = A(3)
    # a.getB()
    # a.p[1].myName()


    # c = [7,8,9]
    # print(np.c_[a,b,c])

    # 曼哈顿距离
    # map(function , iterator（list）)  即让list里面的所有参数都去执行function
    # lambda 自定义函数              lambda i,j : abs(j-i)    lambda i : i**2
    # distance = sum(map(lambda i,j:abs(j-i),a,b))
    # print(distance)
    # c = [[[1,2],[2,3],[3,4]],[[4,5],[5,6],[6,7]]]
    # print(np.ravel(c))
    # p =np.array([[1,2,3],[4,5,6],[7,8,9],[2,3,4]])
    # print(p.shape[0])
    # print(p[2][1])
    # p[2][1] = p[2][1]+1
    # print(p[2][1])
    # p = range(0,5)
    # p = np.asarray(p)
    # print(p)
    # print(type(p))
    # print(p.shape[0])
    # m = np.where( p == 3)
    # print(m)
    # print(m[0])
    # print(m[0][0])
    # p = []
    # p.append([1,2,3,4])
    # p = np.asarray(p)
    # #p.append([1,2,3,4])
    # # print(p)
    # # c = np.delete(p,[0,2],axis=1)
    # # print(c)
    # print(p[:,1:3])
    #print(p[1,3])

    #print(p[0])

    #print(p.shape[0])
    # p.append([1,2,3])
    # p.append([2,2,3])
    # p.append([3,2,3])
    # p.append([4,2,3])
    # print(p.index([2,2,3]))


    # a = np.array([1, 2, 3, 4, 5, 6, 6, 7, 6])
    # b = np.where(a == 6)
    # print(b)
    p = np.asarray([1,2,2,5,6,2])
    # for i in range(0,10):
    #     p .append(np.random.randint(0,2))
    # print(p)
    # c = np.exp(p)
    # p = np.asarray(p)
    # p = p * -1
    # print(p)
    # c = np.exp(p)
    # print(c)
    # print(int(1.2323))
    p = np.asarray([1,3,1,3,2,2,1,3])
    p = np.asarray([2,1,4,1,2,2,2,1])
    print(p)
    #均值
    mu_mean = p.sum()/len(p)
    print("均值",mu_mean)
    # 方差
    temp1 = p - mu_mean
    sum_temp1 = np.power(temp1,2).sum()
    delta = sum_temp1/len(p)
    print("方差",delta)
    #softmax scale
    temp2 = p - mu_mean
    temp2 = temp2/delta
    temp2 = -1 * temp2
    temp2 = np.exp(temp2)
    temp2 = 1 + temp2
    temp2 = 1 / temp2
    print("分布概率",temp2)