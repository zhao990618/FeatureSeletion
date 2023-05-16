import os
import pandas as pd
import  numpy as np

# CSV的
class DataCSV:
    ducPath = ""
    allData = []
    #   返回所有特征数组
    dataAttribute =[]
    #   返回所有样本值，除类标签值
    dataInstance = []
    #   类标签的所有值
    dataClass = []
    #   总样本数量
    numOfInstance = 0
    #   保存了每一列数据
    dataAllColum = []

    def __init__(self,path):
        self.ducPath = path


    def getData(self):
        file_path = self.ducPath
        tempData = []
        with open(file_path,encoding="utf-8") as file_Obj:
            line = file_Obj.readline()
            tempData = line.split(",")
            self.dataAttribute = tempData  # 获取CSV文件里的属性值

        data = pd.read_csv(file_path)          # 获得数组
        #print(data)
        self.allData = data
        data.columns = self.dataAttribute # 设置行标签
       # print(data)
        # 得到数据 除了类以外的所有数据
        x = data[self.dataAttribute[0:len(self.dataAttribute)-1]]
        x = np.asarray(x)
        self.dataInstance = x

        #得到每一个样本的类
        y = data[self.dataAttribute[-1]]
        y = np.asarray(y)
        self.dataClass = y

        #得到总共有多少个样本
        self.numOfInstance = len(y)
        #print(self.numOfInstance)
        # 用于存放每一个需要删除特征的索引
        delectIndex = np.asarray([])
        for index in self.dataAttribute:
            columData =data[index]   # 现在总体数组中去除一列，还保留了数据索引，数据值，和一些文本
            columData = np.asarray(columData) # 将该列的数据中的纯数据取出
            self.dataAllColum.append(columData)

        self.dataAllColum = np.asarray(self.dataAllColum)
        self.allData = np.asarray(self.allData)
        self.dataAttribute = np.array(self.dataAttribute)

        # print(self.allData,self.allData.shape)
        # print(self.dataAllColum,self.dataAllColum.shape)
        # print(self.dataAttribute,self.dataAttribute.shape)

if __name__ == "__main__":
    dataCsv = DataCSV("D:\MachineLearningBackUp\\dataCSV\\Breast.csv")
    dataCsv.getData()

