import os

import numpy as np

from dataProcessing.ReadDataCSV_new import ReadCSV
from mfmode.mfde1 import Genetic

if __name__ == '__main__':
    files = os.listdir("D:\MachineLearningBackUp\dataCSV\\dataCSV_high")

    # eliteProb = [0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    #eliteFront = [0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3]
    ageBorder = [3,6,9,12,15]
    for file in files:
        ducName = dataName = file.split('.')[0]
        path_csv = "D:/MachineLearningBackUp/dataCSV/dataCSV_high/" + ducName + ".csv"

        for m in range(len(ageBorder)):
            # 写入文件的路径
            path_txt = "D:/MachineLearningBackUp/实验/parameterAnalyze/mfmode/ageBorder/" + ducName + "EliteMaxAge.txt"
            # 写入文件的标题
            titleTxt = '   '+str(ageBorder[m])+"  " + ducName + '.csv \n'
            # 将结果写入到文件中去
            with open(path_txt, 'a') as f:
                f.write(titleTxt)
                f.close()

            dataCsv = ReadCSV(path=path_csv)
            print("获取文件数据", file)
            dataCsv.getData()
            genetic = Genetic(dataX=dataCsv.dataX, dataY=dataCsv.dataY,dataName=ducName)

            genetic.setParameter(populationNum=70, iteratorTime=100, eliteProb=0.1,crossoverPro=0.9,F=0.5
                                 ,eliteFront = 0.01,p=0.05,maxAge=ageBorder[m])

            # 循环多少次
            iterateT = 1

            acc = np.zeros(iterateT)
            length = np.zeros(iterateT)
            import time
            start = time.time()
            for i in range(iterateT):
                #print("第",i,"次")
                genetic.goRun()
                acc[i] = 1 - genetic.globalFitness
                length[i] = len(np.where(genetic.globalSolution == 1)[0])
                # 重置
                genetic.population_task1 = np.asarray([])
                genetic.population_task2 = np.asarray([])
                genetic.globalScore = 0


                # 写入文件的值
                stringOfResult = str(acc[i]) + '\t' + str(length[i]) + '\n'
                # 将结果写入到文件中去
                with open(path_txt, 'a') as f:
                    f.write(stringOfResult)
                    f.close()
                print(acc[i], " ", length[i])

            # 向文件中写入均值
            # 写入文件的值
            stringOfResult = str(acc.mean()) + '\t' + '\t' + str(length.mean()) + '\n'
            # 将结果写入到文件中去
            with open(path_txt, 'a') as f:
                f.write('   mean  \n')
                f.write(stringOfResult)
                f.close()



            print("acc:",acc.mean(),"  len:",length.mean())
            print(f" time = {time.time() - start} seconds")
            print("===================================")
            print(" ")