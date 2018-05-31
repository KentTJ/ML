#-*-coding:utf-8-*-

from import_file import *

'读取-------dataMat含label'
def loadDataSet(FileName):
    # 创建两个列表
    dataMat = []
    # 打开文本数据集
    fr = open(FileName)  ####  fr = open('M:/ForTheTimeBeing/AAAANewFile/machinelearninginaction/Ch05/testSet.txt')
    # 遍历文本的每一行
    for line in fr.readlines():
        # 对当前行去除首尾空格，并按空格进行分离
        lineArr = line.strip().split('\t')
        # lineArrFloat = map(float,lineArr)
        # dataMat.append(lineArrFloat)
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
    # 返回数据列表
    return dataMat

#运行代码
# def loadDataSet(fileName):      #general function to parse tab -delimited floats
#     dataMat = []                #assume last column is target value
#     fr = open(fileName)
#     for line in fr.readlines():
#         curLine = line.strip().split('\t')
#         fltLine = map(float,curLine) #map all elements to float()
#         dataMat.append(fltLine)
#     return dataMat

# dataSet= loadDataSet('M:/ForTheTimeBeing/AAAANewFile/machinelearninginaction/Ch09/ex00.txt')
# bb =1


'''按照指定feature的取值来抽取分dataSet'''
# dataSet: 数据集合
# feature: 待切分的特征
# value: 该特征的某个值
def binSplitDataSet(dataSet,featurte,value):


#运行代码
dataSet = mat(eye(4))
mat0,mat1 = binSplitDataSet(dataSet,1,0.5)

int(3.9)