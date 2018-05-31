# -*- coding: UTF-8 -*-
##import numpy as np
from numpy import *
import operator

'''
############problem:  alpha = 0.0001，maxCycle = 1000这样设置收敛有问题，不同数字差异很大
梯度上升优化算法：
input：
'''

'''
(1)读、存数据----书本
'''
# 预处理数据
def loadDataSet():
    # 创建两个列表
    dataMat = [];
    labelMat = []
    # 打开文本数据集
    fr = open('H:/ForTheTimeBeing/AAAANewFile/machinelearninginaction/Ch05/testSet.txt')
    # 遍历文本的每一行
    for line in fr.readlines():
        # 对当前行去除首尾空格，并按空格进行分离
        lineArr = line.strip().split()
        # 将每一行的两个特征x1，x2，加上x0=1,组成列表并添加到数据集列表中
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 将当前行标签添加到标签列表
        labelMat.append(int(lineArr[2]))
    # 返回数据列表，标签列表
    return dataMat, labelMat



'''
(2)定义sigmoid函数
inX
'''


def sigmoid(inX):
    result = 1 / (1 + exp(-inX))
    return result


'''
(3)梯度上升优化算法主体程序----书本
input：
    Label认为是行向量

'''


###(3)梯度上升优化算法主体程序-------书本
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix       #列化
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))  # 初始化1
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)  # matrix mult    #列向量
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights


'''
调用代码：
returnMat, classLabelVector = loadDataSet()
weight = gradAscent(returnMat, classLabelVector)
print(result)
'''

'''
(4)随机梯度上升算法----书本
input：
    Label认为是行向量

'''
# #随机梯度上升算法
# def stocGradAscent(dataMatrix,classLabels):
#     #为便于计算，转为Numpy数组
#     dataMat=array(dataMatrix)
#     #获取数据集的行数和列数
#     m,n=shape(dataMatrix)
#     #初始化权值向量各个参数为1.0
#     weights=ones(n)
#     #设置步长为0.01
#     alpha=0.01
#     #循环m次，每次选取数据集一个样本更新参数
#     for i in range(m):
#         #计算当前样本的sigmoid函数值
#         h=sigmoid(sum(dataMatrix[i]*weights))
#         a = dataMatrix[0]*weights
#         sum(dataMatrix[i] * weights)
#         #计算当前样本的残差(代替梯度)
#         error=(classLabels[i]-h)
#         #更新权值参数
#         weights=weights+alpha*error*array(dataMatrix[i])   #尽量转化为array，少采用list。报错： 'numpy.float64' object cannot be interpreted as an integer
#     return weights

'''
(4)随机梯度上升算法----mine
input：
    Label认为是行向量

'''


def randomGradAscent0(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix       #列化
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones((n, 1))  # 初始化1
    # weightsrecording =zeros((3,m))
    weightsrecording = zeros((m, 3))
    for k in range(m):  # heavy on matrix operations
        h = sigmoid(dataMatrix[k, :] * weights)  # matrix mult    #列向量
        error = (labelMat[k] - h)  # vector subtraction
        weights = weights + alpha * dataMatrix[k, :].transpose() * error  # matrix mult
        weightsrecording[k, :] = weights.T  # 记录weight收敛情况                !!!!!!!!!!!注意：把一列数字存入array只能按行存
    weightsrecording = weightsrecording.T
    return weights, weightsrecording


'''
(4)改进的随机梯度上升算法：                       ！！！！！！！优
#@dataMatrix：数据集列表
#@classLabels：标签列表
#@numIter：迭代次数，默认150
'''


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # from numpy.random import uniform
    # from numpy.random import *
    # 将数据集列表转为Numpy数组
    dataMat = array(dataMatrix)
    # 获取数据集的行数和列数
    m, n = shape(dataMat)
    # 初始化权值参数向量每个维度均为1
    weights = ones(n)
    # 循环每次迭代次数
    for j in range(numIter):
        # 获取数据集行下标列表
        dataIndex = list(range(m))  # !1!!!!!!range返回不是列表
        # 遍历行列表
        for i in range(m):
            # 每次更新参数时设置动态的步长，且为保证多次迭代后对新数据仍然具有一定影响
            # 添加了固定步长0.1
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机获取样本
            randomIndex = int(random.uniform(0, len(dataIndex)))
            # 计算当前sigmoid函数值
            h = sigmoid(sum(dataMat[randomIndex] * weights))
            # 计算权值更新
            # ***********************************************
            error = classLabels[randomIndex] - h
            weights = weights + alpha * error * dataMat[randomIndex]
            # ***********************************************
            # 选取该样本后，将该样本下标删除，确保每次迭代时只使用一次
            del (dataIndex[randomIndex])
    return weights


'''（5）画出决策边界：决策边界对应 权重*dataSet = 0
input
     weight
'''


def plotBestFit(weight):
    import matplotlib.pyplot as plt  # from matplotlib import *
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = dataArr.shape[0]
    x1cord1 = [];
    x2cord1 = []
    x1cord2 = [];
    x2cord2 = []
    for i in range(n):
        if labelMat[i] == 1:  # 两种label分两组画图
            x1cord1.append(dataArr[i, 1]);
            x2cord1.append(dataArr[i, 2])
        else:
            x1cord2.append(dataArr[i, 1]);
            x2cord2.append(dataArr[i, 2])
    fig = plt.figure()  # 开始画图
    # 画点
    ax = fig.add_subplot(111)  # 1X1个图中第一个图
    ax.scatter(x1cord1, x2cord1, s=30, c='red')
    ax.scatter(x1cord2, x2cord2, s=30, c='green')
    # 画分界线
    x1 = arange(-3.0, 3.0, 0.1)  # arange生成array
    x0 = mat(array([1.0] * 60))
    x2 = (-weight[0] * x0 - weight[1] * x1) / weight[2]
    x2 = x2.T
    ax.plot(x1, x2)
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()


'''
(6)画weights收敛情况图
'''


def plotweightscording(weightsrecordings):
    import matplotlib.pyplot as plt
    y1 = weightsrecordings[0, :]
    y2 = weightsrecordings[1, :]
    y3 = weightsrecordings[2, :]
    m = weightsrecordings.shape[1]
    x1cord = arange(m)
    fig = plt.figure()  # 开始画图
    # 画点
    ax1 = fig.add_subplot(131)  # 1X1个图中第一个图
    ax1.scatter(x1cord, y1, s=30, c='red')
    ax2 = fig.add_subplot(131)  # 1X1个图中第一个图
    ax2.scatter(x1cord, y2, s=30, c='green')
    ax3 = fig.add_subplot(131)  # 1X1个图中第一个图
    ax3.scatter(x1cord, y3, s=30, c='yellow')
    plt.xlabel('times');
    plt.ylabel('convergence')
    plt.show()


'''
外部, 调用代码：
import chapter5_logisticRegression as cha
import matplotlib as mpl
import matplotlib.pyplot as plt

weight = gradAscent(returnMat, classLabelVector)
plotBestFit(weight)

'''
returnMat, classLabelVector = loadDataSet()
# weight = gradAscent(returnMat, classLabelVector)
# weight,weightsrecording = randomGradAscent0(returnMat, classLabelVector)
# weight = stocGradAscent(returnMat, classLabelVector)
#weight = stocGradAscent0(returnMat, classLabelVector)
weight = stocGradAscent1(returnMat, classLabelVector,500)
plotBestFit(weight)
#plotweightscording(weightsrecording)

'''Mine'''

'''
(1)读、存数据----mine
'''

# def file2matrix(filename):
#     fr = open('H:/ForTheTimeBeing/AAAANewFile/machinelearninginaction/Ch05/%s'%filename)    #绝对路径
#     #fr = open('AAAANewFile/machinelearninginaction/Ch05/%s'%filename)    #相对路径：加上最后一个文件夹
#     #fr = open(filename)
#     arrayLines = fr.readlines()
#     numberOfLines = len(arrayLines)
#     returnMat = zeros((numberOfLines,3))                #zero里面的括号是元组的括号（行，列）,存入zeros
#     classLabelVector = [] #分类标签是一个列表,append即可，而dataSet是数
#     index = 0
#     for LINE in arrayLines:
#         LINE = LINE.strip()
#         listFromLine = LINE.split()
#         returnMat[index,:] = [1.0]+listFromLine[0:2]                    ##分片，不包括2    添加1.0的一列数字
#         classLabelVector.append(int(listFromLine[-1]))
#         index += 1
#     return returnMat, classLabelVector



# 调用的代码
# returnMat, classLabelVector = file2matrix('testSet.txt')
# print(returnMat,classLabelVector)



'''
(3)梯度上升优化算法主体程序----mine
input：
    Label认为是行向量

'''

# def gradAscent(returnMat,classLabelVector):
#     dataMatrix = mat(returnMat).T             #矩阵化
#     LabelVector = mat(classLabelVector)      #矩阵化，，，行向量
#     alpha = 0.0001
#     maxCycle = 1000
#     n = dataMatrix.shape[0]
#     weight = ones((1,n))
#     h = sigmoid(weight*dataMatrix)
#     for i in range(maxCycle):
#         weight = weight + alpha*(LabelVector-h)*dataMatrix.T
#     return  weight



# 调用代码：
# returnMat, classLabelVector = file2matrix('testSet.txt')
# result = gradAscent(returnMat, classLabelVector)
# print(result)
