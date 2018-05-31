# -*- coding: UTF-8 -*-
from numpy import *



"""注意：虽然能运算，但结果与本书差的太远"""

#分类大量数据计算过程
'''
weights为行向量
'''
def classifyInx(inX,weights):
    #weights为行向量，列化
    weights = mat(weights).T
    inX = mat(inX)
    a= -inX*weights
    result = sigmoid(-inX*weights)
    classresult =[]
    for i in range(len(result)):
        if result[i] >= 0.5 :
            classresult.append(1.0)                      #1.0保证是数字，而不是字符
        else:
            classresult.append(0.0)
    return  classresult


#总：读取、分类大量数据、计算错误率
def colicTest(trainingFileName,testFileName):
    dataMatrixTrain, classLabelsTrain = loadDataSet(trainingFileName)
    weightsTrain = stocGradAscent1(dataMatrixTrain, classLabelsTrain)
    dataMatrixTest, classLabelsTest = loadDataSet(testFileName)
    testResult = classifyInx(dataMatrixTest,weightsTrain)
    index = 0
    m = len(testResult)
    for i in range(m):
        if testResult[i]  != classLabelsTest[i]:
            print(testResult[i],classLabelsTest[i])
            index += 1
    errorRate = index/m
    print('错误率为：%f'%errorRate)
    return testResult

'''
调用函数：

colicTest('horseColicTraining.txt','horseColicTest.txt')
'''



'''
分类器的错误率的判断函数
Parameters:
    dataSet - 特征矩阵
Returns:

Modify:
'''
def datingClassTest():
    #取百分之十
    hoRatio = 0.10
    #文件转化矩阵
    datingDataMat, datingLabels = file2matrix('O:\\ForTheTimeBeing\\AAAANewFile\\datingTestSet2.txt')
    #归一化
    dataSetRanges, ranges, minVals = ToNorm(datingDataMat)
    #抽取
    m = dataSetRanges.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    dataSet = dataSetRanges[numTestVecs+1:m,:]
    labels = datingLabels[numTestVecs+1:m]
    k=3
    for i in range(numTestVecs):
        result = classify0(dataSetRanges[i,:], dataSet, labels, k)
        print('%d正确的结果是：%d' % (result, datingLabels[i]))
        if result != datingLabels[i]: errorCount += 1
        print('%d'%errorCount)
        print('%d' % numTestVecs)
    #errorRate = errorCount/float(numTestVecs)
    errorRate = errorCount / numTestVecs
    print('错误率为：%f'%errorRate)

'''
############problem:  alpha = 0.0001，maxCycle = 1000这样设置收敛有问题，不同数字差异很大
梯度上升优化算法：
input：
'''
'''
(1)读、存数据
'''
# 读取数据——这里内部可以优化
def loadDataSet(filename):
    # 创建两个列表
    dataMat = [];
    labelMat = []
    # 打开文本数据集
    fr = open('H:/ForTheTimeBeing/AAAANewFile/machinelearninginaction/Ch05/%s'%filename)
    m = 0      #记录行数！1！！1！1可以优化
    # 遍历文本的每一行
    for line in fr.readlines():
        m += 1
        # 对当前行去除首尾空格，并按空格进行分离
        lineArr = line.strip().split()
        # 将每一行的两个特征x1，x2，加上x0=1,组成列表并添加到数据集列表中
        for j in range(21):
            dataMat.append( float(lineArr[j]))
        # 将当前行标签添加到标签列表
        labelMat.append(float(lineArr[21]))
    #将列表元素存入array
    dataMatrix = zeros((m,21))
    for i in range(m):              ##!!1!!!!可以优化
        dataMatrix[i,:] = dataMat[21*i:21*i+21]
    # 返回数据列表，标签列表
    return dataMatrix, labelMat


'''
(2)定义sigmoid函数
inX    数字
'''
def sigmoid(inX):
    result = 1 / (1 + exp(-inX))
    return result


'''
(3)改进的随机梯度上升算法：                       ！！！！！！！优
#@dataMatrix：数据集列表
#@classLabels：标签列表
#@numIter：迭代次数，默认150
'''


def stocGradAscent1(dataMatrix, classLabels, numIter=500):
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



colicTest('horseColicTraining.txt','horseColicTest.txt')