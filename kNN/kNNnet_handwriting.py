#-*-coding:utf-8-*-
import numpy as np
import operator
from os import listdir
'''
手写识别系统
'''




"""
函数说明:原始kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果

Modify:
    2017-07-13


"""
def classify0(inX, dataSet, labels, k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet                          # tile把inx的行重复dataSetSize次，列重复1次
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)                                         # axis=1  行加，（）**2+（）**2
    #开方，计算出距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


def imag2matrix(filenameStr):
    returnVector = np.zeros((1,1024))
    #fr = open('H:\\ForTheTimeBeing\\AAAANewFile\\machinelearninginaction\\Ch02\\digits\\trainingDigits\\%s'%filenameStr)
    fr = open('machinelearninginaction\\Ch02\\digits\\trainingDigits\\%s'%filenameStr)
    for i in range(32):
        arrayLines = fr.readline()
        arrayLines = arrayLines.strip()
        for j in range(32):
            returnVector[0,i*32+j] = int(arrayLines[j])          ###3#行和列的下标从0开始
    return returnVector

'''
import kNNnet_handwriting
#testVector = kNNnet_handwriting.imag2matrix(filename)
testVector = kNNnet_handwriting.imag2matrix('0_0.txt')

'''




'''
手写识别系统的测试代码——错误率
'''
    ##读取数据
def dataSetAndLabels(trainingFileName):
    #读取数据存入一个大数组
    trainingFileList = listdir('H:\\ForTheTimeBeing\\AAAANewFile\\machinelearninginaction\\Ch02\\digits\\%s'%trainingFileName)   ##path
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    labels = []
        #读取一个文件
    for i in range(m):
        fileName = trainingFileList[i]
        vector = imag2matrix(fileName)
        trainingMat[i,:] = vector
        #读取LABELS,解析filename
        labels.append(fileName[0])
    return trainingMat,labels,m


#测试代码——错误率
def handwriting_ClassTest():
    trainingMat,labels,m = dataSetAndLabels('trainingDigits')
    testMat,testlabels,mtest= dataSetAndLabels('testDigits')
    errorCount = 0.0
    k = 3
    for i in range(mtest):
        result = classify0(testMat[i], trainingMat, labels, k)
        print('%s正确的结果是：%s' % (result,testlabels[i]))
        if result != testlabels[i]:
            errorCount += 1
        print('%d' % errorCount)
        print('%d' % mtest)
    # errorRate = errorCount/float(numTestVecs)
    errorRate = errorCount / mtest
    print('错误率为：%f' % errorRate)

#def datingClassTest():
#
#    #文件转化矩阵
#    #datingDataMat, datingLabels = file2matrix('O:\\ForTheTimeBeing\\AAAANewFile\\datingTestSet2.txt')
#    #归一化
#    #dataSetRanges, ranges, minVals = ToNorm(datingDataMat)
#    #抽取
#    #m = dataSetRanges.shape[0]
#    #numTestVecs = int(m*hoRatio)


'''调用测试代码——错误率
import kNNnet_handwriting
kNNnet_handwriting.handwriting_ClassTest()
'''

import kNNnet_handwriting
kNNnet_handwriting.handwriting_ClassTest()