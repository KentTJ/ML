# -*- coding: UTF-8 -*-
import numpy as np
import operator

'''微博详述：http://blog.csdn.net/c406495762/article/details/75172850#comments'''


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



'''自定义的训练集'''
def creatDataSet():
    group = np.array([[1.0,1.1],[1,1],[0,0],[0,0.1]])    #需要手动修改的训练集
    labels = ['A','A','B','B']                  #需要手动修改的labels
    return group,labels



'''函数说明：文本txt转换为矩阵'''
def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = np.zeros((numberOfLines,3))                #zero里面的括号是元组的括号（行，列）,存入zeros
    index = 0
    classLabelVector = []                            #分类标签是一个列表,append即可，而dataSet是数
    for LINE in arrayLines:
        LINE = LINE.strip()
        listFromLine = LINE.split()
        returnMat[index,:] = listFromLine[0:3]                    ##分片，不包括3
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector



'''
归一化数值函数:dataSet列向量归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值

Modify:
    2017-03-24
'''
def ToNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    dataSetRanges =  dataSet - np.tile(minVals,(m,1))
    dataSetRanges =  dataSetRanges/np.tile(ranges,(m,1))              #数组相除，对应元素相除
    return dataSetRanges,ranges, minVals


"""
#对于输入的一个人进行判断
Parameters:
     dataSet - 特征矩阵
Returns:


Modify:

"""
def classifyPerson():
    percentTats = float(input('玩视频游戏所耗时间百分比：'))
    fMiles  = float(input('每年获得的飞行常客里程数：'))
    iceCream  = float(input('每周消耗的冰淇淋公升数：'))
    inX = np.array([fMiles,percentTats,iceCream])
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # 引入归一化的数据集
    dataSetRanges, ranges, minVals = ToNorm(datingDataMat)
    k = 5                                                             #k=? 需人为设定
    result = classify0(inX, dataSetRanges,datingLabels, k)
    return result

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
模块外部调用的代码：
 #增加工作路径
import sys
sys.path.append("O:\\ForTheTimeBeing\\AAAANewFile")        #注意：双斜杠！！11！！！
 #导入kNNnet00
import kNNnet00

datingDataMat,datingLabels = kNNnet00.file2matrix('datingTestSet2.txt')
##datingDataMat,datingLabels = kNNnet00.file2matrix('J:\\ForTheTimeBeing\\AAAANewFile\\datingTestSet2.txt')           #注意：双斜杠！！11！！！


#画图：#
# import numpy as np
 #import matplotlib
 #import matplotlib.pyplot as plt
 #fig = plt.figure()
 #ax = fig.add_subplot(111)
 #ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
 #plt.show()


#归一化
dataSetRanges,ranges, minVals = kNNnet00.ToNorm(datingDataMat)


#对于输入的一个人进行判断
#增加工作路径
import sys
sys.path.append("O:\\ForTheTimeBeing\\AAAANewFile")        #注意：双斜杠！！11！！！
 #导入kNNnet00
import kNNnet00
resultPerson = kNNnet00.classifyPerson()


#错误率测试
import sys
sys.path.append("O:\\ForTheTimeBeing\\AAAANewFile")        #注意：双斜杠！！11！！！
 #导入kNNnet00
import kNNnet00
kNNnet00.datingClassTest()

'''
