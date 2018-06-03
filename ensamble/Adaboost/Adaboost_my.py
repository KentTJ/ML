"""Adaboost
adaptive boosting (AdaBoost-SAMME) for classification problems.

"""

# Authors: Gang Chen

from numpy import*

'''简单数据集构造'''

# def loadSimpleData():
#     datMat = matrix([[1.,2.1],
#                     [2.,1.1,],
#                     [1.3,1.],
#                     [1.,1.],
#                     [2.,1.]])
#     classLables = [1.0,1.0,-1.0,-1.0,1.0]
#     return datMat ,classLables
#
# def drawData():
#     datMat, classLables = loadSimpleData()
#     import matplotlib.pyplot as plt
#     fig = plt.figure()  # 开始画图
#     x1 = list(datMat[:, 0])
#     x2 = list(datMat[:, 1])
#     ax = fig.add_subplot(111)  # 1X1个图中第一个图
#     ax.scatter(x1, x2, s=30, c='red')

"""读"""
#数据导数函数
def loadDataSet(filename):
    #获取feature的number
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)             ###保证了同一行的放到同一list元素里
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#xArr,yArr = loadDataSet('horseColicTest2.txt')


def adaBoostTrainDs(dataArr,classLables,numIt = 40):
    """训练一个弱分类器
    args:
        dataArr
        classLables
        numIt  迭代次数
    return:
        weakClassifier
        """
    m = shape(dataArr)[0]
    D = mat(ones(m,1))
    for i in range():
        beatStump,errorRate = buidStump(dataArr,classLables,D)
        print('D:',D)
        alpha =0.5*(log(1-errorRate)/errorRate)
        D 

def adaClassify(datToClass,classifierArr):
    """Adaboost 分类器
    Args:
        datToClass
        classifierArr
    return:
        sign     分类结果
        """
    dataMat = mat(datToClass)
