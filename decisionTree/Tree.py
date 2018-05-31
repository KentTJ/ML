#-*-coding:utf-8-*-
from import_file import *


#计算香农熵

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVet in dataSet:
        label = featVet[-1]
        if label not in labelCounts:
            #创建键和值
            labelCounts[label] = 1
        else:
            labelCounts[label] += 1
    ShannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key]/float(numEntries)
        ShannonEnt += -prob* log(prob,2)
    return ShannonEnt

# def calcShannonEnt(dataSet):
#     numEntries = len(dataSet)    # 总记录数
#     labelCounts = {}    # dataSet中所有出现过的标签值为键，相应标签值出现过的次数作为值
#     for featVec in dataSet:
#         currentLabel = featVec[-1]
#         labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
#     shannonEnt = 0.0
#     for key in labelCounts:
#         prob = float(labelCounts[key])/numEntries
#         shannonEnt += -prob * log(prob, 2)
#     return shannonEnt

#给定dataSet按照某一特征划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #挖掉axis的值
            reduceDataSet = featVec[:axis]
            reduceDataSet.extend(featVec[axis + 1:])
            retDataSet.append(reduceDataSet)
    return  retDataSet

# def splitDataSet(dataSet, axis, value):
#     retDataSet = []
#     for featVec in dataSet:
#         if featVec[axis] == value:
#             reducedFeatVec = featVec[:axis]
#             reducedFeatVec.extend(featVec[axis+1:])
#             retDataSet.append(reducedFeatVec)
#     return retDataSet

#从不同特征中，选择最好的划分方式,选取ShannonEnt最小值，即信息增益最大值
# def chooseBestFeatureToSplit(dataSet):
#     numFeature = len(dataSet[0])- 1
#     numDataSet = len(dataSet)
#     ShannonEnt = 0.0
#     ShannonEntlist = []
#     soleFeatureList = []
#     for i in range(numFeature):
#         #找出所有value值，放入列表
#         for j in range(len(dataSet)):
#             if dataSet[j][i] not in soleFeatureList:
#                 soleFeatureList.append(dataSet[j][i])
#         for value in soleFeatureList:
#             retDataSet = splitDataSet(dataSet,i,value)
#             prob = len(retDataSet)/float(numDataSet)
#             ShannonEnt += -prob * log(prob, 2)
#         ShannonEntlist.append(ShannonEnt)
#     bestFeature = ShannonEntlist.index(min(ShannonEntlist))
#     return bestFeature

'''书本代码'''
# 3-3 选择最好的'数据集划分方式'（特征）
# 一个一个地试每个特征，如果某个按照某个特征分类得到的信息增益（原香农熵-新香农熵）最大，
# 则选这个特征作为最佳数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#多数表决
def majorityCnt(classList):
    classCount = {}
    #统计数据列表中每个特征值出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #根据出现的次数进行排序 key=operator.itemgetter(1) 意思是按照次数进行排序
    #classCount.items() 转换为数据字典 进行排序 reverse = True 表示由大到小排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    #返回次数最多的一项的特征值
    return sortedClassCount[0][0]

# #创建树的函数代码
# def createTree(dataSet,labels):
'''(main） 创建树
input:此处的label是feature的label!!!!!!
'''
def createTree(dataSet, labels):
    # 获取数据集中的最后一列的类标签，存入classList列表
    classList = [example[-1] for example in dataSet]
    # 通过count()函数获取类标签列表中第一个类标签的数目
    # 判断数目是否等于列表长度，相同表面所有类标签相同，属于同一类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有的特征属性，此时数据集的列为1，即只有类标签列
    if len(dataSet[0]) == 1:
        # 多数表决原则，确定类标签
        return majorityCnt(classList)
    # 确定出当前最优的分类特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 在特征标签列表中获取该特征对应的值
    bestFeatLabel = labels[bestFeat]
    # 采用字典嵌套字典的方式，存储分类树信息
    myTree = {bestFeatLabel: {}}

    ##此位置书上写的有误，书上为del(labels[bestFeat])
    ##相当于操作原始列表内容，导致原始列表内容发生改变
    ##按此运行程序，报错'no surfacing'is not in list
    ##以下代码已改正

    # 复制当前特征标签列表，防止改变原始列表的内容
    subLabels = labels[:]
    # 删除属性列表中当前分类数据集特征
    del (subLabels[bestFeat])
    # 获取数据集中最优特征所在列
    featValues = [example[bestFeat] for example in dataSet]
    # 采用set集合性质，获取特征的所有的唯一取值
    uniqueVals = set(featValues)
    # 遍历每一个特征取值
    for value in uniqueVals:
        # 采用递归的方法利用该特征对数据集进行分类
        # @bestFeatLabel 分类特征的特征标签值
        # @dataSet 要分类的数据集
        # @bestFeat 分类特征的标称值
        # @value 标称型特征的取值
        # @subLabels 去除分类特征后的子特征标签列表
        # featureToKeyValuea  = splitDataSet \(dataSet, bestFeat, value
        myTree[bestFeatLabel][value] = createTree(splitDataSet \
                                                      (dataSet, bestFeat, value), subLabels)
    return myTree



# 定义决策树决策结果的属性，用字典来定义
# 下面的字典定义也可写作 decisionNode={boxstyle:'sawtooth',fc:'0.8'}
# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # annotate是关于一个数据点的文本
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


#后面定义了
# def createPlot():
#     fig = plt.figure(1,facecolor='white') # 定义一个画布，背景为白色
#     fig.clf() # 把画布清空
#     # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，
#     #111表示figure中的图有1行1列，即1个，最后的1代表第一个图
#     # frameon表示是否绘制坐标轴矩形
#     createPlot.ax1 = plt.subplot(111,frameon=False)
#     plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
#     plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
#     plt.show()

#获取叶子节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0] #字典的第一个键，即树的一个结点
    secondDict = myTree[firstStr]  #这个键的值，即该结点的所有子树
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

#获取树层数：
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0] #字典的第一个键，即树的一个结点
    secondDict = myTree[firstStr]  #这个键的值，即该结点的所有子树
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            BBB = getTreeDepth(secondDict[key])   #存储下来，后面多次调用到了
            thisDepth = 1 + BBB
        else:
            thisDepth = 1
        if maxDepth < thisDepth:maxDepth = thisDepth
    return maxDepth

#画树枝上文本
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  #当前树的叶子数
    depth = getTreeDepth(myTree) #没有用到这个变量
    firstStr = list(myTree.keys())[0]
    #cntrPt文本中心点   parentPt 指向文本中心的点
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt) #画分支上的键
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #从上往下画
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#如果是字典则是一个判断（内部）结点
            plotTree(secondDict[key],cntrPt,str(key))
        else:   #打印叶子结点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

'''mian画决策树'''
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])# 定义横纵坐标轴，无内容
    #createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) # 绘制图像,无边框,无坐标轴
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))   #全局变量宽度 = 叶子数
    plotTree.totalD = float(getTreeDepth(inTree))  #全局变量高度 = 深度
    #图形的大小是0-1 ，0-1
    plotTree.xOff = -0.5/plotTree.totalW;  #例如绘制3个叶子结点，坐标应为1/3,2/3,3/3
    #但这样会使整个图形偏右因此初始的，将x值向左移一点。
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


dataSet = [[1,1,'yes'],
           [1,1,'yes'],
           [1,0,'no'],
           [0,1,'no'],
           [0,1,'no']]
labels  = ['no surfacing','flippers']



# shannonEnt = calcShannonEnt(dataSet)
# print(shannonEnt)
#
# retData = splitDataSet(dataSet,0,1)
# print(retData)
#
# bestFeature = chooseBestFeatureToSplit(dataSet)
# print(bestFeature)

'测试代码'
# myTree = createTree(dataSet,labels)
# print(myTree)
# print('labels = %s'%labels)
#
#
# # numLeafs0 = getNumLeafs(myTree)
# # numtreesDepth = getNumtreesDepth(myTree)
# # print('numLeafs0 = %d'%numLeafs0)
# # print('numtreesDepth = %d'%numtreesDepth)
#
# aa = createPlot(myTree)

'''测试算法：使用决策树执行分类任务'''
def classifyOneVec(Tree,inputVec,featrureLabels):
    firststr = list(Tree.keys())[0]
    secondDict = Tree[firststr]
    fistfeatureIndex = featrureLabels.index(firststr)
    outPutLabel = []
    featureToKeyValue = secondDict[inputVec[fistfeatureIndex]]
    if type(featureToKeyValue).__name__ == 'dict':
        outPut = classifyOneVec(featureToKeyValue,inputVec,featrureLabels)
    else:
        outPut = featureToKeyValue
    return outPut

#判断一组数据的分类
def classifyDataSet(Tree,DataSet,featureLab):
    labelList = []
    for vector in DataSet:
        outPut = classifyOneVec(Tree,vector,featureLab)
        labelList.append(outPut)
    return labelList


inputdataSet= [[1,1,'yes'],
           [1,1,'yes'],
           [1,0,'no'],
           [0,1,'no'],
           [0,1,'no']]

# #测试代码
# labels  = ['no surfacing','flippers']
# featrureLabel = labels
# # outputLabels = classifyOneVec(myTree,inputdataSet,featrureLabel)
# outputLabels = classifyDataSet(myTree,inputdataSet,featrureLabel)
# print('outputLabels = %s'%outputLabels)


'''（主程序）决策树的存储-----------构造决策树耗时，所以要存储下来，用时直接拿来判断'''
#使用dump()将数据序列化到文件中
def storeTree(inputTree,filename):
    fw = open(filename,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()

#使用load()将数据从文件中序列化读出
def loadTree(fileName):
    fr = open(fileName,'rb')
    return pickle.load(fr)

#测试代码
# storeTree(myTree,'MyTree.txt')
# Trees = loadTree('MyTree.txt')
# print('loadTree = %s'%Trees)



'''示例：使用决策树预测隐形眼镜类型'''
#读取
def loadDataSet(FileName):
    # 创建两个列表
    dataMat = []
    labelMat = []
    # 打开文本数据集
    fr = open(FileName)  ####  fr = open('H:/ForTheTimeBeing/AAAANewFile/machinelearninginaction/Ch05/testSet.txt')
    # 遍历文本的每一行
    for line in fr.readlines():
        # 对当前行去除首尾空格，并按空格进行分离
        lineArr = line.strip().split('\t')
        # 将每一行的两个特征x1，x2，加上x0=1,组成列表并添加到数据集列表中
        dataMat.append([lineArr[0], lineArr[1],lineArr[2],lineArr[3],lineArr[4]])
    # 返回数据列表
    return dataMat

#测试代码
dataMat = loadDataSet('M:/ForTheTimeBeing/AAAANewFile/machinelearninginaction/Ch03/lenses.txt')
lensesLabel = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(dataMat,lensesLabel)
createPlot(lensesTree)
storeTree(lensesTree,'lensesTree')

print(lensesTree)
print(dataMat)
print(labelMat)


