from numpy import *

#数据导数函数（标准）
def loadDataSet(filename):
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

xArr,yArr = loadDataSet('H:/ForTheTimeBeing/AAAANewFile/machinelearninginaction/Ch08/ex0.txt')


#标准线性回归算法（标准）
#ws=(X.T*X).I*(X.T*Y)
def standRegres(xArr,yArr):
    #将列表形式的数据转为numpy矩阵形式
    xMat=mat(xArr);yMat=mat(yArr).T
    #求矩阵的内积
    xTx=xMat.T*xMat
    #numpy线性代数库linalg
    #调用linalg.det()计算矩阵行列式
    #计算矩阵行列式是否为0
    if linalg.det(xTx)==0.0:
        print('This matrix is singular,cannot do inverse')
        return                            #跳出函数
    #如果可逆，根据公式计算回归系数
    ws=xTx.I*(xMat.T*yMat)
    #可以用yHat=xMat*ws计算实际值y的预测值
    #返回归系数
    return ws

ws = standRegres(xArr,yArr)

#计算预测值、画散点图和最佳拟合直线图（标准）
def forYhat(xArr,ws):
    xMat = mat(xArr)
    yHat = xMat*ws
    #画图
    import matplotlib.pyplot as plt
    fig = plt.figure()  # 开始画图
    ## 画点
    x1 = xMat[:,1]
    x1 = x1.flatten()
    x1 = list(x1)
    y1 = yArr
    ax = fig.add_subplot(111)  # 1X1个图中第一个图
    ax.scatter(x1, y1, s=30, c='red')
    ##画线: 坐标范围：1、用DataSet范围 2、或指定范围
    xMatCopy = xMat.copy()                      #！！！排序否则画图出错
    xMatCopy.sort(0)
    x = xMatCopy[:,1]
    yHatSort = (xMatCopy*ws)
    ax.plot(x, yHatSort)                ##！！1！！1！！注意：x,yHatSort最好为列向量或者list（行向量出问题！！！）
    # [<matplotlib.lines.Line2D object at 0x0343F570>]
    plt.show()
    return yHat


yHat = forYhat(xArr,ws)


###计算相关系数：
cor = corrcoef(yHat.T,yArr)
print(cor)