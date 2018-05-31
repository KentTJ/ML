from numpy import *

#数据导数函数（标准,适用性广）
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


#局部加权线性回归算法lwlr（标准）---对单点
#input：手动输入k
#ws=xMat.T*W*xMat.I*(xMat.T*W*yMat)
def lwlr(xArr,yArr,testpoint,k = 0.01):
    #将列表形式的数据转为numpy矩阵形式
    xMat=mat(xArr);yMat=mat(yArr).T
    #权重W，对角矩阵
    m = xMat.shape[0]
    W = mat(eye(m))
    for i in range(m):
        diffxMat = testpoint-xMat[i, :]
        W[i,i] = exp(diffxMat*diffxMat.T/(-2.0*k**2))
    #求矩阵的内积
    xTWx=xMat.T*(W*xMat)
    #numpy线性代数库linalg
    #调用linalg.det()计算矩阵行列式
    #计算矩阵行列式是否为0
    if linalg.det(xTWx)==0.0:
        print('This matrix is singular,cannot do inverse')
        return                            #跳出函数
    #如果可逆，根据公式计算回归系数
    ws=xTWx.I*(xMat.T*W*yMat)
    #可以用yHat=xMat*ws计算实际值y的预测值
    #返回一点处y的预测值
    yHat = testpoint*ws
    return  ws.T,yHat

#
# xMat=mat(xArr)
# testpoint = xMat[0,:]
# ws,y = lwlr(xArr,yArr,testpoint)


#对所有数据点lwlr（标准）
def lwlrForAllpoints(xArr, yArr):
    xMat = mat(xArr)
    m = xMat.shape[0]
    yHat = zeros(m)
    ws = zeros((2,m))
    for i in range(m):
        testpoint = xMat[i, :]
        # ws[:, i] = b                  #!!!!!!!!!!!!!无论，ws按列还是按行存，此处存放的向量b必须行向量？
        ws[:,i],yHat[i]= lwlr(xArr, yArr,testpoint)
    return ws,yHat

wsMat,yHat = lwlrForAllpoints(xArr, yArr)


#计算预测值、画散点图和最佳拟合直线图（标准）
def forYhat(xArr, wsMat):
    xMat = mat(xArr)
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
    index = xMat[:,1].argsort(0)                                     #！！！排序否则画图出错
    x = xMat[index][:,0,:][:,1]                                     #对x进行排序   ！！！！！！！！！[:,0,:]对应[[[]]]三个括号！
    y = yHat[index]                                                 #对y进行排序
    ax.plot(x, y)                ##！！!!！！1！！注意：x,y最好为列向量或者list（行向量出问题！！！）
    # [<matplotlib.lines.Line2D object at 0x0343F570>]
    plt.show()
    return yHat


yHat = forYhat(xArr,wsMat)
#
#
# ###计算相关系数：
# cor = corrcoef(yHat.T,yArr)
# print(cor)