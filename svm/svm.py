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

xArr,yArr = loadDataSet('M:/ForTheTimeBeing/AAAANewFile/machinelearninginaction/Ch06/testSet.txt')
# a = 1

#选取不是i的整数
def selectJrand(i,m):
   j = i
   while (j == i):
       j = int(random.uniform(0,m))
   return j

# i = 3 ;m = 10
# j = selectJrand(i,m)
# a = 1

#保证 L《a << H
def clipAlpha(aj,H,L):
    if aj < L:
        aj = L
    if aj > H:
        aj = H
    return aj

#smo算法简化版
#@dataMat    ：数据列表
#@classLabels：标签列表
#@C          ：权衡因子（增加松弛因子而在目标优化函数中引入了惩罚项）
#@toler      ：容错率
#@Iter       :迭代次数
#@maxIter    ：最大迭代次数
def smoSimple(dataMatin,label,C,toler,maxIter):
    dataMatrix = mat(dataMatin); labelMat = mat(label).T
    b = 0;iter = 0
    m , n =shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # WWWWW = multiply(alphas, labelMat)
            # qqqqq = (dataMatrix * dataMatrix[i, :].T)
            fxi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei = fxi - float(labelMat[i])
            # 如果不满足KKT条件，即labelMat[i]*fXi<1(labelMat[i]*fXi-1<-toler)
            # and alpha<C 或者labelMat[i]*fXi>1(labelMat[i]*fXi-1>toler)and alpha>0
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择第二个变量alphaj
                j = selectJrand(i, m)
                # 计算第二个变量对应数据的预测值
                WWW = multiply(alphas, labelMat).T
                QQQ =(dataMatrix * dataMatrix[j, :].T)
                fXj = float(multiply(alphas, labelMat).T * \
                            (dataMatrix * dataMatrix[j, :].T)) + b
                # 计算与测试与实际值的差值
                Ej = fXj - float(label[j])
                # 记录alphai和alphaj的原始值，便于后续的比较
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 如何两个alpha对应样本的标签不相同
                if (labelMat[i] != labelMat[j]):
                # 求出相应的上下边界
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print('L==H');continue
                # 根据公式计算未经剪辑的alphaj
                # ------------------------------------------
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                # 如果eta>=0,跳出本次循环
                if eta >= 0: print("eta>=0");continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # ------------------------------------------
                # 如果改变后的alphaj值变化不大，跳出本次循环
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough");continue
                # 否则，计算相应的alphai值
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 再分别计算两个alpha情况下对于的b值
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                # 如果0<alphai<C,那么b=b1
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                # 否则如果0<alphai<C,那么b=b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                # 否则，alphai，alphaj=0或C
                else:
                    b = (b1 + b2) / 2.0
                # 如果走到此步，表面改变了一对alpha值
                alphaPairsChanged += 1
                print('iter: %d i:%d,paird changed %d' %(iter, i, alphaPairsChanged))
                # 最后判断是否有改变的alpha对，没有就进行下一次迭代
        if (alphaPairsChanged == 0):
            iter += 1
        # 否则，迭代次   0，继续循环
        else:
            iter = 0
        print("iteration number: %d" % iter)
    # 返回最后的b值和alpha向量
    return b, alphas

b,alphas = smoSimple(xArr,yArr,0.6,0.001,40)

# b,alphas = smoSimple(xArr,yArr,0.6,0.001,40)
# d = alphas[alphas>0]
#
# c =1

#
# def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
#     dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
#     b = 0; m,n = shape(dataMatrix)
#     alphas = mat(zeros((m,1)))
#     iter = 0
#     while (iter < maxIter):·
#         alphaPairsChanged = 0
#         for i in range(m):
#             fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
#             Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
#             if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
#                 j = selectJrand(i,m)
#                 fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
#                 Ej = fXj - float(labelMat[j])
#                 alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
#                 if (labelMat[i] != labelMat[j]):
#                     L = max(0, alphas[j] - alphas[i])
#                     H = min(C, C + alphas[j] - alphas[i])
#                 else:
#                     L = max(0, alphas[j] + alphas[i] - C)
#                     H = min(C, alphas[j] + alphas[i])
#                 if L==H: print ("L==H"); continue
#                 eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
#                 if eta >= 0: print ("eta>=0"); continue
#                 alphas[j] -= labelMat[j]*(Ei - Ej)/eta
#                 alphas[j] = clipAlpha(alphas[j],H,L)
#                 if (abs(alphas[j] - alphaJold) < 0.00001): print( "j not moving enough"); continue
#                 alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
#                                                                         #the update is in the oppostie direction
#                 b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
#                 b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
#                 if (0 < alphas[i]) and (C > alphas[i]): b = b1
#                 elif (0 < alphas[j]) and (C > alphas[j]): b = b2
#                 else: b = (b1 + b2)/2.0
#                 alphaPairsChanged += 1
#                 print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
#         if (alphaPairsChanged == 0): iter += 1
#         else: iter = 0
#         print("iteration number: %d" % iter)
#     return b,alphas



#
# #画散点图和超平面图-------------------------
# def PLOT(xArr, wsMat):
#     xMat = mat(xArr);yMat = mat(wsMat)
#     #按label把数据分开
#     m = len(xArr)
#     x1 = [1];x2 = [1];y1 = [2];y2 = [2]
#     for i in range(m):
#         if (wsMat[i] == -1):
#             #为何这句有问题？append
#             x1.append((xArr[i])[0]);y1.append((xArr[i])[1])
#         else:
#             x2.append((xArr[i])[0]);y2.append((xArr[i])[1])
#     #画图
#     import matplotlib.pyplot as plt
#     fig = plt.figure()  # 开始画图
#     ## 画点scatter
#     ax = fig.add_subplot(111)  # 1X1个图中第一个图
#     ax.scatter(x1, y1, s=30, c='red')
#     ax.scatter(x2, y2, s=30, c='blue')
#     # #画线: 坐标范围：1、用DataSet范围 2、或指定范围
#     ###解方程
#     A = multiply(alphas,yMat).T*mat(xArr)
#     index = xMat[:,0].argsort(0)                                     #！！！排序否则画图出错
#     x = xMat[index][:,0,:][:,0]                                     #对x进行排序   ！！！！！！！！！[:,0,:]对应[[[]]]三个括号！
#     B = ones((m,1))*b[0,0]
#     y = (-B-A[0,0]*x)/A[0,1]
#     ax.plot(x, y)                                        ##！！!!！！1！！注意：x,y最好为列向量或者list（行向量出问题！！！）
#     # [<matplotlib.lines.Line2D object at 0x0343F570>]
#     plt.show()

# aaaa = PLOT(xArr,yArr)




