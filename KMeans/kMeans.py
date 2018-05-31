import numpy as np
from numpy import *

def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

    
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    m,n=shape(dataSet)
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    #参数：dataset,num of cluster,distance func,initCen
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))#store the result matrix,2 cols for index and error
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):#for every points
            minDist = inf;minIndex = -1#init
            for j in range(k):#for every k centers，find the nearest center
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:#if distance is shorter than minDist
                    minDist=distJI;minIndex=j# update distance and index(类别)
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                #此处判断数据点所属类别与之前是否相同（是否变化，只要有一个点变化就重设为True，再次迭代）
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        # update k center
        for cent in range(k):
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment
