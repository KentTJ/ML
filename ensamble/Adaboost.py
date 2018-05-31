"""Adaboost
adaptive boosting (AdaBoost-SAMME) for classification problems.

"""

# Authors: Gang Chen


from numpy import*

'''简单数据集构造'''
def loadSimpleData:
    datMat = matrix([[1.,2.1],
                    [2.,1.1,],
                    [1.3,1.],
                    [1.,1.],
                    [2.,1.]])
    classLables = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat ,classLables


