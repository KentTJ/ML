
import Adaboost

#Standard Scientific Import
#增加工作路径
import sys
sys.path.append("M:\\ForTheTimeBeing\\AAAANewFile\\ML\\ensamble\\Adaboost")


# 马疝病数据集
# 训练集合
dataArr, labelArr = Adaboost.loadDataSet("horseColicTraining2.txt")
weakClassArr, aggClassEst = Adaboost.adaBoostTrainDS(dataArr, labelArr, 40)
print (weakClassArr, '\n-----\n', aggClassEst.T)
# 计算ROC下面的AUC的面积大小
Adaboost.plotROC(aggClassEst.T, labelArr)
# 测试集合
from numpy import *
dataArrTest, labelArrTest = Adaboost.loadDataSet("horseColicTest2.txt")
m = shape(dataArrTest)[0]
predicting10 = Adaboost.adaClassify(dataArrTest, weakClassArr)
errArr = mat(ones((m, 1)))
# 测试：计算总样本数，错误样本数，错误率
print (m, errArr[predicting10 != mat(labelArrTest).T].sum(), errArr[predicting10 != mat(labelArrTest).T].sum()/m )