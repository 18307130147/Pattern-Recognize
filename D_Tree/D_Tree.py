import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
import operator


# read dataset
def readdataset(filename):
    dataset = np.array(pd.read_csv(filename))
    value = []
    dataset = np.delete(dataset, 0, 1)
    for result in dataset[:, 0]:
        if result == 'M':
            value.append(1)
        else:
            value.append(0)
    dataset[:, 0] = value
    for i in range(len(dataset[0, :])-1):
        dataset[:, i+1] = pd.cut(dataset[:, i+1], bins=10, labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])



    return dataset



# 计算信息熵
def calcu_Ent(dataset):
    numEntries = len(dataset)  # 记录的条数
    labelCounts = {}
    # 给所有可能分类创建字典
    for featVec in dataset:
        currentlabel = featVec[1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    Ent = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        Ent = Ent - p * log(p, 2)
    return Ent


# 划分数据集,根据第几列，哪个lable来划分，返回此lable下的那几行，剔除该特征
def splitdataset(dataset, axis, value): # axis是第axis列,由i传递过来
    retdataset = []  # 创建返回的数据集列表
    for featVec in dataset:  # 抽取符合划分特征的值
        if featVec[axis] == value:
            reducedfeatVec = featVec[:axis]  # 去掉axis特征
            reducedfeatVec.extend(featVec[axis + 1:])
            retdataset.append(reducedfeatVec)
    return retdataset


# 使用ID3选择最优特征
def ID3_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 2  # 特征种类数
    baseEnt = calcu_Ent(dataset)  # 此dataset下诊断结果的熵
    bestInfoGain = 0.0  # 熵的减少量(最优解)
    bestFeature = -1  # 目前最佳特征，从-1出发
    for i in range(2, numFeatures+2, 1):  # 遍历所有特征,i代表第i列的特征

        featList = [example[i] for example in dataset]  # 将dataset的第i列提出来featlist
        uniqueVals = set(featList)  # 将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt = 0.0
        for value in uniqueVals:  # 计算每种划分方式的信息熵
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            newEnt += p * calcu_Ent(subdataset)
        infoGain = baseEnt - newEnt
        print(u"ID3中第%d列的特征的信息增益为：%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain  # 计算最好的信息增益
            bestFeature = i
    return bestFeature  # beatFeature是指最优特征的列数


# 使用多数表决
def majorityCnt(classList):
    classCont = {}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote] = 0
        classCont[vote] += 1
    sortedClassCont = sorted(classCont.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCont[0][0]


# 利用ID3算法创建决策树
def ID3_createTree(dataset, labels):
    classList = [example[1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 2:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = ID3_chooseBestFeatureToSplit(dataset)  # 返回信息增益最大的列编号
    bestFeatLabel = labels[bestFeat-2]
    print(u"此时最优索引为：{}".format(bestFeatLabel))
    ID3Tree = {bestFeatLabel: {}}
    del (labels[bestFeat-2])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]  # bestFeat那一列
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        ID3Tree[bestFeatLabel][value] = ID3_createTree(splitdataset(dataset, bestFeat, value), subLabels)
    return ID3Tree


def classify(inputTree, featLabels, testVec):

    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = '0'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def classifytest(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


df = pd.read_csv('data.csv')
df.drop(columns='Unnamed: 32', inplace=True)

# 简化lable集 从1到30 共30个特征
lables = []
for m in range(30):
    str = 'feature{}'.format(m + 1)
    lables.append(str)

# 处理df
value = []
for i in df['diagnosis']:
    if i == 'M':
        value.append(1)
    else:
        value.append(0)
df['diagnosis'] = value
database = np.array(df)

for c in range(2, 32, 1):

    # produce a row
    c_column = []
    for j in df.iloc[:, c:c + 1].values:
        c_column.append(j[0])

    # divide by pd.cut
    c_lables = pd.cut(c_column, bins=10, labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    df.iloc[:, c:c + 1] = c_lables

# print(df)

# 将数据集拆分为train和test
Train, Test = train_test_split(database, random_state=0)
print(Train)
# print(Test)

# Test calcu_Ent
# Ent = calcu_Ent(Train)
# print("Ent of Train is:{}".format(Ent))
x = splitdataset(Train, 1, 0)
print(x)

print("以下为首次寻找最优索引:\n")
first_feature = ID3_chooseBestFeatureToSplit(Train)
print("ID3算法的最优特征索引为:{}".format(first_feature))
print("下面开始创建相应的决策树-------")

# 拷贝，createTree会改变labels
labels_tmp = lables[:]

ID3desicionTree = ID3_createTree(df, labels_tmp)
print('ID3desicionTree:\n', ID3desicionTree)

print("下面为测试数据集结果：")
print('ID3_TestSet_classifyResult:\n', classifytest(ID3desicionTree, lables, Test))
print("---------------------------------------------")
