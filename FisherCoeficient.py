import csv

import numpy as np
import math


with open('trainingData.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
cleanData = []
for i in data:
    if i != []:
        cleanData.append(i)
start = "a"
tempArray = []
splitArray = []
for i in range(0, len(cleanData)):
    if cleanData[i][0] == start:
        tempArray.append(cleanData[i])
    else:
        splitArray.append([start, tempArray])
        start = cleanData[i][0]
        tempArray = []

def getFeatureArray(trainList, feature):
    sampleNo = len(trainList)
    tempTrainList = trainList[0][feature:feature + 1]
    tempTrainList = list(tempTrainList)
    for index in range(1, len(trainList)):
        tempTrainList.extend(trainList[index][feature:feature + 1])
    tempNp = np.array(tempTrainList)
    arrayX = tempNp.reshape(sampleNo, 1)
    return arrayX


def getFeatureMedium(trainList, feature):
    itemcount = 0
    summed = 0
    for p in range(0, len(trainList)):
        x = trainList[p][feature:feature + 1]
        summed += float(x[0])
        itemcount += 1
    mean = summed / itemcount
    return mean


def standardDeviation(featureArray, medium):
    var = 0
    # var= 1/n * sum((xi - medium)^2)
    for n in range(0, len(featureArray)):
        var += math.pow(float(featureArray[n][0]) - medium, 2)
    var = var / len(featureArray)
    #standard deviation:
    sigma = math.sqrt(var)
    return sigma


featuresFisher = []
featuresNormalised = []
for i in range(1, 24):
    featureMediumArray=[]
    temp=0
    for cl in splitArray:
        tempMedium=getFeatureMedium(cl[1],i)
        featureMediumArray.append(tempMedium)
        temp+=tempMedium

    generalMedium = temp / (len(splitArray)+1)
    featureArrays=[]
    for c in splitArray:
        featureArrays.append(getFeatureArray(c[1], i))

    standardDeviationArray=[]
    for j in range(0,len(featureArrays)):
        standardDeviationArray.append(standardDeviation(featureArrays[j], featureMediumArray[j]))
    sbTemp=0
    for k in range(0, len(featureMediumArray)):
        sbTemp+= math.pow(featureMediumArray[k]-generalMedium,2)
    sB = abs(sbTemp)
    swTemp=0
    for l in range(0,len(standardDeviationArray)):
        swTemp+=standardDeviationArray[l]
    sW = abs(swTemp)
    fisher = sB / sW
    featuresFisher.append([fisher, i])


featuresFisher.sort(reverse=True)
chosenFeatures = featuresFisher[0:10]

print(chosenFeatures)
# output:
# [[8.835829139804229, 2], [6.086350247494071, 6], [5.443617369297483, 8],
# [5.436438910573064, 10], [4.95652051481179, 7], [4.602061598390422, 12],
# [4.219478896187501, 1], [4.010558192794001, 13], [3.888987188475382, 9],
# [3.6551584738601104, 5]]

