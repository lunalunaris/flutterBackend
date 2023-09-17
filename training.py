import csv
import math
import random
import time
classes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
           "w",
           "x", "y", "z", "A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_", "I_", "J_", "K_", "L_", "M_", "N_", "O_",
           "P_", "Q_", "R_", "S_",
           "T_", "U_", "W_", "X_", "Y_", "Z_", "pi", "delta", "notEq", "(", "[", "{", "smal", ")", "]", "}", "mod",
           "big",
           "contains", "aprox", "sum", "prod", "sqrt", "beta", "dotDiv", "slash", "minus", "smalEq", "bigEq",
           "alpha", "=", "!", "+", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "divLine",
           ",", "dot"]

timeStart = time.time()
with open('timeTest.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
cleanData = []
for i in data:
    if i != []:
        cleanData.append(i)

trainTemp = cleanData

def getMax(array):
    temp = [0.0] * (len(array[0]))
    for i in range(0, len(array)):
        for j in range(0, len(array[0])):
            if j != 0:
                if temp[j] < (float(array[i][j])):
                    temp[j] = (float(array[i][j]))
    return temp


def getMin(array):
    temp = []
    for i in range(0, len(array[0])):
        if i != 0:
            temp.append((float(array[0][i])))
        else:
            temp.append("")
    for i in range(0, len(array)):
        for j in range(0, len(array[0])):
            if j != 0:
                if temp[j] > float(array[i][j]):
                    temp[j] = float(array[i][j])
    return temp


maximum = getMax(trainTemp)

minumum = getMin(trainTemp)
def normalize(cleanData, maximum, minimum):
    trainNormalized = [[0.0 for x in range(len(cleanData[0]))] for y in range(len(cleanData))]
    for j in range(0, len(cleanData)):
        for k in range(0, len(cleanData[0])):
            temp = cleanData[j][k]
            if k == 0:
                trainNormalized[j][k] = temp
            else:
                trainNormalized[j][k] = ((float(temp)) - minimum[k]) / (maximum[k] - minimum[k])
    start = "a"
    tempArray = []
    splitArray = []
    for i in range(0, len(trainNormalized)):
        if trainNormalized[i][0] == start:
            tempArray.append(trainNormalized[i])
        else:
            splitArray.append([start, tempArray])
            start = trainNormalized[i][0]
            tempArray = []
    return splitArray

splitArray= normalize(cleanData, maximum,minumum)

testData = []
testClassed = []
indexes = []

while len(indexes) < 24 * 0.3:
    index = random.randint(0, 24)
    if index not in indexes:
        indexes.append(index)
print(indexes)

for i in range(0, len(splitArray)):
    for j in range(0, len(indexes)):
        testData.append(splitArray[i][1][j])

start = "a"
tempArray = []
for i in range(0, len(testData)):
    if testData[i][0] == start:
        tempArray.append(testData[i])
    else:
        testClassed.append([start, tempArray])
        start = testData[i][0]
        tempArray = []


def getClosest(dataList, testPoint, k):
    distances = list()

    # from fisher's coeficient, the features were chosen: 2,6,8,10, 7,12,1,13,9,5
    for i in range(len(dataList)):
        sqr =math.pow(float(testPoint[4]) - float(dataList[i][4]), 2) + math.pow(
    float(testPoint[6]) - float(dataList[i][6]),
    2) + math.pow(
    float(testPoint[8]) - float(dataList[i][8]), 2) + math.pow(float(testPoint[12]) - float(dataList[i][12]),
                                                               2) + math.pow(
    float(testPoint[14]) - float(dataList[i][14]), 2) + math.pow(
    float(testPoint[16]) - float(dataList[i][16]), 2) + math.pow(
    float(testPoint[18]) - float(dataList[i][18]), 2)

        distance = math.sqrt(sqr)
        # appending calculated distance with the class name
        distances.append([distance, dataList[0][0]])
    distances.sort()
    distances = distances[:k]
    return distances

def KNN():
    testedData = []
    for point in testData:
        distances = []
        for c in splitArray:
            # we choose K=3 in order to get 3 closest neighbours in each class sample
            distC = getClosest(c[1], point, 3)
            distances += distC
        # sorting again to get the smallest distances
        distances.sort()
        # we choose 3 shortest distances to the test point for all classes
        distances = distances[:3]
        classes = list()
        for d in distances:
            classes.append(d[1])
        # we choose class that appears the most on the list
        maxClass = max(classes, key=classes.count)
        testedData.append([maxClass, point])
    return testedData


evalData = KNN()
correct = 0
for i in range(0, len(evalData)):
    if evalData[i][0] == evalData[i][1][0]:
        correct += 1
print("Test data: " + str(len(testData)))
print("Training data: " + str(len(cleanData)))
print("Correctly classified: " + str(correct))

accuracy = correct / len(testData)
print("Accuracy: " + str(accuracy))
timeEnd = time.time()
totalTime = timeEnd - timeStart
print(totalTime)

# for 12 features:
# Test data: 630
# Training data: 2292
# Correctly classified: 587
# Accuracy: 0.9317460317460318
# for 6 features:
# Test data: 630
# Training data: 2292
# Correctly classified: 587
# Accuracy: 0.9317460317460318

# for 7 classes with no 22 remaining
# Test data: 630
# Training data: 2292
# Correctly classified: 588
# Accuracy: 0.9333333333333333

# with randomised, skeletonised and new clases from Fisher's coeficient, 7 best features:

# Test data: 720
# Training data: 2292
# Correctly classified: 648
# Accuracy: 0.9

# with 8 features:

# Test data: 720
# Training data: 2292
# Correctly classified: 656
# Accuracy: 0.9111111111111111

#with 10 features
# Test data: 720
# Training data: 2292
# Correctly classified: 645
# Accuracy: 0.8958333333333334

# with 9 features:
#     Test
#     data: 720
#     Training
#     data: 2292
#     Correctly
#     classified: 650
#     Accuracy: 0.9027777777777778

#with hand picked features from before skeletonization
# Test data: 720
# Training data: 2292
# Correctly classified: 617
# Accuracy: 0.8569444444444444

#hand picked data after skeletonisation
# Test data: 720
# Training data: 2292
# Correctly classified: 652
# Accuracy: 0.9055555555555556

#with normalization
# Test data: 728
# Training data: 2295
# Correctly classified: 656
# Accuracy: 0.9010989010989011

