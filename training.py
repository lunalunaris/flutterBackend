import csv
import math
classes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
           "w",
           "x", "y", "z", "A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_", "I_", "J_", "K_", "L_", "M_", "N_", "O_",
           "P_", "Q_", "R_", "S_",
           "T_", "U_", "W_", "X_", "Y_", "Z_", "pi", "delta", "notEq", "(", "[", "{", "smal", ")", "]", "}", "mod",
           "big",
           "contains", "aprox", "sum", "prod", "sqrt", "beta", "dotDiv", "slash", "minus", "smalEq", "bigEq",
           "alpha", "=", "!", "+", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "divLine",
           ",", "dot"]

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

testData = []
testClassed = []
for i in range(0, len(splitArray)):
    for j in range(0, len(splitArray[i][1])):
        if j > 0.7 * len(splitArray[i][1]):
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
    # pass data list of all samples in class
    # chosen features indexes: 3	4 5 6 8	9  22
    for i in range(len(dataList)):
        sqr = math.pow(int(testPoint[3]) - int(dataList[i][3]), 2) + math.pow(int(testPoint[4]) - int(dataList[i][4]),2) + math.pow(
            int(testPoint[5]) - int(dataList[i][5]), 2) + math.pow(int(testPoint[6]) - int(dataList[i][6]), 2) + math.pow(
            int(testPoint[8]) - int(dataList[i][8]), 2) + math.pow(int(testPoint[9]) - int(dataList[i][9]),2) + math.pow(int(testPoint[22]) - int(dataList[i][22]), 2)
        distance = math.sqrt(sqr)
        # appending calculated distance with the class name
        distances.append([distance, dataList[0][0]])
    distances.sort()
    distances = distances[:k]
    return distances
#+ math.pow(
            # int(testPoint[11]) - int(dataList[i][11]), 2) + math.pow(int(testPoint[12]) - int(dataList[i][12]), 2) + math.pow(
            # int(testPoint[16]) - int(dataList[i][16]), 2) + math.pow(int(testPoint[18]) - int(dataList[i][18]),2) + math.pow(
            # int(testPoint[19]) - int(dataList[i][19]), 2) +

def KNN():
    testedData = []
    for point in testData:
        distances = []
        for c in splitArray:
            # we choose K=3 in order to get 3 closest neighbours in each class sample
            distC = getClosest(c[1], point, 3)
            distances+=distC
        # sorting again to get the smallest distances
        distances.sort()
        # we choose 3 shortest distances to the test point for all classes
        distances = distances[:3]
        classes = list()
        for d in distances:
            classes.append(d[1])
        # we choose class that appears the most on the list
        maxClass = max(classes,key=classes.count)
        testedData.append([maxClass, point])
    return testedData


evalData = KNN()
correct = 0
for i in range(0, len(evalData)):
    if evalData[i][0] == evalData[i][1][0]:
        correct += 1
print("Test data: "+str(len(testData)))
print("Training data: "+str(len(cleanData)))
print("Correctly classified: "+str(correct))

accuracy = correct / len(testData)
print("Accuracy: "+ str(accuracy))

#for 12 features:
# Test data: 630
# Training data: 2292
# Correctly classified: 587
# Accuracy: 0.9317460317460318
# for 6 features:
# Test data: 630
# Training data: 2292
# Correctly classified: 587
# Accuracy: 0.9317460317460318

#for 7 classes with no 22 remaining
# Test data: 630
# Training data: 2292
# Correctly classified: 588
# Accuracy: 0.9333333333333333