import math
import os
from _csv import writer

import matplotlib.pyplot as plt
import csv
from math import floor
import time
from BoundBox import BoundBox
from Coords import Coords
from flask import Flask, jsonify, request
from skimage import morphology
import numpy as np
from latexdict import latexDict

app = Flask(__name__)
WHITE = 255
BLACK = 0
HEIGHT = 25
WIDTH = 25
classes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
           "w",
           "x", "y", "z", "A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_", "I_", "J_", "K_", "L_", "M_", "N_", "O_",
           "P_", "Q_", "R_", "S_",
           "T_", "U_", "W_", "X_", "Y_", "Z_", "pi", "delta", "notEq", "(", "[", "{", "smal", ")", "]", "}", "mod",
           "big",
           "contains", "aprox", "sum", "prod", "sqrt", "beta", "dotDiv", "slash", "minus", "smalEq", "bigEq",
           "alpha", "=", "!", "+", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "divLine",
           ",", "dot"]

classesBase = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
               "v", "w",
               "x", "y", "z", "(", ")", "1", "2", "3", "4", "5", "6", "7", "8", "9",
               "0", "+", "=", ",", "!", "dot", "[", "]", "|", "/"]


@app.route('/')
def home():
    return "Home"


@app.route('/upload', methods=["POST"])
def uploadPhoto():
    if request.method == "POST":
        image = request.files["image"]
        height = request.form["height"]
        width = request.form["width"]
        userId = request.form["userId"]
        i = plt.imread(image)
        timeStart = time.time()
        result, boxResult = parseImage(i, width, height)
        timeEnd = time.time()
        totalTime = timeEnd - timeStart
        with open(userId + ".csv", "w") as f:
            write = csv.writer(f)
            for i in boxResult:
                write.writerow(i)

        print(totalTime)
        json_file = {'result': result}
        return jsonify(json_file)


@app.route('/verify', methods=["POST"])
def verifyResult():
    if request.method == "POST":
        jsonRes = request.get_json()
        userId = jsonRes["userId"]
        print(userId)
        file = userId + '.csv'
        with open(file, newline='') as csvfile:
            data = list(csv.reader(csvfile))

        with open('trainingData.csv', 'a') as f:
            writer_object = writer(f)
            for box in data:
                writer_object.writerow(box)
            f.close()

        if os.path.exists(file) and os.path.isfile(file):
            os.remove(file)
        json_file = {'response': "OK"}
        return jsonify(json_file)


def parseImage(image, width, height):
    greyArray = transformtoGreyscale(image, width, height)
    cleanArray = applyThreshold(greyArray)
    if width != len(image[0]):
        print("rotating first")
        cleanArray = rotate(cleanArray)
    # lineHorizontal = getProjectionHorizontalOrientation(cleanArray)  # no of rows
    # lineVertical = getProjectionVerticalOrientation(cleanArray)
    # print(lineVertical)
    # print(lineHorizontal)
    # if lineHorizontal > lineVertical:
    #     cleanArray = rotate(cleanArray)
    boxes = getCharacterBoxes(cleanArray)
    boxesMerged = []
    temp = []
    index = 0
    boxesExlude = []
    tempBoxes = boxes
    for i in range(0, len(boxes)):
        # check over and below for boxes contained in left-5, right+5
        if boxes[i] not in boxesMerged and boxes[i].value != "EOL":
            top = boxes[i].coords.topLeft[0]
            bottom = boxes[i].coords.bottomLeft[0]
            left = boxes[i].coords.topLeft[1]
            right = boxes[i].coords.topRight[1]
            index2 = 0
            index += 1
            for secBox in boxes:
                width = right - left
                widthSec = secBox.coords.topRight[1] - secBox.coords.topLeft[1]
                if left - 20 <= secBox.coords.topLeft[
                    1] <= right + 20 and boxes[i].coords.topLeft != secBox.coords.topLeft and width <= widthSec:
                    if top > secBox.coords.bottomLeft[0] >= top - 20 or bottom > secBox.coords.topLeft[0] > bottom + 20:
                        boxesMerged.append(boxes[i])
                        boxesMerged.append(secBox)
                        boxesExlude.append(secBox)
                        resultBox = mergeBoxes([secBox, boxes[i]])
                        tempBoxes[i] = resultBox
                        boxesMerged.append(resultBox)
                index2 += 1
    boxes = [x for x in tempBoxes if x not in boxesExlude]

    boxesSized = []
    for i in range(0, len(boxes)):
        if boxes[i].value != "EOL":
            boxesSized.append(resize(boxes[i], len(boxes[i].pixels[0]), len(boxes[i].pixels), 25, 25))
        else:
            boxesSized.append(boxes[i])
    print(boxesSized[1].pixels)
    boxesSized = skeletonize(boxesSized)
    print(boxesSized[1].pixels)
    boxesSized = getBinary(boxesSized)
    boxesSized = getProjectionHorizontal(boxesSized)
    boxesSized = getProjectionVertical(boxesSized)
    boxesSized = getColumnCrossings(boxesSized)
    boxesSized = getRowCrossings(boxesSized)
    boxesData = extractFeatures(boxesSized)
    trainData = getTrainData()
    testedData = KNN(boxesData, trainData)
    result, boxResult = orderEquation(testedData)
    return result, boxResult


def transformtoGreyscale(image, width, height):
    toupleArr = [[0 for x in range(len(image[0]))] for y in range(len(image))]
    for j in range(0, len(image[0])):
        for i in range(0, len(image)):
            toupleArr[i][j] = int((int(image[i][j][0]) + int(image[i][j][1]) + int(image[i][j][2])) / 3)
    return toupleArr


def reformArray(greyAr, width, height):
    finalArray = []
    index = 0
    subArray = []
    for i in range(0, height):
        for j in range(0, width):
            subArray.append(greyAr[index])
            index += 1
        finalArray.append(subArray)
        subArray = []
    return finalArray


def getThreshold(greyAr):
    tempArray = []
    for i in range(0, len(greyAr)):
        for j in range(0, len(greyAr[0])):
            tempArray.append(greyAr[i][j])
    tempArray.sort()
    lowerBound = 0
    upperBound = 0
    for a in tempArray:
        if a != BLACK:
            lowerBound = a
            break
    tempArray.sort(reverse=True)
    for b in tempArray:
        if b != WHITE:
            upperBound = b
            break
    return lowerBound + (0.4 * (upperBound - lowerBound))


def applyThreshold(greyAr):
    threshold = getThreshold(greyAr)
    for i in range(0, len(greyAr)):
        for j in range(0, len(greyAr[0])):
            if greyAr[i][j] > threshold:
                greyAr[i][j] = WHITE
            else:
                greyAr[i][j] = BLACK
    return greyAr


def getProjectionHorizontalOrientation(array):
    linecount = 0
    countHorizontal = []
    for j in range(0, len(array[0])):
        countHorizontal.append(0)
    for i in range(0, len(array)):
        for j in range(0, len(array[0])):
            if array[i][j] == BLACK:
                countHorizontal[j] = countHorizontal[j] + 1
    for k in range(0, len(countHorizontal) - 1):
        if countHorizontal[k] <= 2 and countHorizontal[k + 1] > 2:
            linecount += 1
    return linecount


def getProjectionVerticalOrientation(array):
    projectionVertical = []
    for j in range(0, len(array)):
        projectionVertical.append(0)
    linecount = 0
    for j in range(0, len(array[0])):
        for i in range(0, len(array)):
            if array[i][j] == BLACK:
                projectionVertical[i] = projectionVertical[i] + 1
    for k in range(0, len(projectionVertical) - 1):
        if projectionVertical[k] <= 2 and projectionVertical[k + 1] > 2:
            linecount += 1
    return linecount


def rotate(array):
    print("rotated")
    temp = [[0 for x in range(len(array))] for y in range(len(array[0]))]
    for i in range(0, len(array)):
        for j in range(0, len(array[0])):
            temp[j][len(array) - 1 - i] = array[i][j]

    return temp


def getCharacterBoxes(finalArray):
    tempPixels = []
    queue = []
    checkedPixels = []
    boxes = []
    count = 0
    EOL = BoundBox(0, 0, coords=Coords([0, 0], [0, 0], [0, 0], [0, 0]), pixels=[])
    EOL.value = "EOL"
    boxes.append(EOL)
    for i in range(0, len(finalArray)):
        for j in range(0, len(finalArray[0])):
            if finalArray[i][j] == BLACK:
                count += 1
            if finalArray[i][j] == BLACK and [i, j] not in checkedPixels:
                checkedPixels.append([i, j])
                tempPixels.append([i, j])
                queue = addToQueue(i, j, queue)
                for point in queue:
                    if len(finalArray) > point[0] >= 0 and len(finalArray[0]) > point[1] >= 0:
                        if finalArray[point[0]][point[1]] == BLACK and [point[0],
                                                                        point[1]] not in checkedPixels:
                            checkedPixels.append([point[0], point[1]])
                            tempPixels.append([point[0], point[1]])
                            queue = addToQueue(point[0], point[1], queue)
                box = getBoundingBox(finalArray, tempPixels)
                if box is not None:
                    boxes.append(box)
                tempPixels = []
                queue = []
        if count == 0 and boxes[-1].value != 'EOL':
            boxes.append(EOL)
        count = 0

    return boxes


def addToQueue(i, j, queue):
    queue.append([i - 1, j - 1])
    queue.append([i - 1, j])
    queue.append([i - 1, j + 1])
    queue.append([i, j - 1])
    queue.append([i, j + 1])
    queue.append([i + 1, j - 1])
    queue.append([i + 1, j])
    queue.append([i + 1, j + 1])
    return queue


def getBoundingBox(finalArray, tempPixels):
    allI = []
    allJ = []
    for k in tempPixels:
        allI.append(k[0])
        allJ.append(k[1])
    minI = min(allI)
    minJ = min(allJ)
    maxI = max(allI)
    maxJ = max(allJ)
    pixels = []
    temp = []
    for i in range(minI, maxI + 1):
        for j in range(minJ, maxJ + 1):
            if [i, j] in tempPixels:
                temp.append(finalArray[i][j])
            else:
                temp.append(WHITE)
        pixels.append(temp)
        temp = []
    if sum([x.count(0) for x in pixels]) > 2:
        return BoundBox(maxI - minI, maxJ - minJ, Coords((minI, minJ), (minI, maxJ), (maxI, minJ), (maxI, maxJ)),
                        pixels)
    else:
        return None


def mergeBoxes(boxes):
    bottom = []
    top = []
    left = []
    right = []
    for box in boxes:
        if box.value != "EOL":
            bottom.append(box.coords.bottomLeft[0])
            top.append(box.coords.topLeft[0])
            left.append(box.coords.topLeft[1])
            right.append(box.coords.topRight[1])
    newTop = min(top)
    newBottom = max(bottom)
    newLeft = min(left)
    newRight = max(right)

    newBox = [[255 for x in range(newRight - newLeft + 1)] for y in range(newBottom - newTop + 1)]

    for box in boxes:
        if box != "EOL":
            iIndex = 0
            for i in range(box.coords.topLeft[0] - newTop, box.coords.bottomLeft[0] - newTop + 1):
                jIndex = 0
                for j in range(box.coords.topLeft[1] - newLeft, box.coords.topRight[1] - newLeft + 1):
                    if box.pixels[iIndex][jIndex] == BLACK:
                        newBox[i][j] = box.pixels[iIndex][jIndex]
                    jIndex += 1
                iIndex += 1
    newBoxfinal = BoundBox(newBottom - newTop, newRight - newLeft,
                           Coords((newTop, newLeft), (newTop, newRight), (newBottom, newLeft),
                                  (newBottom, newRight)),
                           newBox)
    newBoxfinal.value = boxes[0].value
    newBoxfinal.merged = 1
    return newBoxfinal


def resize(box, widthOld, heightOld, widthNew, heightNew):
    expand = []
    # horizontal orientation
    if 1 < heightOld / widthOld:
        wNew = heightOld
        expand = [[255 for x in range(wNew)] for y in range(heightOld)]
        index = 0
        offset = int(floor((wNew - widthOld) / 2))
        for j in range(offset, offset + widthOld):
            for i in range(0, heightOld):
                expand[i][j] = box.pixels[i][index]
            index += 1
    # square
    elif heightOld == widthOld:
        expand = box.pixels
    # vertical orientation
    elif int(heightOld / widthOld) < 1:
        hNew = widthOld
        expand = [[255 for x in range(widthOld)] for y in range(hNew)]
        index = 0
        offset = int(floor((hNew - heightOld) / 2))
        for i in range(offset, offset + heightOld):
            for j in range(0, widthOld):
                expand[i][j] = box.pixels[index][j]
            index += 1
    box.pixels = expand
    return singleResize(box, len(expand[0]), len(expand), widthNew, heightNew)


def singleResize(box, widthOld, heightOld, widthNew, heightNew):
    wScale = float(widthOld / widthNew)
    hScale = float(heightOld / heightNew)
    box.width = box.width * wScale
    box.height = box.height * hScale
    scaled = [[0 for x in range(widthNew)] for y in range(heightNew)]
    for j in range(0, widthNew):
        for i in range(0, heightNew):
            scaled[i][j] = box.pixels[int(i * hScale)][int(j * wScale)]
    box.pixels = scaled
    return box


def skeletonize(boxes):
    for box in boxes:
        if box.value != "EOL":
            tempArray = [[0 for y in range(0, len(box.pixels[0]))] for x in range(0, len(box.pixels))]
            for i in range(0, len(box.pixels)):
                for j in range(0, len(box.pixels[0])):
                    if box.pixels[i][j] == 0:
                        tempArray[i][j] = 1
            npmatrix = (np.array(tempArray)).astype(np.uint8)
            skele = morphology.medial_axis(npmatrix)
            tempArray = [[255 for y in range(0, len(box.pixels[0]))] for x in range(0, len(box.pixels))]
            for i in range(0, len(box.pixels)):
                for j in range(0, len(box.pixels[0])):
                    if skele[i][j] == True:
                        tempArray[i][j] = 0
            box.pixels = tempArray
    return boxes


def getBinary(boxes):
    for box in boxes:
        if box.value != "EOL":
            temp = [[1 for y in range(0, len(box.pixels[0]))] for x in range(0, len(box.pixels))]
            for i in range(0, len(box.pixels)):
                for j in range(0, len(box.pixels[0])):
                    if box.pixels[i][j] == BLACK:
                        temp[i][j] = 1
                    else:
                        temp[i][j] = 0
            box.binary = temp
    return boxes


def getProjectionHorizontal(boxes):
    for box in boxes:
        if box.value != "EOL":
            countHorizontal = [0] * len(box.pixels[0])
            for i in range(0, len(box.pixels)):
                for j in range(0, len(box.pixels[0])):
                    if box.pixels[i][j] == BLACK:
                        countHorizontal[j] = countHorizontal[j] + 1
            temp = []
            for i in range(0, len(countHorizontal), 5):
                if i + 4 < len(countHorizontal):
                    temp.append(
                        countHorizontal[i] + countHorizontal[i + 1] + countHorizontal[i + 2] + countHorizontal[i + 3] +
                        countHorizontal[i + 4])
            box.projectionHorizontal = temp
    return boxes
def getProjectionVertical(boxes):
    for box in boxes:
        if box.value != "EOL":
            projectionVertical = [0] * len(box.pixels)
            for j in range(0, len(box.pixels[0])):
                for i in range(0, len(box.pixels)):
                    if box.pixels[i][j] == BLACK:
                        projectionVertical[i] = projectionVertical[i] + 1
            temp = []
            for i in range(0, len(projectionVertical), 5):
                if i + 4 < len(projectionVertical):
                    temp.append(
                        projectionVertical[i] + projectionVertical[i + 1] + projectionVertical[i + 2] +
                        projectionVertical[
                            i + 3] + projectionVertical[i + 4])
            box.projectionVertical = temp
    return boxes


def getColumnCrossings(boxes):
    for box in boxes:
        if box.value != "EOL":
            crossings = []
            for j in range(0, len(box.pixels[0]), 5):
                if j < len(box.pixels[0]):
                    sum = 0
                    for i in range(0, len(box.pixels)):

                        if i != 0:
                            if box.binary[i][j] == 1 and box.binary[i - 1][j] == 0:
                                sum += 1
                        elif i == 0 and box.binary[i][j] == 1:
                            sum += 1
                    crossings.append(sum)
            box.crossingsVertical = crossings
    return boxes


def getRowCrossings(boxes):
    for box in boxes:
        if box.value != "EOL":
            crossings = []
            for i in range(0, len(box.pixels), 5):
                if i < len(box.pixels):
                    sum = 0
                    for j in range(0, len(box.pixels[0])):
                        if j != 0:
                            if box.binary[i][j] == 1 and box.binary[i][j - 1] == 0:
                                sum += 1
                        elif j == 0 and box.binary[i][j] == 1:
                            sum += 1
                    crossings.append(sum)
            box.crossingsHorizontal = crossings
    return boxes


def extractFeatures(boxes):
    featureArray = [[0 for x in range(128)] for y in range(len(boxes))]
    index = 0
    for box in boxes:
        if box.value != "EOL":
            featureArray[index] = [box.value, box.width, box.height, box.merged]
            featureArray[index] = featureArray[index] + box.projectionVertical
            featureArray[index] = featureArray[index] + box.projectionHorizontal
            featureArray[index] = featureArray[index] + box.crossingsHorizontal
            featureArray[index] = featureArray[index] + box.crossingsVertical
            box.featureArray = featureArray[index]
            index += 1
    return boxes


def getTrainData():
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
    return splitArray


def KNN(boxes, splitArray):
    testedData = []
    for point in boxes:
        if point.value != "EOL":
            distances = []
            for c in splitArray:
                distC = getClosest(c[1], point, 11)
                distances += distC
            distances.sort()
            distances = distances[:11]
            classes = list()
            for d in distances:
                classes.append(d[1])
            maxClass = max(classes, key=classes.count)
            point.value = maxClass
            point.featureArray[0] = str(maxClass)
            print(maxClass)
            point.possibleValues = classes
        testedData.append(point)
    return testedData


def getClosest(dataList, point, k):
    distances = list()
    testPoint = point.featureArray
    for i in range(len(dataList)):
        sqr = math.pow(float(testPoint[4]) - float(dataList[i][4]), 2) \
              + math.pow(float(testPoint[5]) - float(dataList[i][5]),2)\
              + math.pow(float(testPoint[6]) - float(dataList[i][6]), 2) \
              + math.pow(float(testPoint[7]) - float(dataList[i][7]),2)\
              + math.pow(float(testPoint[9]) - float(dataList[i][9]), 2) \
              + math.pow(float(testPoint[19]) - float(dataList[i][19]), 2) \
              + math.pow(float(testPoint[23]) - float(dataList[i][23]), 2)
        distance = math.sqrt(sqr)
        distances.append([distance, dataList[0][0]])
    distances.sort()
    distances = distances[:k]
    return distances


# math.pow(float(testPoint[4]) - float(dataList[i][4]), 2) + math.pow(
#     float(testPoint[6]) - float(dataList[i][6]),
#     2) + math.pow(
#     float(testPoint[8]) - float(dataList[i][8]), 2) + math.pow(float(testPoint[12]) - float(dataList[i][12]),
#                                                                2) + math.pow(
#     float(testPoint[14]) - float(dataList[i][14]), 2) + math.pow(
#     float(testPoint[16]) - float(dataList[i][16]), 2) + math.pow(
#     float(testPoint[18]) - float(dataList[i][18]), 2)


def orderEquation(boxes):
    temp = []
    tempArray = []
    for a in range(0, len(boxes)):
        if boxes[a].value == "EOL":
            if a != 0:
                tempArray.append(temp)
                temp = []
        else:
            temp.append(boxes[a])

    print(tempArray)
    for k in range(0, len(tempArray)):

        for i in range(0, len(tempArray[k])):
            swapped = False
            for j in range(0, len(tempArray[k]) - i - 1):
                if tempArray[k][j + 1].coords.topLeft[1] < tempArray[k][j].coords.topLeft[1]:
                    temp = tempArray[k][j]
                    tempArray[k][j] = tempArray[k][j + 1]
                    tempArray[k][j + 1] = temp
                    swapped = True
            if not swapped:
                break
    boxesFinal = []
    print(tempArray[0][0].value)
    for i in range(0, len(tempArray)):
        for j in range(0, len(tempArray[0])):
            boxesFinal.append(tempArray[i][j])
    #
    line, orderedBoxes = checkSubString(boxesFinal, 0)
    boxResult = []
    for i in range(len(orderedBoxes) - 1, -1, -1):
        if orderedBoxes[i][0] != "=":
            boxResult.append(orderedBoxes[i][1])
        else:
            break
    print(line)
    sets = line.split("=")
    result = sets.pop()
    return result, boxResult


def checkSubString(row, loopNo):
    removed = []
    line = ""
    orderedBoxes = []
    if loopNo <= 2:
        loopNo += 1
        for i in range(len(row)):
            left = row[i].coords.topLeft[1]
            right = row[i].coords.topRight[1]
            top = row[i].coords.topLeft[0]
            bottom = row[i].coords.bottomLeft[0]
            maxLen = len(row)
            value = row[i].value

            if value == "divLine" and row[i] not in removed:
                topAr = []
                bottomAr = []
                if i + 1 < maxLen:
                    if row[i + 1] == "divLine":
                        orderedBoxes.append(["=", row[i].featureArray, row[i + 1].featureArray])
                        line += r"="
                        removed.append(row[i])
                        removed.append(row[i + 1])
                        i += 1
                    else:
                        for t in row:
                            if left < (t.coords.topLeft[0] + t.coords.topRight[0]) / 2 \
                                    < right and t not in removed:
                                if float(t.coords.bottomLeft[1]) < float(top):
                                    topAr.append(t)
                                    removed.append(t)
                                else:
                                    bottomAr.append(t)
                                    removed.append(t)
                            top, tempTop = checkSubString(topAr, loopNo)
                            bottom, tempBottom = checkSubString(bottomAr, loopNo)
                            a = r"\frac{" + top + "}{" + bottom + "}"
                            line += a
                            orderedBoxes.append(["divLine", row[i].featureArray])
                            orderedBoxes = orderedBoxes + tempTop + tempBottom
                            removed.append(row[i])
                else:
                    for t in row:
                        if left < (t.coords.topLeft[0] + t.coords.topRight[0]) / 2 < right and t not in removed:
                            if float(t.coords.bottomLeft[1]) < float(top):
                                topAr.append(t)
                                removed.append(t)
                            else:
                                bottomAr.append(t)
                                removed.append(t)
                        top, tempTop = checkSubString(topAr, loopNo)
                        bottom, tempBottom = checkSubString(bottomAr, loopNo)
                        a = r"\frac{" + top + "}{" + bottom + "}"
                        line += a
                        orderedBoxes.append(["divLine", row[i].featureArray])
                        orderedBoxes = orderedBoxes + tempTop + tempBottom
                        removed.append(row[i])
            elif value == "sqrt" and row[i] not in removed:
                pow = ""
                if i - 1 >= 0:
                    mid = top + ((bottom - top) / 2)
                    if row[i - 1].coords.bottomLeft[0] < mid and row[i - 1] not in removed:
                        if row[i - 1].value in classesBase:
                            pow = "[" + row[i - 1].value + "]"
                        else:
                            pow = "[" + latexDict[row[i - 1].value] + "]"
                        removed.append(row[i - 1])
                        orderedBoxes.append([row[i - 1].value, row[i - 1].featureArray])
                temp = []
                for t in row:
                    if left < t.coords.topLeft[1] < right and top > t.coords.topLeft[0] > bottom and t not in removed:
                        temp.append(t)
                        removed.append(t)
                inside, tempInside = checkSubString(temp, loopNo)
                a = r"\sqrt" + pow + "{" + inside + "}"
                line += a
                removed.append(row[i])
                orderedBoxes.append(["sqrt", row[i].featureArray])
                orderedBoxes = orderedBoxes + tempInside
            elif value == "l" and row[i] not in removed and i + 2 < maxLen:
                orderedBoxes.append(["l", row[i].featureArray])
                if row[i + 1].value == "o" and row[i + 2].value == "g" and row[i + 1] not in removed and row[
                    i + 2] not in removed:
                    line += r"\log"
                    removed.append(row[i])
                    removed.append(row[i + 1])
                    removed.append(row[i + 2])
                    orderedBoxes.append(["0", row[i + 1].featureArray])
                    orderedBoxes.append(["g", row[i + 2].featureArray])
                    i += 2
                else:
                    line += "l"
                    removed.append(row[i])
            elif value == "s" and row[i] not in removed and i + 2 < maxLen:
                orderedBoxes.append(["s", row[i].featureArray])
                if row[i + 1].value == "i" and row[i + 2].value == "n" and \
                    row[i + 1] not in removed and row[
                    i + 2] not in removed:
                    removed.append(row[i])
                    removed.append(row[i + 1])
                    removed.append(row[i + 2])
                    orderedBoxes.append(["i", row[i + 1].featureArray])
                    orderedBoxes.append(["n", row[i + 2].featureArray])
                    line += r"\sin"
                    i += 2
                else:
                    line += "s"
                    removed.append(row[i])

            elif value == "c" and row[i] not in removed and i + 2 < maxLen:
                orderedBoxes.append(["c", row[i].featureArray])
                if row[i + 1].value == "o" and row[i + 2].value == "s" and row[i + 1] not in removed and row[
                    i + 2] not in removed:
                    removed.append(row[i])
                    removed.append(row[i + 1])
                    removed.append(row[i + 2])
                    orderedBoxes.append(["o", row[i + 1].featureArray])
                    orderedBoxes.append(["s", row[i + 2].featureArray])
                    line += r"\cos"
                    i += 2
                elif row[i + 1].value == "t" and row[i + 2].value == "a" and row[i + 1] not in removed and row[
                    i + 2] not in removed and i + 3 < maxLen:

                    if row[i + 3].value == "n" and row[i + 3] not in removed:
                        orderedBoxes.append(["t", row[i + 1].featureArray])
                        removed.append(row[i])
                        removed.append(row[i + 1])
                        removed.append(row[i + 2])
                        removed.append(row[i + 3])
                        orderedBoxes.append(["a", row[i + 2].featureArray])
                        orderedBoxes.append(["n", row[i + 3].featureArray])
                        line += r"\ctan"
                        i += 2
                    else:
                        line += "c"
                        removed.append(row[i])
                else:
                    line += "c"
                    removed.append(row[i])
            elif value == "t" and row[i] not in removed and i + 1 < maxLen:
                orderedBoxes.append(["t", row[i].featureArray])
                if row[i + 1].value == "a" and row[i + 2].value == "n" and row[i + 1] not in removed and row[
                    i + 2] not in removed:
                    removed.append(row[i])
                    removed.append(row[i + 1])
                    removed.append(row[i + 2])
                    orderedBoxes.append(["a", row[i + 1].featureArray])
                    orderedBoxes.append(["n", row[i + 2].featureArray])
                    line += r"\tan"
                    i += 2
                else:
                    line += "t"
                    removed.append(row[i])
            elif value == "dot" and row[i] not in removed:
                mid = row[i + 1].coords.topLeft[0] + (
                        (row[i + 1].coords.bottomLeft[0] - row[i + 1].coords.topLeft[0]) / 2)
                if i + 1 < maxLen:
                    if row[i].coords.topLeft[0] < mid:
                        removed.append(row[i])
                        line += r"\cdot"
                    elif row[i + 1].value == "|" and row[i + 1] not in removed:
                        line += r" ! "
                        removed.append(row[i])
                        removed.append(row[i + 1])
                        i += 1
                    else:
                        removed.append(row[i])
                        line += r"."
                elif i + 2 < maxLen:
                    if row[i + 1].value == "/" and row[i + 2].value == "dot" and row[i + 1] not in removed and row[
                        i + 2] not in removed and row[i].coods.topLeft[0] < mid:
                        line += r"\% "
                        removed.append(row[i])
                        removed.append(row[i + 1])
                        i += 2
                else:
                    line += r"."
                    removed.append(row[i])
            elif value == "|" and i + 1 < maxLen and row[i] not in removed:
                if row[i + 1].value == "dot" and row[i + 1] not in removed:
                    line += r"!"
                    removed.append(row[i])
                    removed.append(row[i + 1])
                    i += 1
                else:
                    line += r"|"
                    removed.append(row[i])

            elif value == "minus" and i + 1 < len(row) and row[i] not in removed:
                mid = row[i + 1].coords.topLeft[1] + (
                        (row[i + 1].coords.topRight[1] - row[i + 1].coords.topLeft[1]) / 2)
                if row[i + 1].value == "minus" and left < mid < right:
                    orderedBoxes.append(["=", row[i].featureArray, row[i + 1].featureArray])
                    line += r"="
                    removed.append(row[i])
                    removed.append(row[i + 1])
                    i += 1
                else:
                    line += r"-"
                    orderedBoxes.append(["minus", row[i].featureArray])
                    removed.append(row[i])
            elif row[i] not in removed:
                orderedBoxes.append([row[i].value, row[i].featureArray])
                if i - 1 >= 0:
                    mid = row[i - 1].coords.topLeft[0] + (
                            (row[i - 1].coords.bottomLeft[0] - row[i - 1].coords.topLeft[0]) / 2)
                    if bottom < mid:
                        if value in classesBase:
                            temp = r"^{" + value + "} "
                        else:
                            temp = r"^{" + str(latexDict[value]) + "} "
                        line += temp
                        # if prev == g and top> topi-1
                    # elif top > mid:
                    #     if value in classesBase:
                    #         temp=r"_{" + value + "} "
                    #     else:
                    #         temp= r"_{" + str(latexDict[value]) + "} "
                    #     line += temp
                    elif value in classesBase:
                        line += value
                    else:
                        line += latexDict[value]
                elif value in classesBase:
                    line += value
                else:
                    line += latexDict[value]
                removed.append(row[i])
    return line, orderedBoxes


if __name__ == '__main__':
    app.run(host="0.0.0.0")
