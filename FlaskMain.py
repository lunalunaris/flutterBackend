import math
import turtle

import matplotlib.pyplot as plt
import csv
import os
from math import floor

import numpy as np
import pandas as pd

from BoundBox import BoundBox
from Coords import Coords
from flask import Flask, jsonify, request

from latexDict import latexDict

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
               "0","+","=",",","!","dot","[","]","|","/"]


@app.route('/')
def home():
    return "Home"


@app.route('/upload', methods=["POST"])
def uploadPhoto():
    if request.method == "POST":
        image = request.files["image"]
        height = request.form["height"]
        width = request.form["width"]
        print(width)
        print(height)
        i = plt.imread(image)
        result =parseImage(i, width, height)
        json_file = {'result': result}
        return jsonify(json_file)


def parseImage(image, width, height):
    greyArray = transformtoGreyscale(image)
    cleanArray = applyThreshold(greyArray)
    lineHorizontal = getProjectionHorizontalOrientation(cleanArray)  # no of rows
    lineVertical = getProjectionVerticalOrientation(cleanArray)
    if lineHorizontal < lineVertical:
        cleanArray = rotate(cleanArray)
    boxes = getCharacterBoxes(cleanArray)

    boxesMerged = []
    temp = []
    index = 0
    for box in boxes:
        # check over and below for boxes contained in left-5, right+5
        if box not in boxesMerged:
            top = box.coords.topLeft[0]
            bottom = box.coords.bottomLeft[0]
            left = box.coords.topLeft[1]
            right = box.coords.topRight[1]
            index2 = 0
            index += 1
            for secBox in boxes:
                width = right - left
                widthSec = secBox.coords.topRight[1] - secBox.coords.topLeft[1]

                if left - 10 <= secBox.coords.topLeft[
                    1] <= right + 10 and box.coords.topLeft != secBox.coords.topLeft and width <= widthSec:
                    if top > secBox.coords.bottomLeft[0] >= top - 10 or bottom > secBox.coords.topLeft[
                        0] > bottom + 10:
                        boxesMerged.append(box)
                        boxesMerged.append(secBox)
                        resultBox = mergeBoxes([secBox, box])
                        temp.append(resultBox)
                index2 += 1
    boxes = [x for x in boxes if x not in boxesMerged]
    boxes = boxes + temp

    boxesSized = []
    for i in range(0, len(boxes)):
        boxesSized.append(resize(boxes[i], len(boxes[i].pixels[0]), len(boxes[i].pixels), 25, 25))
    boxesSized = getZones(boxesSized)
    boxesSized = getProjectionHorizontal(boxesSized)
    boxesSized = getProjectionVertical(boxesSized)
    boxesSized = getColumnCrossings(boxesSized)
    boxesSized = getRowCrossings(boxesSized)
    boxesData = exportToFile(boxesSized)
    print(boxesData[0].pixels)
    print(boxesData[1].pixels)
    trainData = getTrainData()
    testedData = KNN(boxesData, trainData)
    result=orderEquation(testedData)
    return result


def orderEquation(boxes):
    temp = []
    minimum = boxes[0]
    # bubble sort preserves relative order
    for i in range(0, len(boxes)):
        swapped = False
        for j in range(0, len(boxes) - i - 1):
            if boxes[j + 1].coords.topLeft[1] < boxes[j].coords.topLeft[1]:
                temp = boxes[j]
                boxes[j] = boxes[j + 1]
                boxes[j + 1] = temp
                swapped = True
        if not swapped:
            break
        # assume one row only
    line = checkSubString(boxes)
    sets= line.split("=")
    result = sets.pop()
    return result


def checkSubString(row):
    removed = []
    line = ""
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
            for t in row:
                if left < (t.coords.topLeft[1] + t.coords.topRight[1]) / 2 < right and t not in removed:
                    if t[1] < top:
                        topAr += (t[0])
                        removed.append(t[0])
                    else:
                        bottomAr += (t[0])
                        removed.append(t)
            top = checkSubString(topAr)
            bottom = checkSubString(bottomAr)
            a=r"\frac{" + top + "}{" + bottom + "}"
            line += a
            removed.append(row[i])
        elif value == "sqrt" and row[i] not in removed:
            pow = ""
            if i - 1 >= 0:
                mid = (bottom - top) / 2
                if row[i - 1].coords.bottomLeft[0] < mid and row[i - 1] not in removed:
                    if row[i-1].value in classesBase:
                        pow = "[" + row[i - 1].value + "]"
                    else:
                        pow = "[" + latexDict[row[i - 1].value] + "]"
                    removed.append(row[i - 1])
            temp = []
            for t in row:
                if left < t.coords.topLeft[1] < right and top > t.coords.topLeft[0] > bottom and t not in removed:
                    temp.append(t)
                    removed.append(t)
            inside = checkSubString(temp)
            a=r"\sqrt" + pow + "{" + inside + "}"
            line += a
            removed.append(row[i])
        elif value == "l" and value not in removed and i + 2 < maxLen:
            if row[i + 1].value == "o" and row[i + 2].value == "g" and row[i + 1] not in removed and row[
                i + 2] not in removed:
                line += r"\log "
                removed.append(row[i + 1])
                removed.append(row[i + 2])
                i += 2
            else:
                line += "l"
            removed.append(row[i])
        elif value == "s" and value not in removed and i + 2 < maxLen:
            if row[i + 1].value == "i" and row[i + 2].value == "n" and row[i + 1] not in removed and row[
                i + 2] not in removed:
                removed.append(row[i + 1])
                removed.append(row[i + 2])
                line += r"\sin "
                i += 2
            else:
                line += "s"
            removed.append(row[i])
        elif value == "c" and value not in removed and i + 2 < maxLen:
            if row[i + 1].value == "o" and row[i + 2].value == "s" and row[i + 1] not in removed and row[
                i + 2] not in removed:
                removed.append(row[i + 1])
                removed.append(row[i + 2])
                line += r"\cos "
                i += 2
            elif row[i + 1].value == "t" and row[i + 2].value == "a" and row[i + 1] not in removed and row[
                i + 2] not in removed and i + 3 < maxLen:
                if row[i + 3].value == "n" and row[i + 3] not in removed:
                    removed.append(row[i + 1])
                    removed.append(row[i + 2])
                    removed.append(row[i + 3])
                    line += r"\ctan "
                    i += 2
                else:
                    line += "c"
            else:
                line += "c"
            removed.append(row[i])
        elif value == "t" and value not in removed and i + 1 < maxLen:
            if row[i + 1].value == "a" and row[i + 2].value == "n" and row[i + 1] not in removed and row[
                i + 2] not in removed:
                removed.append(row[i])
                removed.append(row[i + 1])
                removed.append(row[i + 2])
                line += r"\tan "
                i += 2
            else:
                line += "t"
        elif value == "dot" and row[i] not in removed:
            if i + 2 < maxLen:
                if row[i + 1].value == "/" and row[i + 2].value == "." and row[i + 1] not in removed and row[
                    i + 2] not in removed:
                    line += r"\% "
                    removed.append(row[i + 1])
                    i += 2
            elif i + 1 < maxLen:
                if row[i + 1].value == "|" and row[i + 1] not in removed:
                    line += r" ! "
                    removed.append(row[i + 1])
                    i += 1
            else:
                line += r"."
            removed.append(row[i])
        elif value == "|" and i + 1 < maxLen and row[i] not in removed:
            if row[i + 1].value == "dot" and row[i + 1] not in removed:
                line += r" ! "
                i += 1
            else:
                line += r" | "
            removed.append(row[i])

        elif value == "minus" and i + 1 < len(row):
            mid = (row[i + 1].coords.topRight[1] - row[i + 1].coords.topLeft[1]) / 2
            if row[i + 1].value == "minus":
                line += r" = "
                i += 1
                removed.append(row[i + 1])
            else:
                line += r" - "
            removed.append(row[i])
        elif row[i] not in removed:
            if i - 1 >= 0:
                mid = (row[i - 1].coords.bottomLeft[0] - row[i - 1].coords.topLeft[0]) / 2
                if top < mid:
                    if value in classesBase:
                        temp=r"^{" + value + "} "
                    else:
                        temp=r"^{" + str(latexDict[value]) + "} "
                    line += temp
                elif top < mid:
                    if value in classesBase:
                        temp=r"_{" + value + "} "
                    else:
                        temp= r"_{" + str(latexDict[value]) + "} "
                    line += temp
                elif value in classesBase:
                    line += value
                    line+=" "
                else:
                    line += latexDict[value]
                    line += " "
            elif value in classesBase:
                line += value
                line += " "
            else:
                line += latexDict[value]
                line += " "
            removed.append(row[i])
    return line


def getClosest(dataList, point, k):
    distances = list()
    # pass data list of all samples in class
    # chosen features indexes: 3	4 5 6 8	9  22
    testPoint = point.featureArray
    for i in range(len(dataList)):
        sqr = math.pow(float(testPoint[4]) - float(dataList[i][4]), 2) + math.pow(
            float(testPoint[5]) - float(dataList[i][5]),
            2) + math.pow(
            float(testPoint[6]) - float(dataList[i][6]), 2) + math.pow(float(testPoint[7]) - float(dataList[i][7]),
                                                                       2) + math.pow(
            float(testPoint[9]) - float(dataList[i][9]), 2) + math.pow(
            float(testPoint[23]) - float(dataList[i][23]), 2)
        distance = math.sqrt(sqr)
        # appending calculated distance with the class name
        distances.append([distance, dataList[0][0]])
    distances.sort()
    distances = distances[:k]
    return distances


def KNN(boxes, splitArray):
    testedData = []
    for point in boxes:
        distances = []
        for c in splitArray:
            # we choose K=3 in order to get 3 closest neighbours in each class sample
            distC = getClosest(c[1], point, 10)
            distances += distC
        # sorting again to get the smallest distances
        distances.sort()
        # we choose 3 shortest distances to the test point for all classes
        distances = distances[:10]
        classes = list()
        for d in distances:
            classes.append(d[1])
        # we choose class that appears the most on the list
        maxClass = max(classes, key=classes.count)
        point.value = maxClass
        print(maxClass)
        point.possibleValues = classes
        testedData.append(point)
    return testedData


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


def rotate(array):
    temp = [[0 for x in range(len(array))] for y in range(len(array[0]))]
    for i in range(0, len(array)):
        for j in range(0, len(array[0])):
            temp[j][len(array) - 1 - i] = array[i][j]

    return temp


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
        if countHorizontal[k] <= 5 and countHorizontal[k + 1] > 5:
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
        if projectionVertical[k] <= 5 and projectionVertical[k + 1] > 5:
            linecount += 1
    return linecount


def transformtoGreyscale(image):
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


def getCharacterBoxes(finalArray):
    tempPixels = []
    queue = []
    checkedPixels = []
    boxes = []
    for i in range(0, len(finalArray)):
        for j in range(0, len(finalArray[0])):
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
    return boxes


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


def resize(box, widthOld, heightOld, widthNew, heightNew):
    expand = []
    # prostokąt orientacja pionowa
    if 1 < heightOld / widthOld:
        wNew = heightOld
        expand = [[255 for x in range(wNew)] for y in range(heightOld)]
        index = 0
        offset = int(floor((wNew - widthOld) / 2))
        for j in range(offset, offset + widthOld):
            for i in range(0, heightOld):
                expand[i][j] = box.pixels[i][index]
            index += 1
    # kwadrat
    elif heightOld == widthOld:
        expand = box.pixels
    # prostokąt orientacja pozioma
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


def getProjectionHorizontal(boxes):
    for box in boxes:
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


def getZones(boxes):
    for box in boxes:
        temp = [[1 for y in range(0, len(box.pixels[0]))] for x in range(0, len(box.pixels))]
        for i in range(0, len(box.pixels)):
            for j in range(0, len(box.pixels[0])):
                if box.pixels[i][j] == BLACK:
                    temp[i][j] = 1
                else:
                    temp[i][j] = 0
        box.binary = temp
    return boxes


def getColumnCrossings(boxes):
    for box in boxes:
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


def exportToFile(boxes):
    featureArray = [[0 for x in range(128)] for y in range(len(boxes))]
    index = 0
    for box in boxes:
        featureArray[index] = [box.value, box.width, box.height, box.merged]
        featureArray[index] = featureArray[index] + box.projectionVertical
        featureArray[index] = featureArray[index] + box.projectionHorizontal
        featureArray[index] = featureArray[index] + box.crossingsHorizontal
        featureArray[index] = featureArray[index] + box.crossingsVertical
        box.featureArray = featureArray[index]
        index += 1

    # for i in range(len(boxes)):
    #     boxes[i].featureArray = featureArray[i]
    return boxes


if __name__ == '__main__':
    app.run(host="0.0.0.0")
