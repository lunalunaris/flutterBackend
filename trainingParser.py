import csv
import os
from math import floor

from matplotlib import pyplot as plt

from BoundBox import BoundBox
from Coords import Coords

WHITE = 255
BLACK = 0
HEIGHT = 50
WIDTH = 50


def load(folder):
    images = []
    for fileName in os.listdir("dataset/" + folder):

        file = plt.imread(os.path.join("dataset/", folder, fileName))
        if file is not None:
            images.append(file)
    return images


def transformToGreyscale(image):
    toupleArr = []
    greyAr = []
    temp = []
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            toupleArr.append((image[i][j][0], image[i][j][1], image[i][j][2]))

    for j in range(0, len(toupleArr)):
        temp = int((int(toupleArr[j][0]) + int(toupleArr[j][1]) + int(toupleArr[j][2])) / 3)
        greyAr.append(temp)
    return greyAr


def getTreshold(greyAr):
    tempArray = []
    for i in range(0, len(greyAr)):
        tempArray.append(greyAr[i])
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
    return lowerBound + (0.5 * (upperBound - lowerBound))


def applyTreshhold(greyAr):
    treshold = getTreshold(greyAr)
    for i in range(0, len(greyAr)):
        if greyAr[i] > treshold:
            greyAr[i] = WHITE
        else:
            greyAr[i] = BLACK
    return greyAr


def reformArray(greyAr):
    finalArray = []
    index = 0
    subArray = []
    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            subArray.append(greyAr[index])
            index += 1
        finalArray.append(subArray)
        subArray = []
    return finalArray


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
    if sum([x.count(0) for x in pixels]) >= 2:
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
    # print (newBox)
    newBoxfinal = BoundBox(newBottom - newTop, newRight - newLeft,
                           Coords((newTop, newLeft), (newTop, newRight), (newBottom, newLeft), (newBottom, newRight)),
                           newBox)
    newBoxfinal.value = boxes[0].value
    newBoxfinal.merged = 1
    return newBoxfinal


def getCharacterBoxes(finalArray, className):
    tempPixels = []
    queue = []
    checkedPixels = []
    boxes = []
    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            if finalArray[i][j] == BLACK and [i, j] not in checkedPixels:
                checkedPixels.append([i, j])
                tempPixels.append([i, j])
                queue = addToQueue(i, j, queue)
                for point in queue:
                    if 50 > point[0] >= 0 and 50 > point[1] >= 0:
                        if finalArray[point[0]][point[1]] == BLACK and [point[0], point[1]] not in checkedPixels:
                            checkedPixels.append([point[0], point[1]])
                            tempPixels.append([point[0], point[1]])
                            queue = addToQueue(point[0], point[1], queue)
                box = getBoundingBox(finalArray, tempPixels)

                if box is not None:
                    box.value = className
                    boxes.append(box)
                tempPixels = []
                queue = []
    if len(boxes) > 1:
        return mergeBoxes(boxes)
    if len(boxes) == 0:
        print(finalArray)
        return None

    return boxes[0]


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
    return scaled


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
                    projectionVertical[i] + projectionVertical[i + 1] + projectionVertical[i + 2] + projectionVertical[
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
        # featureArray[index] = featureArray[index] + box.zones
        index += 1

    with open('test.csv', 'w') as f:
        write = csv.writer(f)
        for row in featureArray:
            write.writerow(row)


classes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
           "w",
           "x", "y", "z", "A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_", "I_", "J_", "K_", "L_", "M_", "N_", "O_",
           "P_", "Q_", "R_", "S_",
           "T_", "U_", "W_", "X_", "Y_", "Z_", "pi", "delta", "notEq", "(", "[", "{", "smal", ")", "]", "}", "mod",
           "big",
           "contains", "aprox", "sum", "prod", "sqrt", "beta", "dotDiv", "slash", "minus", "smalEq", "bigEq",
           "alpha", "=", "!", "+", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "divLine",
           ",", "dot"]
imageArray = []
for cl in classes:
    imageArray.append([cl, load(str(cl))])
boxes = []

for img in imageArray:
    temp = []
    temp2 = []
    temp3 = []

    for i in range(0, len(img[1])):
        temp.append(transformToGreyscale(img[1][i]))
    for i in range(0, len(temp)):
        temp2.append(applyTreshhold(temp[i]))
    for i in range(0, len(temp2)):
        temp3.append(reformArray(temp2[i]))
    for i in range(0, len(temp3)):
        boxes.append(getCharacterBoxes(temp3[i], img[0]))

for b in boxes:
    b = resize(b, len(b.pixels[0]), len(b.pixels), 25, 25)
boxes = getProjectionHorizontal(boxes)
boxes = getProjectionVertical(boxes)
boxes = getZones(boxes)
boxes = getColumnCrossings(boxes)
boxes = getRowCrossings(boxes)
boxes = exportToFile(boxes)

# print (imageArray[0][1])
