import csv
import logging
import sys

import numpy as np
import  pandas as pd
np.set_printoptions(threshold=sys.maxsize)
with open('trainingData.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
cleanData = []
for i in data:
    if i != []:
        cleanData.append(i)
for a in cleanData:
    del(a[0])
print(data)
tem=np.array(cleanData)
df = pd.DataFrame(tem)
t= df.corr()
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(t)



