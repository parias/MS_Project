from sklearn import preprocessing as p
import os
import re
import csv
import glob
import numpy as np
from numpy import mean, std, absolute


#Used to plot figure
figures = []

allFV = np.array((0,0,0,0,0,0,0,0,0))

def readFile():
    for filename in glob.glob('*.csv'):
            with open(filename, 'r') as csvfile:
                # User data
                meanX = []
                meanY = []
                meanZ = []
                stdX = []
                stdY = []
                stdZ = []
                madX = []
                madY = []
                madZ = []
                x = []
                y = []
                z = []
                plots = csv.reader(csvfile, delimiter=',')
                for row in plots:
                    #step.append(float(row[0]))
                    x.append(float(row[1]))
                    y.append(float(row[2]))
                    z.append(float(row[3]))
                extractMean(x,y,z, meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename)
                extractSTD(x,y,z, meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename)
                extractMAD(x,y,z, meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename)
                fv = extractFV(meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename)
                printFV(fv, filename)
                #allFV = np.vstack((allFV, fv))

#Images are sets of 100 points
split = 100

def extractMean(x, y, z, meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename):
    print("X,Y,Z =  ",str(len(x)), str(len(y)), str(len(z)))
    # print("Extracting Mean from " + str(filename))
    for s in range(0, len(x), 50):
        meanX.append(mean(x[s:s+split]))
        meanY.append(mean(y[s:s+split]))
        meanZ.append(mean(z[s:s+split]))

def extractSTD(x, y, z, meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename):
    # print("Extracting Standard Deviation from " + str(filename))
    for s in range(0, len(x), 50):
        stdX.append(std(x[s:s+split]))
        stdY.append(std(y[s:s+split]))
        stdZ.append(std(z[s:s+split]))

def extractMAD(x, y, z, meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename):
    # print("Extracting Median Absolute Deviation from " + str(filename))
    for s in range(0, len(x), 50):
        madX.append(mad(x[s:s+split]))
        madY.append(mad(y[s:s+split]))
        madZ.append(mad(z[s:s+split]))

# Wrong mad
# def mad(data, axis=None):
#     return mean(absolute(data - mean(data, axis)), axis)

def mad(a):
    med = np.median(a)
    return np.median(np.absolute(a - med))


# Counter for images used in naming
def plotimage():
    i = 1
    for s in range(0,len(step),50):
       #print(z)
       plt.plot(step[s:s+split],x[s:s+split])
       plt.axis('off')
       plt.savefig(os.path.splitext(f)[0] + '_' + str(i) + '.png')
       #figure = plt.figure()
       plt.clf()
       plt.close()
       print(os.path.splitext(f)[0] + '_' + str(i) + '.png')
       i += 1

def extractFV(meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename):
    # print(len(meanX))
    print("Extracting Feature Vector from " + str(filename))
    fv = []
    for i in range(len(meanX)):
        # fv.append([alphanum_key(filename)[1], meanX[i], meanY[i], meanZ[i], stdX[i], stdY[i], stdZ[i], madX[i], madY[i], madZ[i]])
        fv.append([meanX[i], meanY[i], meanZ[i], stdX[i], stdY[i], stdZ[i], madX[i], madY[i], madZ[i]])
#    print(len(fv))
    return np.array(fv)

def printFV(fv, filename):
    np.savetxt("./feature_vectors/" + filename, fv, delimiter=',')

# np.loadtext()
def loadFV():
    # TODO
    print("TODO")
    files = []


def normalizeFV():
    max_abs_scaler = p.MaxAbsScaler()
    X_train_maxabs = max_abs_scaler.fit_transform(X_train)

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    return sorted(l, key=alphanum_key)



if __name__ == "__main__":
    readFile()
    # loadFV()
