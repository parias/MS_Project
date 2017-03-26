#import matplotlib.pyplot as plt
from sklearn import preprocessing as p
from os import listdir
import os, csv, glob
import numpy as np
from numpy import mean, std, absolute


#Used to plot figure
figures = []

# f = '/home/arias/walkingdata2/Walking\ Data/csv
#f = listdir('./')[0]

#print(glob.glob('*.csv'))

# X_train = np.array((-1.7431294099999999, 4.6144323000000007, 7.9730880000000006, 1.4377222948387014, 2.7279437935598505, 2.479281955779939, 1.0599271781999999, 2.0640492660000005, 1.9637870399999995)
#allFV = np.array((0,0,0,0,0,0,0,0,0))
#gloabl allFV = None
allFV = np.array((0,0,0,0,0,0,0,0,0))
#user1 = 

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
    #print("Extracting Mean from " + str(filename))
    for s in range(0, len(x), 50):
        meanX.append(mean(x[s:s+split]))
        meanY.append(mean(y[s:s+split]))
        meanZ.append(mean(z[s:s+split]))

def extractSTD(x, y, z, meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename):
    #print("Extracting Standard Deviation from " + str(filename))
    for s in range(0, len(x), 50):
        stdX.append(std(x[s:s+split]))
        stdY.append(std(y[s:s+split]))
        stdZ.append(std(z[s:s+split]))

def extractMAD(x, y, z, meanX, meanY, meanZ, stdX, stdY, stdZ, madX, madY, madZ, filename):
    #print("Extracting Median Absolute Deviation from " + str(filename))
    for s in range(0, len(x), 50):
        madX.append(mad(x[s:s+split]))
        madY.append(mad(y[s:s+split]))
        madZ.append(mad(z[s:s+split]))

def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)

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
    print("Extracting Feature Vector from " + str(filename))
    fv = []
    #print(len(meanX), len(meanY), len(meanZ), len(stdX), len(stdY), len(stdZ), len(madX), len(madY), len(madZ))
    for i in range(len(meanX)):
        fv.append([meanX[i], meanY[i], meanZ[i], stdX[i], stdY[i], stdZ[i], madX[i], madY[i], madZ[i]])
#    print(len(fv))
    return np.array(fv)

def printFV(fv, filename):
    #f = open('feature_vectors/' + filename + '.txt', 'w')
    #print(fv)
    a = np.array((0,0,0,0,0,0,0,0,0))
    c = 1
    np.savetxt("./feature_vectors_np/" + filename, fv, delimiter=',')
    #for i in fv:
       #f.write(str(i))
       #print(type(i))
    #   a = np.vstack((a, i))
       #allFV = np.vstack((allFV, i))
       #print(c)
       #c+=1
    #print(a)
    #max_abs_scaler = p.MaxAbsScaler()
    #print(a)
    #scaled = max_abs_scaler.fit_transform(a)
    #print(scaled)
    #f.close()

# np.loadtext()
def loadFV():
    #TODO
    print("TODO")
    files = []

    #sort files numerically
    for file in glob.glob('*.csv'

def normalizeFV():
    max_abs_scaler = p.MaxAbsScaler()
    X_train_maxabs = max_abs_scaler.fit_transform(X_train)

def tryint(s):
    try:
        return s
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    return sorted(l, key=alphanum_key)



if __name__ == "__main__":
    #readFile()
    #print(allFV)
    loadFV()
