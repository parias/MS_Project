import matplotlib.pyplot as plt
from os import listdir
import os
import csv
import glob

# User data to arrays
step = []
x = []
y = []
z = []

#Used to plot figure
figures = []

# f = '/home/arias/walkingdata2/Walking\ Data/csv
#f = listdir('./')[0]

print(glob.glob('*.csv'))

def readFile():
    #for filename in glob.glob('*.csv'):
        print(filename)
        #with open(f, 'r') as csvfile:
        #    plots = csv.reader(csvfile, delimiter=',')
        #    for row in plots:
        #        step.append(float(row[0]))
        #        x.append(float(row[1]))
        #        y.append(float(row[2]))
        #        z.append(float(row[3]))

#Images are sets of 100 points
split = 100

def extractFeatures():
    meanX = []
    meanY = []
    meanZ = []
    for s in range(0, len(step), 50):
        meanX.append(sum(x[s:s+split]), len(x[s:s+split]))
        meanY.append(sum(y[s:s+split]), len(y[s:s+split]))
        meanZ.append(sum(z[s:s+split]), len(z[s:s+split]))

# Counter for images used in naming
i = 1
def plotimage():
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

if __name__ == "__main__":
    readfile()
