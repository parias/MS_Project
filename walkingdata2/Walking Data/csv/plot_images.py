import matplotlib.pyplot as plt
import os
import re
import csv
import glob


#Images are sets of 100 points
split = 100

# Counter for images used in naming
def plotimage(f):

    with open(f, 'r') as csv_file:
        
        # User data to arrays
        step = []
        x = []
        y = []
        z = []

        print(f)
        plots = csv.reader(csv_file, delimiter=',')
        for row in plots:
            step.append(float(row[0]))
            x.append(float(row[1]))
            y.append(float(row[2]))
            z.append(float(row[3]))

        i = 1 # image number
        file_num = alphanum_key(f)[1]

        # Embarrassingly Parallel...
        for s in range(0,len(step),50):
            # print(s)
            plt.plot(step[s:s+split],x[s:s+split])
            plt.plot(step[s:s+split],y[s:s+split])
            plt.plot(step[s:s+split],z[s:s+split])
            plt.axis('off')
            plt.savefig('./images/' + str(file_num) + '_' + str(i) +  '.png')
            #figure = plt.figure()
            plt.clf()
            plt.close()
            print('./images/' + str(file_num) + '_' +  str(i) + '.png')
            i += 1
 
def load_files():
    files_temp = []
    for f in glob.glob('*.csv'):
        files_temp.append(f)
    return sort_nicely(files_temp)


def try_int(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    return [try_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    return sorted(l, key=alphanum_key)

 
if __name__ == '__main__':
    print('Executing Main()')
    files = load_files()
    print(files)
    for f in files:
        plotimage(f)
