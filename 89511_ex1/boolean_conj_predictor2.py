import numpy as np
import argparse
import sys

#parsing argument to get the path of input file. assume it is the right path.
parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
#collecting matrix out of the input file
trainingSet=np.loadtxt("trainingData\example1.txt")
#collecting num of columns
numcols = len(trainingSet[0])

#initiation of 2 arrays represents literals and their negatives
literals=[]
negative=[]
#initiate all x, not the last column(represents y).
for i in range(numcols-2):
    literals.append(1)
    negative.append(1)

#iterating on trainingSet rows
for i in range(len(trainingSet)):
    #same,but on columns without the y column which represents the true answer
    for j in range(numcols-2):
        #in case of xiTrue=xiLiteral, predict true and continue to next iteration
        if ((trainingSet[i][j]==literals[j])and trainingSet[i][j]!=negative[j]):
            answer=1
        #in case literal!=xiTrue,we predict that statement is false
        else:
            answer=0
            break
    #answers are different
    if (answer!=trainingSet[i][-1]):
        for k in range(numcols-2):
            if (trainingSet[i][k]==1):
                #remove negative
                negative[k]=2;
            if (trainingSet[i][k]==0):
                #remove literal
                literals[k]=2;
#variable that check if we stand at starting point
isFirst=True
for i in range(numcols-2):
    if (negative[i]==1):
        if (isFirst):
            sys.stdout.write("not(x"+repr(i+1)+")")
        else:
            sys.stdout.write(",not(x"+repr(i+1)+")")
    elif (literals[i]==1):
        if(isFirst):
            sys.stdout.write("x"+repr(i+1))
        else:
            sys.stdout.write(",x"+repr(i+1))
    #means that we did print something
    if (negative[i]!=2 or literals[i]!=2):
        isFirst=False
        

