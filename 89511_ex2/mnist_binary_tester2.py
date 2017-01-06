__author__ = 'GROSSMAN'

import numpy as np
import random
import matplotlib.pyplot as plt

data= np.loadtxt("test_data_1_vs_8.dat")
X=data[:, 1:]
Y=data[:, 0]

#convert to hyperline
for y in range(len(Y)):
    if (Y[y]==8):
        Y[y]=1
    else:
        Y[y]=-1

#initiate w
W=[]
for i in range(len(X[0])):
    W=np.append(W,0)

#perceptron
count=0
for j in range(len(X)):
    #taking randomly an index
    rand=random.randint(0,len(X)-1)
    #goal: (yi<w,xi>) >0 for every i
    if (Y[rand]*np.dot(W,X[rand])<=0):
        #means it's an error,fix it
        count=count+1
        W=W+Y[rand]*X[rand]

tmp = 1/(1+np.exp(-10*W/W.max()))
plt.imshow(tmp.reshape(28,28),cmap="gray")
plt.draw()
plt.savefig("final_weight_vector")
plt.imshow(X[1, :].reshape(28, 28), cmap="gray")
plt.draw()

f = open('perceptron_mistakes.txt','w')
f.write("number of mistakes= "+str(count))
f.close()