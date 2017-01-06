
import numpy as np
import random

""" parameters to change """
#expected output: not(x1),x2,not(x4),x5
lits = [3,3,3,3,3,3] #2:pos 1:neg 3:none
rounds = 100


train = []
for i in range(rounds):
    line = []
    y = 1
    for j in range(len(lits)):
        cur = random.choice([0,1])
        line.append(cur)
        if((cur==0 and lits[j]==2) or (cur==1 and lits[j]==1)):
            y = 0;
    line.append(y)
    train.append(line)
    
np.savetxt("trainingData\example1.txt",train,"%i")