import numpy as np
import mlp

anddata = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
xordata = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
#ordata = np.array([[0,0,0,0],[0,0,1,1],[0,1,0,1],[1,0,0,1],[1,0,1,1],[1,1,0,1],[0,1,1,1],[1,1,1,1]])

print '...AND...'
weights1, weights2 = mlp.init_mlp(anddata[:,0:2],anddata[:,2:3],2)
mlp.mlptrain(anddata[:,0:2],anddata[:,2:3],0.25,1001, weights1, weights2)


print '...XOR...'
weights1, weights2 = mlp.init_mlp(xordata[:,0:2],xordata[:,2:3],2)
mlp.mlptrain(xordata[:,0:2],xordata[:,2:3],0.25,5001, weights1, weights2)

#print '...OR...'
#weights1, weights2 = mlp.init_mlp(ordata[:,0:3],ordata[:,3:4],2)
#mlp.mlptrain(ordata[:,0:3],ordata[:,3:4],0.25,5001, weights1, weights2)

