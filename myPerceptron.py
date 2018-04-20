
import random
import numpy as np
import matplotlib.pyplot as plt


######################
# Create training dataset

N = 100 # number of training data points for each sample (=j/2)
D = 2 # dimensionality of data points (=nWeights = i)

dataA = np.random.rand(N,D)
dataB = np.random.rand(N,D)

# create difference between distributions
biasA = +0.0
biasB = +0.5
factor = 10.
for i in range(N) :
  for d in range(D) :
    dataA[i][d] = (dataA[i][d] + biasA) * factor
    dataB[i][d] = (dataB[i][d] + biasB) * factor

# desired output for supervised learning (0 or 1)
data = [[datA, 1] for datA in dataA ] + [[datB, 0] for datB in dataB ]
random.shuffle(data) # for fun
# data looks like [ [[valx, valy, ...] , d] , [[valx, valy, ...] , d] , ... ]



#############################
# define perceptron

# bias of threshold. Good guess is the expected separation.
b = float(factor/2)  

def f(w,x) :
  val = np.inner(w,x)
  if ( val + b ) > 0 : return 1
  return 0

# initialize weights
w = [0.01]*D




################################
# visualisation tools

def plotLine(x, rc, offset) :
  return rc * x + offset

pointcolours = {1 : 'b', 0 : 'r'}

def plotWeights(w,n=2*N) :
  if( not len(w) == 2 ) : return
  if( w[1] == 0 ) : return

  plt.clf()
  pmin = 0.
  pmax = (1. + max(biasA,biasB)) * factor * 1.2
  plt.gca().set_xlim([ pmin, pmax ])
  plt.gca().set_ylim([ pmin, pmax ])

  if(n==2*N) : 
    plt.scatter(np.transpose(dataA)[0], np.transpose(dataA)[1], c='b')
    plt.scatter(np.transpose(dataB)[0], np.transpose(dataB)[1], c='r')
  else :
    for i in range(n+1) :
      plt.scatter(data[i][0][0], data[i][0][1], c=pointcolours[data[i][1]])

  #  edge is at w1*x + w2*y + b = 0
  rc = -w[0]/w[1]
  offset = (-b)/w[1]
  print " Printing line: y = %.3f x + %.3f"%(rc,offset)
  plt.plot( [pmin,pmax] , [plotLine(pmin, rc, offset), plotLine(pmax, rc, offset)] )
  plt.show()





##############################
# Train classifier on data

maxiter = 10         # max. iterations over data
errorThreshold = 0.1 # convergence criterium
weightChange = 0.005 # how fast may the weights change at each update

for loop in range(maxiter) :
  totdiff = 0.

  # loop over all training data points
  for j in range(len(data)) :
    print "using training point N = " + str(j)
    [tp,d] = data[j] # training point tp and desired outcome d
    res = f( w , tp )   # result of the perceptron
    diff = (d - res)    # difference with desired outcome
    totdiff += np.abs(diff)

    # update weights
    for i in range(D) :
      # [old weight] + Delta(y) * x
      w[i] = w[i] + (diff * data[j][0][i]) * weightChange
    
    # output
    #print "diff = %.3f"%(diff)
    #print " > weights: %.3f , %.3f"%(w[0],w[1])
    #plotWeights(w,j)

  totdiff /= len(data)
  print "Iteration " + str(loop)
  print " - TOTDIFF = " + str(totdiff)
  print " --> weights: %.3f , %.3f"%(w[0],w[1])
  if(totdiff < errorThreshold) : 
    print "Converged in " + loop + " iterations over data"
    break


# plot result
plotWeights(w)










