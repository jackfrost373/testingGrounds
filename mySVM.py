
import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


######################
# Create training dataset

N = 100 # number of training data points for each sample (=j/2)
D = 2 # dimensionality of data points (=nWeights = i)

dataA = np.random.rand(N,D)
dataB = np.random.rand(N,D)

# create difference between distributions
biasA = +0.0
biasB = +0.5
factor = 1.
for i in range(N) :
  for d in range(D) :
    dataA[i][d] = (dataA[i][d] + biasA) * factor
    dataB[i][d] = (dataB[i][d] + biasB) * factor

# desired output for supervised learning (0 or 1)
data = [[datA, 1] for datA in dataA ] + [[datB, -1] for datB in dataB ]
random.shuffle(data) # for fun
n = len(data)
# data looks like [ [[valx, valy, ...] , d] , [[valx, valy, ...] , d] , ... ]



#############################
# define SVM

# bias of threshold. Good guess is the expected separation.
b = float(factor/2)  
#b = 0 

# Hyperplane is defined as w . x - b = 0 .
#  b / ||w|| is the offset of the hyperplane along vector w.
#  if linearly separable make 2 planes: w.x-b = 1 for A, w.x-b = -1 for B.
#  for optimal margin, maximize distance (2/||w||) --> minimize ||w||.
#  --> yi(w.xi - b) >= 1 for all datapoints i. Finding minimal w, and b, determine our classifier.

# For not linearly separable problems: 'hinge loss': max(0, 1-yi(w.xi-b)).
#   if separable: result is 0. if not: proportional to sum of distances of wrongly classified points to plane.
#  --> minimize 1/n SUMi[ max(0, 1-yi(w.xi -b) ] + l||w||^2.
#  l = tradeoff between increasing margin-size (points away from plane) vs putting data points xi on OK side of margin.
#   --> make small to ensure OK classification.

#lam = 0.01

def f(w,point,y) :
  # function to minimize
    val = 1-y*np.inner(w,point) - b
    return max(0, 1-val)

# for gradient descent, calculate derivative of f:
#  s(z) = max(0, 1 - yz)
#  g(w) = x.w
#  --> s( g(w) ) = max(0, 1 - y*(x.w) ) is our function.
#  Now ds(g(w)) / dwi = (ds/dg)(dg/dwi)
#  - ds(g) / dg = -y [if x.w < 1] or 0 [if x.w > 1]
#  - dg(w)/dwi = xi
#  --> ds(g(w)) / dwi = -yxi [if yx.w < 1], or 0 [if y(x.w > 1)]

def gradf(w,point,y) :
  # gradient of f at point
  test = y*np.inner(w,point)
  if(test<1)  : return [-y * xi for xi in point]
  if(test>=1) : return [0 for xi in point]



################################
# visualisation tools

def plotLine(x, rc, offset) :
  return rc * x + offset

pointcolours = {1 : 'b', -1 : 'r'}

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

# initialize weights
w = [0.001]*D

maxiter = 5000    # max. iterations over data
minDelta = 0.01   # max. improvement before stop
step = 0.2     # how fast may the weights change at each update
totLoss0 = np.inf
change = np.inf

for loop in range(maxiter) :
  print "LOOP " + str(loop)

  totLoss = 0
  totGrad = [0,0]

  # loop over all training data points
  for j in range(len(data)) :
    #print "using training point N = " + str(j)
    [tp,d] = data[j] # training point tp and desired outcome d

    # print 
    #print "====================="
    #print " Point: " + str(tp) + " (exp: " + str(d) + ")"

    # evaluate function
    loss = f( w , tp , d)  
    totLoss += loss
    #print " Loss: %.3f"%(loss)

    # gradient descent
    grad = gradf( w, tp, d)
    totGrad = [totGrad[i] + grad[i] for i in range(len(grad))]
    #w = [(w[i] - step * grad[i]/np.linalg.norm(grad) / (loop+1)) for i in range(len(w))] # update weights

    # output
    #plotWeights(w,j)

  #print "Total one-loop loss and grad:"
  #print totLoss
  gradLen = sqrt(sum( [i**2 for i in totGrad] ))
  theGrad = [i / gradLen for i in totGrad]
  #print theGrad

  change = totLoss0 - totLoss
  print "--> Delta = " + str(change)
  totLoss0 = totLoss
  
  # update weights
  w = [(w[i] - step * theGrad[i] / (loop+1)) for i in range(len(w))] # update weights
  print " Weights: %.4f , %.4f"%(w[0],w[1])
  plotWeights(w,j)

  if(abs(change) < minDelta) : 
    print "minDelta reached"
    break
  

# plot result
plotWeights(w)










