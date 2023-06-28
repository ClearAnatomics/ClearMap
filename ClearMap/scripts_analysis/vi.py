# Variation of information (VI)
#
# Meila, M. (2007). Comparing clusterings-an information
#   based distance. Journal of Multivariate Analysis, 98,
#   873-895. doi:10.1016/j.jmva.2006.11.013
#
# https://en.wikipedia.org/wiki/Variation_of_information
import numpy as np
from math import log
import random


def getpartition(bs):
  bs1u = np.unique(bs)
  p1 = []
  for b in bs1u:
    p1.append(np.asarray(bs == b).nonzero()[0].tolist())
  return p1

def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.0
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n
      if r > 0.0:
        sigma += r * (log(r / p, 2) + log(r / q, 2))
  return abs(sigma)

def comparePartition(bs1, bs2):
  bs1u = np.unique(bs1)
  bs2u = np.unique(bs2)
  p1 = []
  p2 = []
  for b in bs1u:
    p1.append(np.asarray(bs1 == b).nonzero()[0].tolist())
  for b in bs2u:
    p2.append(np.asarray(bs2 == b).nonzero()[0].tolist())

  v = variation_of_information(p1, p2)
  return v


def computeclusterweights(p1,u1, p2, u2, p2t, u2t):
   W=[]
   for u in u2:
     n2=np.asarray(p2==u).nonzero()[0]
     n1=np.unique(p1[n2])
     n3=np.asarray(p2t==p2t[n2][0]).nonzero()[0]
     
     W.append(n1.shape[0]/n3.shape[0])
       
   sumW=np.sum(np.array(W))
   return W/sumW 
     
def selectSuitedNeighbour(p1, p2, u2, p2i, c, conn):
  # #get neightbours clusters
  # outerEdges=conn[np.logical_or(p2i[conn[:,0]]==c, p2i[conn[:, 1]]==c)].flatten()
  # uoe, coe=np.unique(p2[outerEdges], return_counts=True)
  # 
  # # coe = coe[uoe != c]
  # # uoe = uoe[uoe != c]
  W=[]
  for oe in u2:
    new_clus=np.logical_or(p2==oe, p2i==c).nonzero()[0]

    sbm_eq=p1[new_clus]
    u, count=np.unique(sbm_eq, return_counts=True)
    if count.shape[0]>1:
      W.append(1/(np.sum(count)+np.sum(new_clus)))
    else:
      W.append(count[0])
  sumW = np.sum(np.array(W))
  # print(W, uoe)
  return W / sumW, u2
    



def mergePartition(g, p1, p2, maxstep=1000, epsilon=0.01):

  VIi=comparePartition(p1, p2)
  step=0
  e=1000
  p1t = p1.copy()
  p2t = p2.copy()
  p2i = p2.copy()
  conn=g.edge_connectivity()
  VI0 = VIi
  u1 = np.unique(p1t)
  u2 = np.unique(p2t)
  u2i , c2i= np.unique(p2t, return_counts=True)
  c2=c2i.copy()
  while (step<maxstep or e>=epsilon):
    print('########### step: ',step, e, VI0, u2.shape)
    # print(u2, c2)
    print('###')
    step=step+1
    W=computeclusterweights(p1t,u1, p2i, u2i, p2t, u2)
    n = random.choices(range(u2i.shape[0]), weights=W)
    c=u2i[n]
    # print(c2i)
    # print(W)
    print("trying moving cluster ", c, "of size ", np.sum(p2i==c))

    W_s, neighbours=selectSuitedNeighbour(p1t, p2t, u2, p2i, c, conn)
    # print(neighbours)
    # print(W_s)
    n = random.choices(range(neighbours.shape[0]), weights=W_s)
    n_t=neighbours[n]
    print("in cluster ", n_t, "of size ", np.sum(p2t == n_t))
    m=np.max(p2t)
    p2t_t=p2t.copy()
    p2t_t[p2t==n_t]=m
    p2t_t[p2i==c]=m
    
    VIt=comparePartition(p1, p2t_t)
    e=VI0-VIt
    if e>0:
      p2t=p2t_t
      VI0=VIt
      print(c, n_t, e, 'accepted')
      u1, c1 = np.unique(p1t, return_counts=True)
      u2, c2 = np.unique(p2t, return_counts=True)
      if u2.shape[0]<=u1.shape[0]:
        step=1001
        e=0
    else:
      print(e, 'rejected')
        
        
  
  print(VIi , VIt)
  return p2t_t
    
  
  
  

#
#
# # Identical partitions
# X1 = [ [1,2,3,4,5], [6,7,8,9,10] ]
# Y1 = [ [1,2,3,4,5], [6,7,8,9,10] ]
# print(variation_of_information(X1, Y1))
# # VI = 0
#
# # Similar partitions
# X2 = [ [1,2,3,4], [5,6,7,8,9,10] ]
# Y2 = [ [1,2,3,4,5,6], [7,8,9,10] ]
# print(variation_of_information(X2, Y2))
# # VI = 1.102
#
# # Dissimilar partitions
# X3 = [ [1,2], [3,4,5], [6,7,8], [9,10] ]
# Y3 = [ [10,2,3], [4,5,6,7], [8,9,1] ]
# print(variation_of_information(X3, Y3))
# # VI = 2.302
#
# # Totally different partitions
# X4 = [ [1,2,3,4,5,6,7,8,9,10] ]
# Y4 = [ [1], [2], [3], [4], [5], [6], [7], [8], [9], [10] ]
# print(variation_of_information(X4, Y4))
# # VI = 3.322 (maximum VI is log(N) = log(10) = 3.322)