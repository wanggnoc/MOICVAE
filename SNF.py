import numpy as np
import copy	

def FindDominantSet(W,K):
	m,n = W.shape
	DS = np.zeros((m,n))
	for i in range(m):
		index =  np.argsort(W[i,:])[-K:] # get the closest K neighbors 
		DS[i,index] = W[i,index] # keep only the nearest neighbors 

	#normalize by sum 
	B = np.sum(DS,axis=1)
	B = B.reshape(len(B),1)
	DS = DS/B
	return DS

def normalized(W,ALPHA):
	m,n = W.shape
	W = W+ALPHA*np.identity(m)
	return (W+np.transpose(W))/2

def SNF(Wall,K,t,ALPHA=1):
	C = len(Wall)
	m,n = Wall[0].shape
	for i in range(C):
		B = np.sum(Wall[i],axis=1)
		len_b = len(B)
		B = B.reshape(len_b,1)
		Wall[i] = Wall[i]/B
		Wall[i] = (Wall[i]+np.transpose(Wall[i]))/2
	newW = []
	for i in range(C):
		newW.append(FindDominantSet(Wall[i],K))#K表示邻居
		
	Wsum = np.zeros((m,n))#相似矩阵相加
	for i in range(C):#C表示相似矩阵个数
		Wsum += Wall[i]

	for iteration in range(t):#t表示迭代次数
		Wall0 = []
		for i in range(C):#C代表相似度矩阵个数
			temp = np.dot(np.dot(newW[i], (Wsum - Wall[i])),np.transpose(newW[i]))/(C-1)
			Wall0.append(temp)

		for i in range(C):
			Wall[i] = normalized(Wall0[i],ALPHA)

		Wsum = np.zeros((m,n))
		for i in range (C):
			Wsum+=Wall[i]

	W = Wsum/C
	B = np.sum(W,axis=1)
	B = B.reshape(len(B),1)
	W/=B
	W = (W+np.transpose(W)+np.identity(m))/2
	return W






