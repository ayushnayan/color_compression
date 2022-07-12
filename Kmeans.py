import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import time

img = cv.imread('image3.jpg')
plt.imshow(img)
# plt.show()

image = np.array(img)
h,w,_ = image.shape

class KMean:
	def __init__(self,niters,k):
		self.niters = niters
		self.k = k
	def fit(self,X):
		self.X = X
		self.m,self.n, _ = self.X.shape
		self.X = self.X.reshape(h*w,3)
	def initialise_centroids(self):
		self.cluster = []
		for i in range(self.k):
			self.cluster.append(random.sample(range(256), 3))
		self.cluster = np.asarray(self.cluster)
		self.cluster = self.cluster.reshape((self.k,3))		

	def assign_cluster(self):
		for _ in range(self.niters):
			distances = []
			for i in range(self.k):
				ls = np.array(self.X-self.cluster[i])
				ls = ls**2
				ls = np.sum(ls,axis=1)
				distances.append(ls)
			distances = np.array(distances)	
			distances = distances.T
			self.indx = np.argmin(distances,axis=1)
			for i in range(self.k):
				self.cluster[i] = np.average(self.X[np.where(self.indx == i)],axis = 0)
	def predict(self):
		compressed_img = img.copy()
		for i in range(self.m):
			for j in range(self.n):
				compressed_img[i][j][0] = self.cluster[self.indx[i*w+j]][0]
				compressed_img[i][j][1] = self.cluster[self.indx[i*w+j]][1]
				compressed_img[i][j][2] = self.cluster[self.indx[i*w+j]][2]
		return compressed_img			



km = KMean(10,5)
start = time.time()
km.fit(image)
km.initialise_centroids()
km.assign_cluster()
im = km.predict()
end = time.time()
print(end-start)
plt.imshow(im[:,:,[2,1,0]])
plt.show()
cv.imwrite('image1.jpg',im)	